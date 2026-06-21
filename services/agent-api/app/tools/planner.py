from dataclasses import dataclass, field
import re
from typing import Any

from app.tools.allowlist import TOOL_SPECS, get_tool_spec
from app.tools.catalog import discover_tools, normalize_query, tool_metadata

MAX_SALES_TOOL_STEPS = 6


@dataclass(frozen=True)
class ToolPlanStep:
    tool_name: str
    arguments: dict[str, Any]
    fields: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "fields": self.fields,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ToolPlan:
    strategy: str
    candidate_tools: list[str]
    steps: list[ToolPlanStep] = field(default_factory=list)
    semantic_plan: dict[str, Any] | None = None
    available_tools: list[dict[str, Any]] = field(default_factory=list)
    rejected_tools: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "candidate_tools": self.candidate_tools,
            "steps": [step.to_dict() for step in self.steps],
            "semantic_plan": self.semantic_plan,
            "available_tools": self.available_tools,
            "rejected_tools": self.rejected_tools,
        }


class ToolPlanner:
    """Build conservative, auditable tool plans from the WARO tool catalog."""

    def plan_sales(
        self,
        *,
        question: str,
        period: dict[str, str],
        scopes: tuple[str, ...],
        group_by: str | None = None,
        answer_style: str | None = None,
        semantic_plan: dict[str, Any] | None = None,
    ) -> ToolPlan:
        normalized = normalize_query(question)
        discovery = discover_tools(
            question,
            preferred_domain="sales",
            scopes=scopes,
            limit=12,
        )
        available_tools = discovery["available"]
        rejected_tools = discovery["rejected"]
        candidate_names = [str(tool["name"]) for tool in available_tools]

        requested_tools = self._semantic_tool_names(semantic_plan)
        request_kind = self._semantic_text(semantic_plan, "request_kind")
        dimensions = self._semantic_string_list(semantic_plan, "dimensions")
        requested_metrics = self._semantic_string_list(semantic_plan, "requested_metrics")
        sort_field = self._semantic_text(semantic_plan, "sort_field")
        requested_limit = self._semantic_limit(semantic_plan, default=20)
        metrics_group_by = group_by or self._infer_sales_group_by(normalized)
        if not metrics_group_by and request_kind == "daily_analysis":
            metrics_group_by = "date"
        if not metrics_group_by and "date" in dimensions:
            metrics_group_by = "date"
        if not metrics_group_by and "product" in dimensions:
            metrics_group_by = "product"
        sales_metrics_group_by = None if metrics_group_by == "product" else metrics_group_by
        selected_tool_names = self._select_context_tool_names(
            available_tools=available_tools,
            requested_tools=requested_tools,
            request_kind=request_kind,
            dimensions=dimensions,
            requested_metrics=requested_metrics,
            answer_style=answer_style,
            metrics_group_by=metrics_group_by,
            normalized=normalized,
            scopes=scopes,
        )
        metrics_arguments: dict[str, Any] = {
            "date-from": period["date_from"],
            "date-to": period["date_to"],
        }
        if sales_metrics_group_by:
            metrics_arguments["group-by"] = sales_metrics_group_by

        steps: list[ToolPlanStep] = []
        self._append_step(
            steps,
            ToolPlanStep(
                tool_name="waro.sales.metrics",
                arguments=metrics_arguments,
                fields=["data", "meta", "success"],
                reason="Primary sales metrics are required for sales analysis.",
            ),
        )

        if "waro.financial.products" in selected_tool_names:
            self._append_step(
                steps,
                ToolPlanStep(
                    tool_name="waro.financial.products",
                    arguments={
                        "sort-by": self._financial_sort(normalized, semantic_sort=sort_field),
                        "period": self._period_days(period),
                    },
                    fields=["products", "metrics", "insights"],
                    reason="Product financial context can explain sales performance.",
                ),
            )

        if "waro.analytics.food_cost" in selected_tool_names:
            self._append_step(
                steps,
                ToolPlanStep(
                    tool_name="waro.analytics.food_cost",
                    arguments={
                        "date-from": period["date_from"],
                        "date-to": period["date_to"],
                    },
                    fields=["data", "meta", "success"],
                    reason="Food-cost context can explain margin and profitability questions.",
                ),
            )

        if "waro.analytics.menu" in selected_tool_names:
            self._append_step(
                steps,
                ToolPlanStep(
                    tool_name="waro.analytics.menu",
                    arguments={
                        "date-from": period["date_from"],
                        "date-to": period["date_to"],
                        "limit": requested_limit,
                    },
                    fields=["data", "meta", "success"],
                    reason="Menu analytics can classify product portfolio performance.",
                ),
            )

        if "waro.menu.products" in selected_tool_names:
            self._append_step(
                steps,
                ToolPlanStep(
                    tool_name="waro.menu.products",
                    arguments={"limit": 50},
                    fields=["id", "name", "price", "isAvailable", "category"],
                    reason="Menu product context can clarify product-level questions.",
                ),
            )

        if "waro.customers.metrics" in selected_tool_names:
            self._append_step(
                steps,
                ToolPlanStep(
                    tool_name="waro.customers.metrics",
                    arguments={
                        "date-from": period["date_from"],
                        "date-to": period["date_to"],
                    },
                    fields=["summary", "top_customers"],
                    reason="Customer metrics can explain demand, retention, and frequency.",
                ),
            )
            if "waro.customers.list" in selected_tool_names:
                self._append_step(
                    steps,
                    ToolPlanStep(
                        tool_name="waro.customers.list",
                        arguments={
                            "date-from": period["date_from"],
                            "date-to": period["date_to"],
                            "sort-field": self._customer_sort_field(
                                normalized,
                                semantic_sort=sort_field,
                            ),
                            "sort-direction": "desc",
                            "limit": requested_limit,
                        },
                        fields=[
                            "customer_id",
                            "name",
                            "phone",
                            "order_count",
                            "total_spent",
                            "avg_ticket",
                            "last_order_date",
                            "waros_balance",
                        ],
                        reason="Customer ranking can identify the best customers for the selected period.",
                    ),
                )

        return ToolPlan(
            strategy="catalog_sales_planner_v1",
            candidate_tools=candidate_names,
            steps=steps,
            semantic_plan={
                **(semantic_plan or {}),
                "period": period,
                "group_by": metrics_group_by,
                "sales_metrics_group_by": sales_metrics_group_by,
                "answer_style": answer_style,
                "limit": requested_limit,
            },
            available_tools=available_tools,
            rejected_tools=rejected_tools,
        )

    def _append_step(self, steps: list[ToolPlanStep], step: ToolPlanStep) -> None:
        if len(steps) >= MAX_SALES_TOOL_STEPS:
            return
        if any(existing.tool_name == step.tool_name for existing in steps):
            return
        steps.append(step)

    def _has_scope(self, scope: str, scopes: tuple[str, ...]) -> bool:
        return scope in scopes

    def _semantic_tool_names(self, semantic_plan: dict[str, Any] | None) -> set[str]:
        tools = (semantic_plan or {}).get("tools", [])
        if not isinstance(tools, list):
            return set()
        names: set[str] = set()
        for tool in tools:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                names.add(tool["name"])
        return names

    def _semantic_text(self, semantic_plan: dict[str, Any] | None, key: str) -> str | None:
        value = (semantic_plan or {}).get(key)
        return value if isinstance(value, str) and value else None

    def _semantic_string_list(self, semantic_plan: dict[str, Any] | None, key: str) -> list[str]:
        value = (semantic_plan or {}).get(key)
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    def _semantic_limit(self, semantic_plan: dict[str, Any] | None, *, default: int) -> int:
        try:
            limit = int((semantic_plan or {}).get("limit", default))
        except (TypeError, ValueError):
            return default
        return max(1, min(limit, 50))

    def _select_context_tool_names(
        self,
        *,
        available_tools: list[dict[str, Any]],
        requested_tools: set[str],
        request_kind: str | None,
        dimensions: list[str],
        requested_metrics: list[str],
        answer_style: str | None,
        metrics_group_by: str | None,
        normalized: str,
        scopes: tuple[str, ...],
    ) -> set[str]:
        available_by_name = {
            str(tool.get("name")): tool for tool in available_tools if tool.get("name")
        }
        selected = requested_tools.intersection(available_by_name)
        for tool_name in requested_tools.difference(selected):
            spec = get_tool_spec(tool_name)
            if spec is not None and spec.scope in scopes:
                selected.add(tool_name)
        for name, tool in available_by_name.items():
            if name == "waro.sales.metrics":
                continue
            if self._tool_matches_plan(
                tool,
                request_kind=request_kind,
                dimensions=dimensions,
                requested_metrics=requested_metrics,
                answer_style=answer_style,
                metrics_group_by=metrics_group_by,
                normalized=normalized,
            ):
                selected.add(name)
        for spec in TOOL_SPECS.values():
            if spec.name in selected or spec.name in available_by_name or spec.scope not in scopes:
                continue
            if self._tool_matches_plan(
                tool_metadata(spec),
                request_kind=request_kind,
                dimensions=dimensions,
                requested_metrics=requested_metrics,
                answer_style=answer_style,
                metrics_group_by=metrics_group_by,
                normalized=normalized,
            ):
                selected.add(spec.name)
        if "waro.customers.list" in selected:
            selected.add("waro.customers.metrics")
        if request_kind == "customer_ranking" and "waro.customers.metrics" in available_by_name:
            selected.add("waro.customers.metrics")
            if "waro.customers.list" in available_by_name:
                selected.add("waro.customers.list")
        return selected

    def _tool_matches_plan(
        self,
        tool: dict[str, Any],
        *,
        request_kind: str | None,
        dimensions: list[str],
        requested_metrics: list[str],
        answer_style: str | None,
        metrics_group_by: str | None,
        normalized: str,
    ) -> bool:
        capabilities = tool.get("capabilities") if isinstance(tool.get("capabilities"), dict) else {}
        entity = str(capabilities.get("entity") or "")
        measures = self._string_set(capabilities.get("measures"))
        tool_dimensions = self._string_set(capabilities.get("dimensions"))
        tool_name = str(tool.get("name") or "")
        domain = str(tool.get("domain") or "")
        requested = set(requested_metrics)
        requested_dimensions = set(dimensions)
        if request_kind == "direct_metric" and tool_name not in requested:
            return False

        if entity == "product":
            wants_product = (
                request_kind == "product_ranking"
                or metrics_group_by == "product"
                or "product" in requested_dimensions
                or answer_style == "financial_analysis"
                or bool(requested.intersection(measures))
                or bool(requested.intersection({"product_ranking", "quantity_sold", "gross_profit", "product_profit"}))
                or self._has_discovery_match(tool)
            )
            return wants_product and (
                domain in {"financial", "analytics", "menu", "food_cost"}
                or bool(measures.intersection({"margin", "revenue", "quantity", "cost", "profit"}))
            )

        if entity == "customer" or domain == "customers":
            return (
                request_kind == "customer_ranking"
                or "customer" in requested_dimensions
                or bool(requested.intersection({"frequency", "customer_activity", "retention"}))
                or self._has_discovery_match(tool)
            )

        return tool_name in requested_metrics or tool_name in normalized

    def _has_discovery_match(self, tool: dict[str, Any]) -> bool:
        reasons = tool.get("reasons")
        return isinstance(reasons, list) and any(
            isinstance(reason, str) and reason.startswith("matched_terms:")
            for reason in reasons
        )

    def _string_set(self, value: Any) -> set[str]:
        if not isinstance(value, list | tuple | set | frozenset):
            return set()
        return {str(item) for item in value if isinstance(item, str)}

    def _infer_sales_group_by(self, normalized: str) -> str | None:
        if re.search(r"\b(hora|horas|hour)\b", normalized):
            return "hour"
        if re.search(r"\b(producto|productos|plato|platos|item|items)\b", normalized):
            return "product"
        if re.search(r"\b(pago|pagos|metodo|metodos|payment)\b", normalized):
            return "payment"
        if re.search(
            r"\b(por dia|por dias|por fecha|por fechas|diario|diaria|dia a dia|date)\b",
            normalized,
        ):
            return "date"
        return None

    def _customer_sort_field(self, normalized: str, *, semantic_sort: str | None = None) -> str:
        if semantic_sort in {"total_spent", "order_count", "last_order_date", "avg_ticket"}:
            return semantic_sort
        if re.search(r"\b(ticket|promedio)\b", normalized):
            return "avg_ticket"
        if re.search(r"\b(ordenes?|frecuencia|compran|compras)\b", normalized):
            return "order_count"
        return "total_spent"

    def _financial_sort(self, normalized: str, *, semantic_sort: str | None = None) -> str:
        if semantic_sort in {"margin", "revenue", "cost", "quantity"}:
            return semantic_sort
        if re.search(r"\b(margen|rentabilidad|rentables?|peor(?:es)?)\b", normalized):
            return "margin"
        if re.search(r"\b(ganancia|ganancias|utilidad|utilidades|profit)\b", normalized):
            return "revenue"
        if re.search(r"\b(cantidad|unidades|vendidos?)\b", normalized):
            return "quantity"
        if re.search(r"\b(costo|costos)\b", normalized):
            return "cost"
        if re.search(r"\b(ingreso|ingresos|revenue|ventas?)\b", normalized):
            return "revenue"
        return "margin"

    def _period_days(self, period: dict[str, str]) -> int:
        # Financial products accepts a day window, so keep this conservative when
        # a period is explicit but parsing fails for any reason.
        from datetime import date

        try:
            start = date.fromisoformat(period["date_from"])
            end = date.fromisoformat(period["date_to"])
        except (KeyError, ValueError):
            return 365
        return max(1, min(730, (end - start).days + 1))
