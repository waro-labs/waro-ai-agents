from dataclasses import dataclass, field
import re
from typing import Any

from app.tools.catalog import discover_tools, normalize_query

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
            limit=6,
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

        if (
            "waro.financial.products" in requested_tools
            or request_kind == "product_ranking"
            or answer_style == "financial_analysis"
            or metrics_group_by == "product"
            or "product" in dimensions
            or any(metric in requested_metrics for metric in {"product_ranking", "quantity_sold", "gross_profit", "product_profit"})
            or self._needs_financial_products(normalized)
        ) and self._has_scope("financial:read", scopes):
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

        if (
            "waro.analytics.food_cost" in requested_tools
            or self._needs_food_cost(normalized)
        ) and self._has_scope("analytics:read", scopes):
            self._append_step(
                steps,
                ToolPlanStep(
                    tool_name="waro.analytics.food_cost",
                    arguments={
                        "date-from": period["date_from"],
                        "date-to": period["date_to"],
                    },
                    fields=["product_id", "product_name", "food_cost_pct", "margin_pct"],
                    reason="Food-cost context can explain margin and profitability questions.",
                ),
            )

        if (
            "waro.analytics.menu" in requested_tools
            or self._needs_menu_analysis(normalized)
        ) and self._has_scope("analytics:read", scopes):
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

        if (
            "waro.menu.products" in requested_tools
            or self._needs_menu_products(normalized)
        ) and self._has_scope("menu:read", scopes):
            self._append_step(
                steps,
                ToolPlanStep(
                    tool_name="waro.menu.products",
                    arguments={"limit": 50},
                    fields=["id", "name", "price", "is_available", "category"],
                    reason="Menu product context can clarify product-level questions.",
                ),
            )

        if (
            "waro.customers.metrics" in requested_tools
            or request_kind == "customer_ranking"
            or "customer" in dimensions
            or self._needs_customer_metrics(normalized)
        ) and self._has_scope("customers:read", scopes):
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
            if (
                "waro.customers.list" in requested_tools
                or request_kind == "customer_ranking"
                or self._needs_customer_ranking(normalized)
            ):
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

    def _needs_financial_products(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(productos?|platos?|margen|rentabilidad|rentables?|ingresos?|costo|cantidad|top|peor(?:es)?)\b",
                normalized,
            )
            or re.search(r"\b(financier[ao]s?|utilidad|ganancia|ganancias|profit)\b", normalized)
        )

    def _needs_food_cost(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(food cost|costo de comida|margen|margenes|rentabilidad|rentables?|utilidad|ganancia|ganancias)\b",
                normalized,
            )
        )

    def _needs_menu_analysis(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(portafolio|menu|estrella|bajo rendimiento|performance|desempeno|productos?)\b",
                normalized,
            )
        )

    def _needs_menu_products(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(menu|disponibles?|precio|precios|categoria|categorias)\b",
                normalized,
            )
        )

    def _needs_customer_metrics(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(clientes?|compradores?|retencion|frecuencia|fidelidad|churn|recencia|recompra|mejores?)\b",
                normalized,
            )
        )

    def _needs_customer_ranking(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(clientes?|compradores?)\b",
                normalized,
            )
            and re.search(
                r"\b(mejores?|top|mayor(?:es)?|mas|compran|gasto|ticket)\b",
                normalized,
            )
        )

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
