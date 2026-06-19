from dataclasses import dataclass, field
import re
from typing import Any

from app.tools.catalog import candidate_tools, normalize_query


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "candidate_tools": self.candidate_tools,
            "steps": [step.to_dict() for step in self.steps],
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
    ) -> ToolPlan:
        normalized = normalize_query(question)
        candidates = candidate_tools(
            question,
            preferred_domain="sales",
            scopes=scopes,
            limit=6,
        )
        candidate_names = [spec.name for spec in candidates]

        metrics_group_by = group_by or self._infer_sales_group_by(normalized)
        metrics_arguments: dict[str, Any] = {
            "date-from": period["date_from"],
            "date-to": period["date_to"],
        }
        if metrics_group_by:
            metrics_arguments["group-by"] = metrics_group_by

        steps = [
            ToolPlanStep(
                tool_name="waro.sales.metrics",
                arguments=metrics_arguments,
                fields=["data", "meta", "success"],
                reason="Primary sales metrics are required for sales analysis.",
            )
        ]

        if self._needs_financial_products(normalized) and "financial:read" in scopes:
            steps.append(
                ToolPlanStep(
                    tool_name="waro.financial.products",
                    arguments={
                        "sort-by": self._financial_sort(normalized),
                        "period": self._period_days(period),
                    },
                    fields=["id", "name", "margin", "revenue", "cost", "quantity"],
                    reason="Product financial context can explain sales performance.",
                )
            )

        if self._needs_menu_products(normalized) and "menu:read" in scopes:
            steps.append(
                ToolPlanStep(
                    tool_name="waro.menu.products",
                    arguments={"limit": 50},
                    fields=["id", "name", "price", "is_available", "category"],
                    reason="Menu product context can clarify product-level questions.",
                )
            )

        return ToolPlan(
            strategy="catalog_sales_planner_v1",
            candidate_tools=candidate_names,
            steps=steps,
        )

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
        )

    def _needs_menu_products(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(menu|disponibles?|precio|precios|categoria)\b",
                normalized,
            )
        )

    def _financial_sort(self, normalized: str) -> str:
        if re.search(r"\b(margen|rentabilidad|rentables?|peor(?:es)?)\b", normalized):
            return "margin"
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
