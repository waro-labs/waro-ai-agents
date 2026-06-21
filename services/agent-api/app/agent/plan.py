from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.agent.capabilities import CapabilityMatch, coverage_for_match, required_coverage
from app.agent.intent import QuestionIntent
from app.agent.profiles import profile_for_intent


@dataclass(frozen=True)
class ToolPlanStep:
    tool_name: str
    arguments: dict[str, Any]
    fields: tuple[str, ...]
    purpose: str
    expected_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fields"] = list(self.fields)
        payload["expected_evidence"] = list(self.expected_evidence)
        return payload


@dataclass(frozen=True)
class ToolPlan:
    steps: tuple[ToolPlanStep, ...]
    coverage: tuple[str, ...]
    missing_coverage: tuple[str, ...]
    valid: bool
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "coverage": list(self.coverage),
            "missing_coverage": list(self.missing_coverage),
            "valid": self.valid,
            "blocked_reason": self.blocked_reason,
        }


def build_tool_plan(intent: QuestionIntent, matches: list[CapabilityMatch]) -> ToolPlan:
    accepted = [match for match in matches if match.accepted]
    if not accepted:
        return ToolPlan(
            steps=(),
            coverage=(),
            missing_coverage=tuple(sorted(required_coverage(intent))),
            valid=False,
            blocked_reason="no_compatible_tools",
        )

    selected = _select_matches(intent, accepted)
    steps = tuple(_step_for(intent, match) for match in selected)
    coverage = set()
    for match in selected:
        coverage.update(coverage_for_match(intent, match))
    required = _effective_required_coverage(intent)
    missing = tuple(sorted(required - coverage))
    return ToolPlan(
        steps=steps,
        coverage=tuple(sorted(coverage)),
        missing_coverage=missing,
        valid=not missing and bool(steps),
        blocked_reason=None if not missing and steps else "missing_coverage",
    )


def _select_matches(intent: QuestionIntent, accepted: list[CapabilityMatch]) -> list[CapabilityMatch]:
    profile = profile_for_intent(intent)
    if profile is not None and profile.tool_priorities:
        return _select_profile_matches(accepted, profile.tool_priorities, max_steps=profile.max_steps)
    selected: list[CapabilityMatch] = []
    coverage: set[str] = set()
    required = _effective_required_coverage(intent)
    for match in accepted:
        incremental = coverage_for_match(intent, match) - coverage
        if incremental:
            selected.append(match)
            coverage.update(incremental)
        if required.issubset(coverage) and not intent.requires_cross_tool:
            break
        if required.issubset(coverage) and intent.requires_cross_tool and len({item.capability.domain for item in selected}) >= 2:
            break
    if intent.requires_cross_tool and selected:
        domains = {item.capability.domain for item in selected}
        for match in accepted:
            if match.capability.domain not in domains:
                selected.append(match)
                break
    return selected[:3] if selected else accepted[:1]


def _select_profile_matches(
    accepted: list[CapabilityMatch],
    tool_priorities: tuple[str, ...],
    *,
    max_steps: int,
) -> list[CapabilityMatch]:
    by_name = {match.capability.tool_name: match for match in accepted}
    selected: list[CapabilityMatch] = []
    for tool_name in tool_priorities:
        match = by_name.get(tool_name)
        if match is not None:
            selected.append(match)
        if len(selected) >= max_steps:
            break
    if selected:
        return selected
    return accepted[:max_steps]


def _step_for(intent: QuestionIntent, match: CapabilityMatch) -> ToolPlanStep:
    capability = match.capability
    args: dict[str, Any] = {}
    if intent.time_range.date_from and capability.supports_period:
        args["date-from"] = intent.time_range.date_from
    if intent.time_range.date_to and capability.supports_period:
        args["date-to"] = intent.time_range.date_to
    if capability.tool_name == "waro.financial.products":
        args = {
            "period": _period_days(intent),
            "sort-by": _product_sort(intent),
        }
    elif capability.tool_name == "waro.sales.metrics":
        if intent.entity == "product" or (
            intent.entity != "business" and "product" in intent.dimensions
        ):
            args["group-by"] = "product"
        if "hour" in intent.dimensions:
            args["group-by"] = "hour"
        args["limit"] = 20
        args["sort-by"] = "quantity" if "quantity_sold" in intent.measures else "revenue"
    elif capability.tool_name == "waro.customers.list":
        args["limit"] = 20
        args["sort-field"] = _customer_sort(intent)
        args["sort-direction"] = "desc"
    elif capability.tool_name == "waro.analytics.menu":
        args["limit"] = 20
    elif capability.tool_name == "waro.analytics.waros":
        if "customer" in intent.dimensions or intent.grain == "period_or_customer":
            args["group-by"] = "customer"
        elif "week" in intent.dimensions:
            args["group-by"] = "week"
        else:
            args["group-by"] = "day"
    elif capability.tool_name == "waro.analytics.cohort":
        args.setdefault("period", "weekly")
        args.setdefault("periods", 8)
    elif capability.tool_name == "waro.analytics.churn_risk":
        args.setdefault("limit", 20)
    return ToolPlanStep(
        tool_name=capability.tool_name,
        arguments=args,
        fields=capability.default_fields,
        purpose=_purpose(intent, capability.tool_name),
        expected_evidence=tuple(sorted(coverage_for_match(intent, match))),
    )


def _effective_required_coverage(intent: QuestionIntent) -> set[str]:
    required = required_coverage(intent)
    if intent.entity == "sale":
        required.discard("grain:period")
    if intent.entity == "product":
        required.discard("grain:product_period")
    if intent.entity == "customer":
        required.discard("grain:customer_period")
    if intent.entity == "business":
        return set()
    return required


def _period_days(intent: QuestionIntent) -> int:
    date_from = intent.time_range.date_from
    date_to = intent.time_range.date_to
    if not date_from or not date_to:
        return 365
    try:
        from datetime import date

        start = date.fromisoformat(date_from)
        end = date.fromisoformat(date_to)
        return max(1, min(730, (end - start).days + 1))
    except ValueError:
        return 365


def _product_sort(intent: QuestionIntent) -> str:
    if "quantity_sold" in intent.measures:
        return "quantity"
    if "revenue" in intent.measures or "total_sales" in intent.measures:
        return "revenue"
    if "cost" in intent.measures:
        return "cost"
    return "margin"


def _customer_sort(intent: QuestionIntent) -> str:
    if "order_count" in intent.measures:
        return "order_count"
    if "avg_ticket" in intent.measures:
        return "avg_ticket"
    return "total_spent"


def _purpose(intent: QuestionIntent, tool_name: str) -> str:
    return f"Collect evidence for {intent.entity} using {tool_name}."
