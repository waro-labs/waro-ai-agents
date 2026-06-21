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
    args = _arguments_for(intent, capability)
    return ToolPlanStep(
        tool_name=capability.tool_name,
        arguments=args,
        fields=capability.default_fields,
        purpose=_purpose(intent, capability.tool_name),
        expected_evidence=tuple(sorted(coverage_for_match(intent, match))),
    )


def _arguments_for(intent: QuestionIntent, capability: Any) -> dict[str, Any]:
    """Build tool arguments from the declared CLI argument schema.

    The planner should not need to know every tool by name. If a future CLI tool
    declares familiar arguments such as date-from, limit, sort-by, sort-field or
    group-by, it can participate in planning through its contract.
    """
    args: dict[str, Any] = {}
    properties = _argument_properties(capability.arguments_schema)
    if intent.time_range.date_from and capability.supports_period and "date-from" in properties:
        args["date-from"] = intent.time_range.date_from
    if intent.time_range.date_to and capability.supports_period and "date-to" in properties:
        args["date-to"] = intent.time_range.date_to
    if "period" in properties and capability.supports_period:
        args["period"] = _period_days(intent)
    if "limit" in properties:
        args["limit"] = _bounded_default_limit(properties.get("limit"), default=20)
    sort_by = _best_sort_value(intent, capability, properties.get("sort-by"))
    if sort_by is not None:
        args["sort-by"] = sort_by
    sort_field = _best_sort_value(intent, capability, properties.get("sort-field"))
    if sort_field is not None:
        args["sort-field"] = sort_field
    group_by = _best_group_value(intent, capability, properties.get("group-by"))
    if group_by is not None:
        args["group-by"] = group_by
    if "sort-direction" in properties:
        args["sort-direction"] = "desc"
    if "periods" in properties:
        args["periods"] = _bounded_default_limit(properties.get("periods"), default=8)
    return args


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


def _argument_properties(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    if not isinstance(properties, dict):
        return {}
    return {str(key): value for key, value in properties.items() if isinstance(value, dict)}


def _bounded_default_limit(property_schema: dict[str, Any] | None, *, default: int) -> int:
    maximum = property_schema.get("maximum") if isinstance(property_schema, dict) else None
    minimum = property_schema.get("minimum") if isinstance(property_schema, dict) else None
    value = default
    if isinstance(maximum, int):
        value = min(value, maximum)
    if isinstance(minimum, int):
        value = max(value, minimum)
    return value


def _best_sort_value(
    intent: QuestionIntent,
    capability: Any,
    property_schema: dict[str, Any] | None,
) -> str | None:
    allowed = _schema_enum(property_schema)
    if not allowed:
        return None
    for candidate in _sort_candidates(intent, capability):
        if candidate in allowed:
            return candidate
    return None


def _best_group_value(
    intent: QuestionIntent,
    capability: Any,
    property_schema: dict[str, Any] | None,
) -> str | None:
    allowed = _schema_enum(property_schema)
    if not allowed:
        return None
    candidates = [
        *(_canonical_dimension(dimension) for dimension in intent.dimensions),
        *(_canonical_dimension(dimension) for dimension in capability.dimensions),
    ]
    if intent.grain == "period_or_customer":
        candidates.insert(0, "customer")
    if capability.grain == "period_or_customer" and not any(item in allowed for item in candidates):
        candidates.append("day")
    for candidate in candidates:
        if candidate in allowed:
            return candidate
    return None


def _schema_enum(property_schema: dict[str, Any] | None) -> set[str]:
    if not isinstance(property_schema, dict):
        return set()
    enum = property_schema.get("enum")
    if isinstance(enum, list):
        return {str(item) for item in enum if item is not None}
    any_of = property_schema.get("anyOf")
    if isinstance(any_of, list):
        values: set[str] = set()
        for item in any_of:
            if isinstance(item, dict):
                values.update(_schema_enum(item))
        return values
    return set()


def _sort_candidates(intent: QuestionIntent, capability: Any) -> list[str]:
    raw: list[str] = []
    raw.extend(intent.measures)
    hints = capability.planning_hints if isinstance(capability.planning_hints, dict) else {}
    default_rank = hints.get("default_rank")
    if isinstance(default_rank, list):
        raw.extend(str(item) for item in default_rank if item)
    raw.extend(capability.measures)
    result: list[str] = []
    for item in raw:
        result.extend(_canonical_sort_values(str(item)))
    return _dedupe(result)


def _canonical_sort_values(value: str) -> list[str]:
    normalized = value.replace("-", "_")
    aliases = {
        "quantity_sold": ["quantity", "total_units_sold", "order_count"],
        "total_units_sold": ["quantity", "total_units_sold", "order_count"],
        "quantity": ["quantity", "total_units_sold"],
        "revenue": ["revenue", "total_spent", "total_amount"],
        "total_revenue": ["revenue", "total_spent", "total_amount"],
        "total_sales": ["revenue", "total_spent", "total_amount"],
        "total_spent": ["total_spent", "revenue"],
        "order_count": ["order_count"],
        "total_orders": ["order_count"],
        "avg_ticket": ["avg_ticket"],
        "margin": ["margin", "profit_margin_pct"],
        "profit_margin_pct": ["margin", "profit_margin_pct"],
        "profit_margin_real_pct": ["margin", "profit_margin_real_pct"],
        "cost": ["cost", "estimated_cost"],
        "estimated_cost": ["cost", "estimated_cost"],
        "last_order_date": ["last_order_date", "order_date"],
        "order_date": ["order_date", "last_order_date"],
        "totalamount": ["total_amount", "revenue"],
    }
    return aliases.get(normalized, [normalized])


def _canonical_dimension(value: str) -> str:
    normalized = value.replace("-", "_")
    aliases = {
        "product": "product",
        "name": "product",
        "id": "product",
        "customer_id": "customer",
        "customer": "customer",
        "date": "date",
        "weekday": "weekday",
        "month": "month",
        "week": "week",
        "hour": "hour",
        "payment": "payment",
        "payment_method": "payment",
        "ticket": "ticket",
    }
    return aliases.get(normalized, normalized)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _purpose(intent: QuestionIntent, tool_name: str) -> str:
    return f"Collect evidence for {intent.entity} using {tool_name}."
