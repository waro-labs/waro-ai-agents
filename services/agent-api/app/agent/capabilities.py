from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.agent.intent import QuestionIntent, normalize_measure, normalize_text
from app.agent.queryspec import query_tool_requested
from app.tools.allowlist import ToolSpec
from app.tools.response_contract import ResponseContract


@dataclass(frozen=True)
class ToolCapability:
    tool_name: str
    scope: str
    domain: str
    entity: str
    grain: str
    measures: tuple[str, ...]
    dimensions: tuple[str, ...]
    operations: tuple[str, ...]
    supports_period: bool
    default_fields: tuple[str, ...]
    arguments_schema: dict[str, Any]
    planning_hints: dict[str, Any] = field(default_factory=dict)
    response_contract: dict[str, Any] | None = None
    can_answer_patterns: tuple[str, ...] = ()
    cannot_answer_patterns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapabilityMatch:
    capability: ToolCapability
    score: int
    accepted: bool
    reasons: tuple[str, ...]
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "tool_name": self.capability.tool_name,
            "score": self.score,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "capability": self.capability.to_dict(),
        }
        if self.rejected_reason:
            payload["rejected_reason"] = self.rejected_reason
        return payload


def capability_from_spec(
    spec: ToolSpec,
    *,
    arguments_schema: dict[str, Any] | None = None,
    response_contract: ResponseContract | None = None,
) -> ToolCapability:
    raw = dict(spec.capabilities or {})
    measures = tuple(_dedupe(normalize_measure(str(item)) for item in _list(raw.get("measures"))))
    dimensions = tuple(_dedupe(normalize_measure(str(item)) for item in _list(raw.get("dimensions"))))
    operations = tuple(_dedupe(normalize_measure(str(item)) for item in _list(raw.get("supported_operations"))))
    return ToolCapability(
        tool_name=spec.name,
        scope=spec.scope,
        domain=str(spec.domain),
        entity=str(raw.get("entity") or spec.domain),
        grain=str(raw.get("grain") or "unknown"),
        measures=measures,
        dimensions=dimensions,
        operations=operations,
        supports_period=bool(raw.get("supports_period", False)),
        default_fields=tuple(response_contract.default_fields if response_contract else spec.default_fields),
        arguments_schema=arguments_schema or spec.args_model.model_json_schema(by_alias=True),
        planning_hints=raw,
        response_contract=(
            {
                "shape": response_contract.shape,
                "row_path": response_contract.row_path,
                "fields": list(response_contract.fields),
                "default_fields": list(response_contract.default_fields),
                "top_level_keys": list(response_contract.top_level_keys),
            }
            if response_contract
            else None
        ),
        can_answer_patterns=tuple(spec.examples),
        cannot_answer_patterns=(),
    )


def match_tools(
    intent: QuestionIntent,
    capabilities: list[ToolCapability],
    *,
    scopes: tuple[str, ...],
) -> list[CapabilityMatch]:
    scope_set = set(scopes)
    matches = [_score_capability(intent, capability, scope_set=scope_set) for capability in capabilities]
    matches.sort(key=lambda item: (-int(item.accepted), -item.score, item.capability.tool_name))
    return matches


def search_capabilities(
    intent: QuestionIntent,
    capabilities: list[ToolCapability],
    *,
    scopes: tuple[str, ...],
    limit: int = 8,
) -> list[CapabilityMatch]:
    """Return compatible capability candidates for planning.

    This keeps discovery dynamic: new CLI capabilities participate through their
    declared contract instead of hardcoded module routing.
    """
    matches = match_tools(intent, capabilities, scopes=scopes)
    accepted = [match for match in matches if match.accepted]
    return (accepted or matches)[:limit]


def _score_capability(
    intent: QuestionIntent,
    capability: ToolCapability,
    *,
    scope_set: set[str],
) -> CapabilityMatch:
    reasons: list[str] = []
    score = 0
    if capability.scope not in scope_set:
        return CapabilityMatch(
            capability=capability,
            score=0,
            accepted=False,
            reasons=(),
            rejected_reason=f"missing_scope:{capability.scope}",
        )

    is_query_tool = query_tool_requested(capability)
    if not _entity_compatible(intent, capability):
        return CapabilityMatch(
            capability=capability,
            score=0,
            accepted=False,
            reasons=(),
            rejected_reason=f"entity_mismatch:{capability.entity}",
        )
    score += 8
    reasons.append("entity")

    if _grain_compatible(intent, capability):
        score += 5
        reasons.append("grain")
    elif intent.grain in {"product_period", "customer_period"} and capability.grain == "order":
        return CapabilityMatch(
            capability=capability,
            score=score,
            accepted=False,
            reasons=tuple(reasons),
            rejected_reason=f"grain_mismatch:{capability.grain}",
        )

    measure_hits = _measure_hits(intent, capability)
    dimension_hits = set(intent.dimensions).intersection(capability.dimensions)
    operation_hits = set(intent.operations).intersection(capability.operations)
    if measure_hits:
        score += 4 * len(measure_hits)
        reasons.append("measures:" + ",".join(sorted(measure_hits)))
    if dimension_hits:
        score += 2 * len(dimension_hits)
        reasons.append("dimensions:" + ",".join(sorted(dimension_hits)))
    if operation_hits:
        score += len(operation_hits)
        reasons.append("operations:" + ",".join(sorted(operation_hits)))
    if intent.entity == "product" and "margin" in intent.measures:
        profitability_context = set(capability.measures).intersection({"cost", "profit"})
        if profitability_context:
            score += 2
            reasons.append("profitability_context:" + ",".join(sorted(profitability_context)))
    if intent.time_range.date_from and capability.supports_period:
        score += 2
        reasons.append("period")

    accepted = score >= 8 and (
        bool(measure_hits)
        or _summary_tool_ok(intent, capability)
        or (intent.entity == "business" and bool(operation_hits))
    )
    if is_query_tool:
        accepted = accepted and _query_tool_has_analytical_fit(intent, capability, measure_hits)
    rejected_reason = None if accepted else "missing_required_measure"
    return CapabilityMatch(
        capability=capability,
        score=score,
        accepted=accepted,
        reasons=tuple(reasons),
        rejected_reason=rejected_reason,
    )


def required_coverage(intent: QuestionIntent) -> set[str]:
    required = set(intent.measures)
    if intent.entity:
        required.add(f"entity:{intent.entity}")
    if intent.grain:
        required.add(f"grain:{intent.grain}")
    return required


def coverage_for_match(intent: QuestionIntent, match: CapabilityMatch) -> set[str]:
    capability = match.capability
    coverage: set[str] = _measure_hits(intent, capability)
    if _entity_compatible(intent, capability):
        coverage.add(f"entity:{intent.entity}")
    if _grain_compatible(intent, capability):
        coverage.add(f"grain:{intent.grain}")
    if query_tool_requested(capability):
        coverage.add(f"entity:{intent.entity}")
        coverage.add(f"grain:{intent.grain}")
    if intent.entity == "product" and capability.entity == "product":
        coverage.add("entity:product")
        coverage.add("grain:product_period")
    if intent.entity == "customer" and capability.entity == "customer":
        coverage.add("entity:customer")
        coverage.add("grain:customer_period")
    return coverage


def _entity_compatible(intent: QuestionIntent, capability: ToolCapability) -> bool:
    if intent.entity == "unknown":
        return True
    if query_tool_requested(capability):
        return intent.entity in {
            "sale",
            "order",
            "product",
            "customer",
            "business",
            "ingredient",
            "inventory",
            "purchase",
            "supplier",
            "procurement",
        }
    if intent.entity == "business":
        return capability.entity in {
            "sale",
            "order",
            "product",
            "customer",
            "loyalty_transaction",
            "ingredient",
            "inventory",
            "purchase",
            "supplier",
            "procurement",
        }
    if intent.entity == capability.entity:
        return True
    if intent.entity == "sale" and capability.entity in {"sale", "order"}:
        return True
    if intent.entity == "product" and capability.entity in {"product", "menu_item"}:
        return True
    if intent.entity == "inventory" and capability.entity in {"ingredient", "inventory"}:
        return True
    if intent.entity == "procurement" and capability.entity in {"purchase", "supplier", "procurement"}:
        return True
    return False


def _grain_compatible(intent: QuestionIntent, capability: ToolCapability) -> bool:
    if intent.grain == "unknown":
        return True
    if query_tool_requested(capability):
        return intent.grain in {
            "period",
            "product_period",
            "customer_period",
            "business_period",
            "inventory_snapshot",
            "inventory_movement",
            "purchase_period",
            "supplier_period",
        }
    if intent.grain == "business_period":
        return capability.supports_period or capability.grain in {
            "period",
            "period_or_group",
            "product_period",
            "customer_period",
            "customer_period_summary",
            "cohort_period",
            "period_or_customer",
            "inventory_snapshot",
            "inventory_movement",
            "purchase_period",
            "supplier_period",
        }
    if intent.grain == capability.grain:
        return True
    if intent.grain == "period" and capability.grain in {"period_or_group", "daily_series"}:
        return True
    if intent.grain == "product_period" and capability.grain in {"product_period", "period_or_group"}:
        return True
    if intent.grain == "customer_period" and capability.grain in {"customer_period", "customer_period_summary"}:
        return True
    if intent.grain == "inventory_snapshot" and capability.grain in {"inventory_snapshot", "ingredient", "period_or_group"}:
        return True
    if intent.grain == "inventory_movement" and capability.grain in {"inventory_movement", "period_or_group"}:
        return True
    if intent.grain == "purchase_period" and capability.grain in {"purchase_period", "period_or_group"}:
        return True
    if intent.grain == "supplier_period" and capability.grain in {"supplier_period", "period_or_group"}:
        return True
    return False


def _summary_tool_ok(intent: QuestionIntent, capability: ToolCapability) -> bool:
    return intent.entity == capability.entity and capability.grain.endswith("_summary")


def _measure_hits(intent: QuestionIntent, capability: ToolCapability) -> set[str]:
    if not query_tool_requested(capability):
        return set(intent.measures).intersection(capability.measures)
    capability_measures = set(capability.measures)
    hits: set[str] = set()
    for measure in intent.measures:
        aliases = _query_measure_aliases(measure)
        if capability_measures.intersection(aliases):
            hits.add(measure)
    return hits


def _query_tool_has_analytical_fit(
    intent: QuestionIntent,
    capability: ToolCapability,
    measure_hits: set[str],
) -> bool:
    if not measure_hits:
        return False
    if intent.entity == "business":
        return len(measure_hits) >= 2 or bool(set(intent.operations).intersection({"diagnose", "compare"}))
    if intent.requires_cross_tool:
        return len(measure_hits) >= 2
    if set(intent.operations).intersection({"compare", "diagnose"}):
        return bool(set(intent.dimensions).intersection(capability.dimensions)) or len(measure_hits) >= 2
    if intent.entity in {"product", "customer"} and intent.grain in {"product_period", "customer_period"}:
        return bool(measure_hits)
    return bool(set(intent.operations).intersection({"rank", "group", "sort", "filter"}))


def _query_measure_aliases(measure: str) -> set[str]:
    normalized = measure.replace("-", "_")
    aliases = {
        "margin": {"margin", "profit_margin_pct", "profit_margin_real_pct", "profit_margin_operativo_pct"},
        "cost": {"cost", "profit_per_unit"},
        "profit": {"total_profit", "profit"},
        "total_sales": {"total_sales", "revenue"},
        "total_revenue": {"total_revenue", "revenue"},
        "total_orders": {"total_orders", "order_count", "orders_count"},
        "quantity": {"quantity", "quantity_sold"},
        "total_units_sold": {"total_units_sold", "quantity_sold"},
        "current_stock": {"current_stock", "quantity", "stock"},
        "minimum_stock": {"minimum_stock", "low_stock"},
        "quantity_purchased": {"quantity_purchased", "quantity"},
        "purchase_count": {"purchase_count", "count"},
        "avg_unit_cost": {"avg_unit_cost", "unit_cost", "cost"},
        "total_cost": {"total_cost", "cost"},
        "movement_count": {"movement_count", "count"},
        "net_quantity": {"net_quantity", "quantity"},
    }
    return aliases.get(normalized, {normalized})


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dedupe(values) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = normalize_text(str(value)).replace("-", "_").replace(" ", "_")
        if normalized and normalized not in result:
            result.append(normalized)
    return result
