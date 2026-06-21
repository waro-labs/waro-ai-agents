from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.agent.intent import QuestionIntent, normalize_measure, normalize_text
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

    measure_hits = set(intent.measures).intersection(capability.measures)
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
    if intent.time_range.date_from and capability.supports_period:
        score += 2
        reasons.append("period")

    accepted = score >= 8 and (
        bool(measure_hits)
        or _summary_tool_ok(intent, capability)
        or (intent.entity == "business" and bool(operation_hits))
    )
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
    coverage: set[str] = set(intent.measures).intersection(capability.measures)
    if _entity_compatible(intent, capability):
        coverage.add(f"entity:{intent.entity}")
    if _grain_compatible(intent, capability):
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
    if intent.entity == "business":
        return capability.entity in {
            "sale",
            "order",
            "product",
            "customer",
            "loyalty_transaction",
        }
    if intent.entity == capability.entity:
        return True
    if intent.entity == "sale" and capability.entity in {"sale", "order"}:
        return True
    if intent.entity == "product" and capability.entity in {"product", "menu_item"}:
        return True
    return False


def _grain_compatible(intent: QuestionIntent, capability: ToolCapability) -> bool:
    if intent.grain == "unknown":
        return True
    if intent.grain == "business_period":
        return capability.supports_period or capability.grain in {
            "period",
            "period_or_group",
            "product_period",
            "customer_period",
            "customer_period_summary",
            "cohort_period",
            "period_or_customer",
        }
    if intent.grain == capability.grain:
        return True
    if intent.grain == "period" and capability.grain in {"period_or_group", "daily_series"}:
        return True
    if intent.grain == "product_period" and capability.grain in {"product_period", "period_or_group"}:
        return True
    if intent.grain == "customer_period" and capability.grain in {"customer_period", "customer_period_summary"}:
        return True
    return False


def _summary_tool_ok(intent: QuestionIntent, capability: ToolCapability) -> bool:
    return intent.entity == capability.entity and capability.grain.endswith("_summary")


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dedupe(values) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = normalize_text(str(value)).replace("-", "_").replace(" ", "_")
        if normalized and normalized not in result:
            result.append(normalized)
    return result
