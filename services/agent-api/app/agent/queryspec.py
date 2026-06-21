from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.agent.intent import QuestionIntent


QUERY_TOOL_NAME = "waro.queries.run"


class QuerySpecValidationError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class QueryDateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date_from: str | None = Field(default=None, alias="from")
    date_to: str | None = Field(default=None, alias="to")


class QueryFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date_range: QueryDateRange | None = None


class QueryOrderBy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    direction: Literal["asc", "desc"] = "desc"

    @field_validator("field")
    @classmethod
    def field_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("order_by.field is required")
        return value


class QuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dataset: str
    measures: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    order_by: list[QueryOrderBy] = Field(default_factory=list)
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("dataset")
    @classmethod
    def dataset_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset is required")
        return value

    @field_validator("measures", "dimensions")
    @classmethod
    def clean_string_list(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            raise ValueError("must include at least one value")
        return list(dict.fromkeys(cleaned))

    @model_validator(mode="after")
    def require_sortable_order_by(self) -> QuerySpec:
        if not self.order_by and self.measures:
            self.order_by = [QueryOrderBy(field=self.measures[0], direction="desc")]
        return self


@dataclass(frozen=True)
class QueryDatasetRule:
    measures: frozenset[str]
    dimensions: frozenset[str]
    filters: frozenset[str] = frozenset({"date_range"})
    max_limit: int = 100

    @property
    def sortable_fields(self) -> frozenset[str]:
        return self.measures | self.dimensions


DATASET_RULES: dict[str, QueryDatasetRule] = {
    "sales_items": QueryDatasetRule(
        measures=frozenset({"quantity_sold", "revenue", "orders_count", "avg_price"}),
        dimensions=frozenset({"product", "product_id", "category", "day"}),
    ),
    "customers": QueryDatasetRule(
        measures=frozenset({"order_count", "total_spent", "avg_ticket", "waros_balance"}),
        dimensions=frozenset({"customer", "customer_id", "day"}),
    ),
    "product_profitability": QueryDatasetRule(
        measures=frozenset(
            {
                "quantity_sold",
                "revenue",
                "profit_per_unit",
                "profit_margin_pct",
                "profit_margin_real_pct",
                "profit_margin_operativo_pct",
                "total_profit",
            }
        ),
        dimensions=frozenset({"product", "product_id", "category", "classification", "day"}),
    ),
}


def is_query_tool(tool_name: str) -> bool:
    return tool_name == QUERY_TOOL_NAME or tool_name.endswith(".queries.run")


def query_tool_requested(capability: Any) -> bool:
    return is_query_tool(str(getattr(capability, "tool_name", "")))


def build_queryspec_for_intent(intent: QuestionIntent, capability: Any) -> dict[str, Any]:
    dataset = _dataset_for_intent(intent)
    rule = DATASET_RULES[dataset]
    measures = _query_measures(intent, rule=rule, capability=capability)
    dimensions = _query_dimensions(intent, rule=rule)
    order_field = _order_field(intent, measures=measures, rule=rule)
    spec: dict[str, Any] = {
        "dataset": dataset,
        "measures": measures,
        "dimensions": dimensions,
        "filters": {},
        "order_by": [{"field": order_field, "direction": "desc"}],
        "limit": 20,
    }
    if intent.time_range.date_from or intent.time_range.date_to:
        spec["filters"]["date_range"] = {
            "from": intent.time_range.date_from,
            "to": intent.time_range.date_to,
        }
    return validate_queryspec(spec).model_dump(by_alias=True, mode="json", exclude_none=True)


def validate_queryspec_payload(value: Any) -> QuerySpec:
    raw = _parse_queryspec_payload(value)
    return validate_queryspec(raw)


def validate_queryspec(value: dict[str, Any]) -> QuerySpec:
    try:
        spec = QuerySpec.model_validate(value)
    except ValidationError as exc:
        raise QuerySpecValidationError(_validation_reason(exc)) from exc
    rule = DATASET_RULES.get(spec.dataset)
    if rule is None:
        raise QuerySpecValidationError(f"invalid_dataset:{spec.dataset}")
    _require_allowed("measure", spec.measures, rule.measures)
    _require_allowed("dimension", spec.dimensions, rule.dimensions)
    filter_payload = spec.filters.model_dump(by_alias=True, exclude_none=True)
    _require_allowed("filter", filter_payload.keys(), rule.filters)
    for order in spec.order_by:
        if order.field not in rule.sortable_fields:
            raise QuerySpecValidationError(f"invalid_order_by:{order.field}")
    if spec.limit > rule.max_limit:
        raise QuerySpecValidationError(f"limit_too_high:{spec.limit}")
    return spec


def query_trace_attributes_from_args(arguments: dict[str, Any], *, valid: bool, rejected_reason: str = "") -> dict[str, Any]:
    try:
        spec = validate_queryspec_payload(arguments.get("spec"))
        return {
            "waro.queries.dataset": spec.dataset,
            "waro.queries.measures": ",".join(spec.measures),
            "waro.queries.dimensions": ",".join(spec.dimensions),
            "waro.queries.valid": valid,
            "waro.queries.rejected_reason": rejected_reason,
        }
    except QuerySpecValidationError as exc:
        raw = _parse_queryspec_payload(arguments.get("spec"), strict=False)
        return {
            "waro.queries.dataset": str(raw.get("dataset") or ""),
            "waro.queries.measures": ",".join(str(item) for item in _list(raw.get("measures"))),
            "waro.queries.dimensions": ",".join(str(item) for item in _list(raw.get("dimensions"))),
            "waro.queries.valid": False,
            "waro.queries.rejected_reason": rejected_reason or exc.reason,
        }


def _dataset_for_intent(intent: QuestionIntent) -> str:
    measures = set(intent.measures)
    if intent.entity == "customer":
        return "customers"
    if intent.entity == "product" and measures.intersection(
        {
            "margin",
            "cost",
            "profit_margin_pct",
            "profit_margin_real_pct",
            "profit_margin_operativo_pct",
            "total_profit",
        }
    ):
        return "product_profitability"
    return "sales_items"


def _query_measures(intent: QuestionIntent, *, rule: QueryDatasetRule, capability: Any) -> list[str]:
    candidates = []
    for measure in intent.measures:
        candidates.extend(_canonical_query_measures(measure))
    hints = getattr(capability, "planning_hints", {})
    default_rank = hints.get("default_rank") if isinstance(hints, dict) else None
    if isinstance(default_rank, list):
        candidates.extend(str(item) for item in default_rank)
    candidates.extend(rule.measures)
    return [item for item in _dedupe(candidates) if item in rule.measures][:4]


def _query_dimensions(intent: QuestionIntent, *, rule: QueryDatasetRule) -> list[str]:
    candidates = [_canonical_query_dimension(item) for item in intent.dimensions]
    if intent.entity == "product":
        candidates.insert(0, "product")
    if intent.entity == "customer":
        candidates.insert(0, "customer")
    if intent.entity == "sale":
        candidates.insert(0, "day")
    return [item for item in _dedupe(candidates) if item in rule.dimensions][:3]


def _order_field(intent: QuestionIntent, *, measures: list[str], rule: QueryDatasetRule) -> str:
    for measure in intent.measures:
        for candidate in _canonical_query_measures(measure):
            if candidate in rule.sortable_fields:
                return candidate
    return measures[0]


def _canonical_query_measures(value: str) -> list[str]:
    normalized = value.replace("-", "_")
    aliases = {
        "margin": ["profit_margin_pct", "profit_margin_real_pct"],
        "cost": ["profit_per_unit"],
        "profit": ["total_profit"],
        "quantity": ["quantity_sold"],
        "total_units_sold": ["quantity_sold"],
        "total_sales": ["revenue"],
        "total_revenue": ["revenue"],
        "total_orders": ["order_count", "orders_count"],
    }
    return aliases.get(normalized, [normalized])


def _canonical_query_dimension(value: str) -> str:
    normalized = value.replace("-", "_")
    return {
        "date": "day",
        "name": "product",
        "id": "product_id",
    }.get(normalized, normalized)


def _parse_queryspec_payload(value: Any, *, strict: bool = True) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            if strict:
                raise QuerySpecValidationError("invalid_json") from exc
            return {}
        if isinstance(parsed, dict):
            return parsed
    if strict:
        raise QuerySpecValidationError("spec_must_be_object")
    return {}


def _require_allowed(kind: str, values, allowed: frozenset[str]) -> None:
    for value in values:
        if value not in allowed:
            raise QuerySpecValidationError(f"invalid_{kind}:{value}")


def _validation_reason(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(item) for item in first.get("loc", ())) or "queryspec"
    return f"invalid_{loc}"


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dedupe(values) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
