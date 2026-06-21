import json
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from app.agent.classifier import classify_complexity, heuristic_complexity
from app.agent.capabilities import ToolCapability, capability_from_spec, match_tools
from app.agent.evidence import build_evidence_artifact, deterministic_evidence_summary
from app.agent.intent import coerce_intent, heuristic_intent, parse_question_intent, resolve_contextual_intent
from app.agent.loop import AgentLoop
from app.agent.plan import ToolPlanStep, build_tool_plan
from app.agent.profiles import profile_for_intent, rows_for_group
from app.agent.strategy import heuristic_answer_strategy
from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMError, LLMMessage, LLMResponse
from app.tools.allowlist import TOOL_SPECS
from app.tools.models import ToolCallResponse
from app.tools.response_contract import ResponseContract
from app.tools.registry import ToolRegistry, set_tool_registry


class FakeLLM:
    provider = "kimi"
    decisions = [
        {"action": "call_tool", "tool_name": "waro.sales.metrics", "arguments": {}, "reason": "metrics"},
        {"action": "finish", "tool_name": None, "arguments": {}, "reason": "done"},
    ]
    verify = {"safe_to_answer": True, "missing": "", "needs_more_tools": False}

    def __init__(self):
        self.calls = 0

    async def complete(self, *, messages, temperature=0.2, model=None):
        self.calls += 1
        system = messages[0].content if messages else ""
        if "Verifica si el artifact" in system:
            return LLMResponse(content=json.dumps(self.verify), model=model or "test", provider="kimi")
        decision = self.decisions[min(self.calls - 1, len(self.decisions) - 1)]
        return LLMResponse(content=json.dumps(decision), model=model or "test", provider="kimi")


class IntentLLM:
    provider = "kimi"

    def __init__(self, payload):
        self.payload = payload

    async def complete(self, *, messages, temperature=0.2, model=None):
        return LLMResponse(content=json.dumps(self.payload), model=model or "test", provider="kimi")


class FakeGateway:
    async def call(self, *, request, context):
        if request.tool_name == "waro.financial.products":
            result = {
                "products": [
                    {"name": "Burger", "quantity": 50, "revenue": 1000000, "margin": 12},
                    {"name": "Pizza", "quantity": 30, "revenue": 900000, "margin": 35},
                ],
                "metrics": {},
            }
        elif request.tool_name == "waro.analytics.food_cost":
            result = {
                "data": {
                    "products": [
                        {"name": "Burger", "total_units_sold": 50, "total_revenue": 1000000, "profit_margin_pct": 12}
                    ]
                }
            }
        else:
            result = {
                "success": True,
                "data": {"totalSales": 1000, "totalOrders": 10, "avgTicket": 100},
            }
        return ToolCallResponse(
            tool_call_id=uuid4(),
            tool_name=request.tool_name,
            status="succeeded",
            result=result,
            result_summary="metrics ok",
        )


class RecordingGateway:
    def __init__(self):
        self.calls = []

    async def call(self, *, request, context):
        self.calls.append(request)
        return ToolCallResponse(
            tool_call_id=uuid4(),
            tool_name=request.tool_name,
            status="succeeded",
            result={"rows": [{"product": "Burger", "revenue": 1000}]},
            result_summary="query ok",
        )


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class FailingLLM:
    provider = "kimi"

    async def complete(self, *, messages, temperature=0.2, model=None):
        raise LLMError("Kimi completion request failed.")


@pytest.mark.asyncio
async def test_heuristic_complexity_simple():
    assert heuristic_complexity("dame las ventas de ayer") == "simple"


@pytest.mark.asyncio
async def test_classify_complexity_without_llm():
    result = await classify_complexity(
        settings=Settings(LLM_PROVIDER="disabled"),
        llm_adapter=FakeLLM(),
        question="dame las ventas de ayer",
    )
    assert result["complexity"] == "simple"
    assert result["source"] == "heuristic"


@pytest.mark.asyncio
async def test_classify_complexity_falls_back_when_llm_fails():
    result = await classify_complexity(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test"),
        llm_adapter=FailingLLM(),
        question="dame las ventas de ayer",
    )
    assert result["complexity"] == "simple"
    assert result["source"] == "heuristic"
    assert result["reason"].startswith("heuristic_llm_error")


@pytest.mark.asyncio
async def test_question_intent_sales_ticket():
    intent = heuristic_intent("¿Cuánto vendí ayer y cuál fue el ticket promedio?")
    assert intent.entity == "sale"
    assert "total_sales" in intent.measures
    assert "avg_ticket" in intent.measures
    assert "aggregate" in intent.operations


@pytest.mark.asyncio
async def test_question_intent_product_margin_cross_tool():
    intent = heuristic_intent("Dime qué productos vendieron mucho este mes pero tienen bajo margen.")
    assert intent.entity == "product"
    assert "quantity_sold" in intent.measures
    assert "margin" in intent.measures
    assert intent.requires_cross_tool is True


@pytest.mark.asyncio
async def test_question_intent_customer_frequency_vs_spend():
    intent = heuristic_intent("Compara clientes frecuentes contra clientes con mayor valor comprado este mes.")
    assert intent.entity == "customer"
    assert "order_count" in intent.measures
    assert "total_spent" in intent.measures
    assert "compare" in intent.operations


@pytest.mark.asyncio
async def test_question_intent_business_behavior_analysis():
    intent = heuristic_intent("quiero que me hables mas de este negocio que comportamientos tiene segun sus ventas")
    assert intent.entity == "business"
    assert intent.grain == "business_period"
    assert "total_sales" in intent.measures
    assert "margin" in intent.measures
    assert "diagnose" in intent.operations
    assert intent.requires_cross_tool is True


@pytest.mark.asyncio
async def test_question_intent_waros_customer_generation():
    intent = heuristic_intent("¿Qué clientes han generado más WAROS este mes?")
    assert intent.entity == "loyalty_transaction"
    assert intent.grain == "period_or_customer"
    assert "total_issued" in intent.measures
    assert "customer" in intent.dimensions
    assert "rank" in intent.operations


@pytest.mark.asyncio
async def test_question_intent_keeps_specific_waros_fallback_when_llm_is_generic():
    fallback = heuristic_intent("¿Qué clientes han generado más WAROS este mes?")
    intent = coerce_intent(
        {
            "entity": "customer",
            "grain": "customer_period",
            "measures": ["total_spent"],
            "operations": ["rank"],
        },
        fallback=fallback,
        source="llm",
    )
    assert intent.entity == "loyalty_transaction"
    assert intent.grain == "period_or_customer"
    assert "total_issued" in intent.measures


@pytest.mark.asyncio
async def test_question_intent_customer_cohort_retention():
    intent = heuristic_intent("¿Cómo está la retención de clientes por cohortes este mes?")
    assert intent.entity == "customer"
    assert intent.grain == "cohort_period"
    assert "retention_pct" in intent.measures
    assert "cohort" in intent.dimensions


def test_coerce_intent_lets_llm_replace_generic_fallback_measures():
    fallback = heuristic_intent("diagnostico del negocio")
    intent = coerce_intent(
        {
            "entity": "product",
            "grain": "product_period",
            "measures": ["quantity_sold"],
            "dimensions": ["product"],
            "operations": ["rank"],
        },
        fallback=fallback,
        source="llm",
    )
    assert intent.entity == "product"
    assert intent.measures == ("quantity_sold",)
    assert "margin" not in intent.measures
    assert intent.operations == ("rank",)


@pytest.mark.asyncio
async def test_parse_question_intent_applies_context_to_llm_result():
    intent = await parse_question_intent(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test"),
        llm_adapter=IntentLLM(
            {
                "entity": "sale",
                "grain": "period",
                "measures": ["margin"],
                "dimensions": [],
                "operations": ["aggregate"],
                "time_range": {
                    "date_from": None,
                    "date_to": None,
                    "timezone": "America/Bogota",
                    "label": "",
                },
                "answer_goal": "que margen tienen",
                "requires_cross_tool": False,
                "confidence": 0.7,
                "ambiguities": [],
            }
        ),
        question="que margen tienen",
        conversation_state={
            "source": "artifact",
            "active_entity": "product",
            "active_grain": "product_period",
            "active_period": {
                "date_from": None,
                "date_to": None,
                "timezone": "America/Bogota",
                "label": "",
            },
        },
        capability_hints=[capability_from_spec(TOOL_SPECS["waro.financial.products"]).to_dict()],
    )
    assert intent.entity == "product"
    assert intent.grain == "product_period"
    assert "margin" in intent.measures


def test_contextual_intent_inherits_product_entity_and_period():
    previous = heuristic_intent("dime los productos mas vendidos del ano")
    intent = resolve_contextual_intent(
        heuristic_intent("que margen tienen"),
        question="que margen tienen",
        conversation_state={
            "source": "artifact",
            "active_entity": previous.entity,
            "active_grain": previous.grain,
            "active_period": previous.time_range.to_dict(),
            "active_measures": list(previous.measures),
            "active_dimensions": list(previous.dimensions),
        },
    )
    assert intent.entity == "product"
    assert intent.grain == "product_period"
    assert "margin" in intent.measures
    assert intent.time_range.label == previous.time_range.label


def test_answer_strategy_recommends_from_contextual_follow_up():
    strategy = heuristic_answer_strategy(
        question="que puedo hacer con esto",
        intent=heuristic_intent("que puedo hacer con esto"),
        artifact={"safe_to_answer": True},
        conversation_state={"source": "artifact", "active_entity": "customer"},
    )
    assert strategy.type == "recommendation"
    assert strategy.use_previous_artifact is True


def test_answer_strategy_business_follow_up_is_diagnosis_not_comparison():
    strategy = heuristic_answer_strategy(
        question="que mas me puedes decir del negocio",
        intent=heuristic_intent("que mas me puedes decir del negocio"),
        artifact={"safe_to_answer": True},
        conversation_state={"source": "artifact", "active_entity": "business"},
    )
    assert strategy.type == "diagnosis"
    assert strategy.avoid_repeating is True


def _dynamic_capability(
    *,
    tool_name: str,
    entity: str,
    grain: str,
    measures: tuple[str, ...],
    dimensions: tuple[str, ...],
    operations: tuple[str, ...],
    scope: str = "analytics:read",
    default_fields: tuple[str, ...] = ("data",),
    supports_period: bool = True,
    arguments_schema: dict | None = None,
    planning_hints: dict | None = None,
) -> ToolCapability:
    return ToolCapability(
        tool_name=tool_name,
        scope=scope,
        domain="analytics",
        entity=entity,
        grain=grain,
        measures=measures,
        dimensions=dimensions,
        operations=operations,
        supports_period=supports_period,
        default_fields=default_fields,
        arguments_schema=arguments_schema or {},
        planning_hints=planning_hints or {},
    )


def _queries_capability() -> ToolCapability:
    return _dynamic_capability(
        tool_name="waro.queries.run",
        entity="query_row",
        grain="dynamic_dataset_row",
        measures=(
            "quantity_sold",
            "revenue",
            "order_count",
            "total_spent",
            "avg_ticket",
            "profit_margin_pct",
            "total_profit",
        ),
        dimensions=("product", "customer", "category", "day"),
        operations=("filter", "aggregate", "group", "rank", "sort", "limit", "compare"),
        default_fields=("rows", "meta"),
        arguments_schema={
            "properties": {
                "spec": {"type": "string"},
                "dry-run": {"type": "boolean"},
            },
            "required": ["spec"],
        },
        planning_hints={"default_rank": ["revenue", "quantity_sold", "total_spent"]},
    )


def _queries_schema_payload():
    return {
        "schema_version": "waro.agent.v2",
        "tools": [
            {
                "name": "waro.queries.run",
                "command": ["queries", "run"],
                "scope": "analytics:read",
                "domain": "queries",
                "description": "Run a safe QuerySpec",
                "capabilities": _queries_capability().planning_hints
                | {
                    "entity": "query_row",
                    "grain": "dynamic_dataset_row",
                    "measures": list(_queries_capability().measures),
                    "dimensions": list(_queries_capability().dimensions),
                    "supported_operations": list(_queries_capability().operations),
                    "supports_period": True,
                },
                "arguments": _queries_capability().arguments_schema,
                "response": {
                    "shape": "rows",
                    "row_path": "rows",
                    "fields": ["rows", "meta"],
                    "default_fields": ["rows", "meta"],
                    "top_level_keys": ["rows", "meta"],
                },
            }
        ],
    }


def test_capability_matcher_routes_waros_question_to_waros_tool():
    intent = heuristic_intent("¿Qué clientes han generado más WAROS este mes?")
    waros = _dynamic_capability(
        tool_name="waro.analytics.waros",
        entity="loyalty_transaction",
        grain="period_or_customer",
        measures=("total_issued", "total_redeemed", "redemption_rate_pct"),
        dimensions=("customer", "period"),
        operations=("aggregate", "group", "rank", "summarize"),
        default_fields=("groups", "summary"),
        arguments_schema={
            "properties": {
                "group-by": {
                    "enum": ["day", "week", "customer"],
                    "type": "string",
                }
            }
        },
    )
    customers = capability_from_spec(TOOL_SPECS["waro.customers.list"])

    matches = match_tools(
        intent,
        [customers, waros],
        scopes=("analytics:read", "customers:read"),
    )
    by_name = {match.capability.tool_name: match for match in matches}
    assert by_name["waro.analytics.waros"].accepted is True
    assert by_name["waro.customers.list"].accepted is False

    plan = build_tool_plan(intent, matches)
    assert plan.valid is True
    assert [step.tool_name for step in plan.steps] == ["waro.analytics.waros"]
    assert plan.steps[0].arguments["group-by"] == "customer"


def test_capability_matcher_routes_cohort_question_to_cohort_tool():
    intent = heuristic_intent("¿Cómo está la retención de clientes por cohortes este mes?")
    cohort = _dynamic_capability(
        tool_name="waro.analytics.cohort",
        entity="customer",
        grain="cohort_period",
        measures=("cohort_size", "retention_pct"),
        dimensions=("cohort", "cohort_date"),
        operations=("aggregate", "summarize", "compare"),
        default_fields=("cohorts", "period", "periods"),
    )
    customers = capability_from_spec(TOOL_SPECS["waro.customers.list"])

    matches = match_tools(
        intent,
        [customers, cohort],
        scopes=("analytics:read", "customers:read"),
    )
    by_name = {match.capability.tool_name: match for match in matches}
    assert by_name["waro.analytics.cohort"].accepted is True
    assert by_name["waro.customers.list"].accepted is False

    plan = build_tool_plan(intent, matches)
    assert plan.valid is True
    assert [step.tool_name for step in plan.steps] == ["waro.analytics.cohort"]


def test_capability_matcher_prefers_profitability_context_for_product_margin():
    intent = resolve_contextual_intent(
        heuristic_intent("que margen tienen"),
        question="que margen tienen",
        conversation_state={"source": "artifact", "active_entity": "product", "active_grain": "product_period"},
    )
    matches = match_tools(
        intent,
        [
            capability_from_spec(TOOL_SPECS["waro.analytics.menu"]),
            capability_from_spec(TOOL_SPECS["waro.financial.products"]),
        ],
        scopes=("analytics:read", "financial:read"),
    )
    assert matches[0].capability.tool_name == "waro.financial.products"


def test_business_analysis_plan_uses_multiple_capability_domains():
    intent = heuristic_intent("quiero que me hables mas de este negocio que comportamientos tiene segun sus ventas")
    profile = profile_for_intent(intent)
    assert profile is not None
    assert profile.id == "business_analyst"
    capabilities = [
        capability_from_spec(TOOL_SPECS["waro.sales.metrics"]),
        capability_from_spec(TOOL_SPECS["waro.financial.products"]),
        capability_from_spec(TOOL_SPECS["waro.analytics.food_cost"]),
        capability_from_spec(TOOL_SPECS["waro.customers.list"]),
        _dynamic_capability(
            tool_name="waro.analytics.cohort",
            entity="customer",
            grain="cohort_period",
            measures=("cohort_size", "retention_pct"),
            dimensions=("cohort",),
            operations=("aggregate", "summarize", "diagnose"),
            default_fields=("cohorts", "period", "periods"),
        ),
    ]
    matches = match_tools(
        intent,
        capabilities,
        scopes=("orders:read", "financial:read", "analytics:read", "customers:read"),
    )
    plan = build_tool_plan(intent, matches)
    assert plan.valid is True
    assert [step.tool_name for step in plan.steps] == [
        "waro.sales.metrics",
        "waro.financial.products",
        "waro.analytics.food_cost",
        "waro.customers.list",
        "waro.analytics.cohort",
    ]


def test_profile_groups_can_match_semantic_evidence_from_unknown_tools():
    intent = heuristic_intent("diagnostico del negocio")
    profile = profile_for_intent(intent)
    rows = rows_for_group(
        tables=[
            {
                "tool": "waro.experimental.product_rankings",
                "expected_evidence": ["entity:product", "grain:product_period", "margin"],
                "rows": [{"name": "Burger", "margin": 12}],
            }
        ],
        profile=profile,
        group="products",
    )
    assert rows == [{"name": "Burger", "margin": 12}]


def test_capability_matcher_rejects_sales_list_for_product_margin():
    intent = heuristic_intent("Dime qué productos vendieron mucho este mes pero tienen bajo margen.")
    capabilities = [capability_from_spec(spec) for spec in TOOL_SPECS.values()]
    matches = match_tools(
        intent,
        capabilities,
        scopes=("orders:read", "financial:read", "analytics:read", "menu:read", "customers:read"),
    )
    by_name = {match.capability.tool_name: match for match in matches}
    assert by_name["waro.sales.list"].accepted is False
    assert by_name["waro.financial.products"].accepted is True
    assert by_name["waro.analytics.food_cost"].accepted is True


def test_plan_validator_blocks_sales_list_only_for_product_margin():
    intent = heuristic_intent("Dime qué productos vendieron mucho este mes pero tienen bajo margen.")
    sales_list = capability_from_spec(TOOL_SPECS["waro.sales.list"])
    matches = match_tools(intent, [sales_list], scopes=("orders:read",))
    plan = build_tool_plan(intent, matches)
    assert plan.valid is False
    assert plan.blocked_reason == "no_compatible_tools"


def test_plan_validator_accepts_financial_and_food_cost_for_product_margin():
    intent = heuristic_intent("Dime qué productos vendieron mucho este mes pero tienen bajo margen.")
    capabilities = [
        capability_from_spec(TOOL_SPECS["waro.financial.products"]),
        capability_from_spec(TOOL_SPECS["waro.analytics.food_cost"]),
    ]
    matches = match_tools(intent, capabilities, scopes=("financial:read", "analytics:read"))
    plan = build_tool_plan(intent, matches)
    assert plan.valid is True
    assert {step.tool_name for step in plan.steps} == {
        "waro.financial.products",
        "waro.analytics.food_cost",
    }


def test_plan_builds_arguments_from_schema_for_unknown_compatible_tool():
    intent = heuristic_intent("dime los productos mas vendidos este mes")
    capability = _dynamic_capability(
        tool_name="waro.experimental.product_rankings",
        entity="product",
        grain="product_period",
        measures=("quantity_sold", "revenue", "margin"),
        dimensions=("product", "category"),
        operations=("rank", "sort", "limit"),
        scope="analytics:read",
        arguments_schema={
            "properties": {
                "date-from": {"type": "string"},
                "date-to": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "sort-by": {"type": "string", "enum": ["quantity", "revenue", "margin"]},
            }
        },
        planning_hints={"default_rank": ["quantity"]},
    )
    matches = match_tools(intent, [capability], scopes=("analytics:read",))
    plan = build_tool_plan(intent, matches)
    assert plan.valid is True
    assert plan.steps[0].tool_name == "waro.experimental.product_rankings"
    assert plan.steps[0].arguments["limit"] == 20
    assert plan.steps[0].arguments["sort-by"] == "quantity"
    assert "date-from" in plan.steps[0].arguments


def test_plan_prefers_queries_for_product_sales_with_margin():
    intent = heuristic_intent("dime los productos mas vendidos del año y su margen")
    matches = match_tools(
        intent,
        [capability_from_spec(TOOL_SPECS["waro.financial.products"]), _queries_capability()],
        scopes=("financial:read", "analytics:read"),
    )
    plan = build_tool_plan(intent, matches)
    spec = json.loads(plan.steps[0].arguments["spec"])

    assert plan.valid is True
    assert [step.tool_name for step in plan.steps] == ["waro.queries.run"]
    assert spec["dataset"] == "product_profitability"
    assert "quantity_sold" in spec["measures"]
    assert "profit_margin_pct" in spec["measures"]
    assert "product" in spec["dimensions"]
    assert "sql" not in json.dumps(spec).lower()


def test_plan_builds_customer_queries_for_comparison():
    intent = heuristic_intent("Compara clientes frecuentes contra clientes con mayor valor comprado este mes.")
    matches = match_tools(
        intent,
        [capability_from_spec(TOOL_SPECS["waro.customers.list"]), _queries_capability()],
        scopes=("customers:read", "analytics:read"),
    )
    plan = build_tool_plan(intent, matches)
    spec = json.loads(plan.steps[0].arguments["spec"])

    assert plan.valid is True
    assert [step.tool_name for step in plan.steps] == ["waro.queries.run"]
    assert spec["dataset"] == "customers"
    assert {"order_count", "total_spent"}.issubset(set(spec["measures"]))
    assert spec["order_by"][0]["field"] in {"order_count", "total_spent"}


def test_plan_uses_queries_for_contextual_product_margin_follow_up():
    intent = resolve_contextual_intent(
        heuristic_intent("que margen tienen"),
        question="que margen tienen",
        conversation_state={"source": "artifact", "active_entity": "product", "active_grain": "product_period"},
    )
    matches = match_tools(intent, [_queries_capability()], scopes=("analytics:read",))
    plan = build_tool_plan(intent, matches)
    spec = json.loads(plan.steps[0].arguments["spec"])

    assert plan.valid is True
    assert plan.steps[0].tool_name == "waro.queries.run"
    assert spec["dataset"] == "product_profitability"
    assert "profit_margin_pct" in spec["measures"]


def test_plan_validator_accepts_customer_frequency_vs_spend_without_revenue():
    intent = heuristic_intent("Compara clientes frecuentes contra clientes con mayor valor comprado este mes.")
    assert "revenue" not in intent.measures
    assert {"order_count", "total_spent"}.issubset(set(intent.measures))
    capabilities = [
        capability_from_spec(TOOL_SPECS["waro.customers.metrics"]),
        capability_from_spec(TOOL_SPECS["waro.customers.list"]),
    ]
    matches = match_tools(intent, capabilities, scopes=("customers:read",))
    plan = build_tool_plan(intent, matches)
    assert plan.valid is True
    assert "revenue" not in plan.missing_coverage


def test_evidence_merges_product_sales_and_margin_rows():
    intent = heuristic_intent("Dime qué productos vendieron mucho este mes pero tienen bajo margen.")
    plan = build_tool_plan(
        intent,
        match_tools(
            intent,
            [
                capability_from_spec(TOOL_SPECS["waro.financial.products"]),
                capability_from_spec(TOOL_SPECS["waro.analytics.food_cost"]),
            ],
            scopes=("financial:read", "analytics:read"),
        ),
    )
    artifact = build_evidence_artifact(
        question="Dime qué productos vendieron mucho este mes pero tienen bajo margen.",
        intent=intent,
        plan=plan,
        observations=[
            {
                "tool_name": "waro.financial.products",
                "status": "succeeded",
                "result": {"products": [{"name": "Burger", "quantity": 50, "revenue": 1000000}]},
            },
            {
                "tool_name": "waro.analytics.food_cost",
                "status": "succeeded",
                "result": {"data": {"products": [{"name": "Burger", "profit_margin_pct": 12}]}},
            },
        ],
    )
    summary = deterministic_evidence_summary(artifact)
    assert "50 unidades" in summary
    assert "$1.000.000 vendido" in summary
    assert "margen 12%" in summary


def test_product_ranking_summary_does_not_add_margin_when_not_requested():
    intent = heuristic_intent("dime los productos mas vendidos del ano")
    plan = build_tool_plan(
        intent,
        match_tools(
            intent,
            [capability_from_spec(TOOL_SPECS["waro.financial.products"])],
            scopes=("financial:read",),
        ),
    )
    artifact = build_evidence_artifact(
        question="dime los productos mas vendidos del ano",
        intent=intent,
        plan=plan,
        observations=[
            {
                "tool_name": "waro.financial.products",
                "status": "succeeded",
                "result": {
                    "products": [
                        {"name": "Burger", "quantity": 50, "revenue": 1000000, "margin": 12}
                    ]
                },
            }
        ],
    )
    summary = deterministic_evidence_summary(artifact)
    assert "Productos por unidades vendidas" in summary
    assert "bajo margen" not in summary
    assert "margen 12%" not in summary
    assert "50 unidades" in summary


def test_product_follow_up_margin_uses_previous_ranked_rows_when_current_rows_missing():
    intent = resolve_contextual_intent(
        heuristic_intent("que margen tienen"),
        question="que margen tienen",
        conversation_state={
            "source": "artifact",
            "active_entity": "product",
            "active_grain": "product_period",
            "last_artifact": {
                "ranked_rows": [
                    {"name": "Burger", "quantity": 50, "revenue": 1000000, "margin": 12}
                ]
            },
        },
    )
    plan = build_tool_plan(
        intent,
        match_tools(
            intent,
            [capability_from_spec(TOOL_SPECS["waro.financial.products"])],
            scopes=("financial:read",),
        ),
    )
    artifact = build_evidence_artifact(
        question="que margen tienen",
        intent=intent,
        plan=plan,
        observations=[],
        conversation_state={
            "source": "artifact",
            "active_entity": "product",
            "last_artifact": {
                "ranked_rows": [
                    {"name": "Burger", "quantity": 50, "revenue": 1000000, "margin": 12}
                ]
            },
        },
    )
    artifact["safe_to_answer"] = True
    artifact["answer_strategy"] = {"type": "follow_up", "use_previous_artifact": True}
    summary = deterministic_evidence_summary(artifact)
    assert "Burger" in summary
    assert "margen 12%" in summary


def test_evidence_summarizes_waros_customer_rows():
    intent = heuristic_intent("¿Qué clientes han generado más WAROS este mes?")
    waros = _dynamic_capability(
        tool_name="waro.analytics.waros",
        entity="loyalty_transaction",
        grain="period_or_customer",
        measures=("total_issued", "total_redeemed"),
        dimensions=("customer",),
        operations=("rank", "group"),
        default_fields=("groups", "summary"),
    )
    plan = build_tool_plan(
        intent,
        match_tools(intent, [waros], scopes=("analytics:read",)),
    )
    artifact = build_evidence_artifact(
        question="¿Qué clientes han generado más WAROS este mes?",
        intent=intent,
        plan=plan,
        observations=[
            {
                "tool_name": "waro.analytics.waros",
                "status": "succeeded",
                "result": {
                    "rows": [
                        {
                            "name": "Ana",
                            "total_earned": 120,
                            "total_redeemed": 20,
                            "transaction_count": 3,
                        }
                    ]
                },
            }
        ],
    )
    summary = deterministic_evidence_summary(artifact)
    assert "Clientes que mas generaron WAROS" in summary
    assert "Ana" in summary
    assert "120 WAROS generados" in summary


def test_evidence_summarizes_cohort_rows():
    intent = heuristic_intent("¿Cómo está la retención de clientes por cohortes este mes?")
    cohort = _dynamic_capability(
        tool_name="waro.analytics.cohort",
        entity="customer",
        grain="cohort_period",
        measures=("cohort_size", "retention_pct"),
        dimensions=("cohort",),
        operations=("aggregate", "summarize"),
        default_fields=("cohorts", "period", "periods"),
    )
    plan = build_tool_plan(
        intent,
        match_tools(intent, [cohort], scopes=("analytics:read",)),
    )
    artifact = build_evidence_artifact(
        question="¿Cómo está la retención de clientes por cohortes este mes?",
        intent=intent,
        plan=plan,
        observations=[
            {
                "tool_name": "waro.analytics.cohort",
                "status": "succeeded",
                "result": {
                    "rows": [
                        {
                            "cohort_label": "2026-W24",
                            "cohort_size": 12,
                            "retention": [{"pct": 25.0}],
                        }
                    ]
                },
            }
        ],
    )
    summary = deterministic_evidence_summary(artifact)
    assert "Retencion por cohortes" in summary
    assert "2026-W24" in summary
    assert "12 clientes iniciales" in summary


def test_evidence_summarizes_business_analysis_patterns():
    intent = heuristic_intent("quiero que me hables mas de este negocio que comportamientos tiene segun sus ventas")
    plan = build_tool_plan(
        intent,
        match_tools(
            intent,
            [
                capability_from_spec(TOOL_SPECS["waro.sales.metrics"]),
                capability_from_spec(TOOL_SPECS["waro.financial.products"]),
                capability_from_spec(TOOL_SPECS["waro.customers.list"]),
            ],
            scopes=("orders:read", "financial:read", "customers:read"),
        ),
    )
    artifact = build_evidence_artifact(
        question="quiero que me hables mas de este negocio que comportamientos tiene segun sus ventas",
        intent=intent,
        plan=plan,
        observations=[
            {
                "tool_name": "waro.sales.metrics",
                "status": "succeeded",
                "result": {"data": {"totalSales": 98155500, "avgTicket": 24266, "totalOrders": 404}},
            },
            {
                "tool_name": "waro.financial.products",
                "status": "succeeded",
                "result": {
                    "products": [
                        {"name": "HOT DOG SENCILLO", "quantity": 80, "revenue": 1600000, "margin": 35}
                    ]
                },
            },
            {
                "tool_name": "waro.customers.list",
                "status": "succeeded",
                "result": {
                    "rows": [
                        {"name": "Genérico", "order_count": 388, "total_spent": 10648000},
                        {"name": "Ana", "order_count": 2, "total_spent": 94000},
                    ]
                },
            },
        ],
    )
    summary = deterministic_evidence_summary(artifact)
    assert "Analisis del negocio" in summary
    assert "Datos base" in summary
    assert "Comportamientos detectados" in summary
    assert "Genérico" in summary
    assert "margen bajo" in summary
    assert "Acciones recomendadas" in summary


def test_capability_uses_schema_default_fields_over_legacy_defaults():
    contract = ResponseContract(
        command=("analytics", "menu"),
        shape="rows",
        row_path="data",
        fields=("data",),
        default_fields=("data",),
        top_level_keys=("data",),
    )
    capability = capability_from_spec(
        TOOL_SPECS["waro.analytics.menu"],
        response_contract=contract,
    )
    assert capability.default_fields == ("data",)


@pytest.mark.asyncio
async def test_agent_loop_builds_evidence_artifact_for_simple_question():
    settings = Settings(TOOL_CATALOG_SOURCE="static", LLM_PROVIDER="disabled")
    registry = ToolRegistry(settings)
    set_tool_registry(registry)
    await registry.refresh(force=True)
    loop = AgentLoop(
        settings=settings,
        gateway=FakeGateway(),
        registry=registry,
        llm_adapter=FakeLLM(),
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-agent",
        member_id=None,
        scopes=("orders:read", "financial:read", "menu:read", "customers:read", "analytics:read"),
    )
    artifact = await loop.run(
        question="dame las ventas de ayer",
        context=context,
        run_id=uuid4(),
        complexity="simple",
    )
    assert artifact["agent_mode"] is True
    assert artifact["observations"]
    assert artifact["agent_engine_version"] == "intent-capability-v1"
    assert artifact["metrics"]["total_sales"] == 1000


@pytest.mark.asyncio
async def test_agent_loop_uses_catalog_fallback_when_llm_step_fails():
    settings = Settings(TOOL_CATALOG_SOURCE="static", LLM_PROVIDER="kimi", KIMI_API_KEY="test")
    registry = ToolRegistry(settings)
    set_tool_registry(registry)
    await registry.refresh(force=True)
    loop = AgentLoop(
        settings=settings,
        gateway=FakeGateway(),
        registry=registry,
        llm_adapter=FailingLLM(),
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-agent-fallback",
        member_id=None,
        scopes=("orders:read", "financial:read", "menu:read", "customers:read", "analytics:read"),
    )
    artifact = await loop.run(
        question="dime qué productos venden mucho pero tienen bajo margen",
        context=context,
        run_id=uuid4(),
        complexity="complex",
    )
    assert artifact["agent_mode"] is True
    assert artifact["safe_to_answer"] is True
    assert artifact["observations"]
    assert "waro.sales.list" not in {obs["tool_name"] for obs in artifact["observations"]}


@pytest.mark.asyncio
async def test_agent_loop_blocks_invalid_queryspec_before_gateway(monkeypatch):
    async def fake_load(self):
        return _queries_schema_payload()

    settings = Settings(TOOL_CATALOG_SOURCE="cli", LLM_PROVIDER="disabled")
    registry = ToolRegistry(settings)
    monkeypatch.setattr(ToolRegistry, "_load_cli_schema", fake_load)
    await registry.refresh(force=True)
    gateway = RecordingGateway()
    loop = AgentLoop(
        settings=settings,
        gateway=gateway,
        registry=registry,
        llm_adapter=FakeLLM(),
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-invalid-query",
        member_id=None,
        scopes=("analytics:read",),
    )
    span = RecordingSpan()
    observations = await loop._execute_plan(
        plan_steps=(
            ToolPlanStep(
                tool_name="waro.queries.run",
                arguments={
                    "spec": json.dumps(
                        {
                            "dataset": "raw_sql",
                            "measures": ["revenue"],
                            "dimensions": ["product"],
                            "order_by": [{"field": "revenue", "direction": "desc"}],
                            "limit": 10,
                        }
                    )
                },
                fields=("rows", "meta"),
                purpose="test invalid query",
                expected_evidence=("revenue",),
            ),
        ),
        context=context,
        run_id=uuid4(),
        span=span,
    )

    assert gateway.calls == []
    assert observations[0]["status"] == "failed"
    assert observations[0]["error"]["rejected_reason"] == "invalid_dataset:raw_sql"
    assert span.attributes["waro.queries.valid"] is False
    assert span.attributes["waro.queries.rejected_reason"] == "invalid_dataset:raw_sql"


@pytest.mark.asyncio
async def test_sales_workflow_react_mode_uses_agent_runner():
    from app.workflows.sales import SalesWorkflow

    class RecordingRunner:
        async def execute(self, **kwargs):
            return {
                "agent_mode": True,
                "safe_to_answer": True,
                "summary": "Respuesta agentica",
                "observations": [{"tool_name": "waro.sales.metrics", "status": "succeeded"}],
                "tables": [],
                "complexity": "simple",
            }

        async def execute_shadow(self, **kwargs):
            legacy = kwargs["legacy_artifact"]
            legacy["agent_shadow_artifact"] = {"shadow": True}
            return legacy

    @asynccontextmanager
    async def connection_factory():
        class FakeConnection:
            async def fetchrow(self, query, *args):
                return {"id": uuid4()}

            async def execute(self, *args, **kwargs):
                return None

        yield FakeConnection()

    workflow = SalesWorkflow(
        settings=Settings(AGENT_MODE="react", LLM_PROVIDER="disabled"),
        gateway=FakeGateway(),
        llm_adapter=FakeLLM(),
        connection_factory=connection_factory,
    )
    workflow.kali_runner = RecordingRunner()
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-react",
        member_id=None,
        scopes=("orders:read",),
    )
    state = await workflow._run_kali_agent_node(
        {
            "request": type("Req", (), {"question": "dame ventas de ayer"})(),
            "context": context,
            "run_id": uuid4(),
            "sales_intent": "sales_metrics",
        }
    )
    assert state["summary"] == "Respuesta agentica"
    assert state["artifact"]["agent_mode"] is True


@pytest.mark.asyncio
async def test_sales_workflow_stream_react_mode_uses_graph():
    from app.workflows.sales import SalesWorkflow

    class RecordingRunner:
        async def execute(self, **kwargs):
            return {
                "agent_mode": True,
                "safe_to_answer": True,
                "summary": "Respuesta agentica en stream",
                "observations": [
                    {
                        "tool_name": "waro.sales.metrics",
                        "status": "succeeded",
                        "result_summary": "Returned 1 rows.",
                    }
                ],
                "tables": [],
                "complexity": "simple",
            }

        async def execute_shadow(self, **kwargs):
            legacy = kwargs["legacy_artifact"]
            legacy["agent_shadow_artifact"] = {"shadow": True}
            return legacy

    @asynccontextmanager
    async def connection_factory():
        class FakeConnection:
            async def fetchrow(self, query, *args):
                return {"id": uuid4()}

            async def execute(self, *args, **kwargs):
                return None

        yield FakeConnection()

    workflow = SalesWorkflow(
        settings=Settings(AGENT_MODE="react", LLM_PROVIDER="disabled"),
        gateway=FakeGateway(),
        llm_adapter=FakeLLM(),
        connection_factory=connection_factory,
    )
    workflow.kali_runner = RecordingRunner()
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-react-stream",
        member_id=None,
        scopes=("orders:read",),
    )
    events = [
        event
        async for event in workflow.stream(
            request=type("Req", (), {"question": "dame ventas de ayer", "conversation_id": None})(),
            context=context,
        )
    ]

    assert [event.event for event in events] == [
        "run_started",
        "step_started",
        "tool_started",
        "tool_finished",
        "final",
    ]
    assert events[-1].data["summary"] == "Respuesta agentica en stream"
    assert events[-1].data["artifact_summary"]["agent_mode"] is True
