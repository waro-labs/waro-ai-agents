import json
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from app.agent.classifier import classify_complexity, heuristic_complexity
from app.agent.capabilities import capability_from_spec, match_tools
from app.agent.intent import heuristic_intent
from app.agent.loop import AgentLoop
from app.agent.plan import build_tool_plan
from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMError, LLMMessage, LLMResponse
from app.tools.allowlist import TOOL_SPECS
from app.tools.models import ToolCallResponse
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
