from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest

from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm import LLMResponse
from app.tools.models import ToolCallResponse
from app.workflows.agent import AgentWorkflow
from app.workflows.models import AgentQuestionRequest


class FakeConnection:
    def __init__(self):
        self.fetches = []
        self.executes = []
        self.ids = [uuid4() for _ in range(40)]

    async def fetchrow(self, query, *args):
        self.fetches.append((query, args))
        return {"id": self.ids.pop(0)}

    async def execute(self, query, *args):
        self.executes.append((query, args))


class FakeGateway:
    def __init__(self):
        self.calls = []

    async def call(self, *, request, context):
        self.calls.append(request)
        if request.tool_name == "waro.sales.metrics":
            result = {
                "data": {
                    "totalSales": 431500.0,
                    "totalOrders": 14,
                    "avgTicket": 30821.428571428572,
                },
                "meta": {"dateFrom": "2026-06-17", "dateTo": "2026-06-17"},
                "success": True,
            }
        elif request.tool_name == "waro.analytics.food_cost":
            result = {
                "data": [
                    {
                        "product_id": "p1",
                        "product_name": "Arepa",
                        "food_cost_pct": 42,
                        "margin_pct": 18,
                        "revenue": 120000,
                        "cost": 50400,
                    }
                ]
            }
        elif request.tool_name == "waro.financial.products":
            result = {
                "data": [
                    {
                        "id": "p1",
                        "name": "Arepa",
                        "margin": 18,
                        "revenue": 120000,
                        "cost": 50400,
                        "quantity": 40,
                    }
                ]
            }
        else:
            result = {"data": [{"id": "p1", "name": "Arepa", "price": 9000}]}
        return ToolCallResponse(
            tool_call_id=uuid4(),
            tool_name=request.tool_name,
            status="succeeded",
            result=result,
            result_summary="Returned data.",
        )


class FakeLLMAdapter:
    provider = "fake"

    def __init__(self, content: str):
        self.content = content
        self.calls = []
        self.call_kwargs = []

    async def complete(self, *, messages, temperature=0.2, model=None):
        self.calls.append(messages)
        self.call_kwargs.append({"temperature": temperature, "model": model})
        return LLMResponse(
            content=self.content,
            model=model or "fake-model",
            provider=self.provider,
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
            estimated_cost_usd=None,
        )

    async def stream_complete(self, *, messages, temperature=0.2, model=None):
        raise NotImplementedError


def build_context() -> InternalRequestContext:
    return InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-agent",
        member_id=None,
        scopes=("orders:read", "analytics:read", "menu:read", "financial:read"),
    )


def build_workflow(connection: FakeConnection, gateway: FakeGateway) -> AgentWorkflow:
    @asynccontextmanager
    async def connection_factory():
        yield connection

    return AgentWorkflow(
        settings=Settings(),
        gateway=gateway,
        connection_factory=connection_factory,
    )


def test_agent_workflow_routes_by_explicit_workflow_and_keywords():
    workflow = AgentWorkflow(settings=Settings())

    explicit = workflow._route(
        AgentQuestionRequest(question="dame algo", workflow="food_cost")
    )
    assert explicit.workflow == "food_cost"
    assert explicit.confidence == 1.0

    sales = workflow._route(AgentQuestionRequest(question="dame las ventas de ayer"))
    assert sales.workflow == "sales"
    assert sales.reason == "sales_keyword"

    food_cost = workflow._route(
        AgentQuestionRequest(question="que productos tienen peor margen")
    )
    assert food_cost.workflow == "food_cost"
    assert food_cost.reason == "food_cost_keyword"


@pytest.mark.asyncio
async def test_agent_workflow_uses_router_model_for_hybrid_prerouting():
    llm = FakeLLMAdapter(
        '{"workflow":"food_cost","confidence":0.91,"reason":"pregunta de recetas"}'
    )
    workflow = AgentWorkflow(
        settings=Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_MODEL="kimi-default",
            KIMI_ROUTER_MODEL="kimi-cheap-router",
        ),
        llm_adapter=llm,
    )

    route = await workflow._route_hybrid(
        AgentQuestionRequest(question="que recetas tienen mayor costo")
    )

    assert route.workflow == "food_cost"
    assert route.reason.startswith("llm_prerouter:")
    assert llm.call_kwargs[0]["model"] == "kimi-cheap-router"
    assert llm.call_kwargs[0]["temperature"] == 0


@pytest.mark.asyncio
async def test_agent_workflow_falls_back_to_deterministic_router_on_low_confidence():
    llm = FakeLLMAdapter(
        '{"workflow":"food_cost","confidence":0.4,"reason":"ambiguo"}'
    )
    workflow = AgentWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        llm_adapter=llm,
    )

    route = await workflow._route_hybrid(
        AgentQuestionRequest(question="dame las ventas de ayer")
    )

    assert route.workflow == "sales"
    assert route.reason.startswith("deterministic_low_llm_confidence:")


@pytest.mark.asyncio
async def test_agent_workflow_dispatches_sales_and_preserves_group_by():
    connection = FakeConnection()
    gateway = FakeGateway()
    workflow = build_workflow(connection, gateway)

    response = await workflow.run(
        request=AgentQuestionRequest(
            question="dame ventas por hora",
            date_from="2026-06-17",
            date_to="2026-06-17",
            group_by="hour",
        ),
        context=build_context(),
    )

    assert response.status == "completed"
    assert response.workflow == "sales"
    assert isinstance(response.run_id, UUID)
    assert [call.tool_name for call in gateway.calls] == ["waro.sales.metrics"]
    assert gateway.calls[0].arguments == {
        "date-from": "2026-06-17",
        "date-to": "2026-06-17",
        "group-by": "hour",
    }
    assert response.artifact["metrics"]["total_sales"] == 431500.0


@pytest.mark.asyncio
async def test_agent_workflow_dispatches_food_cost_and_preserves_compare_to():
    connection = FakeConnection()
    gateway = FakeGateway()
    workflow = build_workflow(connection, gateway)

    response = await workflow.run(
        request=AgentQuestionRequest(
            question="productos con peor food cost",
            workflow="food_cost",
            date_from="2026-06-01",
            date_to="2026-06-18",
            compare_to="previous_period",
        ),
        context=build_context(),
    )

    assert response.status == "completed"
    assert response.workflow == "food_cost"
    assert [call.tool_name for call in gateway.calls] == [
        "waro.analytics.food_cost",
        "waro.menu.products",
        "waro.financial.products",
    ]
    assert gateway.calls[0].arguments == {
        "date-from": "2026-06-01",
        "date-to": "2026-06-18",
        "compare-to": "previous_period",
    }
    assert response.artifact["low_margin_products"][0]["name"] == "Arepa"
