from contextlib import asynccontextmanager
import json
from uuid import UUID, uuid4

import pytest

from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMMessage, LLMResponse, LLMStreamChunk
from app.tools.models import ToolCallResponse
from app.workflows.food_cost import FoodCostWorkflow
from app.workflows.models import FoodCostQuestionRequest


def parse_sse_frame(frame: str) -> tuple[str, dict]:
    lines = frame.splitlines()
    event_line = next(line for line in lines if line.startswith("event: "))
    data_line = next(line for line in lines if line.startswith("data: "))
    return event_line.removeprefix("event: "), json.loads(
        data_line.removeprefix("data: ")
    )


class FakeConnection:
    def __init__(self):
        self.fetches = []
        self.executes = []
        self.ids = [uuid4() for _ in range(20)]

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
        if request.tool_name == "waro.analytics.food_cost":
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
            result_summary="Returned 1 rows.",
        )


class FakeLLMAdapter:
    provider = "fake"

    def __init__(
        self,
        content: str | None = None,
        error: Exception | None = None,
        response: LLMResponse | None = None,
    ):
        self.content = content or "Resumen generado por LLM."
        self.error = error
        self.response = response
        self.calls: list[list[LLMMessage]] = []

    async def complete(self, *, messages, temperature=0.2):
        self.calls.append(messages)
        if self.error:
            raise self.error
        if self.response:
            return self.response
        return LLMResponse(content=self.content, model="fake-model", provider=self.provider)


class FakeStreamingLLMAdapter(FakeLLMAdapter):
    async def stream_complete(self, *, messages, temperature=0.2):
        self.calls.append(messages)
        yield LLMStreamChunk(text="Resumen")
        yield LLMStreamChunk(text=" Kimi")
        yield LLMStreamChunk(text=" de food cost.")
        yield LLMStreamChunk(
            response=LLMResponse(
                content="Resumen Kimi de food cost.",
                model="fake-model",
                provider=self.provider,
                input_tokens=100,
                output_tokens=40,
                total_tokens=140,
                estimated_cost_usd=0.0001,
                cost_source="test",
            )
        )


@pytest.mark.asyncio
async def test_food_cost_workflow_persists_run_tools_message_summary_and_evals():
    connection = FakeConnection()
    gateway = FakeGateway()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = FoodCostWorkflow(
        settings=Settings(),
        gateway=gateway,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-1",
        member_id=None,
        scopes=("analytics:read", "menu:read", "financial:read"),
    )

    response = await workflow.run(
        request=FoodCostQuestionRequest(
            question="Que productos tienen el food cost mas alto?",
            date_from="2026-06-01",
            date_to="2026-06-18",
        ),
        context=context,
    )

    assert response.status == "completed"
    assert isinstance(response.run_id, UUID)
    assert [call.tool_name for call in gateway.calls] == [
        "waro.analytics.food_cost",
        "waro.menu.products",
        "waro.financial.products",
    ]
    assert gateway.calls[0].arguments == {
        "date-from": "2026-06-01",
        "date-to": "2026-06-18",
    }
    assert response.artifact["low_margin_products"][0]["name"] == "Arepa"
    assert response.artifact["recommendations"]
    assert "question" not in response.artifact
    assert {eval_result.evaluator_name for eval_result in response.evals} == {
        "food_cost_tool_usage",
        "food_cost_safety",
        "food_cost_business_usefulness",
    }
    assert all(eval_result.passed for eval_result in response.evals)

    executed_sql = "\n".join(query for query, _ in connection.executes)
    fetched_sql = "\n".join(query for query, _ in connection.fetches)
    assert "INSERT INTO ai.conversations" in fetched_sql
    assert "INSERT INTO ai.messages" in fetched_sql
    assert "INSERT INTO ai.runs" in fetched_sql
    assert "INSERT INTO ai.steps" in fetched_sql
    assert "INSERT INTO ai.context_summaries" in executed_sql
    assert executed_sql.count("INSERT INTO ai.eval_results") == 3
    assert "status = 'completed'" in executed_sql
    step_payloads = " ".join(
        str(args)
        for query, args in connection.fetches
        if "INSERT INTO ai.steps" in query
    )
    assert "Que productos tienen el food cost mas alto?" not in step_payloads
    assert "question_length" in step_payloads


@pytest.mark.asyncio
async def test_food_cost_workflow_uses_llm_summary_when_enabled():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(content="Resumen Kimi de food cost.")

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = FoodCostWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-llm",
        member_id=None,
        scopes=("analytics:read", "menu:read", "financial:read"),
    )

    response = await workflow.run(
        request=FoodCostQuestionRequest(question="Resume food cost."),
        context=context,
    )

    assert response.summary == "Resumen Kimi de food cost."
    assert len(llm.calls) == 1
    executed_sql = "\n".join(query for query, _ in connection.executes)
    assert "INSERT INTO ai.context_summaries" in executed_sql
    fetched_sql = "\n".join(query for query, _ in connection.fetches)
    assert "food_cost_summary" in str(connection.fetches)
    assert "INSERT INTO ai.steps" in fetched_sql


@pytest.mark.asyncio
async def test_food_cost_workflow_streams_event_order_and_final_payload():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(content="Resumen Kimi de food cost.")

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = FoodCostWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-food-cost-stream",
        member_id=None,
        scopes=("analytics:read", "menu:read", "financial:read"),
    )

    events = [
        event
        async for event in workflow.stream(
            request=FoodCostQuestionRequest(
                question="Que productos tienen el food cost mas alto?",
                date_from="2026-06-01",
                date_to="2026-06-18",
            ),
            context=context,
        )
    ]

    assert [event.event for event in events] == [
        "run_started",
        "step_started",
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
        "llm_started",
        "final",
    ]
    assert [call.tool_name for call in gateway.calls] == [
        "waro.analytics.food_cost",
        "waro.menu.products",
        "waro.financial.products",
    ]

    final_event_name, final_payload = parse_sse_frame(events[-1].to_sse())
    assert final_event_name == "final"
    assert final_payload["status"] == "completed"
    assert final_payload["summary"] == "Resumen Kimi de food cost."
    assert final_payload["artifact_summary"]["period"] == {
        "date_from": "2026-06-01",
        "date_to": "2026-06-18",
        "compare_to": None,
    }
    assert final_payload["artifact_summary"]["low_margin_products"][0]["name"] == "Arepa"
    assert [
        call["tool_name"]
        for call in final_payload["artifact_summary"]["tool_calls"]
    ] == [
        "waro.analytics.food_cost",
        "waro.menu.products",
        "waro.financial.products",
    ]
    assert "question" not in final_payload["artifact_summary"]

    executed_sql = "\n".join(query for query, _ in connection.executes)
    fetched_sql = "\n".join(query for query, _ in connection.fetches)
    assert "INSERT INTO ai.runs" in fetched_sql
    assert "INSERT INTO ai.steps" in fetched_sql
    assert executed_sql.count("INSERT INTO ai.eval_results") == 3
    assert "status = 'completed'" in executed_sql


@pytest.mark.asyncio
async def test_food_cost_workflow_streams_llm_token_events_before_final_payload():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeStreamingLLMAdapter()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = FoodCostWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-food-cost-token-stream",
        member_id=None,
        scopes=("analytics:read", "menu:read", "financial:read"),
    )

    events = [
        event
        async for event in workflow.stream(
            request=FoodCostQuestionRequest(question="Resume food cost."),
            context=context,
        )
    ]

    event_names = [event.event for event in events]
    assert event_names[-5:] == ["llm_started", "token", "token", "token", "final"]
    token_payloads = [
        parse_sse_frame(event.to_sse())[1]
        for event in events
        if event.event == "token"
    ]
    assert [payload["text"] for payload in token_payloads] == [
        "Resumen",
        " Kimi",
        " de food cost.",
    ]
    assert all("content" not in payload for payload in token_payloads)
    assert (
        parse_sse_frame(events[-1].to_sse())[1]["summary"]
        == "Resumen Kimi de food cost."
    )
    assert len(llm.calls) == 1
    llm_step_args = [
        args
        for query, args in connection.fetches
        if "INSERT INTO ai.steps" in query and args[2] == "llm"
    ][0]
    llm_step_json = f"{llm_step_args[4]} {llm_step_args[5]}"
    assert "Resume food cost" not in llm_step_json
    assert "test-key" not in llm_step_json


@pytest.mark.asyncio
async def test_food_cost_workflow_persists_llm_usage_cost_metadata():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(
        response=LLMResponse(
            content="Resumen con costo.",
            model="kimi-k2.7-code",
            provider="kimi",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.00249,
            cost_source="static:estimated-kimi-pricing-2026-06-18",
        )
    )

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = FoodCostWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-llm-cost",
        member_id=None,
        scopes=("analytics:read", "menu:read", "financial:read"),
    )

    response = await workflow.run(
        request=FoodCostQuestionRequest(question="Resume food cost."),
        context=context,
    )

    assert response.summary == "Resumen con costo."
    llm_step_args = [
        args
        for query, args in connection.fetches
        if "INSERT INTO ai.steps" in query and args[2] == "llm"
    ][0]
    output_json = llm_step_args[5]
    assert '"input_count": 1000' in output_json
    assert '"output_count": 500' in output_json
    assert '"total_count": 1500' in output_json
    assert '"estimated_cost_usd": 0.00249' in output_json
    assert "static:estimated-kimi-pricing-2026-06-18" in output_json
    assert "test-key" not in output_json
    assert "Resume food cost" not in output_json


@pytest.mark.asyncio
async def test_food_cost_workflow_falls_back_when_llm_fails():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(error=RuntimeError("provider unavailable"))

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = FoodCostWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-llm-fallback",
        member_id=None,
        scopes=("analytics:read", "menu:read", "financial:read"),
    )

    response = await workflow.run(
        request=FoodCostQuestionRequest(question="Resume food cost."),
        context=context,
    )

    assert response.summary.startswith("Food-cost workflow flagged")
    assert len(llm.calls) == 1
