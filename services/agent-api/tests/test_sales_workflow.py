from contextlib import asynccontextmanager
import json
from uuid import UUID, uuid4

import pytest

from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMMessage, LLMResponse
from app.tools.models import ToolCallResponse
from app.workflows.models import SalesQuestionRequest
from app.workflows.sales import SalesWorkflow


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
        return ToolCallResponse(
            tool_call_id=uuid4(),
            tool_name=request.tool_name,
            status="succeeded",
            result={
                "data": [
                    {
                        "totalSales": 431500.0,
                        "orderCount": 14,
                        "avgTicket": 30821.428571428572,
                        "series": [{"date": "2026-06-17", "totalSales": 431500.0}],
                    }
                ],
                "success": True,
            },
            result_summary="Returned 1 rows.",
        )


class FakeLLMAdapter:
    provider = "fake"

    def __init__(self, content: str = "Resumen de ventas generado por LLM."):
        self.content = content
        self.calls: list[list[LLMMessage]] = []

    async def complete(self, *, messages, temperature=0.2):
        self.calls.append(messages)
        return LLMResponse(
            content=self.content,
            model="fake-model",
            provider=self.provider,
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            estimated_cost_usd=0.0001,
            prompt_cost_usd=0.00004,
            completion_cost_usd=0.00006,
            cost_source="test",
        )


@pytest.mark.asyncio
async def test_sales_workflow_persists_run_tools_message_summary_and_evals():
    connection = FakeConnection()
    gateway = FakeGateway()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(),
        gateway=gateway,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales",
        member_id=None,
        scopes=("orders:read",),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(
            question="Cuanto vendi ayer?",
            date_from="2026-06-17",
            date_to="2026-06-17",
        ),
        context=context,
    )

    assert response.status == "completed"
    assert isinstance(response.run_id, UUID)
    assert [call.tool_name for call in gateway.calls] == ["waro.sales.metrics"]
    assert gateway.calls[0].arguments == {
        "date-from": "2026-06-17",
        "date-to": "2026-06-17",
        "group-by": "date",
    }
    assert gateway.calls[0].fields == ["totalSales", "orderCount", "avgTicket", "series"]
    assert response.artifact["metrics"]["total_sales"] == 431500.0
    assert response.artifact["metrics"]["order_count"] == 14
    assert response.artifact["metrics"]["avg_ticket"] == 30821.428571428572
    assert response.evals[0].evaluator_name == "sales_tool_usage"
    assert all(eval_result.passed for eval_result in response.evals)

    executed_sql = "\n".join(query for query, _ in connection.executes)
    fetched_sql = "\n".join(query for query, _ in connection.fetches)
    assert "INSERT INTO ai.conversations" in fetched_sql
    assert "INSERT INTO ai.messages" in fetched_sql
    assert "INSERT INTO ai.runs" in fetched_sql
    assert "INSERT INTO ai.steps" in fetched_sql
    assert "INSERT INTO ai.context_summaries" in executed_sql
    assert executed_sql.count("INSERT INTO ai.eval_results") == 2
    assert "status = 'completed'" in executed_sql


@pytest.mark.asyncio
async def test_sales_workflow_uses_llm_summary_when_enabled():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(content="Ayer vendiste $431.500 con ticket promedio de $30.821.")

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-llm",
        member_id=None,
        scopes=("orders:read",),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(
            question="Cuanto vendi ayer?",
            date_from="2026-06-17",
            date_to="2026-06-17",
        ),
        context=context,
    )

    assert response.summary == "Ayer vendiste $431.500 con ticket promedio de $30.821."
    assert len(llm.calls) == 1
    assert llm.calls[0][0].role == "system"
    assert "analista de ventas" in llm.calls[0][0].content
    fetched_sql = "\n".join(query for query, _ in connection.fetches)
    assert "sales_summary" in str(connection.fetches)
    assert "INSERT INTO ai.steps" in fetched_sql


@pytest.mark.asyncio
async def test_sales_workflow_streams_event_order_and_final_payload():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(content="Ayer vendiste $431.500 con ticket promedio de $30.821.")

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-stream",
        member_id=None,
        scopes=("orders:read",),
    )

    events = [
        event
        async for event in workflow.stream(
            request=SalesQuestionRequest(
                question="Cuanto vendi ayer?",
                date_from="2026-06-17",
                date_to="2026-06-17",
            ),
            context=context,
        )
    ]

    assert [event.event for event in events] == [
        "run_started",
        "step_started",
        "tool_started",
        "tool_finished",
        "llm_started",
        "final",
    ]
    assert [call.tool_name for call in gateway.calls] == ["waro.sales.metrics"]

    final_event_name, final_payload = parse_sse_frame(events[-1].to_sse())
    assert final_event_name == "final"
    assert final_payload["status"] == "completed"
    assert final_payload["summary"] == "Ayer vendiste $431.500 con ticket promedio de $30.821."
    assert final_payload["artifact_summary"]["metrics"] == {
        "total_sales": 431500.0,
        "order_count": 14,
        "avg_ticket": 30821.428571428572,
    }
    assert final_payload["artifact_summary"]["tool_calls"][0]["tool_name"] == "waro.sales.metrics"
    assert "question" not in final_payload["artifact_summary"]

    executed_sql = "\n".join(query for query, _ in connection.executes)
    fetched_sql = "\n".join(query for query, _ in connection.fetches)
    assert "INSERT INTO ai.runs" in fetched_sql
    assert "INSERT INTO ai.steps" in fetched_sql
    assert executed_sql.count("INSERT INTO ai.eval_results") == 2
    assert "status = 'completed'" in executed_sql
