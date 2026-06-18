from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest

from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.tools.models import ToolCallResponse
from app.workflows.food_cost import FoodCostWorkflow
from app.workflows.models import FoodCostQuestionRequest


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
