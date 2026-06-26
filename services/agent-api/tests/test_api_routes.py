import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.dependencies.internal_auth import InternalRequestContext
from app.main import (
    app,
    get_food_cost_workflow,
    get_sales_workflow,
    require_internal_request,
)
from app.streaming import stream_event
from app.workflows.models import (
    FoodCostWorkflowResponse,
    SalesWorkflowResponse,
)


class FakeSalesWorkflow:
    def __init__(self):
        self.run_requests = []
        self.stream_requests = []

    async def run(self, *, request, context):
        self.run_requests.append((request, context))
        return SalesWorkflowResponse(
            conversation_id=uuid4(),
            run_id=uuid4(),
            input_message_id=uuid4(),
            output_message_id=uuid4(),
            status="completed",
            artifact={
                "period": {"date_from": request.date_from, "date_to": request.date_to},
                "metrics": {"total_sales": 431500.0},
            },
            summary="Ayer vendiste 431500.",
            evals=[],
        )

    async def stream(self, *, request, context):
        self.stream_requests.append((request, context))
        run_id = uuid4()
        yield stream_event(
            "run_started",
            run_id=run_id,
            data={"workflow": "sales_metrics", "status": "running"},
        )
        yield stream_event(
            "final",
            run_id=run_id,
            data={
                "status": "completed",
                "summary": "Ayer vendiste 431500.",
                "artifact_summary": {"metrics": {"total_sales": 431500.0}},
            },
        )


class FakeFoodCostWorkflow:
    def __init__(self):
        self.run_requests = []
        self.stream_requests = []

    async def run(self, *, request, context):
        self.run_requests.append((request, context))
        return FoodCostWorkflowResponse(
            conversation_id=uuid4(),
            run_id=uuid4(),
            input_message_id=uuid4(),
            output_message_id=uuid4(),
            status="completed",
            artifact={
                "period": {
                    "date_from": request.date_from,
                    "date_to": request.date_to,
                    "compare_to": request.compare_to,
                },
                "low_margin_products": [{"name": "Arepa"}],
            },
            summary="Arepa necesita revision.",
            evals=[],
        )

    async def stream(self, *, request, context):
        self.stream_requests.append((request, context))
        run_id = uuid4()
        yield stream_event(
            "run_started",
            run_id=run_id,
            data={"workflow": "food_cost_analysis", "status": "running"},
        )
        yield stream_event(
            "final",
            run_id=run_id,
            data={
                "status": "completed",
                "summary": "Arepa necesita revision.",
                "artifact_summary": {"low_margin_products": [{"name": "Arepa"}]},
            },
        )


@pytest.fixture
def route_client():
    sales_workflow = FakeSalesWorkflow()
    food_cost_workflow = FakeFoodCostWorkflow()

    async def internal_context():
        return InternalRequestContext(
            tenant_id=str(uuid4()),
            profile_id=str(uuid4()),
            request_id="req-route-test",
            member_id=None,
            scopes=("orders:read", "analytics:read", "menu:read", "financial:read"),
            timezone="America/Mexico_City",
        )

    app.dependency_overrides[require_internal_request] = internal_context
    app.dependency_overrides[get_sales_workflow] = lambda: sales_workflow
    app.dependency_overrides[get_food_cost_workflow] = lambda: food_cost_workflow

    try:
        with TestClient(app) as client:
            yield client, sales_workflow, food_cost_workflow
    finally:
        app.dependency_overrides.clear()


def parse_sse_frames(body: str) -> list[tuple[str, dict]]:
    frames = []
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        event = next(line for line in lines if line.startswith("event: "))
        data = next(line for line in lines if line.startswith("data: "))
        frames.append(
            (
                event.removeprefix("event: "),
                json.loads(data.removeprefix("data: ")),
            )
        )
    return frames


def test_sales_sync_endpoint_remains_available_for_compatibility(route_client):
    client, sales_workflow, _ = route_client

    response = client.post(
        "/internal/ai/sales/messages",
        json={
            "question": "Cuanto vendi ayer?",
            "date_from": "2026-06-17",
            "date_to": "2026-06-17",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["summary"] == "Ayer vendiste 431500."
    assert data["artifact"]["metrics"]["total_sales"] == 431500.0
    assert sales_workflow.run_requests[0][0].question == "Cuanto vendi ayer?"
    assert sales_workflow.run_requests[0][1].timezone == "America/Mexico_City"


def test_food_cost_sync_endpoint_remains_available_for_compatibility(route_client):
    client, _, food_cost_workflow = route_client

    response = client.post(
        "/internal/ai/food-cost/messages",
        json={
            "question": "Que productos tienen food cost alto?",
            "date_from": "2026-06-01",
            "date_to": "2026-06-18",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["summary"] == "Arepa necesita revision."
    assert data["artifact"]["low_margin_products"][0]["name"] == "Arepa"
    assert food_cost_workflow.run_requests[0][0].date_from == "2026-06-01"


def test_sales_stream_endpoint_returns_sse_frames(route_client):
    client, sales_workflow, _ = route_client

    response = client.post(
        "/internal/ai/sales/messages/stream",
        json={"question": "Cuanto vendi ayer?"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse_frames(response.text)
    assert [event for event, _ in frames] == ["run_started", "final"]
    assert frames[-1][1]["status"] == "completed"
    assert frames[-1][1]["summary"] == "Ayer vendiste 431500."
    assert "question" not in frames[-1][1]["artifact_summary"]
    assert sales_workflow.stream_requests[0][0].question == "Cuanto vendi ayer?"
    assert sales_workflow.stream_requests[0][1].timezone == "America/Mexico_City"


def test_food_cost_stream_endpoint_returns_sse_frames(route_client):
    client, _, food_cost_workflow = route_client

    response = client.post(
        "/internal/ai/food-cost/messages/stream",
        json={"question": "Resume food cost."},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse_frames(response.text)
    assert [event for event, _ in frames] == ["run_started", "final"]
    assert frames[-1][1]["status"] == "completed"
    assert frames[-1][1]["summary"] == "Arepa necesita revision."
    assert "question" not in frames[-1][1]["artifact_summary"]
    assert food_cost_workflow.stream_requests[0][0].question == "Resume food cost."
