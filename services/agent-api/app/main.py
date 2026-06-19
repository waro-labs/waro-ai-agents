from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.database import DatabasePool
from app.dependencies.internal_auth import InternalRequestContext, require_internal_request
from app.streaming import streaming_response
from app.telemetry import configure_tracing, instrument_app
from app.tools import ToolCallRequest, ToolCallResponse, ToolGateway
from app.tools.catalog import tool_catalog
from app.workflows.agent import AgentWorkflow
from app.workflows.food_cost import FoodCostWorkflow
from app.workflows.models import (
    AgentQuestionRequest,
    AgentWorkflowResponse,
    FoodCostQuestionRequest,
    FoodCostWorkflowResponse,
    SalesQuestionRequest,
    SalesWorkflowResponse,
)
from app.workflows.sales import SalesWorkflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
    configure_tracing(app.state.settings)
    yield
    await DatabasePool.close_pool()


app = FastAPI(
    title="WARO AI Agent API",
    version="0.1.0",
    description="Internal API boundary for WARO AI agent workflows.",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
instrument_app(app, get_settings())  # HTTP generates spans for internal requests


# Factories
def get_tool_gateway() -> ToolGateway:
    return ToolGateway(settings=get_settings())


def get_agent_workflow() -> AgentWorkflow:
    return AgentWorkflow(settings=get_settings())


def get_food_cost_workflow() -> FoodCostWorkflow:
    return FoodCostWorkflow(settings=get_settings())


def get_sales_workflow() -> SalesWorkflow:
    return SalesWorkflow(settings=get_settings())


@app.get("/health", tags=["health"])
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.otel_service_name,
        "environment": settings.environment,
        "dependencies": {
            "postgres": "configured",
            "redis": "configured",
            "phoenix": "configured",
            "tracing": "enabled" if settings.otel_enabled else "disabled",
        },
        "internal_auth": {
            "signature_secret": (
                "configured"
                if settings.is_signature_verification_enabled
                else "not_configured"
            ),
            "signature_verification": (
                "implemented"
                if settings.is_signature_verification_enabled
                else "not_configured"
            ),
        },
    }


@app.post("/internal/tools/call", response_model=ToolCallResponse, tags=["tools"])
async def call_tool(
    request: ToolCallRequest,
    context: InternalRequestContext = Depends(require_internal_request),
    gateway: ToolGateway = Depends(get_tool_gateway),
) -> ToolCallResponse:
    return await gateway.call(request=request, context=context)


@app.get("/internal/tools/catalog", tags=["tools"])
async def list_tool_catalog(
    _: InternalRequestContext = Depends(require_internal_request),
) -> dict[str, Any]:
    return {"tools": tool_catalog()}


@app.post(
    "/internal/ai/messages",
    response_model=AgentWorkflowResponse,
    tags=["ai"],
)
async def ask_agent(
    request: AgentQuestionRequest,
    context: InternalRequestContext = Depends(require_internal_request),
    workflow: AgentWorkflow = Depends(get_agent_workflow),
) -> AgentWorkflowResponse:
    return await workflow.run(request=request, context=context)


@app.post("/internal/ai/messages/stream", tags=["ai"])
async def stream_agent(
    request: AgentQuestionRequest,
    context: InternalRequestContext = Depends(require_internal_request),
    workflow: AgentWorkflow = Depends(get_agent_workflow),
) -> StreamingResponse:
    return streaming_response(workflow.stream(request=request, context=context))


@app.post(
    "/internal/ai/food-cost/messages",
    response_model=FoodCostWorkflowResponse,
    tags=["ai"],
)
async def ask_food_cost(
    request: FoodCostQuestionRequest,
    context: InternalRequestContext = Depends(require_internal_request),
    workflow: FoodCostWorkflow = Depends(get_food_cost_workflow),
) -> FoodCostWorkflowResponse:
    return await workflow.run(request=request, context=context)


@app.post("/internal/ai/food-cost/messages/stream", tags=["ai"])
async def stream_food_cost(
    request: FoodCostQuestionRequest,
    context: InternalRequestContext = Depends(require_internal_request),
    workflow: FoodCostWorkflow = Depends(get_food_cost_workflow),
) -> StreamingResponse:
    return streaming_response(workflow.stream(request=request, context=context))


@app.post(
    "/internal/ai/sales/messages",
    response_model=SalesWorkflowResponse,
    tags=["ai"],
)
async def ask_sales(
    request: SalesQuestionRequest,
    context: InternalRequestContext = Depends(require_internal_request),
    workflow: SalesWorkflow = Depends(get_sales_workflow),
) -> SalesWorkflowResponse:
    return await workflow.run(request=request, context=context)


@app.post("/internal/ai/sales/messages/stream", tags=["ai"])
async def stream_sales(
    request: SalesQuestionRequest,
    context: InternalRequestContext = Depends(require_internal_request),
    workflow: SalesWorkflow = Depends(get_sales_workflow),
) -> StreamingResponse:
    return streaming_response(workflow.stream(request=request, context=context))
