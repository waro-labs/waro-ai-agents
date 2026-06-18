from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI

from app.config import get_settings
from app.database import DatabasePool
from app.dependencies.internal_auth import InternalRequestContext, require_internal_request
from app.telemetry import configure_tracing, instrument_app
from app.tools import ToolCallRequest, ToolCallResponse, ToolGateway


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
instrument_app(app, get_settings())


def get_tool_gateway() -> ToolGateway:
    return ToolGateway(settings=get_settings())


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
