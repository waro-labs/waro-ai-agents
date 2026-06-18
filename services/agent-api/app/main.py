from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.config import get_settings
from app.database import DatabasePool


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
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
        },
        "internal_auth": {
            "signature_secret": (
                "configured"
                if settings.is_signature_verification_enabled
                else "not_configured"
            ),
            "signature_verification": "not_implemented",
        },
    }
