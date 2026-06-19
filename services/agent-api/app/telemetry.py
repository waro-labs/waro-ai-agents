from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import Settings

_CONFIGURED = False


def configure_tracing(settings: Settings) -> None:
    global _CONFIGURED
    if _CONFIGURED or not settings.otel_enabled:
        return

    resource = Resource.create(
        {
            SERVICE_NAME: settings.otel_service_name,
            DEPLOYMENT_ENVIRONMENT: settings.environment,
        }
    )
    provider = TracerProvider(resource=resource)
    headers = None
    if settings.phoenix_api_key:
        headers = (("authorization", f"Bearer {settings.phoenix_api_key}"),)
    exporter = OTLPSpanExporter(
        endpoint=settings.phoenix_collector_endpoint,
        headers=headers,
        insecure=settings.phoenix_collector_endpoint.startswith("http://"),
        timeout=settings.otel_export_timeout_seconds,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def current_trace_ids() -> tuple[str | None, str | None]:
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return None, None
    return f"{context.trace_id:032x}", f"{context.span_id:016x}"


def mark_span_error(span: Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


class RequestTraceMiddleware:
    def __init__(self, app: ASGIApp, service_name: str):
        self.app = app
        self.service_name = service_name
        self.tracer = trace.get_tracer(__name__)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET"))
        path = str(scope.get("path", ""))
        span_name = f"{method} {path}"
        status_code: int | None = None

        async def send_with_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("service.name", self.service_name)
            span.set_attribute("http.request.method", method)
            span.set_attribute("url.path", path)
            try:
                await self.app(scope, receive, send_with_status)
            except Exception as exc:
                mark_span_error(span, exc)
                raise
            finally:
                if status_code is not None:
                    span.set_attribute("http.response.status_code", status_code)
                    if status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR))


def instrument_app(app: Any, settings: Settings) -> None:
    if not settings.otel_enabled:
        return
    app.add_middleware(RequestTraceMiddleware, service_name=settings.otel_service_name)
