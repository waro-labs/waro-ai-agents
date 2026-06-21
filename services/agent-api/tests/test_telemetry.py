import pytest

from app.telemetry import RequestTraceMiddleware


class RecordingTracer:
    def __init__(self):
        self.names = []

    def start_as_current_span(self, name):
        self.names.append(name)
        return RecordingSpan()


class RecordingSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_attribute(self, *_args):
        return None

    def set_status(self, *_args):
        return None

    def record_exception(self, *_args):
        return None


async def ok_app(_scope, _receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def send(_message):
    return None


@pytest.mark.asyncio
async def test_request_trace_middleware_skips_health_spans():
    middleware = RequestTraceMiddleware(ok_app, service_name="test")
    tracer = RecordingTracer()
    middleware.tracer = tracer

    await middleware(
        {"type": "http", "method": "GET", "path": "/health"},
        receive,
        send,
    )

    assert tracer.names == []


@pytest.mark.asyncio
async def test_request_trace_middleware_traces_non_health_requests():
    middleware = RequestTraceMiddleware(ok_app, service_name="test")
    tracer = RecordingTracer()
    middleware.tracer = tracer

    await middleware(
        {"type": "http", "method": "POST", "path": "/internal/ai/sales/messages/stream"},
        receive,
        send,
    )

    assert tracer.names == ["POST /internal/ai/sales/messages/stream"]
