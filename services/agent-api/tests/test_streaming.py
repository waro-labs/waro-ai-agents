import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.streaming import (
    STREAM_EVENT_NAMES,
    STREAM_MEDIA_TYPE,
    StreamEvent,
    format_sse_event,
    iter_sse_events,
    stream_event,
    streaming_response,
    terminal_error_event,
)


def parse_sse_frame(frame: str) -> tuple[str, dict]:
    lines = frame.splitlines()
    event_line = next(line for line in lines if line.startswith("event: "))
    data_line = next(line for line in lines if line.startswith("data: "))
    return event_line.removeprefix("event: "), json.loads(
        data_line.removeprefix("data: ")
    )


def test_stream_event_names_define_initial_contract():
    assert STREAM_EVENT_NAMES == (
        "run_started",
        "step_started",
        "tool_started",
        "tool_finished",
        "llm_started",
        "token",
        "final",
        "error",
    )


def test_stream_event_formats_sse_frame_with_json_payload():
    run_id = uuid4()
    event = stream_event(
        "tool_finished",
        run_id=run_id,
        data={
            "tool_name": "waro.menu.products",
            "status": "succeeded",
            "tool_call_id": str(uuid4()),
            "result_summary": "Returned 1 row.",
        },
    )

    frame = format_sse_event(event)
    event_name, data = parse_sse_frame(frame)

    assert frame.endswith("\n\n")
    assert event_name == "tool_finished"
    assert data["event"] == "tool_finished"
    assert data["run_id"] == str(run_id)
    assert data["tool_name"] == "waro.menu.products"
    assert data["status"] == "succeeded"
    assert data["result_summary"] == "Returned 1 row."
    assert "emitted_at" in data


def test_stream_event_escapes_newlines_inside_data_json():
    frame = stream_event(
        "final",
        data={"summary": "Linea 1\nLinea 2", "artifact_summary": {"rows": 1}},
    ).to_sse()

    assert frame.count("data: ") == 1
    assert "Linea 1\\nLinea 2" in frame
    _, data = parse_sse_frame(frame)
    assert data["summary"] == "Linea 1\nLinea 2"


def test_terminal_error_event_uses_sanitized_terminal_shape():
    event = terminal_error_event(
        error_type="RuntimeError",
        error_message="Tool execution failed.",
        run_id="run-1",
    )

    event_name, data = parse_sse_frame(event.to_sse())

    assert event_name == "error"
    assert data == {
        "event": "error",
        "emitted_at": data["emitted_at"],
        "run_id": "run-1",
        "status": "failed",
        "error": {
            "type": "RuntimeError",
            "message": "Tool execution failed.",
        },
    }


def test_stream_event_rejects_reserved_payload_keys():
    with pytest.raises(ValidationError, match="reserved keys"):
        StreamEvent(event="run_started", data={"event": "final"})


def test_stream_event_rejects_prompt_or_message_content_payloads():
    with pytest.raises(ValidationError, match="message content"):
        StreamEvent(
            event="llm_started",
            data={
                "provider": "kimi",
                "messages": [{"role": "user", "content": "Cuanto vendi ayer?"}],
            },
        )


@pytest.mark.asyncio
async def test_iter_sse_events_yields_formatted_frames_in_order():
    async def events():
        yield stream_event("run_started", data={"workflow": "sales"})
        yield stream_event("final", data={"status": "completed", "summary": "Listo."})

    frames = [frame async for frame in iter_sse_events(events())]

    assert [parse_sse_frame(frame)[0] for frame in frames] == ["run_started", "final"]


def test_streaming_response_sets_sse_headers():
    async def events():
        yield stream_event("run_started", data={"workflow": "food_cost"})

    response = streaming_response(events())

    assert response.media_type == STREAM_MEDIA_TYPE
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
