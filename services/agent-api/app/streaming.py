from collections.abc import AsyncIterable, AsyncIterator, Mapping
from datetime import datetime, timezone
import json
from typing import Any, Literal
from uuid import UUID

from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


StreamEventName = Literal[
    "run_started",
    "step_started",
    "agent_step",
    "tool_started",
    "tool_finished",
    "llm_started",
    "token",
    "final",
    "error",
]

STREAM_EVENT_NAMES: tuple[StreamEventName, ...] = (
    "run_started",
    "step_started",
    "agent_step",
    "tool_started",
    "tool_finished",
    "llm_started",
    "token",
    "final",
    "error",
)

STREAM_MEDIA_TYPE = "text/event-stream"
MESSAGE_CONTENT_KEYS = frozenset(
    {
        "content",
        "contents",
        "prompt",
        "prompts",
        "messages",
        "completion",
        "completions",
    }
)
RESERVED_PAYLOAD_KEYS = frozenset({"event", "run_id", "emitted_at"})


class StreamEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: StreamEventName
    data: dict[str, Any] = Field(default_factory=dict)
    run_id: UUID | str | None = None
    emitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("data")
    @classmethod
    def validate_safe_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved_keys = RESERVED_PAYLOAD_KEYS.intersection(value)
        if reserved_keys:
            keys = ", ".join(sorted(reserved_keys))
            raise ValueError(f"stream event data uses reserved keys: {keys}")
        unsafe_path = _find_message_content_key(value)
        if unsafe_path:
            raise ValueError(
                f"stream event data must not include message content at {unsafe_path}"
            )
        return value

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": self.event,
            "emitted_at": self.emitted_at.isoformat(),
        }
        if self.run_id is not None:
            payload["run_id"] = str(self.run_id)
        payload.update(jsonable_encoder(self.data))
        return payload

    def to_sse(self) -> str:
        return format_sse_event(self)


def stream_event(
    event: StreamEventName,
    *,
    data: dict[str, Any] | None = None,
    run_id: UUID | str | None = None,
) -> StreamEvent:
    return StreamEvent(event=event, data=data or {}, run_id=run_id)


def terminal_error_event(
    *,
    error_type: str,
    error_message: str,
    run_id: UUID | str | None = None,
) -> StreamEvent:
    return stream_event(
        "error",
        run_id=run_id,
        data={
            "status": "failed",
            "error": {
                "type": error_type,
                "message": error_message,
            },
        },
    )


def format_sse_event(event: StreamEvent) -> str:
    data = json.dumps(
        event.payload(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event.event}\ndata: {data}\n\n"


async def iter_sse_events(events: AsyncIterable[StreamEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield format_sse_event(event)


def streaming_response(events: AsyncIterable[StreamEvent]) -> StreamingResponse:
    return StreamingResponse(
        iter_sse_events(events),
        media_type=STREAM_MEDIA_TYPE,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _find_message_content_key(value: Any, path: str = "data") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text.lower() in MESSAGE_CONTENT_KEYS:
                return child_path
            unsafe_path = _find_message_content_key(child, child_path)
            if unsafe_path:
                return unsafe_path
    elif isinstance(value, list):
        for index, child in enumerate(value):
            unsafe_path = _find_message_content_key(child, f"{path}[{index}]")
            if unsafe_path:
                return unsafe_path
    return None
