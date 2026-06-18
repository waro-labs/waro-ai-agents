from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    step_id: UUID | None = None
    tool_name: str = Field(min_length=1, max_length=120)
    arguments: dict[str, Any] = Field(default_factory=dict)
    fields: list[str] | None = None
    idempotency_key: str | None = Field(default=None, max_length=200)
    dry_run: bool = False

    @field_validator("fields")
    @classmethod
    def fields_must_not_be_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("fields must contain at least one field")
        return value


class ToolCallResponse(BaseModel):
    tool_call_id: UUID | None
    tool_name: str
    status: str
    result: Any = None
    result_summary: str | None = None
    error: dict[str, Any] | None = None
