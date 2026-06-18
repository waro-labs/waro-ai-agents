import json
from typing import Any
from uuid import UUID

from app.dependencies.internal_auth import InternalRequestContext
from app.tools.sanitize import sanitize_value, truncate_text


def summarize_result(result: Any) -> str:
    if isinstance(result, dict):
        if isinstance(result.get("data"), list):
            return f"Returned {len(result['data'])} rows."
        if isinstance(result.get("data"), dict):
            keys = ", ".join(sorted(result["data"].keys())[:8])
            return f"Returned data object with keys: {keys}."
        keys = ", ".join(sorted(result.keys())[:8])
        return f"Returned object with keys: {keys}."
    if isinstance(result, list):
        return f"Returned {len(result)} rows."
    return "Returned scalar result."


def bound_json(value: Any, max_bytes: int) -> Any:
    encoded = json.dumps(value, default=str, ensure_ascii=False)
    if len(encoded.encode("utf-8")) <= max_bytes:
        return value
    return {
        "truncated": True,
        "summary": summarize_result(value),
    }


class ToolCallAudit:
    def __init__(self, connection: Any):
        self.connection = connection

    async def start(
        self,
        *,
        context: InternalRequestContext,
        run_id: UUID,
        step_id: UUID | None,
        tool_name: str,
        arguments: dict[str, Any],
        dry_run: bool,
        idempotency_key: str | None,
    ) -> UUID:
        row = await self.connection.fetchrow(
            """
            INSERT INTO ai.tool_calls (
                run_id,
                step_id,
                tenant_id,
                tool_name,
                arguments_json,
                dry_run,
                status,
                idempotency_key
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'running', $7)
            RETURNING id
            """,
            run_id,
            step_id,
            UUID(context.tenant_id),
            tool_name,
            json.dumps(sanitize_value(arguments), default=str),
            dry_run,
            idempotency_key,
        )
        tool_call_id = row["id"]
        await self.action_event(
            context=context,
            run_id=run_id,
            tool_call_id=tool_call_id,
            action="tool_call_started",
            status="running",
            payload={"tool_name": tool_name, "dry_run": dry_run},
        )
        return tool_call_id

    async def finish_success(
        self,
        *,
        context: InternalRequestContext,
        run_id: UUID,
        tool_call_id: UUID,
        result: Any,
        result_summary: str,
        max_bytes: int,
    ) -> None:
        await self.connection.execute(
            """
            UPDATE ai.tool_calls
            SET result_json = $1::jsonb,
                result_summary = $2,
                status = 'succeeded',
                finished_at = now()
            WHERE id = $3
            """,
            json.dumps(bound_json(sanitize_value(result), max_bytes), default=str),
            truncate_text(result_summary, 1000),
            tool_call_id,
        )
        await self.action_event(
            context=context,
            run_id=run_id,
            tool_call_id=tool_call_id,
            action="tool_call_finished",
            status="succeeded",
            payload={"summary": result_summary},
        )

    async def finish_error(
        self,
        *,
        context: InternalRequestContext,
        run_id: UUID,
        tool_call_id: UUID,
        error: dict[str, Any],
    ) -> None:
        sanitized_error = sanitize_value(error)
        await self.connection.execute(
            """
            UPDATE ai.tool_calls
            SET error_json = $1::jsonb,
                status = 'failed',
                finished_at = now()
            WHERE id = $2
            """,
            json.dumps(sanitized_error, default=str),
            tool_call_id,
        )
        await self.action_event(
            context=context,
            run_id=run_id,
            tool_call_id=tool_call_id,
            action="tool_call_failed",
            status="failed",
            payload=sanitized_error,
        )

    async def action_event(
        self,
        *,
        context: InternalRequestContext,
        run_id: UUID,
        tool_call_id: UUID,
        action: str,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        await self.connection.execute(
            """
            INSERT INTO audit.ai_action_events (
                tenant_id,
                run_id,
                tool_call_id,
                actor_profile_id,
                action,
                risk_level,
                status,
                payload
            )
            VALUES ($1, $2, $3, $4, $5, 'low', $6, $7::jsonb)
            """,
            UUID(context.tenant_id),
            run_id,
            tool_call_id,
            UUID(context.profile_id),
            action,
            status,
            json.dumps(sanitize_value(payload), default=str),
        )
