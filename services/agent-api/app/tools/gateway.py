from typing import Any

from fastapi import HTTPException, status
from pydantic import ValidationError

from app.config import Settings
from app.database import get_db_connection
from app.dependencies.internal_auth import InternalRequestContext
from app.tools.allowlist import coerce_args, get_tool_spec, resolve_fields
from app.tools.audit import ToolCallAudit, summarize_result
from app.tools.models import ToolCallRequest, ToolCallResponse
from app.tools.runner import ToolRunError, WaroCliRunner
from app.tools.sanitize import sanitize_text


class ToolGateway:
    def __init__(
        self,
        *,
        settings: Settings,
        runner: WaroCliRunner | None = None,
        connection_factory: Any = get_db_connection,
    ):
        self.settings = settings
        self.runner = runner or WaroCliRunner(settings)
        self.connection_factory = connection_factory

    async def call(
        self,
        *,
        request: ToolCallRequest,
        context: InternalRequestContext,
    ) -> ToolCallResponse:
        spec = get_tool_spec(request.tool_name)
        if spec is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "unknown_tool", "tool_name": request.tool_name},
            )

        if spec.scope not in context.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "missing_scope", "required_scope": spec.scope},
            )

        try:
            args = coerce_args(spec, request.arguments)
            fields = resolve_fields(spec, request.fields)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "invalid_tool_arguments", "message": str(exc)},
            ) from exc

        sanitized_arguments = {
            "arguments": args.model_dump(by_alias=True, mode="json", exclude_none=True),
            "fields": list(fields),
        }

        async with self.connection_factory() as connection:
            audit = ToolCallAudit(connection)
            tool_call_id = await audit.start(
                context=context,
                run_id=request.run_id,
                step_id=request.step_id,
                tool_name=spec.name,
                arguments=sanitized_arguments,
                dry_run=request.dry_run,
                idempotency_key=request.idempotency_key,
            )
            try:
                run_result = await self.runner.run(
                    spec=spec,
                    args=args,
                    fields=fields,
                    dry_run=request.dry_run,
                )
            except ToolRunError as exc:
                error_context = exc.to_context()
                for key in ("stderr", "stdout", "message"):
                    if isinstance(error_context.get(key), str):
                        error_context[key] = sanitize_text(
                            error_context[key],
                            secrets=[self.settings.waro_api_key or ""],
                        )
                await audit.finish_error(
                    context=context,
                    run_id=request.run_id,
                    tool_call_id=tool_call_id,
                    error=error_context,
                )
                return ToolCallResponse(
                    tool_call_id=tool_call_id,
                    tool_name=spec.name,
                    status="failed",
                    error=error_context,
                )

            result_summary = summarize_result(run_result.result)
            await audit.finish_success(
                context=context,
                run_id=request.run_id,
                tool_call_id=tool_call_id,
                result=run_result.result,
                result_summary=result_summary,
                max_bytes=self.settings.tool_result_max_bytes,
            )
            return ToolCallResponse(
                tool_call_id=tool_call_id,
                tool_name=spec.name,
                status="succeeded",
                result=run_result.result,
                result_summary=result_summary,
            )
