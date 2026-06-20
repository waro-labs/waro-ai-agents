from typing import Any

from fastapi import HTTPException, status
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import ValidationError

from app.config import Settings
from app.database import get_db_connection
from app.dependencies.internal_auth import InternalRequestContext
from app.telemetry import current_trace_ids, mark_span_error
from app.tools.allowlist import coerce_args, get_tool_spec, resolve_fields
from app.tools.audit import ToolCallAudit, summarize_result
from app.tools.contracts import WaroContractRegistry
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
        self.contracts = WaroContractRegistry(settings)
        self.connection_factory = connection_factory
        self.tracer = trace.get_tracer(__name__)

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
            contract = await self.contracts.get(spec.name)
            fields = resolve_fields(spec, request.fields, contract=contract)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "invalid_tool_arguments", "message": str(exc)},
            ) from exc

        sanitized_arguments = {
            "arguments": args.model_dump(by_alias=True, mode="json", exclude_none=True),
            "fields": list(fields),
        }

        with self.tracer.start_as_current_span(f"tool.{spec.name}") as span:
            span.set_attribute("waro.tool.name", spec.name)
            span.set_attribute("waro.tool.scope", spec.scope)
            span.set_attribute("waro.tool.dry_run", request.dry_run)
            span.set_attribute("waro.tool.contract.source", "cli_schema" if contract else "fallback")
            span.set_attribute(
                "waro.tool.fields",
                ",".join(fields),
            )
            span.set_attribute(
                "waro.tool.arguments",
                sanitize_text(
                    str(sanitized_arguments["arguments"]),
                    secrets=[self.settings.waro_api_key or ""],
                ),
            )
            span.set_attribute("waro.tenant_id", context.tenant_id)
            span.set_attribute("waro.request_id", context.request_id)
            span.set_attribute("waro.run_id", str(request.run_id))
            if request.step_id:
                span.set_attribute("waro.step_id", str(request.step_id))

            trace_id, span_id = current_trace_ids()
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
                    trace_id=trace_id,
                    span_id=span_id,
                )
                span.set_attribute("waro.tool_call_id", str(tool_call_id))
                span.add_event(
                    "cli.started",
                    {
                        "waro.tool.name": spec.name,
                        "waro.tool.fields": ",".join(fields),
                    },
                )
                try:
                    run_result = await self.runner.run(
                        spec=spec,
                        args=args,
                        fields=fields,
                        dry_run=request.dry_run,
                    )
                except ToolRunError as exc:
                    mark_span_error(span, exc)
                    error_context = exc.to_context()
                    for key in ("stderr", "stdout", "message"):
                        if isinstance(error_context.get(key), str):
                            error_context[key] = sanitize_text(
                                error_context[key],
                                secrets=[self.settings.waro_api_key or ""],
                            )
                    span.set_attribute("waro.tool.status", "failed")
                    span.set_attribute("waro.tool.error_type", type(exc).__name__)
                    span.set_attribute("waro.tool.returncode", exc.returncode or 0)
                    span.add_event(
                        "cli.failed",
                        {
                            "waro.tool.name": spec.name,
                            "waro.tool.error_type": type(exc).__name__,
                            "waro.tool.returncode": exc.returncode or 0,
                        },
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
                result_shape = self._result_shape(run_result.result)
                span.set_attribute("waro.tool.status", "succeeded")
                span.set_attribute("waro.tool.result_summary", result_summary)
                span.set_attribute("waro.tool.result.kind", result_shape["kind"])
                if result_shape.get("row_count") is not None:
                    span.set_attribute("waro.tool.result.row_count", result_shape["row_count"])
                if result_shape.get("data_count") is not None:
                    span.set_attribute("waro.tool.result.data_count", result_shape["data_count"])
                if result_shape.get("products_count") is not None:
                    span.set_attribute(
                        "waro.tool.result.products_count",
                        result_shape["products_count"],
                    )
                span.add_event(
                    "cli.completed",
                    {
                        "waro.tool.name": spec.name,
                        "waro.tool.result.kind": result_shape["kind"],
                        "waro.tool.result.row_count": result_shape.get("row_count") or 0,
                        "waro.tool.result.data_count": result_shape.get("data_count") or 0,
                        "waro.tool.result.products_count": result_shape.get("products_count") or 0,
                    },
                )
                span.set_status(Status(StatusCode.OK))
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

    def _result_shape(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            rows = result.get("rows")
            data = result.get("data")
            products = result.get("products")
            return {
                "kind": "dict",
                "row_count": len(rows) if isinstance(rows, list) else None,
                "data_count": len(data) if isinstance(data, list) else None,
                "products_count": len(products) if isinstance(products, list) else None,
            }
        if isinstance(result, list):
            return {
                "kind": "list",
                "row_count": len(result),
                "data_count": None,
                "products_count": None,
            }
        return {
            "kind": type(result).__name__,
            "row_count": None,
            "data_count": None,
            "products_count": None,
        }
