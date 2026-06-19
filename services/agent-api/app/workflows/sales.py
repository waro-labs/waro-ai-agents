from collections.abc import AsyncIterator
import json
from datetime import datetime, timedelta
from typing import Any, TypedDict
from uuid import UUID
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import Settings
from app.database import get_db_connection
from app.dependencies.internal_auth import InternalRequestContext
from app.evals.sales import evaluate_sales_artifact
from app.llm import LLMAdapter, get_llm_adapter
from app.llm.prompts import sales_summary_messages
from app.streaming import StreamEvent, stream_event, terminal_error_event
from app.tools import ToolCallRequest, ToolCallResponse, ToolGateway
from app.tools.sanitize import sanitize_value, truncate_text
from app.workflows.models import SalesQuestionRequest, SalesWorkflowResponse


class SalesGraphState(TypedDict, total=False):
    request: SalesQuestionRequest
    context: InternalRequestContext
    conversation_id: UUID
    input_message_id: UUID
    run_id: UUID
    tool_calls: list[dict[str, Any]]
    artifact: dict[str, Any]
    summary: str
    evals: list[Any]
    output_message_id: UUID


class SalesWorkflow:
    agent_name = "sales_agent"
    graph_name = "sales_langgraph_v1"

    def __init__(
        self,
        *,
        settings: Settings,
        gateway: ToolGateway | None = None,
        llm_adapter: LLMAdapter | None = None,
        connection_factory: Any = get_db_connection,
    ):
        self.settings = settings
        self.connection_factory = connection_factory
        self.gateway = gateway or ToolGateway(
            settings=settings,
            connection_factory=connection_factory,
        )
        self.llm_adapter = llm_adapter or get_llm_adapter(settings)
        self.tracer = trace.get_tracer(__name__)
        self.graph = self._build_graph()

    async def run(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
    ) -> SalesWorkflowResponse:
        conversation_id, input_message_id, run_id = await self._start_run(
            request=request,
            context=context,
        )
        try:
            state = await self.graph.ainvoke(
                {
                    "request": request,
                    "context": context,
                    "conversation_id": conversation_id,
                    "input_message_id": input_message_id,
                    "run_id": run_id,
                }
            )
            return SalesWorkflowResponse(
                conversation_id=conversation_id,
                run_id=run_id,
                input_message_id=input_message_id,
                output_message_id=state["output_message_id"],
                status="completed",
                artifact=state["artifact"],
                summary=state["summary"],
                evals=state["evals"],
            )
        except Exception as exc:
            await self._mark_failed(
                context=context,
                run_id=run_id,
                error={"message": str(exc), "type": type(exc).__name__},
            )
            raise

    async def stream(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
    ) -> AsyncIterator[StreamEvent]:
        conversation_id, input_message_id, run_id = await self._start_run(
            request=request,
            context=context,
        )
        yield stream_event(
            "run_started",
            run_id=run_id,
            data={
                "workflow": self.graph_name,
                "agent_name": self.agent_name,
                "conversation_id": str(conversation_id),
                "input_message_id": str(input_message_id),
                "status": "running",
            },
        )

        try:
            yield stream_event(
                "step_started",
                run_id=run_id,
                data={"step_type": "router", "name": "sales_intent_router"},
            )
            await self._route_sales(
                {
                    "request": request,
                    "context": context,
                    "run_id": run_id,
                }
            )
            tool_calls: list[dict[str, Any]] = []
            async for event in self._stream_required_tools(
                request=request,
                context=context,
                run_id=run_id,
                tool_calls=tool_calls,
            ):
                yield event

            artifact = self._build_artifact(request=request, tool_calls=tool_calls)
            if self.settings.llm_provider != "disabled":
                yield stream_event(
                    "llm_started",
                    run_id=run_id,
                    data={
                        "provider": self.llm_adapter.provider,
                        "model": self.settings.kimi_model
                        if self.settings.llm_provider == "kimi"
                        else None,
                    },
                )
            summary_holder: dict[str, str] = {}
            async for event in self._stream_summary_with_llm(
                artifact=artifact,
                context=context,
                run_id=run_id,
                summary_holder=summary_holder,
            ):
                yield event
            summary = summary_holder["summary"]
            evals = evaluate_sales_artifact(artifact)
            output_message_id = await self._finish_run(
                context=context,
                conversation_id=conversation_id,
                input_message_id=input_message_id,
                run_id=run_id,
                artifact=artifact,
                summary=summary,
                evals=evals,
            )
            yield stream_event(
                "final",
                run_id=run_id,
                data={
                    "status": "completed",
                    "conversation_id": str(conversation_id),
                    "input_message_id": str(input_message_id),
                    "output_message_id": str(output_message_id),
                    "summary": summary,
                    "artifact_summary": self._stream_artifact_summary(artifact),
                    "evals": [
                        {
                            "evaluator_name": eval_result.evaluator_name,
                            "score": eval_result.score,
                            "passed": eval_result.passed,
                        }
                        for eval_result in evals
                    ],
                },
            )
        except Exception as exc:
            await self._mark_failed(
                context=context,
                run_id=run_id,
                error={"message": str(exc), "type": type(exc).__name__},
            )
            yield terminal_error_event(
                run_id=run_id,
                error_type=type(exc).__name__,
                error_message=truncate_text(str(exc), 240),
            )

    def _build_graph(self):
        graph = StateGraph(SalesGraphState)
        graph.add_node("route_sales", self._route_sales)
        graph.add_node("call_sales_tools", self._call_sales_tools_node)
        graph.add_node("build_artifact", self._build_artifact_node)
        graph.add_node("finish_run", self._finish_run_node)
        graph.add_edge(START, "route_sales")
        graph.add_edge("route_sales", "call_sales_tools")
        graph.add_edge("call_sales_tools", "build_artifact")
        graph.add_edge("build_artifact", "finish_run")
        graph.add_edge("finish_run", END)
        return graph.compile()

    async def _route_sales(self, state: SalesGraphState) -> SalesGraphState:
        await self._record_step(
            run_id=state["run_id"],
            tenant_id=state["context"].tenant_id,
            step_type="router",
            name="sales_intent_router",
            input_json={"question": state["request"].question},
            output_json={"intent": "sales_metrics"},
            output_summary="Routed request to the sales workflow.",
        )
        return {}

    async def _call_sales_tools_node(self, state: SalesGraphState) -> SalesGraphState:
        return {
            "tool_calls": await self._call_required_tools(
                request=state["request"],
                context=state["context"],
                run_id=state["run_id"],
            )
        }

    async def _build_artifact_node(self, state: SalesGraphState) -> SalesGraphState:
        artifact = self._build_artifact(
            request=state["request"],
            tool_calls=state["tool_calls"],
        )
        summary = await self._build_summary_with_llm(
            artifact=artifact,
            context=state["context"],
            run_id=state["run_id"],
        )
        return {
            "artifact": artifact,
            "summary": summary,
            "evals": evaluate_sales_artifact(artifact),
        }

    async def _finish_run_node(self, state: SalesGraphState) -> SalesGraphState:
        output_message_id = await self._finish_run(
            context=state["context"],
            conversation_id=state["conversation_id"],
            input_message_id=state["input_message_id"],
            run_id=state["run_id"],
            artifact=state["artifact"],
            summary=state["summary"],
            evals=state["evals"],
        )
        return {"output_message_id": output_message_id}

    async def _start_run(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
    ) -> tuple[UUID, UUID, UUID]:
        async with self.connection_factory() as connection:
            conversation_id = request.conversation_id
            if conversation_id is None:
                row = await connection.fetchrow(
                    """
                    INSERT INTO ai.conversations (
                        tenant_id,
                        created_by_profile_id,
                        title,
                        metadata
                    )
                    VALUES ($1, $2, $3, $4::jsonb)
                    RETURNING id
                    """,
                    UUID(context.tenant_id),
                    UUID(context.profile_id),
                    truncate_text(request.question, 120),
                    json.dumps({"source": "sales_workflow"}),
                )
                conversation_id = row["id"]

            message_row = await connection.fetchrow(
                """
                INSERT INTO ai.messages (
                    conversation_id,
                    tenant_id,
                    role,
                    content,
                    content_sanitized,
                    metadata
                )
                VALUES ($1, $2, 'user', $3, $4, $5::jsonb)
                RETURNING id
                """,
                conversation_id,
                UUID(context.tenant_id),
                request.question,
                truncate_text(request.question),
                json.dumps({"workflow": self.graph_name}),
            )
            run_row = await connection.fetchrow(
                """
                INSERT INTO ai.runs (
                    conversation_id,
                    tenant_id,
                    input_message_id,
                    agent_name,
                    graph_name,
                    status
                )
                VALUES ($1, $2, $3, $4, $5, 'running')
                RETURNING id
                """,
                conversation_id,
                UUID(context.tenant_id),
                message_row["id"],
                self.agent_name,
                self.graph_name,
            )
            await connection.execute(
                """
                UPDATE ai.conversations
                SET updated_at = now()
                WHERE id = $1
                """,
                conversation_id,
            )
            return conversation_id, message_row["id"], run_row["id"]

    async def _record_step(
        self,
        *,
        run_id: UUID,
        tenant_id: str,
        step_type: str,
        name: str,
        input_json: dict[str, Any] | None = None,
        output_json: dict[str, Any] | None = None,
        output_summary: str | None = None,
    ) -> UUID:
        async with self.connection_factory() as connection:
            row = await connection.fetchrow(
                """
                INSERT INTO ai.steps (
                    run_id,
                    tenant_id,
                    step_type,
                    name,
                    input_json,
                    output_json,
                    output_summary
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7)
                RETURNING id
                """,
                run_id,
                UUID(tenant_id),
                step_type,
                name,
                json.dumps(sanitize_value(input_json or {}), default=str),
                json.dumps(sanitize_value(output_json or {}), default=str),
                output_summary,
            )
            return row["id"]

    async def _call_required_tools(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
    ) -> list[dict[str, Any]]:
        _, response = await self._call_sales_metrics(
            request=request,
            context=context,
            run_id=run_id,
        )
        return [self._tool_call_record(response)]

    async def _stream_required_tools(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
        tool_calls: list[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        step_id, arguments, fields = await self._prepare_sales_metrics_call(
            request=request,
            context=context,
            run_id=run_id,
        )
        yield stream_event(
            "tool_started",
            run_id=run_id,
            data={
                "step_id": str(step_id),
                "tool_name": "waro.sales.metrics",
            },
        )
        response = await self.gateway.call(
            request=ToolCallRequest(
                run_id=run_id,
                step_id=step_id,
                tool_name="waro.sales.metrics",
                arguments=arguments,
                fields=fields,
                idempotency_key=f"{run_id}:1:waro.sales.metrics",
            ),
            context=context,
        )
        tool_calls.append(self._tool_call_record(response))
        yield stream_event(
            "tool_finished",
            run_id=run_id,
            data={
                "step_id": str(step_id),
                "tool_name": response.tool_name,
                "status": response.status,
                "tool_call_id": str(response.tool_call_id)
                if response.tool_call_id
                else None,
                "result_summary": response.result_summary,
                "error_type": self._stream_error_type(response.error),
            },
        )

    async def _call_sales_metrics(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
    ) -> tuple[UUID, ToolCallResponse]:
        step_id, arguments, fields = await self._prepare_sales_metrics_call(
            request=request,
            context=context,
            run_id=run_id,
        )
        response = await self.gateway.call(
            request=ToolCallRequest(
                run_id=run_id,
                step_id=step_id,
                tool_name="waro.sales.metrics",
                arguments=arguments,
                fields=fields,
                idempotency_key=f"{run_id}:1:waro.sales.metrics",
            ),
            context=context,
        )
        return step_id, response

    async def _prepare_sales_metrics_call(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
    ) -> tuple[UUID, dict[str, Any], list[str]]:
        period = self._resolve_period(request)
        arguments = {
            "date-from": period["date_from"],
            "date-to": period["date_to"],
            "group-by": request.group_by,
        }
        fields = ["totalSales", "orderCount", "avgTicket", "series"]
        step_id = await self._record_step(
            run_id=run_id,
            tenant_id=context.tenant_id,
            step_type="tool",
            name="waro.sales.metrics",
            input_json={"arguments": arguments, "fields": fields},
        )
        return step_id, arguments, fields

    def _tool_call_record(self, response: ToolCallResponse) -> dict[str, Any]:
        return {
            "tool_name": response.tool_name,
            "status": response.status,
            "tool_call_id": str(response.tool_call_id) if response.tool_call_id else None,
            "result": sanitize_value(response.result),
            "result_summary": response.result_summary,
            "error": sanitize_value(response.error),
        }

    def _build_artifact(
        self,
        *,
        request: SalesQuestionRequest,
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        period = self._resolve_period(request)
        rows = self._rows_for(tool_calls, "waro.sales.metrics")
        first_row = rows[0] if rows else {}
        metrics = sanitize_value(
            {
                "total_sales": first_row.get("totalSales"),
                "order_count": first_row.get("orderCount"),
                "avg_ticket": first_row.get("avgTicket"),
                "series": first_row.get("series"),
            }
        )
        highlights = self._highlights(metrics)
        return {
            "question": request.question,
            "period": period,
            "metrics": metrics,
            "highlights": highlights,
            "explanation": "Sales analysis uses the WARO sales metrics tool for the requested period.",
            "tool_calls": [
                {
                    "tool_name": call["tool_name"],
                    "status": call["status"],
                    "tool_call_id": call["tool_call_id"],
                    "result_summary": call["result_summary"],
                    "error": call["error"],
                }
                for call in tool_calls
            ],
        }

    def _stream_artifact_summary(self, artifact: dict[str, Any]) -> dict[str, Any]:
        metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
        return {
            "period": artifact.get("period"),
            "metrics": {
                "total_sales": metrics.get("total_sales"),
                "order_count": metrics.get("order_count"),
                "avg_ticket": metrics.get("avg_ticket"),
            },
            "tool_calls": [
                {
                    "tool_name": call.get("tool_name"),
                    "status": call.get("status"),
                    "tool_call_id": call.get("tool_call_id"),
                    "result_summary": call.get("result_summary"),
                }
                for call in artifact.get("tool_calls", [])
                if isinstance(call, dict)
            ],
        }

    def _stream_error_type(self, error: Any) -> str | None:
        if not error:
            return None
        if isinstance(error, dict):
            value = error.get("type") or error.get("error") or error.get("code")
            return str(value) if value else "tool_error"
        return type(error).__name__

    def _resolve_period(self, request: SalesQuestionRequest) -> dict[str, str]:
        if request.date_from and request.date_to:
            return {"date_from": request.date_from, "date_to": request.date_to}
        yesterday = datetime.now(ZoneInfo("America/Bogota")).date() - timedelta(days=1)
        value = yesterday.isoformat()
        return {"date_from": request.date_from or value, "date_to": request.date_to or value}

    def _rows_for(self, tool_calls: list[dict[str, Any]], tool_name: str) -> list[dict[str, Any]]:
        for call in tool_calls:
            if call["tool_name"] != tool_name or call["status"] != "succeeded":
                continue
            result = call.get("result")
            if isinstance(result, dict) and isinstance(result.get("data"), list):
                return [row for row in result["data"] if isinstance(row, dict)]
            if isinstance(result, list):
                return [row for row in result if isinstance(row, dict)]
        return []

    def _highlights(self, metrics: dict[str, Any]) -> list[str]:
        highlights = []
        if metrics.get("total_sales") is not None:
            highlights.append(f"Total sales: {metrics['total_sales']}.")
        if metrics.get("order_count") is not None:
            highlights.append(f"Orders: {metrics['order_count']}.")
        if metrics.get("avg_ticket") is not None:
            highlights.append(f"Average ticket: {metrics['avg_ticket']}.")
        return highlights or ["No sales metrics were returned for the selected period."]

    def _build_summary(self, artifact: dict[str, Any]) -> str:
        metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
        total_sales = metrics.get("total_sales")
        avg_ticket = metrics.get("avg_ticket")
        if total_sales is None and avg_ticket is None:
            return "Sales workflow completed without returned sales metrics."
        parts = []
        if total_sales is not None:
            parts.append(f"total sales {total_sales}")
        if avg_ticket is not None:
            parts.append(f"average ticket {avg_ticket}")
        return "Sales workflow completed with " + " and ".join(parts) + "."

    async def _build_summary_with_llm(
        self,
        *,
        artifact: dict[str, Any],
        context: InternalRequestContext,
        run_id: UUID,
    ) -> str:
        fallback_summary = self._build_summary(artifact)
        if self.settings.llm_provider == "disabled":
            return fallback_summary

        with self.tracer.start_as_current_span("llm.sales.summary") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", self.llm_adapter.provider)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            if self.settings.llm_provider == "kimi":
                span.set_attribute("llm.model", self.settings.kimi_model)
                span.set_attribute("llm.model_name", self.settings.kimi_model)

            try:
                response = await self.llm_adapter.complete(
                    messages=sales_summary_messages(artifact),
                    temperature=0.2,
                )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                await self._record_step(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    step_type="llm",
                    name="sales_summary",
                    input_json={
                        "provider": self.settings.llm_provider,
                        "model": self.settings.kimi_model
                        if self.settings.llm_provider == "kimi"
                        else None,
                    },
                    output_json={"fallback": True, "error_type": type(exc).__name__},
                    output_summary=fallback_summary,
                )
                return fallback_summary

            summary = response.content.strip() or fallback_summary
            span.set_attribute("llm.model", response.model)
            span.set_attribute("llm.model_name", response.model)
            span.set_attribute("llm.response.provider", response.provider)
            if response.input_tokens is not None:
                span.set_attribute("llm.usage.prompt_tokens", response.input_tokens)
                span.set_attribute("llm.token_count.prompt", response.input_tokens)
            if response.output_tokens is not None:
                span.set_attribute("llm.usage.completion_tokens", response.output_tokens)
                span.set_attribute("llm.token_count.completion", response.output_tokens)
            if response.total_tokens is not None:
                span.set_attribute("llm.usage.total_tokens", response.total_tokens)
                span.set_attribute("llm.token_count.total", response.total_tokens)
            if response.prompt_cost_usd is not None:
                span.set_attribute("llm.cost.prompt", response.prompt_cost_usd)
            if response.completion_cost_usd is not None:
                span.set_attribute("llm.cost.completion", response.completion_cost_usd)
            if response.estimated_cost_usd is not None:
                span.set_attribute("llm.cost.estimated_usd", response.estimated_cost_usd)
                span.set_attribute("llm.cost.total", response.estimated_cost_usd)
            span.set_attribute("llm.cost.source", response.cost_source)
            span.set_status(Status(StatusCode.OK))
            await self._record_step(
                run_id=run_id,
                tenant_id=context.tenant_id,
                step_type="llm",
                name="sales_summary",
                input_json={
                    "provider": response.provider,
                    "model": response.model,
                    "message_count": 2,
                },
                output_json={
                    "fallback": summary == fallback_summary,
                    "llm_usage": {
                        "input_count": response.input_tokens,
                        "output_count": response.output_tokens,
                        "total_count": response.total_tokens,
                    },
                    "llm_cost": {
                        "estimated_cost_usd": response.estimated_cost_usd,
                        "source": response.cost_source,
                    },
                },
                output_summary=summary,
            )
            return summary

    async def _stream_summary_with_llm(
        self,
        *,
        artifact: dict[str, Any],
        context: InternalRequestContext,
        run_id: UUID,
        summary_holder: dict[str, str],
    ) -> AsyncIterator[StreamEvent]:
        stream_complete = getattr(self.llm_adapter, "stream_complete", None)
        if self.settings.llm_provider == "disabled" or stream_complete is None:
            summary_holder["summary"] = await self._build_summary_with_llm(
                artifact=artifact,
                context=context,
                run_id=run_id,
            )
            return

        fallback_summary = self._build_summary(artifact)
        with self.tracer.start_as_current_span("llm.sales.summary") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", self.llm_adapter.provider)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            if self.settings.llm_provider == "kimi":
                span.set_attribute("llm.model", self.settings.kimi_model)
                span.set_attribute("llm.model_name", self.settings.kimi_model)

            try:
                response = None
                async for chunk in stream_complete(
                    messages=sales_summary_messages(artifact),
                    temperature=0.2,
                ):
                    if chunk.text:
                        yield stream_event(
                            "token",
                            run_id=run_id,
                            data={
                                "provider": self.llm_adapter.provider,
                                "model": self.settings.kimi_model
                                if self.settings.llm_provider == "kimi"
                                else None,
                                "text": chunk.text,
                            },
                        )
                    if chunk.response is not None:
                        response = chunk.response
                if response is None:
                    raise RuntimeError("LLM stream did not include a final response.")
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                await self._record_step(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    step_type="llm",
                    name="sales_summary",
                    input_json={
                        "provider": self.settings.llm_provider,
                        "model": self.settings.kimi_model
                        if self.settings.llm_provider == "kimi"
                        else None,
                    },
                    output_json={"fallback": True, "error_type": type(exc).__name__},
                    output_summary=fallback_summary,
                )
                summary_holder["summary"] = fallback_summary
                return

            summary = response.content.strip() or fallback_summary
            span.set_attribute("llm.model", response.model)
            span.set_attribute("llm.model_name", response.model)
            span.set_attribute("llm.response.provider", response.provider)
            if response.input_tokens is not None:
                span.set_attribute("llm.usage.prompt_tokens", response.input_tokens)
                span.set_attribute("llm.token_count.prompt", response.input_tokens)
            if response.output_tokens is not None:
                span.set_attribute("llm.usage.completion_tokens", response.output_tokens)
                span.set_attribute("llm.token_count.completion", response.output_tokens)
            if response.total_tokens is not None:
                span.set_attribute("llm.usage.total_tokens", response.total_tokens)
                span.set_attribute("llm.token_count.total", response.total_tokens)
            if response.prompt_cost_usd is not None:
                span.set_attribute("llm.cost.prompt", response.prompt_cost_usd)
            if response.completion_cost_usd is not None:
                span.set_attribute("llm.cost.completion", response.completion_cost_usd)
            if response.estimated_cost_usd is not None:
                span.set_attribute("llm.cost.estimated_usd", response.estimated_cost_usd)
                span.set_attribute("llm.cost.total", response.estimated_cost_usd)
            span.set_attribute("llm.cost.source", response.cost_source)
            span.set_status(Status(StatusCode.OK))
            await self._record_step(
                run_id=run_id,
                tenant_id=context.tenant_id,
                step_type="llm",
                name="sales_summary",
                input_json={
                    "provider": response.provider,
                    "model": response.model,
                    "message_count": 2,
                },
                output_json={
                    "fallback": summary == fallback_summary,
                    "llm_usage": {
                        "input_count": response.input_tokens,
                        "output_count": response.output_tokens,
                        "total_count": response.total_tokens,
                    },
                    "llm_cost": {
                        "estimated_cost_usd": response.estimated_cost_usd,
                        "source": response.cost_source,
                    },
                },
                output_summary=summary,
            )
            summary_holder["summary"] = summary

    async def _finish_run(
        self,
        *,
        context: InternalRequestContext,
        conversation_id: UUID,
        input_message_id: UUID,
        run_id: UUID,
        artifact: dict[str, Any],
        summary: str,
        evals: list[Any],
    ) -> UUID:
        async with self.connection_factory() as connection:
            step_row = await connection.fetchrow(
                """
                INSERT INTO ai.steps (
                    run_id,
                    tenant_id,
                    step_type,
                    name,
                    output_json,
                    output_summary
                )
                VALUES ($1, $2, 'agent', 'sales_artifact', $3::jsonb, $4)
                RETURNING id
                """,
                run_id,
                UUID(context.tenant_id),
                json.dumps(sanitize_value(artifact), default=str),
                summary,
            )
            message_row = await connection.fetchrow(
                """
                INSERT INTO ai.messages (
                    conversation_id,
                    tenant_id,
                    role,
                    content,
                    content_sanitized,
                    metadata
                )
                VALUES ($1, $2, 'assistant', $3, $4, $5::jsonb)
                RETURNING id
                """,
                conversation_id,
                UUID(context.tenant_id),
                json.dumps(artifact, default=str, ensure_ascii=False),
                summary,
                json.dumps({"workflow": self.graph_name, "step_id": str(step_row["id"])}),
            )
            await connection.execute(
                """
                INSERT INTO ai.context_summaries (
                    conversation_id,
                    tenant_id,
                    summary,
                    covered_from_message_id,
                    covered_to_message_id,
                    summary_type,
                    metadata
                )
                VALUES ($1, $2, $3, $4, $5, 'compact', $6::jsonb)
                """,
                conversation_id,
                UUID(context.tenant_id),
                summary,
                input_message_id,
                message_row["id"],
                json.dumps({"workflow": self.graph_name}),
            )
            for eval_result in evals:
                await connection.execute(
                    """
                    INSERT INTO ai.eval_results (
                        run_id,
                        tenant_id,
                        evaluator_name,
                        score,
                        passed,
                        result_json
                    )
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    run_id,
                    UUID(context.tenant_id),
                    eval_result.evaluator_name,
                    eval_result.score,
                    eval_result.passed,
                    json.dumps(sanitize_value(eval_result.result), default=str),
                )
            await connection.execute(
                """
                UPDATE ai.runs
                SET output_message_id = $1,
                    status = 'completed',
                    finished_at = now()
                WHERE id = $2
                """,
                message_row["id"],
                run_id,
            )
            await connection.execute(
                """
                UPDATE ai.conversations
                SET updated_at = now()
                WHERE id = $1
                """,
                conversation_id,
            )
            return message_row["id"]

    async def _mark_failed(
        self,
        *,
        context: InternalRequestContext,
        run_id: UUID,
        error: dict[str, Any],
    ) -> None:
        async with self.connection_factory() as connection:
            await connection.execute(
                """
                UPDATE ai.runs
                SET status = 'failed',
                    error = $1::jsonb,
                    finished_at = now()
                WHERE id = $2 AND tenant_id = $3
                """,
                json.dumps(sanitize_value(error), default=str),
                run_id,
                UUID(context.tenant_id),
            )
