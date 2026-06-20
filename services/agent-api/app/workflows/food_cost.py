from collections.abc import AsyncIterator
import json
from typing import Any, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import Settings
from app.database import get_db_connection
from app.dependencies.internal_auth import InternalRequestContext
from app.evals.food_cost import evaluate_food_cost_artifact
from app.llm import LLMAdapter, get_llm_adapter
from app.llm.prompts import food_cost_summary_messages
from app.streaming import StreamEvent, stream_event, terminal_error_event
from app.telemetry import current_trace_ids
from app.tools import ToolCallRequest, ToolCallResponse, ToolGateway
from app.tools.sanitize import sanitize_value, truncate_text
from app.workflows.models import FoodCostQuestionRequest, FoodCostWorkflowResponse


class FoodCostGraphState(TypedDict, total=False):
    request: FoodCostQuestionRequest
    context: InternalRequestContext
    conversation_id: UUID
    input_message_id: UUID
    run_id: UUID
    tool_calls: list[dict[str, Any]]
    artifact: dict[str, Any]
    summary: str
    evals: list[Any]
    output_message_id: UUID


class FoodCostWorkflow:
    agent_name = "food_cost_agent"
    graph_name = "food_cost_langgraph_v1"

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
        request: FoodCostQuestionRequest,
        context: InternalRequestContext,
    ) -> FoodCostWorkflowResponse:
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
            return FoodCostWorkflowResponse(
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
        request: FoodCostQuestionRequest,
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
                data={"step_type": "router", "name": "food_cost_intent_router"},
            )
            await self._route_food_cost(
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
                summary_model = self.settings.llm_composer_model
                yield stream_event(
                    "llm_started",
                    run_id=run_id,
                    data={
                        "provider": self.llm_adapter.provider,
                        "model": summary_model
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
            evals = evaluate_food_cost_artifact(artifact)
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
        graph = StateGraph(FoodCostGraphState)
        graph.add_node("route_food_cost", self._route_food_cost)
        graph.add_node("call_food_cost_tools", self._call_food_cost_tools_node)
        graph.add_node("build_artifact", self._build_artifact_node)
        graph.add_node("finish_run", self._finish_run_node)
        graph.add_edge(START, "route_food_cost")
        graph.add_edge("route_food_cost", "call_food_cost_tools")
        graph.add_edge("call_food_cost_tools", "build_artifact")
        graph.add_edge("build_artifact", "finish_run")
        graph.add_edge("finish_run", END)
        return graph.compile()

    async def _route_food_cost(self, state: FoodCostGraphState) -> FoodCostGraphState:
        with self.tracer.start_as_current_span("food_cost.route_intent") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.run_id", str(state["run_id"]))
            span.set_attribute("waro.tenant_id", state["context"].tenant_id)
            span.set_attribute("waro.food_cost.intent", "food_cost")
            span.set_attribute("waro.request.question_length", len(state["request"].question))
            await self._record_step(
                run_id=state["run_id"],
                tenant_id=state["context"].tenant_id,
                step_type="router",
                name="food_cost_intent_router",
                input_json={"question_length": len(state["request"].question)},
                output_json={"intent": "food_cost"},
                output_summary="Routed request to the food-cost workflow.",
            )
            span.set_status(Status(StatusCode.OK))
        return {}

    async def _call_food_cost_tools_node(
        self,
        state: FoodCostGraphState,
    ) -> FoodCostGraphState:
        return {
            "tool_calls": await self._call_required_tools(
                request=state["request"],
                context=state["context"],
                run_id=state["run_id"],
            )
        }

    async def _build_artifact_node(self, state: FoodCostGraphState) -> FoodCostGraphState:
        with self.tracer.start_as_current_span("food_cost.build_artifact") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.run_id", str(state["run_id"]))
            span.set_attribute("waro.tool.call_count", len(state["tool_calls"]))
            artifact = self._build_artifact(
                request=state["request"],
                tool_calls=state["tool_calls"],
            )
            low_margin_products = artifact.get("low_margin_products", [])
            if isinstance(low_margin_products, list):
                span.set_attribute("waro.food_cost.low_margin_count", len(low_margin_products))
            summary = await self._build_summary_with_llm(
                artifact=artifact,
                context=state["context"],
                run_id=state["run_id"],
            )
            span.set_status(Status(StatusCode.OK))
        return {
            "artifact": artifact,
            "summary": summary,
            "evals": evaluate_food_cost_artifact(artifact),
        }

    async def _finish_run_node(self, state: FoodCostGraphState) -> FoodCostGraphState:
        with self.tracer.start_as_current_span("food_cost.finish_run") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.run_id", str(state["run_id"]))
            span.set_attribute("waro.eval.count", len(state["evals"]))
            output_message_id = await self._finish_run(
                context=state["context"],
                conversation_id=state["conversation_id"],
                input_message_id=state["input_message_id"],
                run_id=state["run_id"],
                artifact=state["artifact"],
                summary=state["summary"],
                evals=state["evals"],
            )
            span.set_attribute("waro.output_message_id", str(output_message_id))
            span.set_status(Status(StatusCode.OK))
        return {"output_message_id": output_message_id}

    async def _start_run(
        self,
        *,
        request: FoodCostQuestionRequest,
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
                    json.dumps({"source": "food_cost_workflow"}),
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
        _, span_id = current_trace_ids()
        step_span_id = span_id if step_type != "tool" else None
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
                    output_summary,
                    span_id
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8)
                RETURNING id
                """,
                run_id,
                UUID(tenant_id),
                step_type,
                name,
                json.dumps(sanitize_value(input_json or {}), default=str),
                json.dumps(sanitize_value(output_json or {}), default=str),
                output_summary,
                step_span_id,
            )
            return row["id"]

    async def _call_required_tools(
        self,
        *,
        request: FoodCostQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, (tool_name, _, _) in enumerate(
            self._required_tool_calls(request),
            start=1,
        ):
            _, response = await self._call_food_cost_tool(
                request=request,
                context=context,
                run_id=run_id,
                index=index,
                tool_name=tool_name,
            )
            results.append(self._tool_call_record(response))
        return results

    async def _stream_required_tools(
        self,
        *,
        request: FoodCostQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
        tool_calls: list[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        for index, (tool_name, _, _) in enumerate(
            self._required_tool_calls(request),
            start=1,
        ):
            step_id, arguments, fields = await self._prepare_food_cost_call(
                request=request,
                context=context,
                run_id=run_id,
                tool_name=tool_name,
            )
            yield stream_event(
                "tool_started",
                run_id=run_id,
                data={
                    "step_id": str(step_id),
                    "tool_name": tool_name,
                },
            )
            response = await self.gateway.call(
                request=ToolCallRequest(
                    run_id=run_id,
                    step_id=step_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    fields=fields,
                    idempotency_key=f"{run_id}:{index}:{tool_name}",
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

    async def _call_food_cost_tool(
        self,
        *,
        request: FoodCostQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
        index: int,
        tool_name: str,
    ) -> tuple[UUID, ToolCallResponse]:
        step_id, arguments, fields = await self._prepare_food_cost_call(
            request=request,
            context=context,
            run_id=run_id,
            tool_name=tool_name,
        )
        response = await self.gateway.call(
            request=ToolCallRequest(
                run_id=run_id,
                step_id=step_id,
                tool_name=tool_name,
                arguments=arguments,
                fields=fields,
                idempotency_key=f"{run_id}:{index}:{tool_name}",
            ),
            context=context,
        )
        return step_id, response

    async def _prepare_food_cost_call(
        self,
        *,
        request: FoodCostQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
        tool_name: str,
    ) -> tuple[UUID, dict[str, Any], list[str]]:
        with self.tracer.start_as_current_span("food_cost.plan_tool") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            span.set_attribute("waro.tool.name", tool_name)
            call = next(
                (
                    (arguments, fields)
                    for name, arguments, fields in self._required_tool_calls(request)
                    if name == tool_name
                ),
                None,
            )
            if call is None:
                raise ValueError(f"Unknown food-cost tool: {tool_name}")
            arguments, fields = call
            span.set_attribute("waro.resolver.period.date_from", request.date_from or "")
            span.set_attribute("waro.resolver.period.date_to", request.date_to or "")
            span.set_attribute("waro.resolver.compare_to", request.compare_to or "")
            span.set_attribute("waro.tool.fields", ",".join(fields))
            step_id = await self._record_step(
                run_id=run_id,
                tenant_id=context.tenant_id,
                step_type="tool",
                name=tool_name,
                input_json={"arguments": arguments, "fields": fields},
            )
            span.set_attribute("waro.step_id", str(step_id))
            span.set_status(Status(StatusCode.OK))
        return step_id, arguments, fields

    def _required_tool_calls(
        self,
        request: FoodCostQuestionRequest,
    ) -> list[tuple[str, dict[str, Any], list[str]]]:
        period_args = {
            key: value
            for key, value in {
                "date-from": request.date_from,
                "date-to": request.date_to,
                "compare-to": request.compare_to,
            }.items()
            if value is not None
        }
        return [
            (
                "waro.analytics.food_cost",
                period_args,
                ["product_id", "product_name", "food_cost_pct", "margin_pct", "revenue", "cost"],
            ),
            (
                "waro.menu.products",
                {"limit": 100},
                ["id", "name", "price", "cost", "margin"],
            ),
            (
                "waro.financial.products",
                {"sort-by": "margin"},
                ["id", "name", "margin", "revenue", "cost", "quantity"],
            ),
        ]

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
        request: FoodCostQuestionRequest,
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        food_cost_rows = self._rows_for(tool_calls, "waro.analytics.food_cost")
        financial_rows = self._rows_for(tool_calls, "waro.financial.products")
        low_margin_products = sorted(
            [self._product_snapshot(row) for row in food_cost_rows],
            key=lambda row: (
                row.get("margin_pct") if row.get("margin_pct") is not None else 999999,
                -(row.get("food_cost_pct") or 0),
            ),
        )[:5]
        if not low_margin_products:
            low_margin_products = [
                self._product_snapshot(row)
                for row in sorted(
                    financial_rows,
                    key=lambda row: row.get("margin") if row.get("margin") is not None else 999999,
                )[:5]
            ]

        return {
            "period": {
                "date_from": request.date_from,
                "date_to": request.date_to,
                "compare_to": request.compare_to,
            },
            "low_margin_products": low_margin_products,
            "recommendations": self._recommendations(low_margin_products),
            "explanation": (
                "Food-cost analysis uses analytics first, then menu and financial "
                "product context to explain margin pressure."
            ),
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
        return {
            "period": artifact.get("period"),
            "low_margin_products": artifact.get("low_margin_products", []),
            "recommendations": artifact.get("recommendations", []),
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

    def _rows_for(self, tool_calls: list[dict[str, Any]], tool_name: str) -> list[dict[str, Any]]:
        for call in tool_calls:
            if call["tool_name"] != tool_name or call["status"] != "succeeded":
                continue
            result = call.get("result")
            if isinstance(result, dict) and isinstance(result.get("rows"), list):
                return [row for row in result["rows"] if isinstance(row, dict)]
            if isinstance(result, dict) and isinstance(result.get("data"), list):
                return [row for row in result["data"] if isinstance(row, dict)]
            if isinstance(result, list):
                return [row for row in result if isinstance(row, dict)]
        return []

    def _product_snapshot(self, row: dict[str, Any]) -> dict[str, Any]:
        return sanitize_value(
            {
                "id": row.get("product_id") or row.get("id"),
                "name": row.get("product_name") or row.get("name"),
                "food_cost_pct": row.get("food_cost_pct"),
                "margin_pct": row.get("margin_pct"),
                "margin": row.get("margin"),
                "revenue": row.get("revenue"),
                "cost": row.get("cost"),
            }
        )

    def _recommendations(self, products: list[dict[str, Any]]) -> list[str]:
        if not products:
            return [
                "Review recipes with missing food-cost data before making pricing decisions."
            ]
        names = ", ".join(
            str(product.get("name") or product.get("id"))
            for product in products[:3]
            if product.get("name") or product.get("id")
        )
        return [
            f"Prioritize recipe and supplier-cost review for {names}.",
            "Compare current menu price against ingredient cost movement before changing prices.",
            "Use the next operational review to decide price, portion, or supplier actions.",
        ]

    def _build_summary(self, artifact: dict[str, Any]) -> str:
        products = artifact.get("low_margin_products", [])
        if not products:
            return "Food-cost workflow completed without finding product-level margin rows."
        product_names = ", ".join(
            str(product.get("name") or product.get("id"))
            for product in products[:3]
            if product.get("name") or product.get("id")
        )
        return f"Food-cost workflow flagged {len(products)} low-margin products: {product_names}."

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

        with self.tracer.start_as_current_span("llm.food_cost.summary") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", self.llm_adapter.provider)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            summary_model = self.settings.llm_composer_model
            if self.settings.llm_provider == "kimi":
                span.set_attribute("llm.model", summary_model)
                span.set_attribute("llm.model_name", summary_model)

            try:
                response = await self.llm_adapter.complete(
                    messages=food_cost_summary_messages(artifact),
                    temperature=0.2,
                    model=summary_model,
                )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                await self._record_step(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    step_type="llm",
                    name="food_cost_summary",
                    input_json={
                        "provider": self.settings.llm_provider,
                        "model": summary_model
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
                name="food_cost_summary",
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
        with self.tracer.start_as_current_span("llm.food_cost.summary") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", self.llm_adapter.provider)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            summary_model = self.settings.llm_composer_model
            if self.settings.llm_provider == "kimi":
                span.set_attribute("llm.model", summary_model)
                span.set_attribute("llm.model_name", summary_model)

            try:
                response = None
                async for chunk in stream_complete(
                    messages=food_cost_summary_messages(artifact),
                    temperature=0.2,
                    model=summary_model,
                ):
                    if chunk.text:
                        yield stream_event(
                            "token",
                            run_id=run_id,
                            data={
                                "provider": self.llm_adapter.provider,
                                "model": summary_model
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
                    name="food_cost_summary",
                    input_json={
                        "provider": self.settings.llm_provider,
                        "model": summary_model
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
                name="food_cost_summary",
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
        _, span_id = current_trace_ids()
        async with self.connection_factory() as connection:
            step_row = await connection.fetchrow(
                """
                INSERT INTO ai.steps (
                    run_id,
                    tenant_id,
                    step_type,
                    name,
                    output_json,
                    output_summary,
                    span_id
                )
                VALUES ($1, $2, 'agent', 'food_cost_artifact', $3::jsonb, $4, $5)
                RETURNING id
                """,
                run_id,
                UUID(context.tenant_id),
                json.dumps(sanitize_value(artifact), default=str),
                summary,
                span_id,
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
