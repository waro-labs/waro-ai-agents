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
from app.tools import ToolCallRequest, ToolGateway
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
        await self._record_step(
            run_id=state["run_id"],
            tenant_id=state["context"].tenant_id,
            step_type="router",
            name="food_cost_intent_router",
            input_json={"question": state["request"].question},
            output_json={"intent": "food_cost"},
            output_summary="Routed request to the food-cost workflow.",
        )
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
            "evals": evaluate_food_cost_artifact(artifact),
        }

    async def _finish_run_node(self, state: FoodCostGraphState) -> FoodCostGraphState:
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
        request: FoodCostQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
    ) -> list[dict[str, Any]]:
        period_args = {
            key: value
            for key, value in {
                "date-from": request.date_from,
                "date-to": request.date_to,
                "compare-to": request.compare_to,
            }.items()
            if value is not None
        }
        calls = [
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
        results: list[dict[str, Any]] = []
        for index, (tool_name, arguments, fields) in enumerate(calls, start=1):
            step_id = await self._record_step(
                run_id=run_id,
                tenant_id=context.tenant_id,
                step_type="tool",
                name=tool_name,
                input_json={"arguments": arguments, "fields": fields},
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
            results.append(
                {
                    "tool_name": response.tool_name,
                    "status": response.status,
                    "tool_call_id": str(response.tool_call_id) if response.tool_call_id else None,
                    "result": sanitize_value(response.result),
                    "result_summary": response.result_summary,
                    "error": sanitize_value(response.error),
                }
            )
        return results

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
            "question": request.question,
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
            span.set_attribute("llm.provider", self.llm_adapter.provider)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            if self.settings.llm_provider == "kimi":
                span.set_attribute("llm.model", self.settings.kimi_model)

            try:
                response = await self.llm_adapter.complete(
                    messages=food_cost_summary_messages(artifact),
                    temperature=0.2,
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
            span.set_attribute("llm.response.provider", response.provider)
            if response.input_tokens is not None:
                span.set_attribute("llm.usage.prompt_tokens", response.input_tokens)
            if response.output_tokens is not None:
                span.set_attribute("llm.usage.completion_tokens", response.output_tokens)
            if response.total_tokens is not None:
                span.set_attribute("llm.usage.total_tokens", response.total_tokens)
            if response.estimated_cost_usd is not None:
                span.set_attribute("llm.cost.estimated_usd", response.estimated_cost_usd)
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
                VALUES ($1, $2, 'agent', 'food_cost_artifact', $3::jsonb, $4)
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
