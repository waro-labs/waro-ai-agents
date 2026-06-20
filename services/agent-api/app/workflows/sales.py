from collections.abc import AsyncIterator
import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any, Literal, TypedDict
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
from app.llm.prompts import sales_planner_messages, sales_summary_messages
from app.streaming import StreamEvent, stream_event, terminal_error_event
from app.telemetry import current_trace_ids
from app.tools import ToolCallRequest, ToolCallResponse, ToolGateway
from app.tools.catalog import tool_catalog
from app.tools.planner import ToolPlan, ToolPlanner, ToolPlanStep
from app.tools.sanitize import sanitize_value, truncate_text
from app.workflows.models import SalesQuestionRequest, SalesWorkflowResponse


SMALL_TALK_PATTERNS = (
    r"\b(hola|buenas|buenos dias|buenas tardes|buenas noches)\b",
    r"\b(como estas|que tal|como vas)\b",
    r"\b(funcionas|estas funcionando|me escuchas)\b",
    r"\b(gracias|ok|listo)\b",
)

SALES_SIGNAL_PATTERNS = (
    r"\bventas?\b",
    r"\bvend[ií]|vendimos|vendido\b",
    r"\bingresos?\b",
    r"\bfactur",
    r"\border(?:es|nes)?\b",
    r"\bticket\b",
    r"\bultim[oa]s?\s+\d{1,3}\s+dias?\b",
    r"\b(hoy|ayer|este mes|del mes|mes actual|esta semana|semana actual)\b",
)

SPANISH_NUMBER_WORDS = {
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "treinta": 30,
}

SPANISH_MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

AnswerStyle = Literal[
    "direct_metric",
    "business_analysis",
    "financial_analysis",
    "diagnostic",
    "follow_up",
]


class SalesGraphState(TypedDict, total=False):
    request: SalesQuestionRequest
    context: InternalRequestContext
    conversation_id: UUID
    input_message_id: UUID
    run_id: UUID
    sales_intent: str
    tool_plan: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    artifact: dict[str, Any]
    summary: str
    evals: list[Any]
    output_message_id: UUID
    previous_context_summary: str | None


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
        self.tool_planner = ToolPlanner()
        self.tracer = trace.get_tracer(__name__)
        self.graph = self._build_graph()

    def _printf_step(self, name: str, payload: dict[str, Any] | None = None) -> None:
        safe_payload = sanitize_value(payload or {})
        labels = {
            "run_started": ("🚀", "Run started"),
            "run_completed": ("✅", "Run completed"),
            "run_failed": ("❌", "Run failed"),
            "sales_intent_router": ("🧭", "Router"),
            "load_conversation_context": ("📚", "Conversation context"),
            "sales_semantic_planner": ("🧠", "Semantic planner"),
            "sales_context_resolver": ("🧩", "Context resolver"),
            "sales_tool_planner": ("🛠️", "Tool planner"),
            "tool_started": ("🔧", "Tool started"),
            "tool_finished": ("📦", "Tool finished"),
            "analysis_artifact": ("📊", "Analysis artifact"),
            "answer_composer": ("✍️", "Answer composer"),
        }
        icon, label = labels.get(name, ("•", name.replace("_", " ").title()))
        print(
            f"[agent-api:sales] {icon} {label} ({name}) "
            f"{json.dumps(safe_payload, ensure_ascii=False, default=str)}",
            flush=True,
        )

    def _set_response_contract_span_attributes(
        self,
        span: Any,
        response_contract: dict[str, Any],
    ) -> None:
        if not response_contract:
            return
        span.set_attribute("waro.request.kind", str(response_contract.get("request_kind", "")))
        span.set_attribute("waro.data.status", str(response_contract.get("data_status", "")))
        span.set_attribute(
            "waro.safe_to_answer",
            bool(response_contract.get("safe_to_answer", False)),
        )
        span.set_attribute(
            "waro.missing_required_data",
            ",".join(str(item) for item in response_contract.get("missing_required_data", [])),
        )
        failed_tools = response_contract.get("failed_tools")
        if isinstance(failed_tools, list):
            span.set_attribute(
                "waro.failed_tools",
                ",".join(
                    str(tool.get("tool_name"))
                    for tool in failed_tools
                    if isinstance(tool, dict) and tool.get("tool_name")
                ),
            )

    def _tool_result_shape(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            data = result.get("data")
            products = result.get("products")
            return {
                "kind": "dict",
                "keys": sorted(str(key) for key in result.keys())[:20],
                "data_type": type(data).__name__ if data is not None else None,
                "data_count": len(data) if isinstance(data, list) else None,
                "products_count": len(products) if isinstance(products, list) else None,
                "success": result.get("success"),
            }
        if isinstance(result, list):
            return {"kind": "list", "row_count": len(result)}
        return {"kind": type(result).__name__}

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
        self._printf_step(
            "run_started",
            {
                "run_id": str(run_id),
                "conversation_id": str(conversation_id),
                "tenant_id": context.tenant_id,
                "question_length": len(request.question),
                "scopes": list(context.scopes),
            },
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
            self._printf_step(
                "run_completed",
                {
                    "run_id": str(run_id),
                    "answer_style": state["artifact"].get("answer_style"),
                    "tool_count": len(state["artifact"].get("tool_calls", [])),
                },
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
            self._printf_step(
                "run_failed",
                {
                    "run_id": str(run_id),
                    "error_type": type(exc).__name__,
                    "message": truncate_text(str(exc), 240),
                },
            )
            raise

    async def stream(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
    ) -> AsyncIterator[StreamEvent]:
        with self.tracer.start_as_current_span("sales.stream") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.agent.name", self.agent_name)
            span.set_attribute("waro.tenant_id", context.tenant_id)
            span.set_attribute("waro.request.question_length", len(request.question))
            conversation_id, input_message_id, run_id = await self._start_run(
                request=request,
                context=context,
            )
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.conversation_id", str(conversation_id))
            self._printf_step(
                "run_started",
                {
                    "run_id": str(run_id),
                    "conversation_id": str(conversation_id),
                    "tenant_id": context.tenant_id,
                    "question_length": len(request.question),
                    "scopes": list(context.scopes),
                    "mode": "stream",
                },
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
                route_state = await self._route_sales(
                    {
                        "request": request,
                        "context": context,
                        "run_id": run_id,
                    }
                )
                sales_intent = route_state.get("sales_intent", "sales_metrics")
                span.set_attribute("waro.sales.intent", sales_intent)
                previous_context_summary = await self._load_latest_context_summary(
                    request=request,
                    context=context,
                )
                tool_calls: list[dict[str, Any]] = []
                tool_plan: dict[str, Any] | None = None
                if sales_intent == "sales_metrics":
                    plan = await self._plan_sales_tool_calls(
                        request=request,
                        context=context,
                        run_id=run_id,
                        previous_context_summary=previous_context_summary,
                    )
                    tool_plan = self._audit_tool_plan(plan)
                    async for event in self._stream_required_tools(
                        context=context,
                        run_id=run_id,
                        plan=plan,
                        tool_calls=tool_calls,
                    ):
                        yield event

                artifact = self._build_artifact(
                    request=request,
                    tool_calls=tool_calls,
                    intent=sales_intent,
                    tool_plan=tool_plan,
                    previous_context_summary=previous_context_summary,
                )
                self._printf_step(
                    "analysis_artifact",
                    {
                        "run_id": str(run_id),
                        "intent": artifact.get("intent"),
                        "answer_style": artifact.get("answer_style"),
                        "analysis_area": (
                            artifact.get("analysis_request", {}).get("area")
                            if isinstance(artifact.get("analysis_request"), dict)
                            else None
                        ),
                        "analysis_objective": (
                            artifact.get("analysis_request", {}).get("objective")
                            if isinstance(artifact.get("analysis_request"), dict)
                            else None
                        ),
                        "request_kind": (
                            artifact.get("response_contract", {}).get("request_kind")
                            if isinstance(artifact.get("response_contract"), dict)
                            else None
                        ),
                        "data_status": (
                            artifact.get("response_contract", {}).get("data_status")
                            if isinstance(artifact.get("response_contract"), dict)
                            else None
                        ),
                        "safe_to_answer": (
                            artifact.get("response_contract", {}).get("safe_to_answer")
                            if isinstance(artifact.get("response_contract"), dict)
                            else None
                        ),
                        "tool_count": len(tool_calls),
                    },
                )
                span.set_attribute("waro.answer.style", artifact.get("answer_style", ""))
                span.set_attribute("waro.tool.call_count", len(tool_calls))
                stream_analysis_request = (
                    artifact.get("analysis_request")
                    if isinstance(artifact.get("analysis_request"), dict)
                    else {}
                )
                stream_response_contract = (
                    artifact.get("response_contract")
                    if isinstance(artifact.get("response_contract"), dict)
                    else {}
                )
                span.set_attribute(
                    "waro.analysis.area",
                    str(stream_analysis_request.get("area", "")),
                )
                span.set_attribute(
                    "waro.analysis.objective",
                    str(stream_analysis_request.get("objective", "")),
                )
                self._set_response_contract_span_attributes(
                    span,
                    stream_response_contract,
                )
                span.add_event(
                    "response_contract.evaluated",
                    {
                        "waro.request.kind": str(
                            stream_response_contract.get("request_kind", "")
                        ),
                        "waro.data.status": str(
                            stream_response_contract.get("data_status", "")
                        ),
                        "waro.safe_to_answer": bool(
                            stream_response_contract.get("safe_to_answer", False)
                        ),
                    },
                )
                if self.settings.llm_provider != "disabled" and self._should_use_llm(artifact):
                    summary_model = self._summary_model_for(artifact)
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
                self._printf_step(
                    "answer_composer",
                    {
                        "run_id": str(run_id),
                        "summary_length": len(summary),
                        "used_llm": self._should_use_llm(artifact)
                        and self.settings.llm_provider != "disabled",
                    },
                )
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
                self._printf_step(
                    "run_completed",
                    {
                        "run_id": str(run_id),
                        "output_message_id": str(output_message_id),
                        "status": "completed",
                    },
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
                span.set_status(Status(StatusCode.OK))
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                await self._mark_failed(
                    context=context,
                    run_id=run_id,
                    error={"message": str(exc), "type": type(exc).__name__},
                )
                self._printf_step(
                    "run_failed",
                    {
                        "run_id": str(run_id),
                        "error_type": type(exc).__name__,
                        "message": truncate_text(str(exc), 240),
                    },
                )
                yield stream_event(
                    "error",
                    run_id=run_id,
                    data={
                        "error_type": type(exc).__name__,
                        "message": truncate_text(str(exc), 240),
                    },
                )

    def _build_graph(self):
        graph = StateGraph(SalesGraphState)
        graph.add_node("route_sales", self._route_sales)
        graph.add_node("call_sales_tools", self._call_sales_tools_node)
        graph.add_node("build_artifact", self._build_artifact_node)
        graph.add_node("finish_run", self._finish_run_node)
        graph.add_edge(START, "route_sales")
        graph.add_conditional_edges(
            "route_sales",
            self._next_after_route,
            {
                "call_sales_tools": "call_sales_tools",
                "build_artifact": "build_artifact",
            },
        )
        graph.add_edge("call_sales_tools", "build_artifact")
        graph.add_edge("build_artifact", "finish_run")
        graph.add_edge("finish_run", END)
        return graph.compile()

    async def _route_sales(self, state: SalesGraphState) -> SalesGraphState:
        with self.tracer.start_as_current_span("sales.route_intent") as span:
            has_conversation_context = state["request"].conversation_id is not None
            intent = self._resolve_sales_intent(
                state["request"].question,
                has_conversation_context=has_conversation_context,
            )
            self._printf_step(
                "sales_intent_router",
                {
                    "run_id": str(state["run_id"]),
                    "intent": intent,
                    "has_conversation_context": has_conversation_context,
                    "question_length": len(state["request"].question),
                    "tool_planned": intent == "sales_metrics",
                },
            )
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.run_id", str(state["run_id"]))
            span.set_attribute("waro.tenant_id", state["context"].tenant_id)
            span.set_attribute("waro.sales.intent", intent)
            span.set_attribute("waro.sales.tool_planned", intent == "sales_metrics")
            span.set_attribute("waro.conversation.has_context", has_conversation_context)
            span.set_attribute("waro.request.question_length", len(state["request"].question))
            await self._record_step(
                run_id=state["run_id"],
                tenant_id=state["context"].tenant_id,
                step_type="router",
                name="sales_intent_router",
                input_json={
                    "question_length": len(state["request"].question),
                    "has_conversation_context": has_conversation_context,
                },
                output_json={
                    "intent": intent,
                    "tool_planned": intent == "sales_metrics",
                },
                output_summary=f"Routed request to {intent}.",
            )
            span.set_status(Status(StatusCode.OK))
        return {"sales_intent": intent}

    def _next_after_route(self, state: SalesGraphState) -> str:
        if state.get("sales_intent") == "sales_metrics":
            return "call_sales_tools"
        return "build_artifact"

    async def _call_sales_tools_node(self, state: SalesGraphState) -> SalesGraphState:
        previous_context_summary = await self._load_latest_context_summary(
            request=state["request"],
            context=state["context"],
        )
        plan = await self._plan_sales_tool_calls(
            request=state["request"],
            context=state["context"],
            run_id=state["run_id"],
            previous_context_summary=previous_context_summary,
        )
        return {
            "tool_plan": self._audit_tool_plan(plan),
            "tool_calls": await self._call_required_tools(
                context=state["context"],
                run_id=state["run_id"],
                plan=plan,
            ),
            "previous_context_summary": previous_context_summary,
        }

    async def _build_artifact_node(self, state: SalesGraphState) -> SalesGraphState:
        with self.tracer.start_as_current_span("sales.build_artifact") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.run_id", str(state["run_id"]))
            tool_calls = state.get("tool_calls", [])
            tool_plan = state.get("tool_plan")
            intent = state.get("sales_intent", "sales_metrics")
            previous_context_summary = state.get("previous_context_summary")
            if previous_context_summary is None and intent == "follow_up":
                previous_context_summary = await self._load_latest_context_summary(
                    request=state["request"],
                    context=state["context"],
                )
            span.set_attribute("waro.sales.intent", intent)
            span.set_attribute("waro.tool.call_count", len(tool_calls))
            artifact = self._build_artifact(
                request=state["request"],
                tool_calls=tool_calls,
                intent=intent,
                tool_plan=tool_plan,
                previous_context_summary=previous_context_summary,
            )
            self._printf_step(
                "analysis_artifact",
                {
                    "run_id": str(state["run_id"]),
                    "intent": artifact.get("intent"),
                    "answer_style": artifact.get("answer_style"),
                    "analysis_area": (
                        artifact.get("analysis_request", {}).get("area")
                        if isinstance(artifact.get("analysis_request"), dict)
                        else None
                    ),
                        "analysis_objective": (
                            artifact.get("analysis_request", {}).get("objective")
                            if isinstance(artifact.get("analysis_request"), dict)
                            else None
                        ),
                        "request_kind": (
                            artifact.get("response_contract", {}).get("request_kind")
                            if isinstance(artifact.get("response_contract"), dict)
                            else None
                        ),
                        "data_status": (
                            artifact.get("response_contract", {}).get("data_status")
                            if isinstance(artifact.get("response_contract"), dict)
                            else None
                        ),
                        "safe_to_answer": (
                            artifact.get("response_contract", {}).get("safe_to_answer")
                            if isinstance(artifact.get("response_contract"), dict)
                            else None
                        ),
                        "tool_count": len(tool_calls),
                    },
                )
            span.set_attribute("waro.answer.style", artifact.get("answer_style", ""))
            analysis_request = (
                artifact.get("analysis_request")
                if isinstance(artifact.get("analysis_request"), dict)
                else {}
            )
            response_contract = (
                artifact.get("response_contract")
                if isinstance(artifact.get("response_contract"), dict)
                else {}
            )
            span.set_attribute("waro.analysis.area", str(analysis_request.get("area", "")))
            span.set_attribute(
                "waro.analysis.objective",
                str(analysis_request.get("objective", "")),
            )
            span.set_attribute(
                "waro.analysis.dimensions",
                ",".join(str(item) for item in analysis_request.get("dimensions", [])),
            )
            self._set_response_contract_span_attributes(span, response_contract)
            span.add_event(
                "response_contract.evaluated",
                {
                    "waro.request.kind": str(response_contract.get("request_kind", "")),
                    "waro.data.status": str(response_contract.get("data_status", "")),
                    "waro.safe_to_answer": bool(response_contract.get("safe_to_answer", False)),
                    "waro.missing_required_data": ",".join(
                        str(item)
                        for item in response_contract.get("missing_required_data", [])
                    ),
                },
            )
            if response_contract and not response_contract.get("safe_to_answer", True):
                span.add_event(
                    "response_contract.blocked",
                    {
                        "waro.error_message": truncate_text(
                            str(response_contract.get("error_message") or ""),
                            240,
                        ),
                    },
                )
            metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
            if metrics.get("total_sales") is not None:
                span.set_attribute("waro.sales.total_sales", metrics["total_sales"])
            if metrics.get("order_count") is not None:
                span.set_attribute("waro.sales.order_count", metrics["order_count"])
            summary = await self._build_summary_with_llm(
                artifact=artifact,
                context=state["context"],
                run_id=state["run_id"],
            )
            self._printf_step(
                "answer_composer",
                {
                    "run_id": str(state["run_id"]),
                    "summary_length": len(summary),
                    "used_llm": self._should_use_llm(artifact)
                    and self.settings.llm_provider != "disabled",
                },
            )
            span.set_status(Status(StatusCode.OK))
        return {
            "artifact": artifact,
            "summary": summary,
            "evals": evaluate_sales_artifact(artifact),
        }

    async def _finish_run_node(self, state: SalesGraphState) -> SalesGraphState:
        with self.tracer.start_as_current_span("sales.finish_run") as span:
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
        request: SalesQuestionRequest,
        context: InternalRequestContext,
    ) -> tuple[UUID, UUID, UUID]:
        trace_id, _ = current_trace_ids()
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
                    trace_id,
                    agent_name,
                    graph_name,
                    status
                )
                VALUES ($1, $2, $3, $4, $5, $6, 'running')
                RETURNING id
                """,
                conversation_id,
                UUID(context.tenant_id),
                message_row["id"],
                trace_id,
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
        context: InternalRequestContext,
        run_id: UUID,
        plan: ToolPlan,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, step in enumerate(plan.steps, start=1):
            self._printf_step(
                "tool_started",
                {
                    "run_id": str(run_id),
                    "index": index,
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                    "fields": step.fields,
                },
            )
            _, response = await self._call_planned_tool(
                context=context,
                run_id=run_id,
                step=step,
                index=index,
            )
            self._printf_step(
                "tool_finished",
                {
                    "run_id": str(run_id),
                    "index": index,
                    "tool_name": response.tool_name,
                    "status": response.status,
                    "summary": response.result_summary,
                    "error": response.error,
                },
            )
            records.append(self._tool_call_record(response))
        return records

    async def _stream_required_tools(
        self,
        *,
        context: InternalRequestContext,
        run_id: UUID,
        plan: ToolPlan,
        tool_calls: list[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        for index, step in enumerate(plan.steps, start=1):
            self._printf_step(
                "tool_started",
                {
                    "run_id": str(run_id),
                    "index": index,
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                    "fields": step.fields,
                },
            )
            step_id = await self._record_planned_tool_step(
                run_id=run_id,
                tenant_id=context.tenant_id,
                step=step,
                plan=plan,
            )
            yield stream_event(
                "tool_started",
                run_id=run_id,
                data={
                    "step_id": str(step_id),
                    "tool_name": step.tool_name,
                    "reason": step.reason,
                },
            )
            response = await self.gateway.call(
                request=ToolCallRequest(
                    run_id=run_id,
                    step_id=step_id,
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    fields=step.fields,
                    idempotency_key=f"{run_id}:{index}:{step.tool_name}",
                ),
                context=context,
            )
            tool_calls.append(self._tool_call_record(response))
            self._printf_step(
                "tool_finished",
                {
                    "run_id": str(run_id),
                    "index": index,
                    "tool_name": response.tool_name,
                    "status": response.status,
                    "summary": response.result_summary,
                    "error": response.error,
                },
            )
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

    async def _call_planned_tool(
        self,
        *,
        context: InternalRequestContext,
        run_id: UUID,
        step: ToolPlanStep,
        index: int,
    ) -> tuple[UUID, ToolCallResponse]:
        step_id = await self._record_planned_tool_step(
            run_id=run_id,
            tenant_id=context.tenant_id,
            step=step,
            plan=None,
        )
        response = await self.gateway.call(
            request=ToolCallRequest(
                run_id=run_id,
                step_id=step_id,
                tool_name=step.tool_name,
                arguments=step.arguments,
                fields=step.fields,
                idempotency_key=f"{run_id}:{index}:{step.tool_name}",
            ),
            context=context,
        )
        return step_id, response

    async def _plan_sales_tool_calls(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
        previous_context_summary: str | None = None,
    ) -> ToolPlan:
        with self.tracer.start_as_current_span("sales.tool_planner") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            semantic_plan = await self._resolve_semantic_sales_plan(
                request=request,
                context=context,
                run_id=run_id,
            )
            context_resolution = self._resolve_contextual_sales_plan(
                request=request,
                semantic_plan=semantic_plan,
                previous_context_summary=previous_context_summary,
            )
            self._printf_step(
                "sales_context_resolver",
                {
                    "run_id": str(run_id),
                    "used_context": context_resolution.get("used_context", False),
                    "inherited_period": context_resolution.get("inherited_period", False),
                    "period": context_resolution.get("period"),
                    "answer_style": context_resolution.get("answer_style"),
                    "tool_name": context_resolution.get("tool_name"),
                    "reason": context_resolution.get("reason"),
                },
            )
            if context_resolution.get("inherited_period"):
                semantic_plan = {
                    **semantic_plan,
                    "period": context_resolution["period"],
                    "period_source": "conversation_context",
                    "source": "conversation_context",
                }
            if context_resolution.get("answer_style"):
                semantic_plan = {
                    **semantic_plan,
                    "answer_style": context_resolution["answer_style"],
                }
            if context_resolution.get("group_by"):
                semantic_plan = {
                    **semantic_plan,
                    "group_by": context_resolution["group_by"],
                }
            if context_resolution.get("tool_name"):
                semantic_plan = {
                    **semantic_plan,
                    "tools": [
                        *[
                            tool
                            for tool in semantic_plan.get("tools", [])
                            if isinstance(tool, dict)
                        ],
                        {
                            "name": context_resolution["tool_name"],
                            "reason": "conversation_context",
                        },
                    ],
                }
            if context_resolution.get("used_context"):
                semantic_plan = {
                    **semantic_plan,
                    "context_resolution": context_resolution,
                }
                span.set_attribute("waro.context.used", True)
                span.set_attribute(
                    "waro.context.inherited_period",
                    bool(context_resolution.get("inherited_period")),
                )
                await self._record_step(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    step_type="agent",
                    name="sales_context_resolver",
                    input_json={
                        "question_length": len(request.question),
                        "has_previous_context": bool(previous_context_summary),
                    },
                    output_json=context_resolution,
                    output_summary="Resolved sales context from previous conversation.",
                )
            period = semantic_plan["period"]
            group_by = request.group_by or semantic_plan.get("group_by")
            answer_style = str(
                semantic_plan.get("answer_style")
                or self._resolve_answer_style(request.question)
            )
            plan = self.tool_planner.plan_sales(
                question=request.question,
                period=period,
                scopes=context.scopes,
                group_by=group_by if isinstance(group_by, str) else None,
                answer_style=answer_style,
                semantic_plan=semantic_plan,
            )
            self._printf_step(
                "sales_tool_planner",
                {
                    "run_id": str(run_id),
                    "period": period,
                    "answer_style": answer_style,
                    "group_by": group_by,
                    "steps": [
                        {
                            "tool_name": step.tool_name,
                            "arguments": step.arguments,
                            "fields": step.fields,
                        }
                        for step in plan.steps
                    ],
                    "available_tools": [
                        tool.get("name") for tool in plan.available_tools
                    ],
                    "rejected_tools": [
                        {
                            "name": tool.get("name"),
                            "reason": tool.get("rejected_reason"),
                        }
                        for tool in plan.rejected_tools
                    ],
                },
            )
            span.set_attribute("waro.resolver.period.date_from", period["date_from"])
            span.set_attribute("waro.resolver.period.date_to", period["date_to"])
            span.set_attribute("waro.resolver.source", str(semantic_plan.get("source", "")))
            span.set_attribute("waro.resolver.confidence", float(semantic_plan.get("confidence", 0)))
            if group_by:
                span.set_attribute("waro.resolver.group_by", str(group_by))
            span.set_attribute("waro.answer.style", answer_style)
            span.set_attribute("waro.tool.plan.strategy", plan.strategy)
            span.set_attribute("waro.tool.plan.step_count", len(plan.steps))
            span.set_attribute(
                "waro.tool.selected",
                ",".join(step.tool_name for step in plan.steps),
            )
            span.set_attribute("waro.tool.candidates", ",".join(plan.candidate_tools))
            span.set_attribute(
                "waro.tool.available",
                ",".join(str(tool.get("name")) for tool in plan.available_tools),
            )
            span.set_attribute(
                "waro.tool.rejected",
                ",".join(
                    f"{tool.get('name')}:{tool.get('rejected_reason')}"
                    for tool in plan.rejected_tools
                ),
            )
            span.add_event(
                "tool_plan.selected",
                {
                    "waro.tool.selected": ",".join(step.tool_name for step in plan.steps),
                    "waro.tool.step_count": len(plan.steps),
                    "waro.group_by": str(group_by or ""),
                    "waro.answer.style": answer_style,
                },
            )
            if plan.rejected_tools:
                span.add_event(
                    "tool_plan.rejected",
                    {
                        "waro.tool.rejected": ",".join(
                            f"{tool.get('name')}:{tool.get('rejected_reason')}"
                            for tool in plan.rejected_tools
                        )
                    },
                )
            audit_semantic_plan = self._audit_semantic_plan(semantic_plan)
            audit_tool_plan = self._audit_tool_plan(plan)
            await self._record_step(
                run_id=run_id,
                tenant_id=context.tenant_id,
                step_type="agent",
                name="sales_tool_planner",
                input_json={
                    "question_length": len(request.question),
                    "period": period,
                    "available_scopes": list(context.scopes),
                    "answer_style": answer_style,
                    "semantic_plan": audit_semantic_plan,
                },
                output_json=audit_tool_plan,
                output_summary=f"Planned {len(plan.steps)} tool call(s).",
            )
            span.set_status(Status(StatusCode.OK))
            return plan

    async def _resolve_semantic_sales_plan(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
        run_id: UUID,
    ) -> dict[str, Any]:
        fallback = self._fallback_semantic_sales_plan(request)
        if self.settings.llm_provider == "disabled":
            with self.tracer.start_as_current_span("sales.semantic_planner") as span:
                span.set_attribute("openinference.span.kind", "CHAIN")
                span.set_attribute("waro.workflow.name", self.graph_name)
                span.set_attribute("waro.run_id", str(run_id))
                span.set_attribute("waro.tenant_id", context.tenant_id)
                span.set_attribute("waro.planner.source", "deterministic_fallback")
                span.set_attribute("waro.answer.style", str(fallback.get("answer_style", "")))
                span.set_attribute("waro.group_by", str(fallback.get("group_by") or ""))
                period = fallback.get("period") if isinstance(fallback.get("period"), dict) else {}
                span.set_attribute("waro.period.date_from", str(period.get("date_from", "")))
                span.set_attribute("waro.period.date_to", str(period.get("date_to", "")))
                span.add_event(
                    "semantic_plan.fallback_used",
                    {"waro.reason": "llm_provider_disabled"},
                )
                span.set_status(Status(StatusCode.OK))
            self._printf_step(
                "sales_semantic_planner",
                {
                    "run_id": str(run_id),
                    "source": "deterministic_fallback",
                    "period": fallback.get("period"),
                    "answer_style": fallback.get("answer_style"),
                    "group_by": fallback.get("group_by"),
                },
            )
            return fallback

        today = datetime.now(ZoneInfo("America/Bogota")).date()
        planner_model = self.settings.llm_planner_model
        with self.tracer.start_as_current_span("llm.sales.planner") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", self.llm_adapter.provider)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            span.set_attribute("waro.request.question_length", len(request.question))
            if self.settings.llm_provider == "kimi":
                span.set_attribute("llm.model", planner_model)
                span.set_attribute("llm.model_name", planner_model)

            try:
                response = await self.llm_adapter.complete(
                    messages=sales_planner_messages(
                        question=request.question,
                        today=today.isoformat(),
                        timezone="America/Bogota",
                        tool_catalog=tool_catalog(),
                    ),
                    temperature=0,
                    model=planner_model,
                )
                raw_plan = self._parse_planner_json(response.content)
                validated = self._validate_semantic_sales_plan(
                    raw_plan,
                    request=request,
                    fallback=fallback,
                    today=today,
                )
                span.add_event(
                    "semantic_plan.validated",
                    {
                        "waro.planner.source": str(validated.get("source", "")),
                        "waro.answer.style": str(validated.get("answer_style", "")),
                        "waro.group_by": str(validated.get("group_by") or ""),
                        "waro.period.date_from": str(validated["period"]["date_from"]),
                        "waro.period.date_to": str(validated["period"]["date_to"]),
                    },
                )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.add_event(
                    "semantic_plan.fallback_used",
                    {
                        "waro.reason": "planner_error",
                        "waro.error_type": type(exc).__name__,
                    },
                )
                await self._record_step(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    step_type="llm",
                    name="sales_semantic_planner",
                    input_json={
                        "provider": self.settings.llm_provider,
                        "model": planner_model
                        if self.settings.llm_provider == "kimi"
                        else None,
                    },
                    output_json={
                        "fallback": True,
                        "error_type": type(exc).__name__,
                        "semantic_plan": self._audit_semantic_plan(fallback),
                    },
                    output_summary="Semantic planner fell back to deterministic resolver.",
                )
                self._printf_step(
                    "sales_semantic_planner",
                    {
                        "run_id": str(run_id),
                        "source": "fallback_error",
                        "error_type": type(exc).__name__,
                        "period": fallback.get("period"),
                        "answer_style": fallback.get("answer_style"),
                    },
                )
                return fallback

            self._printf_step(
                "sales_semantic_planner",
                {
                    "run_id": str(run_id),
                    "source": validated.get("source"),
                    "period": validated.get("period"),
                    "answer_style": validated.get("answer_style"),
                    "group_by": validated.get("group_by"),
                    "confidence": validated.get("confidence"),
                    "tools": [
                        tool.get("name")
                        for tool in validated.get("tools", [])
                        if isinstance(tool, dict)
                    ],
                },
            )
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
            if response.estimated_cost_usd is not None:
                span.set_attribute("llm.cost.estimated_usd", response.estimated_cost_usd)
                span.set_attribute("llm.cost.total", response.estimated_cost_usd)
            span.set_attribute("llm.cost.source", response.cost_source)
            span.set_attribute("waro.resolver.source", str(validated.get("source", "")))
            span.set_attribute("waro.resolver.confidence", float(validated.get("confidence", 0)))
            span.set_attribute("waro.answer.style", str(validated.get("answer_style", "")))
            span.set_attribute("waro.group_by", str(validated.get("group_by") or ""))
            span.set_attribute("waro.period.date_from", str(validated["period"]["date_from"]))
            span.set_attribute("waro.period.date_to", str(validated["period"]["date_to"]))
            span.set_attribute(
                "waro.requested_tools",
                ",".join(
                    str(tool.get("name"))
                    for tool in validated.get("tools", [])
                    if isinstance(tool, dict) and tool.get("name")
                ),
            )
            span.set_status(Status(StatusCode.OK))
            await self._record_step(
                run_id=run_id,
                tenant_id=context.tenant_id,
                step_type="llm",
                name="sales_semantic_planner",
                input_json={
                    "provider": response.provider,
                    "model": response.model,
                    "message_count": 2,
                },
                output_json={
                    "fallback": validated.get("source") != "llm_structured",
                    "semantic_plan": self._audit_semantic_plan(validated),
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
                output_summary=(
                    f"Resolved {validated['period']['date_from']} to "
                    f"{validated['period']['date_to']} as {validated.get('answer_style')}."
                ),
            )
            return validated

    async def _record_planned_tool_step(
        self,
        *,
        run_id: UUID,
        tenant_id: str,
        step: ToolPlanStep,
        plan: ToolPlan | None,
    ) -> UUID:
        with self.tracer.start_as_current_span("sales.plan_tool") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", tenant_id)
            span.set_attribute("waro.tool.name", step.tool_name)
            span.set_attribute("waro.tool.fields", ",".join(step.fields))
            span.set_attribute("waro.tool.reason", step.reason)
            step_id = await self._record_step(
                run_id=run_id,
                tenant_id=tenant_id,
                step_type="tool",
                name=step.tool_name,
                input_json={
                    "arguments": step.arguments,
                    "fields": step.fields,
                    "reason": step.reason,
                    "plan": self._audit_tool_plan(plan) if plan else None,
                },
            )
            span.set_attribute("waro.step_id", str(step_id))
            span.set_status(Status(StatusCode.OK))
            return step_id

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
        intent: str = "sales_metrics",
        tool_plan: dict[str, Any] | None = None,
        previous_context_summary: str | None = None,
    ) -> dict[str, Any]:
        if intent == "small_talk":
            return {
                "intent": "small_talk",
                "answer_style": "small_talk",
                "period": None,
                "metrics": {},
                "highlights": ["Small talk handled without querying sales metrics."],
                "explanation": "Sales metrics were not queried because the message was conversational.",
                "tool_plan": None,
                "tool_calls": [],
            }
        if intent == "follow_up":
            return {
                "intent": "follow_up",
                "answer_style": "follow_up",
                "period": None,
                "metrics": {},
                "previous_context_summary": truncate_text(previous_context_summary or "", 1200),
                "highlights": ["Follow-up handled from conversation context."],
                "explanation": "Sales metrics were not queried because the message refers to prior context.",
                "tool_plan": None,
                "tool_calls": [],
            }

        semantic_plan = self._semantic_plan_from_tool_plan(tool_plan)
        period = (
            semantic_plan.get("period")
            if isinstance(semantic_plan.get("period"), dict)
            else self._resolve_period(request)
        )
        answer_style = (
            str(semantic_plan.get("answer_style"))
            if semantic_plan.get("answer_style")
            else self._resolve_answer_style(request.question)
        )
        metric_data = self._metrics_data_for(tool_calls, "waro.sales.metrics")
        metrics = sanitize_value(
            {
                "total_sales": metric_data.get("totalSales"),
                "order_count": metric_data.get("totalOrders")
                if metric_data.get("totalOrders") is not None
                else metric_data.get("orderCount"),
                "avg_ticket": metric_data.get("avgTicket"),
                "series": metric_data.get("series"),
            }
        )
        highlights = self._highlights(metrics)
        auxiliary_context = self._auxiliary_context(tool_calls)
        analysis_request = self._analysis_request_context(
            request=request,
            answer_style=answer_style,
            period=period,
            metrics=metrics,
            auxiliary_context=auxiliary_context,
            tool_calls=tool_calls,
        )
        financial_analysis = self._financial_analysis_context(
            metrics=metrics,
            auxiliary_context=auxiliary_context,
        )
        response_contract = self._response_contract(
            request=request,
            answer_style=answer_style,
            metrics=metrics,
            analysis_request=analysis_request,
            auxiliary_context=auxiliary_context,
            tool_calls=tool_calls,
        )
        return {
            "intent": "sales_metrics",
            "answer_style": answer_style,
            "period": period,
            "metrics": metrics,
            "analysis_request": analysis_request,
            "response_contract": response_contract,
            "auxiliary_context": auxiliary_context,
            "financial_analysis": financial_analysis
            if answer_style == "financial_analysis"
            else None,
            "highlights": highlights,
            "explanation": "Sales analysis uses the WARO sales metrics tool for the requested period.",
            "semantic_plan": semantic_plan,
            "tool_plan": tool_plan,
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
            "intent": artifact.get("intent"),
            "answer_style": artifact.get("answer_style"),
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
            "tool_plan": artifact.get("tool_plan"),
            "semantic_plan": artifact.get("semantic_plan"),
            "response_contract": artifact.get("response_contract"),
        }

    def _stream_error_type(self, error: Any) -> str | None:
        if not error:
            return None
        if isinstance(error, dict):
            value = error.get("type") or error.get("error") or error.get("code")
            return str(value) if value else "tool_error"
        return type(error).__name__

    def _resolve_sales_intent(
        self,
        question: str,
        *,
        has_conversation_context: bool = False,
    ) -> Literal["sales_metrics", "small_talk", "follow_up"]:
        normalized = self._normalize_question(question)
        if self._is_small_talk(normalized) and not self._has_sales_signal(normalized):
            return "small_talk"
        if (
            has_conversation_context
            and self._is_follow_up(normalized)
            and not self._has_sales_signal(normalized)
        ):
            return "follow_up"
        return "sales_metrics"

    def _fallback_semantic_sales_plan(self, request: SalesQuestionRequest) -> dict[str, Any]:
        period = self._resolve_period(request)
        normalized = self._normalize_question(request.question)
        group_by = request.group_by
        if group_by is None and self._needs_daily_breakdown(normalized):
            group_by = "date"
        return {
            "intent": self._resolve_sales_intent(
                request.question,
                has_conversation_context=request.conversation_id is not None,
            ),
            "period": period,
            "group_by": group_by,
            "answer_style": self._resolve_answer_style(request.question),
            "tools": [{"name": "waro.sales.metrics", "reason": "fallback_default"}],
            "confidence": 0.55,
            "reason": "deterministic_fallback",
            "source": "deterministic_fallback",
        }

    def _parse_planner_json(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Planner response must be a JSON object.")
        return parsed

    def _validate_semantic_sales_plan(
        self,
        plan: dict[str, Any],
        *,
        request: SalesQuestionRequest,
        fallback: dict[str, Any],
        today: date,
    ) -> dict[str, Any]:
        normalized = self._normalize_question(request.question)
        allowed_styles = {"direct_metric", "business_analysis", "financial_analysis", "diagnostic"}
        allowed_groups = {"date", "weekday", "hour", "product", "payment", "ticket"}
        intent = plan.get("intent") if plan.get("intent") in {"small_talk", "sales_metrics"} else fallback["intent"]
        answer_style = (
            plan.get("answer_style")
            if plan.get("answer_style") in allowed_styles
            else fallback["answer_style"]
        )
        resolved_answer_style = self._resolve_answer_style(request.question)
        if resolved_answer_style == "direct_metric":
            answer_style = "direct_metric"
        elif answer_style == "direct_metric":
            answer_style = resolved_answer_style

        period = self._period_from_planner(plan, fallback=fallback)
        period_source = "llm"
        if request.date_from and request.date_to:
            period = {"date_from": request.date_from, "date_to": request.date_to}
            period_source = "request_override"
        elif re.search(r"\bmes\s+pasad[oa]\b", normalized):
            period = self._previous_month_period(today)
            period_source = "guardrail_previous_month"
        elif period is fallback["period"]:
            period_source = "fallback"
        if period.get("date_to") and period["date_to"] > today.isoformat():
            period = {
                "date_from": period["date_from"],
                "date_to": today.isoformat(),
            }
            period_source = "guardrail_future_clamped"

        group_by = plan.get("group_by")
        if group_by not in allowed_groups:
            group_by = fallback.get("group_by")
        if request.group_by:
            group_by = request.group_by
        elif self._needs_daily_breakdown(normalized):
            group_by = "date"

        confidence = plan.get("confidence")
        if not isinstance(confidence, int | float):
            confidence = fallback.get("confidence", 0.55)
        confidence = max(0.0, min(1.0, float(confidence)))
        tools = plan.get("tools") if isinstance(plan.get("tools"), list) else fallback["tools"]
        return {
            "intent": intent,
            "period": period,
            "group_by": group_by,
            "answer_style": answer_style,
            "tools": sanitize_value(tools),
            "confidence": confidence,
            "reason": str(plan.get("reason") or fallback["reason"])[:300],
            "source": "llm_structured" if period_source == "llm" else period_source,
            "period_source": period_source,
        }

    def _resolve_contextual_sales_plan(
        self,
        *,
        request: SalesQuestionRequest,
        semantic_plan: dict[str, Any],
        previous_context_summary: str | None,
    ) -> dict[str, Any]:
        if request.conversation_id is None or not previous_context_summary:
            return {"used_context": False}
        normalized = self._normalize_question(request.question)
        today = datetime.now(ZoneInfo("America/Bogota")).date()
        if self._request_has_explicit_period(request, normalized=normalized, today=today):
            return {"used_context": False, "reason": "request_has_period"}
        if not self._is_contextual_drilldown(normalized):
            return {"used_context": False, "reason": "not_contextual_drilldown"}

        period = self._period_from_context_summary(
            previous_context_summary,
            today=today,
        )
        if not period:
            return {"used_context": False, "reason": "no_context_period"}

        resolution: dict[str, Any] = {
            "used_context": True,
            "inherited_period": True,
            "period": period,
            "previous_context_summary_length": len(previous_context_summary),
        }
        if self._is_product_drilldown(normalized):
            resolution["group_by"] = "product"
            resolution["tool_name"] = "waro.financial.products"
        if self._is_financial_drilldown(normalized):
            resolution["answer_style"] = "financial_analysis"
        elif semantic_plan.get("answer_style") == "direct_metric":
            resolution["answer_style"] = "business_analysis"
        return resolution

    def _request_has_explicit_period(
        self,
        request: SalesQuestionRequest,
        *,
        normalized: str,
        today: date,
    ) -> bool:
        if request.date_from and request.date_to:
            return True
        return self._infer_period_from_question(normalized, today=today) is not None

    def _is_contextual_drilldown(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(cuales?|cuantos?|muestrame|dime|detalle|lista|top|mayor(?:es)?|"
                r"menor(?:es)?|mejor(?:es)?|peor(?:es)?|productos?|platos?|clientes?|"
                r"margen|ganancia|ganancias|utilidad|rentabilidad|costo|costos|"
                r"rendimiento|performance)\b",
                normalized,
            )
        )

    def _is_product_drilldown(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(productos?|platos?|items?|margen|ganancia|ganancias|utilidad|"
                r"rentabilidad|costo|costos|rendimiento|performance)\b",
                normalized,
            )
        )

    def _is_financial_drilldown(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(margen|ganancia|ganancias|utilidad|rentabilidad|costo|costos|"
                r"financier[ao]s?|profit|performance|rendimiento)\b",
                normalized,
            )
        )

    def _period_from_context_summary(
        self,
        summary: str,
        *,
        today: date,
    ) -> dict[str, str] | None:
        normalized = self._normalize_question(summary)
        iso_range = re.search(
            r"\b(?:del|desde)\s+(?P<from>\d{4}-\d{2}-\d{2})\s+"
            r"(?:al|hasta)\s+(?P<to>\d{4}-\d{2}-\d{2})\b",
            normalized,
        )
        if iso_range:
            return {
                "date_from": iso_range.group("from"),
                "date_to": iso_range.group("to"),
            }
        iso_single = re.search(r"\bel\s+(?P<date>\d{4}-\d{2}-\d{2})\b", normalized)
        if iso_single:
            value = iso_single.group("date")
            return {"date_from": value, "date_to": value}

        month_names = "|".join(SPANISH_MONTHS)
        spanish_range = re.search(
            rf"\b(?:periodo\s+)?(?:del|desde(?:\s+el)?)\s+"
            rf"(?P<day_from>\d{{1,2}})\s+de\s+(?P<month_from>{month_names})\s+"
            rf"(?:al|hasta(?:\s+el)?)\s+"
            rf"(?P<day_to>\d{{1,2}})\s+de\s+(?P<month_to>{month_names})"
            rf"(?:\s+de\s+(?P<year>\d{{4}}))?\b",
            normalized,
        )
        if not spanish_range:
            return None
        year = int(spanish_range.group("year") or today.year)
        try:
            start = date(
                year,
                SPANISH_MONTHS[spanish_range.group("month_from")],
                int(spanish_range.group("day_from")),
            )
            end = date(
                year,
                SPANISH_MONTHS[spanish_range.group("month_to")],
                int(spanish_range.group("day_to")),
            )
        except (KeyError, ValueError):
            return None
        if start > end:
            return None
        return {"date_from": start.isoformat(), "date_to": end.isoformat()}

    def _period_from_planner(
        self,
        plan: dict[str, Any],
        *,
        fallback: dict[str, Any],
    ) -> dict[str, str]:
        date_from = plan.get("date_from")
        date_to = plan.get("date_to")
        if not isinstance(date_from, str) or not isinstance(date_to, str):
            return fallback["period"]
        try:
            start = date.fromisoformat(date_from)
            end = date.fromisoformat(date_to)
        except ValueError:
            return fallback["period"]
        if start > end:
            return fallback["period"]
        return {"date_from": start.isoformat(), "date_to": end.isoformat()}

    def _semantic_plan_from_tool_plan(self, tool_plan: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(tool_plan, dict):
            return {}
        semantic_plan = tool_plan.get("semantic_plan")
        if isinstance(semantic_plan, dict):
            return semantic_plan
        return {}

    def _audit_tool_plan(self, plan: ToolPlan) -> dict[str, Any]:
        payload = plan.to_dict()
        semantic_plan = payload.get("semantic_plan")
        if isinstance(semantic_plan, dict):
            payload["semantic_plan"] = self._audit_semantic_plan(semantic_plan)
        return payload

    def _audit_semantic_plan(self, semantic_plan: dict[str, Any]) -> dict[str, Any]:
        payload = dict(semantic_plan)
        reason = payload.pop("reason", None)
        if isinstance(reason, str) and reason:
            payload["reason_length"] = len(reason)
        return payload

    def _resolve_answer_style(self, question: str) -> AnswerStyle:
        normalized = self._normalize_question(question)
        if re.search(
            r"\b(analisis financiero|financiero|rentabilidad|margen|margenes|"
            r"costos?|utilidad|ganancia|ganancias|profit)\b",
            normalized,
        ):
            return "financial_analysis"
        if re.search(r"\b(productos?|platos?|items?)\b", normalized) and re.search(
            r"\b(top|mayor(?:es)?|mejor(?:es)?|peor(?:es)?|menor(?:es)?|mas|"
            r"vendid[oa]s?|ventas?|cantidad|unidades|ranking)\b",
            normalized,
        ):
            return "business_analysis"
        if self._needs_daily_breakdown(normalized) or re.search(
            r"\b(clientes?|compradores?|mejores?|top|ranking)\b",
            normalized,
        ):
            return "business_analysis"
        if re.search(
            r"\b(analiza|analisis|explica|explicame|detalle|detallado|"
            r"como vamos|que significa|recomendaciones?|acciones?)\b",
            normalized,
        ):
            return "business_analysis"
        if re.search(
            r"\b(error|problema|fall[ao]|no hay datos|revisar|diagnostico)\b",
            normalized,
        ):
            return "diagnostic"
        return "direct_metric"

    def _is_small_talk(self, normalized: str) -> bool:
        return any(re.search(pattern, normalized) for pattern in SMALL_TALK_PATTERNS)

    def _is_follow_up(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(como asi|por que|porque|explicame|explica eso|que significa|"
                r"a que te refieres|amplia|ampliame|detalle eso|y eso|eso que quiere decir)\b",
                normalized,
            )
        )

    def _has_sales_signal(self, normalized: str) -> bool:
        return any(re.search(pattern, normalized) for pattern in SALES_SIGNAL_PATTERNS)

    def _resolve_period(self, request: SalesQuestionRequest) -> dict[str, str]:
        if request.date_from and request.date_to:
            return {"date_from": request.date_from, "date_to": request.date_to}
        today = datetime.now(ZoneInfo("America/Bogota")).date()
        inferred = self._infer_period_from_question(request.question, today=today)
        if inferred:
            return {
                "date_from": request.date_from or inferred["date_from"],
                "date_to": request.date_to or inferred["date_to"],
            }
        yesterday = today - timedelta(days=1)
        value = yesterday.isoformat()
        return {"date_from": request.date_from or value, "date_to": request.date_to or value}

    def _infer_period_from_question(
        self,
        question: str,
        *,
        today: date,
    ) -> dict[str, str] | None:
        normalized = self._normalize_question(question)
        explicit_date = self._explicit_date_from_question(normalized, today=today)
        if explicit_date:
            value = explicit_date.isoformat()
            return {"date_from": value, "date_to": value}
        if re.search(r"\b(hoy|dia actual)\b", normalized):
            value = today.isoformat()
            return {"date_from": value, "date_to": value}
        if re.search(r"\bayer\b", normalized):
            value = (today - timedelta(days=1)).isoformat()
            return {"date_from": value, "date_to": value}
        if re.search(r"\bmes\s+pasad[oa]\b", normalized):
            return self._previous_month_period(today)
        if re.search(r"\b(este mes|del mes|mes actual|month to date|mtd)\b", normalized):
            start = today.replace(day=1)
            return {"date_from": start.isoformat(), "date_to": today.isoformat()}
        if re.search(r"\b(esta semana|semana actual)\b", normalized):
            start = today - timedelta(days=today.weekday())
            return {"date_from": start.isoformat(), "date_to": today.isoformat()}
        last_days = self._last_days_count(normalized)
        if last_days:
            start = today - timedelta(days=last_days - 1)
            return {"date_from": start.isoformat(), "date_to": today.isoformat()}
        return None

    def _previous_month_period(self, today: date) -> dict[str, str]:
        first_this_month = today.replace(day=1)
        last_previous_month = first_this_month - timedelta(days=1)
        first_previous_month = last_previous_month.replace(day=1)
        return {
            "date_from": first_previous_month.isoformat(),
            "date_to": last_previous_month.isoformat(),
        }

    def _needs_daily_breakdown(self, normalized: str) -> bool:
        return bool(
            re.search(
                r"\b(promedio\s+(?:de\s+)?ventas?\s+por\s+dia|por\s+dia|"
                r"dia\s+a\s+dia|diari[ao]|tendencia\s+diaria)\b",
                normalized,
            )
        )

    def _last_days_count(self, normalized: str) -> int | None:
        match = re.search(
            r"\b(?:los\s+|las\s+)?ultim[oa]s?\s+(?P<count>\d{1,3}|"
            r"uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|treinta"
            r")\s+dias?\b",
            normalized,
        )
        if not match:
            return None
        count_text = match.group("count")
        count = SPANISH_NUMBER_WORDS.get(count_text, int(count_text) if count_text.isdigit() else 0)
        if 1 <= count <= 90:
            return count
        return None

    def _explicit_date_from_question(self, normalized: str, *, today: date) -> date | None:
        months = {
            "enero": 1,
            "febrero": 2,
            "marzo": 3,
            "abril": 4,
            "mayo": 5,
            "junio": 6,
            "julio": 7,
            "agosto": 8,
            "septiembre": 9,
            "setiembre": 9,
            "octubre": 10,
            "noviembre": 11,
            "diciembre": 12,
        }
        month_names = "|".join(months)
        match = re.search(
            rf"\b(?:dia\s+)?(?P<day>\d{{1,2}})\s+de\s+(?P<month>{month_names})(?:\s+(?:de|del)\s+(?P<year>\d{{4}}))?\b",
            normalized,
        )
        if match:
            day = int(match.group("day"))
            month = months[match.group("month")]
            year = int(match.group("year") or today.year)
            try:
                return date(year, month, day)
            except ValueError:
                return None

        match = re.search(
            r"\b(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b",
            normalized,
        )
        if not match:
            match = re.search(
                r"\b(?P<day>\d{1,2})[-/](?P<month>\d{1,2})(?:[-/](?P<year>\d{2,4}))?\b",
                normalized,
            )
        if match:
            day = int(match.group("day"))
            month = int(match.group("month"))
            year_text = match.group("year")
            year = today.year if not year_text else int(year_text)
            if year < 100:
                year += 2000
            try:
                return date(year, month, day)
            except ValueError:
                return None

        correction = re.search(r"\b(?P<day>\d{1,2})\s+no\s+\d{1,2}\b", normalized)
        if correction:
            day = int(correction.group("day"))
            try:
                return date(today.year, today.month, day)
            except ValueError:
                return None
        return None

    def _normalize_question(self, question: str) -> str:
        folded = question.casefold().strip()
        without_accents = "".join(
            char
            for char in unicodedata.normalize("NFKD", folded)
            if not unicodedata.combining(char)
        )
        return " ".join(without_accents.split())

    def _metrics_data_for(
        self,
        tool_calls: list[dict[str, Any]],
        tool_name: str,
    ) -> dict[str, Any]:
        for call in tool_calls:
            if call["tool_name"] != tool_name or call["status"] != "succeeded":
                continue
            result = call.get("result")
            if isinstance(result, dict):
                data = result.get("data")
                if isinstance(data, dict):
                    return data
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    rows = [row for row in data if isinstance(row, dict) and row]
                    totals = self._aggregate_metric_rows(rows)
                    return {"series": rows, **totals} if rows else {}
                return result
            if isinstance(result, list) and result and isinstance(result[0], dict):
                rows = [row for row in result if isinstance(row, dict) and row]
                totals = self._aggregate_metric_rows(rows)
                return {"series": rows, **totals} if rows else {}
        return {}

    def _aggregate_metric_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        total_sales = 0.0
        total_orders = 0.0
        has_sales = False
        has_orders = False
        for row in rows:
            sales_value = self._first_numeric(
                row,
                ("totalSales", "total_sales", "sales", "revenue", "total_revenue"),
            )
            order_value = self._first_numeric(
                row,
                ("totalOrders", "total_orders", "orderCount", "orders", "order_count"),
            )
            if sales_value is not None:
                total_sales += sales_value
                has_sales = True
            if order_value is not None:
                total_orders += order_value
                has_orders = True
        output: dict[str, Any] = {}
        if has_sales:
            output["totalSales"] = total_sales
        if has_orders:
            output["totalOrders"] = total_orders
        if has_sales and has_orders and total_orders:
            output["avgTicket"] = total_sales / total_orders
        return output

    def _first_numeric(self, row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = self._numeric_value(row.get(key))
            if value is not None:
                return value
        return None

    def _rows_for(self, tool_calls: list[dict[str, Any]], tool_name: str) -> list[dict[str, Any]]:
        for call in tool_calls:
            if call["tool_name"] != tool_name or call["status"] != "succeeded":
                continue
            result = call.get("result")
            if isinstance(result, dict) and isinstance(result.get("data"), list):
                return [row for row in result["data"] if isinstance(row, dict)]
            if isinstance(result, dict) and isinstance(result.get("products"), list):
                return [row for row in result["products"] if isinstance(row, dict)]
            if isinstance(result, list):
                return [row for row in result if isinstance(row, dict)]
        return []

    def _result_for(
        self,
        tool_calls: list[dict[str, Any]],
        tool_name: str,
    ) -> dict[str, Any]:
        for call in tool_calls:
            if call["tool_name"] != tool_name or call["status"] != "succeeded":
                continue
            result = call.get("result")
            if isinstance(result, dict):
                return result
        return {}

    def _auxiliary_context(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        financial_rows = self._rows_for(tool_calls, "waro.financial.products")
        financial_result = self._result_for(tool_calls, "waro.financial.products")
        menu_rows = self._rows_for(tool_calls, "waro.menu.products")
        context: dict[str, Any] = {}
        if financial_result:
            context["financial_metrics"] = sanitize_value(financial_result.get("metrics", {}))
            context["financial_insights"] = sanitize_value(financial_result.get("insights", {}))
        if financial_rows:
            context["financial_products"] = [
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "margin": row.get("margin"),
                    "revenue": row.get("revenue")
                    if row.get("revenue") is not None
                    else row.get("total_revenue"),
                    "cost": row.get("cost"),
                    "quantity": row.get("quantity")
                    if row.get("quantity") is not None
                    else row.get("sales"),
                    "profit": row.get("profit"),
                    "price": row.get("price"),
                    "classification": row.get("classification"),
                }
                for row in financial_rows[:10]
            ]
        if menu_rows:
            context["menu_products"] = [
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "price": row.get("price"),
                    "is_available": row.get("is_available"),
                    "category": row.get("category"),
                }
                for row in menu_rows[:10]
            ]
        return sanitize_value(context)

    def _analysis_request_context(
        self,
        *,
        request: SalesQuestionRequest,
        answer_style: str,
        period: dict[str, Any],
        metrics: dict[str, Any],
        auxiliary_context: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized = self._normalize_question(request.question)
        area = self._analysis_area(normalized=normalized, answer_style=answer_style)
        dimensions = self._analysis_dimensions(normalized=normalized, area=area)
        requested_metrics = self._analysis_metrics(
            normalized=normalized,
            area=area,
            metrics=metrics,
            auxiliary_context=auxiliary_context,
        )
        data_available = self._analysis_data_available(
            metrics=metrics,
            auxiliary_context=auxiliary_context,
            tool_calls=tool_calls,
        )
        return sanitize_value(
            {
                "request_type": "analysis"
                if answer_style in {"business_analysis", "financial_analysis", "diagnostic"}
                else "metric_lookup",
                "area": area,
                "objective": self._analysis_objective(
                    normalized=normalized,
                    area=area,
                    answer_style=answer_style,
                ),
                "period": period,
                "dimensions": dimensions,
                "requested_metrics": requested_metrics,
                "data_available": data_available,
                "depth": self._analysis_depth(normalized),
                "missing_data_policy": "state_limitations",
            }
        )

    def _analysis_area(self, *, normalized: str, answer_style: str) -> str:
        if answer_style == "financial_analysis" or re.search(
            r"\b(financier[ao]s?|margen|rentabilidad|ganancia|ganancias|utilidad|costos?)\b",
            normalized,
        ):
            return "finance"
        if re.search(r"\b(clientes?|retencion|frecuencia|fidelidad|recompra)\b", normalized):
            return "customers"
        if re.search(r"\b(menu|productos?|platos?|recetas?|categorias?)\b", normalized):
            return "menu"
        return "commercial"

    def _analysis_objective(
        self,
        *,
        normalized: str,
        area: str,
        answer_style: str,
    ) -> str:
        if re.search(r"\b(compara|comparar|vs|contra|diferencia)\b", normalized):
            return "compare_periods_or_segments"
        if re.search(
            r"\b(top|mayor(?:es)?|mejor(?:es)?|peor(?:es)?|menor(?:es)?|mas|"
            r"vendid[oa]s?|cantidad|unidades|ranking)\b",
            normalized,
        ):
            return "rank_entities"
        if answer_style == "diagnostic":
            return "diagnose_issue"
        if area == "finance":
            return "evaluate_gross_profitability"
        if area == "customers":
            return "understand_customer_behavior"
        if area == "menu":
            return "evaluate_menu_performance"
        return "evaluate_sales_performance"

    def _analysis_dimensions(self, *, normalized: str, area: str) -> list[str]:
        dimensions = ["overall"]
        if area in {"finance", "menu"} or re.search(r"\b(productos?|platos?|items?)\b", normalized):
            dimensions.append("product")
        if re.search(r"\b(categoria|categorias)\b", normalized):
            dimensions.append("category")
        if re.search(r"\b(clientes?|cliente)\b", normalized):
            dimensions.append("customer")
        if re.search(r"\b(dia|dias|fecha|diari[ao])\b", normalized):
            dimensions.append("date")
        if re.search(r"\b(hora|horas)\b", normalized):
            dimensions.append("hour")
        return list(dict.fromkeys(dimensions))

    def _analysis_metrics(
        self,
        *,
        normalized: str,
        area: str,
        metrics: dict[str, Any],
        auxiliary_context: dict[str, Any],
    ) -> list[str]:
        requested = []
        if metrics.get("total_sales") is not None or re.search(r"\b(ventas?|ingresos?)\b", normalized):
            requested.append("sales")
        if metrics.get("avg_ticket") is not None or "ticket" in normalized:
            requested.append("average_ticket")
        if area == "finance" or re.search(r"\b(margen|rentabilidad|ganancia|utilidad|costos?)\b", normalized):
            requested.extend(["gross_margin", "gross_profit", "product_profit"])
        if auxiliary_context.get("financial_products"):
            requested.append("product_ranking")
        if area == "customers":
            requested.extend(["customer_activity", "retention", "frequency"])
        return list(dict.fromkeys(requested))

    def _analysis_data_available(
        self,
        *,
        metrics: dict[str, Any],
        auxiliary_context: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, bool]:
        tool_names = {call.get("tool_name") for call in tool_calls if isinstance(call, dict)}
        return {
            "sales": metrics.get("total_sales") is not None,
            "orders": metrics.get("order_count") is not None,
            "average_ticket": metrics.get("avg_ticket") is not None,
            "product_financials": bool(auxiliary_context.get("financial_products")),
            "product_margin": bool(auxiliary_context.get("financial_metrics")),
            "food_cost": "waro.analytics.food_cost" in tool_names,
            "menu": "waro.menu.products" in tool_names or "waro.analytics.menu" in tool_names,
            "customers": "waro.customers.metrics" in tool_names
            or "waro.customers.list" in tool_names,
            "labor_cost": False,
            "fixed_expenses": False,
            "inventory": False,
            "cash_flow": False,
        }

    def _response_contract(
        self,
        *,
        request: SalesQuestionRequest,
        answer_style: str,
        metrics: dict[str, Any],
        analysis_request: dict[str, Any],
        auxiliary_context: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request_kind = self._response_request_kind(
            normalized=self._normalize_question(request.question),
            answer_style=answer_style,
            analysis_request=analysis_request,
        )
        required_data = self._required_data_for_request_kind(request_kind)
        data_available = (
            analysis_request.get("data_available", {})
            if isinstance(analysis_request.get("data_available"), dict)
            else {}
        )
        if isinstance(metrics.get("series"), list) and metrics["series"]:
            data_available = {**data_available, "sales_series": True}
        failed_tools = [
            {
                "tool_name": call.get("tool_name"),
                "error_type": self._stream_error_type(call.get("error")),
            }
            for call in tool_calls
            if isinstance(call, dict) and call.get("status") != "succeeded"
        ]
        missing_required_data = [
            item for item in required_data if not bool(data_available.get(item))
        ]
        if failed_tools:
            data_status = "failed"
        elif missing_required_data:
            data_status = "empty" if not any(data_available.values()) else "partial"
        else:
            data_status = "ok"

        safe_to_answer = data_status in {"ok", "partial"} and (
            request_kind == "financial_analysis" or not missing_required_data
        )
        error_message = None
        if not safe_to_answer:
            error_message = self._contract_error_message(
                request_kind=request_kind,
                data_status=data_status,
                missing_required_data=missing_required_data,
                failed_tools=failed_tools,
                period=analysis_request.get("period"),
            )
        return sanitize_value(
            {
                "request_kind": request_kind,
                "required_data": required_data,
                "data_status": data_status,
                "safe_to_answer": safe_to_answer,
                "missing_required_data": missing_required_data,
                "failed_tools": failed_tools,
                "error_message": error_message,
            }
        )

    def _response_request_kind(
        self,
        *,
        normalized: str,
        answer_style: str,
        analysis_request: dict[str, Any],
    ) -> str:
        dimensions = analysis_request.get("dimensions")
        objective = analysis_request.get("objective")
        area = analysis_request.get("area")
        if isinstance(dimensions, list) and "product" in dimensions and objective == "rank_entities":
            return "product_ranking"
        if isinstance(dimensions, list) and "customer" in dimensions:
            return "customer_ranking"
        if isinstance(dimensions, list) and "date" in dimensions and re.search(
            r"\b(por dia|por dias|dia a dia|diari[ao]|fecha|fechas|promedio de ventas por dia)\b",
            normalized,
        ):
            return "daily_analysis"
        if answer_style == "financial_analysis" or area == "finance":
            return "financial_analysis"
        if answer_style == "direct_metric":
            return "direct_metric"
        return "business_analysis"

    def _required_data_for_request_kind(self, request_kind: str) -> list[str]:
        if request_kind == "product_ranking":
            return ["product_financials"]
        if request_kind == "customer_ranking":
            return ["customers"]
        if request_kind == "daily_analysis":
            return ["sales_series"]
        if request_kind == "financial_analysis":
            return ["sales"]
        if request_kind == "direct_metric":
            return ["sales"]
        return ["sales"]

    def _contract_error_message(
        self,
        *,
        request_kind: str,
        data_status: str,
        missing_required_data: list[str],
        failed_tools: list[dict[str, Any]],
        period: Any,
    ) -> str:
        label = self._period_label(period) if isinstance(period, dict) else "el periodo consultado"
        failed_names = [
            str(tool.get("tool_name"))
            for tool in failed_tools
            if isinstance(tool, dict) and tool.get("tool_name")
        ]
        if request_kind == "product_ranking":
            subject = "ranking de productos"
        elif request_kind == "customer_ranking":
            subject = "ranking de clientes"
        elif request_kind == "daily_analysis":
            subject = "analisis diario de ventas"
        elif request_kind == "financial_analysis":
            subject = "analisis financiero"
        else:
            subject = "consulta de ventas"

        if data_status == "failed" and failed_names:
            return (
                f"No pude completar el {subject} para {label}. "
                f"Fallo la consulta: {', '.join(failed_names)}."
            )
        if missing_required_data:
            return (
                f"No pude completar el {subject} para {label}: "
                f"faltan datos requeridos ({', '.join(missing_required_data)})."
            )
        return f"No pude completar el {subject} para {label}."

    def _analysis_depth(self, normalized: str) -> str:
        if re.search(r"\b(detallad[ao]|profundo|profundiza|completo)\b", normalized):
            return "deep"
        if re.search(r"\b(resumen|rapido|breve|concreto)\b", normalized):
            return "brief"
        return "executive"

    def _financial_analysis_context(
        self,
        *,
        metrics: dict[str, Any],
        auxiliary_context: dict[str, Any],
    ) -> dict[str, Any]:
        financial_metrics = (
            auxiliary_context.get("financial_metrics")
            if isinstance(auxiliary_context.get("financial_metrics"), dict)
            else {}
        )
        financial_products = (
            auxiliary_context.get("financial_products")
            if isinstance(auxiliary_context.get("financial_products"), list)
            else []
        )
        product_revenue = self._numeric_value(financial_metrics.get("total_revenue"))
        product_profit = self._numeric_value(financial_metrics.get("total_profit"))
        gross_margin_estimate = None
        if product_revenue and product_profit is not None:
            gross_margin_estimate = round((product_profit / product_revenue) * 100, 2)

        top_products_by_profit = sorted(
            [
                product
                for product in financial_products
                if isinstance(product, dict)
                and self._numeric_value(product.get("profit")) is not None
            ],
            key=lambda product: self._numeric_value(product.get("profit")) or 0,
            reverse=True,
        )[:5]

        available_metrics = []
        if metrics.get("total_sales") is not None:
            available_metrics.append("sales")
        if metrics.get("avg_ticket") is not None:
            available_metrics.append("average_ticket")
        if product_profit is not None:
            available_metrics.append("product_gross_profit")
        if gross_margin_estimate is not None:
            available_metrics.append("gross_margin_estimate")
        if financial_products:
            available_metrics.append("product_profit_ranking")

        missing_metrics = [
            "labor_cost",
            "fixed_expenses",
            "net_profit",
            "cash_flow",
            "break_even",
            "inventory_turnover",
            "prime_cost",
        ]

        return sanitize_value(
            {
                "analysis_quality": "partial",
                "analysis_scope": "commercial_finance_gross_margin",
                "available_metrics": available_metrics,
                "missing_metrics": missing_metrics,
                "supported_metrics": {
                    "total_sales": metrics.get("total_sales"),
                    "order_count": metrics.get("order_count"),
                    "avg_ticket": metrics.get("avg_ticket"),
                    "product_revenue": product_revenue,
                    "product_gross_profit": product_profit,
                    "gross_margin_estimate_pct": gross_margin_estimate,
                    "active_products": financial_metrics.get("active_products"),
                    "low_performance_count": financial_metrics.get("low_performance_count"),
                },
                "top_products_by_profit": [
                    {
                        "name": product.get("name"),
                        "profit": product.get("profit"),
                        "revenue": product.get("revenue"),
                        "quantity": product.get("quantity"),
                        "margin": product.get("margin"),
                        "classification": product.get("classification"),
                    }
                    for product in top_products_by_profit
                ],
                "allowed_conclusions": [
                    "Evaluate sales, average ticket, gross product profit, and product-level contribution.",
                    "Rank products by observed gross profit when product profit is available.",
                    "Identify data gaps before making net profitability claims.",
                ],
                "unsupported_conclusions": [
                    "Do not claim net profitability, EBITDA, cash flow health, or break-even status.",
                    "Do not conclude long-term viability without labor, fixed expenses, inventory, and cash-flow data.",
                    "Do not equate Low Performance classification with negative net profitability.",
                ],
                "recommended_next_data": [
                    "labor_cost",
                    "fixed_expenses",
                    "inventory",
                    "delivery_commissions",
                    "discounts_and_refunds",
                ],
            }
        )

    def _numeric_value(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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
        if artifact.get("intent") == "small_talk":
            return (
                "Hola, estoy funcionando. Puedes preguntarme por ventas, por ejemplo: "
                "dame las ventas de ayer o dime las ventas de los ultimos 15 dias."
            )
        if artifact.get("intent") == "follow_up":
            previous = str(artifact.get("previous_context_summary") or "").strip()
            if previous:
                return f"Claro. Me referia a esto: {previous}"
            return (
                "Claro. Necesito un poco mas de contexto para explicar ese punto. "
                "Preguntame sobre la ultima metrica, un producto, una fecha o un margen especifico."
            )
        metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
        answer_style = artifact.get("answer_style")
        total_sales = metrics.get("total_sales")
        order_count = metrics.get("order_count")
        avg_ticket = metrics.get("avg_ticket")
        response_contract = (
            artifact.get("response_contract")
            if isinstance(artifact.get("response_contract"), dict)
            else {}
        )
        if response_contract and not response_contract.get("safe_to_answer", True):
            error_message = str(response_contract.get("error_message") or "").strip()
            if error_message:
                return error_message
        if answer_style != "direct_metric" and self._is_product_analysis_artifact(artifact):
            return self._build_product_fallback_summary(artifact)
        if answer_style != "direct_metric" and response_contract.get("request_kind") == "daily_analysis":
            return self._build_daily_fallback_summary(artifact)
        if answer_style == "direct_metric":
            period = artifact.get("period") if isinstance(artifact.get("period"), dict) else {}
            label = self._period_label(period)
            if total_sales is None and order_count is None and avg_ticket is None:
                return f"{label}: no encontre metricas de ventas para ese periodo."
            parts = []
            if total_sales is not None:
                parts.append(f"vendiste {self._format_cop(total_sales)}")
            if order_count is not None:
                parts.append(f"en {int(order_count)} ordenes")
            summary = f"{label}: " + " ".join(parts) + "."
            if avg_ticket is not None:
                summary += f" Ticket promedio: {self._format_cop(avg_ticket)}."
            return summary
        if total_sales is None and avg_ticket is None:
            return "Sales workflow completed without returned sales metrics."
        parts = []
        if total_sales is not None:
            parts.append(f"total sales {total_sales}")
        if avg_ticket is not None:
            parts.append(f"average ticket {avg_ticket}")
        return "Sales workflow completed with " + " and ".join(parts) + "."

    def _is_product_analysis_artifact(self, artifact: dict[str, Any]) -> bool:
        analysis_request = (
            artifact.get("analysis_request")
            if isinstance(artifact.get("analysis_request"), dict)
            else {}
        )
        dimensions = analysis_request.get("dimensions")
        requested_metrics = analysis_request.get("requested_metrics")
        return (
            isinstance(dimensions, list)
            and "product" in dimensions
            and (
                analysis_request.get("objective") == "rank_entities"
                or (
                    isinstance(requested_metrics, list)
                    and any(
                        metric in requested_metrics
                        for metric in {
                            "product_ranking",
                            "product_profit",
                            "gross_profit",
                            "gross_margin",
                        }
                    )
                )
            )
        )

    def _build_product_fallback_summary(self, artifact: dict[str, Any]) -> str:
        auxiliary_context = (
            artifact.get("auxiliary_context")
            if isinstance(artifact.get("auxiliary_context"), dict)
            else {}
        )
        products = (
            auxiliary_context.get("financial_products")
            if isinstance(auxiliary_context.get("financial_products"), list)
            else []
        )
        period = artifact.get("period") if isinstance(artifact.get("period"), dict) else {}
        label = self._period_label(period)
        if not products:
            failed_product_tools = [
                call.get("tool_name")
                for call in artifact.get("tool_calls", [])
                if isinstance(call, dict)
                and call.get("tool_name")
                in {"waro.financial.products", "waro.analytics.menu", "waro.menu.products"}
                and call.get("status") != "succeeded"
            ]
            if failed_product_tools:
                tools = ", ".join(str(tool) for tool in failed_product_tools)
                return (
                    f"No pude completar el analisis de productos para {label}. "
                    f"Fallo la consulta de datos de producto: {tools}."
                )
            return (
                f"No pude completar el ranking de productos para {label}: "
                "la consulta no devolvio datos de producto."
            )

        ranked = sorted(
            [product for product in products if isinstance(product, dict)],
            key=lambda product: self._numeric_value(product.get("quantity"))
            or self._numeric_value(product.get("revenue"))
            or self._numeric_value(product.get("profit"))
            or 0,
            reverse=True,
        )[:5]
        lines = [f"No pude generar el analisis completo, pero encontre estos productos para {label}:"]
        for index, product in enumerate(ranked, start=1):
            name = str(product.get("name") or "Producto sin nombre")
            details = []
            quantity = self._numeric_value(product.get("quantity"))
            revenue = self._numeric_value(product.get("revenue"))
            profit = self._numeric_value(product.get("profit"))
            if quantity is not None:
                details.append(f"{quantity:g} unidades")
            if revenue is not None:
                details.append(f"{self._format_cop(revenue)} en ventas")
            if profit is not None:
                details.append(f"{self._format_cop(profit)} de ganancia bruta")
            suffix = f" ({', '.join(details)})" if details else ""
            lines.append(f"{index}. {name}{suffix}")
        return "\n".join(lines)

    def _build_daily_fallback_summary(self, artifact: dict[str, Any]) -> str:
        metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
        series = metrics.get("series") if isinstance(metrics.get("series"), list) else []
        period = artifact.get("period") if isinstance(artifact.get("period"), dict) else {}
        label = self._period_label(period)
        if not series:
            return f"No pude completar el analisis diario de ventas para {label}: no hay serie diaria."
        total_sales = metrics.get("total_sales")
        order_count = metrics.get("order_count")
        avg_ticket = metrics.get("avg_ticket")
        parts = [f"{label}: encontre {len(series)} dias con datos."]
        if total_sales is not None:
            parts.append(f"Ventas totales: {self._format_cop(total_sales)}.")
        if order_count is not None:
            parts.append(f"Ordenes: {int(float(order_count))}.")
        if avg_ticket is not None:
            parts.append(f"Ticket promedio: {self._format_cop(avg_ticket)}.")
        return " ".join(parts)

    def _should_use_llm(self, artifact: dict[str, Any]) -> bool:
        return (
            artifact.get("intent") not in {"small_talk", "follow_up"}
            and artifact.get("answer_style") != "direct_metric"
        )

    def _summary_model_for(self, artifact: dict[str, Any]) -> str:
        if artifact.get("answer_style") == "financial_analysis":
            return self.settings.llm_analysis_model
        return self.settings.llm_composer_model

    def _format_cop(self, value: Any) -> str:
        try:
            numeric = round(float(value))
        except (TypeError, ValueError):
            return str(value)
        return "$" + f"{numeric:,}".replace(",", ".")

    def _period_label(self, period: dict[str, Any]) -> str:
        date_from = period.get("date_from")
        date_to = period.get("date_to")
        if not date_from and not date_to:
            return "Periodo consultado"
        if date_from == date_to:
            return f"El {date_from}"
        return f"Del {date_from} al {date_to}"

    async def _build_summary_with_llm(
        self,
        *,
        artifact: dict[str, Any],
        context: InternalRequestContext,
        run_id: UUID,
    ) -> str:
        fallback_summary = self._build_summary(artifact)
        summary_model = self._summary_model_for(artifact)
        if self.settings.llm_provider == "disabled" or not self._should_use_llm(artifact):
            with self.tracer.start_as_current_span("sales.summary_fallback") as span:
                span.set_attribute("openinference.span.kind", "CHAIN")
                span.set_attribute("waro.workflow.name", self.graph_name)
                span.set_attribute("waro.run_id", str(run_id))
                span.set_attribute("waro.answer.style", str(artifact.get("answer_style", "")))
                response_contract = (
                    artifact.get("response_contract")
                    if isinstance(artifact.get("response_contract"), dict)
                    else {}
                )
                self._set_response_contract_span_attributes(span, response_contract)
                span.add_event(
                    "fallback.used",
                    {
                        "waro.reason": "llm_disabled_or_not_required",
                        "waro.summary_length": len(fallback_summary),
                    },
                )
                span.set_status(Status(StatusCode.OK))
            return fallback_summary

        with self.tracer.start_as_current_span("llm.sales.summary") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", self.llm_adapter.provider)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            span.set_attribute("waro.answer.style", artifact.get("answer_style", ""))
            response_contract = (
                artifact.get("response_contract")
                if isinstance(artifact.get("response_contract"), dict)
                else {}
            )
            self._set_response_contract_span_attributes(span, response_contract)
            if self.settings.llm_provider == "kimi":
                span.set_attribute("llm.model", summary_model)
                span.set_attribute("llm.model_name", summary_model)

            try:
                response = await self.llm_adapter.complete(
                    messages=sales_summary_messages(artifact),
                    temperature=0.2,
                    model=summary_model,
                )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.add_event(
                    "fallback.used",
                    {
                        "waro.reason": "llm_error",
                        "waro.error_type": type(exc).__name__,
                        "waro.summary_length": len(fallback_summary),
                    },
                )
                await self._record_step(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    step_type="llm",
                    name="sales_summary",
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
            if summary == fallback_summary:
                span.add_event(
                    "fallback.used",
                    {
                        "waro.reason": "empty_or_matching_llm_response",
                        "waro.summary_length": len(summary),
                    },
                )
            else:
                span.add_event(
                    "llm.summary_completed",
                    {"waro.summary_length": len(summary)},
                )
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
                    "answer_style": artifact.get("answer_style"),
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
        fallback_summary = self._build_summary(artifact)
        summary_model = self._summary_model_for(artifact)
        if self.settings.llm_provider == "disabled" or not self._should_use_llm(artifact):
            with self.tracer.start_as_current_span("sales.summary_fallback") as span:
                span.set_attribute("openinference.span.kind", "CHAIN")
                span.set_attribute("waro.workflow.name", self.graph_name)
                span.set_attribute("waro.run_id", str(run_id))
                span.set_attribute("waro.answer.style", str(artifact.get("answer_style", "")))
                response_contract = (
                    artifact.get("response_contract")
                    if isinstance(artifact.get("response_contract"), dict)
                    else {}
                )
                self._set_response_contract_span_attributes(span, response_contract)
                span.add_event(
                    "fallback.used",
                    {
                        "waro.reason": "llm_disabled_or_not_required",
                        "waro.summary_length": len(fallback_summary),
                    },
                )
                span.set_status(Status(StatusCode.OK))
            summary_holder["summary"] = fallback_summary
            return

        stream_complete = getattr(self.llm_adapter, "stream_complete", None)
        if stream_complete is None:
            summary_holder["summary"] = await self._build_summary_with_llm(
                artifact=artifact,
                context=context,
                run_id=run_id,
            )
            return

        with self.tracer.start_as_current_span("llm.sales.summary") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", self.llm_adapter.provider)
            span.set_attribute("waro.run_id", str(run_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            span.set_attribute("waro.answer.style", artifact.get("answer_style", ""))
            response_contract = (
                artifact.get("response_contract")
                if isinstance(artifact.get("response_contract"), dict)
                else {}
            )
            self._set_response_contract_span_attributes(span, response_contract)
            if self.settings.llm_provider == "kimi":
                span.set_attribute("llm.model", summary_model)
                span.set_attribute("llm.model_name", summary_model)

            try:
                response = None
                async for chunk in stream_complete(
                    messages=sales_summary_messages(artifact),
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
                span.add_event(
                    "fallback.used",
                    {
                        "waro.reason": "llm_stream_error",
                        "waro.error_type": type(exc).__name__,
                        "waro.summary_length": len(fallback_summary),
                    },
                )
                await self._record_step(
                    run_id=run_id,
                    tenant_id=context.tenant_id,
                    step_type="llm",
                    name="sales_summary",
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
            if summary == fallback_summary:
                span.add_event(
                    "fallback.used",
                    {
                        "waro.reason": "empty_or_matching_llm_response",
                        "waro.summary_length": len(summary),
                    },
                )
            else:
                span.add_event(
                    "llm.summary_completed",
                    {"waro.summary_length": len(summary)},
                )
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
                    "answer_style": artifact.get("answer_style"),
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

    async def _load_latest_context_summary(
        self,
        *,
        request: SalesQuestionRequest,
        context: InternalRequestContext,
    ) -> str | None:
        if request.conversation_id is None:
            self._printf_step(
                "load_conversation_context",
                {"context_found": False, "reason": "no_conversation_id"},
            )
            return None
        with self.tracer.start_as_current_span("sales.load_conversation_context") as span:
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            span.set_attribute("waro.conversation_id", str(request.conversation_id))
            span.set_attribute("waro.tenant_id", context.tenant_id)
            async with self.connection_factory() as connection:
                row = await connection.fetchrow(
                    """
                    SELECT summary
                    FROM ai.context_summaries
                    WHERE conversation_id = $1
                      AND tenant_id = $2
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    request.conversation_id,
                    UUID(context.tenant_id),
                )
            value = row["summary"] if row and "summary" in row else None
            summary = value if isinstance(value, str) else None
            self._printf_step(
                "load_conversation_context",
                {
                    "conversation_id": str(request.conversation_id),
                    "context_found": bool(summary),
                    "summary_length": len(summary or ""),
                },
            )
            span.set_attribute("waro.conversation.context_found", bool(summary))
            span.set_status(Status(StatusCode.OK))
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
                VALUES ($1, $2, 'agent', 'sales_artifact', $3::jsonb, $4, $5)
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
