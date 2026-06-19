from __future__ import annotations

from collections.abc import AsyncIterator
import re
import unicodedata
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import Settings
from app.database import get_db_connection
from app.dependencies.internal_auth import InternalRequestContext
from app.llm import LLMAdapter
from app.streaming import StreamEvent, stream_event, terminal_error_event
from app.tools import ToolGateway
from app.tools.sanitize import truncate_text
from app.workflows.food_cost import FoodCostWorkflow
from app.workflows.models import (
    AgentQuestionRequest,
    AgentRoute,
    AgentWorkflowResponse,
    FoodCostQuestionRequest,
    SalesQuestionRequest,
)
from app.workflows.sales import SalesWorkflow


class AgentGraphState(TypedDict, total=False):
    request: AgentQuestionRequest
    context: InternalRequestContext
    route: AgentRoute
    response: AgentWorkflowResponse


class AgentWorkflow:
    graph_name = "agent_router_langgraph_v1"

    def __init__(
        self,
        *,
        settings: Settings,
        gateway: ToolGateway | None = None,
        llm_adapter: LLMAdapter | None = None,
        connection_factory: Any = get_db_connection,
    ):
        self.settings = settings
        self.tracer = trace.get_tracer(__name__)
        self.sales_workflow = SalesWorkflow(
            settings=settings,
            gateway=gateway,
            llm_adapter=llm_adapter,
            connection_factory=connection_factory,
        )
        self.food_cost_workflow = FoodCostWorkflow(
            settings=settings,
            gateway=gateway,
            llm_adapter=llm_adapter,
            connection_factory=connection_factory,
        )
        self.graph = self._build_graph()

    async def run(
        self,
        *,
        request: AgentQuestionRequest,
        context: InternalRequestContext,
    ) -> AgentWorkflowResponse:
        with self.tracer.start_as_current_span("agent.workflow") as span:
            span.set_attribute("waro.workflow.name", self.graph_name)
            span.set_attribute("waro.tenant_id", context.tenant_id)
            span.set_attribute("waro.request_id", context.request_id)
            try:
                state = await self.graph.ainvoke({"request": request, "context": context})
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            response = state["response"]
            span.set_attribute("waro.agent.domain", response.workflow)
            span.set_attribute("waro.agent.route.confidence", response.route.confidence)
            span.set_attribute("waro.run_id", str(response.run_id))
            span.set_status(Status(StatusCode.OK))
            return response

    async def stream(
        self,
        *,
        request: AgentQuestionRequest,
        context: InternalRequestContext,
    ) -> AsyncIterator[StreamEvent]:
        route = self._route(request)
        yield stream_event(
            "step_started",
            data={
                "step_type": "router",
                "name": "agent_domain_router",
                "workflow": route.workflow,
                "confidence": route.confidence,
            },
        )
        try:
            async for event in self._stream_routed_workflow(
                request=request,
                context=context,
                route=route,
            ):
                yield event
        except Exception as exc:
            yield terminal_error_event(
                error_type=type(exc).__name__,
                error_message=truncate_text(str(exc), 240),
            )

    def _build_graph(self):
        graph = StateGraph(AgentGraphState)
        graph.add_node("route_agent", self._route_agent_node)
        graph.add_node("dispatch_workflow", self._dispatch_workflow_node)
        graph.add_edge(START, "route_agent")
        graph.add_edge("route_agent", "dispatch_workflow")
        graph.add_edge("dispatch_workflow", END)
        return graph.compile()

    async def _route_agent_node(self, state: AgentGraphState) -> AgentGraphState:
        with self.tracer.start_as_current_span("agent.route") as span:
            route = self._route(state["request"])
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.agent.domain", route.workflow)
            span.set_attribute("waro.agent.route.confidence", route.confidence)
            span.set_attribute("waro.agent.route.reason", route.reason)
            span.set_attribute("waro.request.question_length", len(state["request"].question))
            span.set_status(Status(StatusCode.OK))
            return {"route": route}

    async def _dispatch_workflow_node(self, state: AgentGraphState) -> AgentGraphState:
        route = state["route"]
        with self.tracer.start_as_current_span(f"agent.dispatch.{route.workflow}") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.agent.domain", route.workflow)
            span.set_attribute("waro.agent.route.confidence", route.confidence)
            span.set_attribute("waro.tenant_id", state["context"].tenant_id)
            try:
                child_response = await self._run_routed_workflow(
                    request=state["request"],
                    context=state["context"],
                    route=route,
                )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            span.set_attribute("waro.run_id", str(child_response.run_id))
            span.set_attribute("waro.conversation_id", str(child_response.conversation_id))
            span.set_status(Status(StatusCode.OK))
            return {"response": child_response}

    async def _run_routed_workflow(
        self,
        *,
        request: AgentQuestionRequest,
        context: InternalRequestContext,
        route: AgentRoute,
    ) -> AgentWorkflowResponse:
        if route.workflow == "food_cost":
            response = await self.food_cost_workflow.run(
                request=FoodCostQuestionRequest(
                    question=request.question,
                    conversation_id=request.conversation_id,
                    date_from=request.date_from,
                    date_to=request.date_to,
                    compare_to=request.compare_to,
                ),
                context=context,
            )
        else:
            response = await self.sales_workflow.run(
                request=SalesQuestionRequest(
                    question=request.question,
                    conversation_id=request.conversation_id,
                    date_from=request.date_from,
                    date_to=request.date_to,
                    group_by=request.group_by,
                ),
                context=context,
            )
        return AgentWorkflowResponse(
            conversation_id=response.conversation_id,
            run_id=response.run_id,
            input_message_id=response.input_message_id,
            output_message_id=response.output_message_id,
            status=response.status,
            workflow=route.workflow,
            route=route,
            artifact=response.artifact,
            summary=response.summary,
            evals=response.evals,
        )

    async def _stream_routed_workflow(
        self,
        *,
        request: AgentQuestionRequest,
        context: InternalRequestContext,
        route: AgentRoute,
    ) -> AsyncIterator[StreamEvent]:
        with self.tracer.start_as_current_span(f"agent.stream.{route.workflow}") as span:
            span.set_attribute("waro.agent.domain", route.workflow)
            span.set_attribute("waro.agent.route.confidence", route.confidence)
            if route.workflow == "food_cost":
                stream = self.food_cost_workflow.stream(
                    request=FoodCostQuestionRequest(
                        question=request.question,
                        conversation_id=request.conversation_id,
                        date_from=request.date_from,
                        date_to=request.date_to,
                        compare_to=request.compare_to,
                    ),
                    context=context,
                )
            else:
                stream = self.sales_workflow.stream(
                    request=SalesQuestionRequest(
                        question=request.question,
                        conversation_id=request.conversation_id,
                        date_from=request.date_from,
                        date_to=request.date_to,
                        group_by=request.group_by,
                    ),
                    context=context,
                )
            async for event in stream:
                yield event
            span.set_status(Status(StatusCode.OK))

    def _route(self, request: AgentQuestionRequest) -> AgentRoute:
        if request.workflow:
            return AgentRoute(
                workflow=request.workflow,
                confidence=1.0,
                reason="explicit_workflow",
            )

        normalized = self._normalize(request.question)
        if self._matches(normalized, FOOD_COST_HINTS):
            return AgentRoute(
                workflow="food_cost",
                confidence=0.9,
                reason="food_cost_keyword",
            )
        if self._matches(normalized, SALES_HINTS):
            return AgentRoute(
                workflow="sales",
                confidence=0.85,
                reason="sales_keyword",
            )
        return AgentRoute(
            workflow="sales",
            confidence=0.5,
            reason="default_sales",
        )

    def _normalize(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value.lower())
        without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
        return re.sub(r"\s+", " ", without_accents).strip()

    def _matches(self, normalized: str, hints: tuple[str, ...]) -> bool:
        return any(re.search(pattern, normalized) for pattern in hints)


WorkflowName = Literal["sales", "food_cost"]

FOOD_COST_HINTS = (
    r"\bfood\s*cost\b",
    r"\bcosto\s+de\s+comida\b",
    r"\bcostos?\b",
    r"\bmargen(?:es)?\b",
    r"\brentabilidad\b",
    r"\brecetas?\b",
    r"\bproducto[s]?\s+(?:menos|mas)\s+rentable",
)

SALES_HINTS = (
    r"\bventas?\b",
    r"\bvend[ií]\b",
    r"\bvendimos\b",
    r"\bingresos?\b",
    r"\bfactur",
    r"\border(?:es|nes)?\b",
    r"\bticket\b",
)
