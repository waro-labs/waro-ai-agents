from __future__ import annotations

from collections.abc import AsyncIterator
import json
import re
import unicodedata
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.config import Settings
from app.database import get_db_connection
from app.dependencies.internal_auth import InternalRequestContext
from app.llm import LLMAdapter, get_llm_adapter
from app.llm.prompts import agent_router_messages
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
        self.llm_adapter = llm_adapter or get_llm_adapter(settings)
        self.sales_workflow = SalesWorkflow(
            settings=settings,
            gateway=gateway,
            llm_adapter=self.llm_adapter,
            connection_factory=connection_factory,
        )
        self.food_cost_workflow = FoodCostWorkflow(
            settings=settings,
            gateway=gateway,
            llm_adapter=self.llm_adapter,
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
        route = await self._route_hybrid(request)
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
            route = await self._route_hybrid(state["request"])
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("waro.agent.domain", route.workflow)
            span.set_attribute("waro.agent.route.confidence", route.confidence)
            span.set_attribute("waro.agent.route.reason", route.reason)
            span.set_attribute("waro.request.question_length", len(state["request"].question))
            span.set_status(Status(StatusCode.OK))
            return {"route": route}

    async def _route_hybrid(self, request: AgentQuestionRequest) -> AgentRoute:
        deterministic_route = self._route(request)
        if request.workflow or self.settings.llm_provider == "disabled":
            return deterministic_route

        router_model = self.settings.llm_router_model
        with self.tracer.start_as_current_span("llm.agent.router") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("llm.provider", self.llm_adapter.provider)
            span.set_attribute("llm.model", router_model)
            span.set_attribute("llm.model_name", router_model)
            span.set_attribute(
                "waro.agent.deterministic_domain",
                deterministic_route.workflow,
            )
            span.set_attribute(
                "waro.agent.deterministic_reason",
                deterministic_route.reason,
            )
            try:
                response = await self.llm_adapter.complete(
                    messages=agent_router_messages(question=request.question),
                    temperature=0,
                    model=router_model,
                )
                self._set_llm_response_span_attributes(span, response)
                route = self._parse_llm_route(response.content)
            except Exception as exc:
                span.record_exception(exc)
                span.set_attribute("waro.agent.route.source", "deterministic_after_llm_error")
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                return AgentRoute(
                    workflow=deterministic_route.workflow,
                    confidence=deterministic_route.confidence,
                    reason=f"deterministic_after_llm_error:{deterministic_route.reason}",
                )

            span.set_attribute("waro.agent.llm_domain", route.workflow)
            span.set_attribute("waro.agent.llm_confidence", route.confidence)
            span.set_attribute("waro.agent.llm_reason", route.reason)
            span.add_event(
                "agent_router.llm_completed",
                {
                    "workflow": route.workflow,
                    "confidence": route.confidence,
                    "reason": route.reason,
                },
            )

            if route.confidence < 0.7:
                span.set_attribute("waro.agent.route.source", "deterministic_low_llm_confidence")
                span.set_status(Status(StatusCode.OK))
                return AgentRoute(
                    workflow=deterministic_route.workflow,
                    confidence=deterministic_route.confidence,
                    reason=f"deterministic_low_llm_confidence:{deterministic_route.reason}",
                )

            span.set_attribute("waro.agent.route.source", "llm_prerouter")
            span.set_status(Status(StatusCode.OK))
            return route

    def _parse_llm_route(self, content: str) -> AgentRoute:
        parsed = json.loads(content)
        workflow = parsed.get("workflow")
        if workflow not in {"sales", "food_cost"}:
            raise ValueError("LLM router returned unsupported workflow.")
        confidence = float(parsed.get("confidence", 0))
        return AgentRoute(
            workflow=workflow,
            confidence=max(0, min(confidence, 1)),
            reason=f"llm_prerouter:{str(parsed.get('reason', '')).strip() or 'classified'}",
        )

    def _set_llm_response_span_attributes(self, span: Any, response: Any) -> None:
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
        if self._is_sales_margin_question(normalized):
            return AgentRoute(
                workflow="sales",
                confidence=0.9,
                reason="sales_margin_keyword",
            )
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

    def _is_sales_margin_question(self, normalized: str) -> bool:
        has_margin_signal = bool(
            re.search(
                r"\b(margen(?:es)?|rentabilidad|rentables?|bajo margen)\b",
                normalized,
            )
        )
        has_sales_product_signal = bool(
            re.search(
                r"\b(productos?|platos?|items?)\b",
                normalized,
            )
            and re.search(
                r"\b(venden|vendidos?|vendid[oa]s?|ventas?|ingresos?|cantidad|unidades)\b",
                normalized,
            )
        )
        has_recipe_signal = bool(
            re.search(
                r"\b(food\s*cost|costo\s+de\s+comida|recetas?|insumos?|ingredientes?|preparacion)\b",
                normalized,
            )
        )
        return has_margin_signal and has_sales_product_signal and not has_recipe_signal


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
    r"\bvenden\b",
    r"\bvendid[oa]s?\b",
    r"\bvendimos\b",
    r"\bingresos?\b",
    r"\bfactur",
    r"\border(?:es|nes)?\b",
    r"\bticket\b",
)
