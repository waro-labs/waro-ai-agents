from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from uuid import UUID

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.agent.advisor_state import merge_advisor_state
from app.agent.analyst import analyze_agent_artifact
from app.agent.capabilities import capability_from_spec, match_tools, search_capabilities
from app.agent.conversation_planner import ConversationPlan, plan_conversation
from app.agent.evidence import build_evidence_artifact
from app.agent.intent import QuestionIntent, parse_question_intent
from app.agent.plan import ToolPlan, build_tool_plan
from app.agent.queryspec import (
    QuerySpecValidationError,
    is_query_tool,
    query_dataset_rule_source,
    query_dataset_rules_for_capability,
    query_trace_attributes_from_args,
    validate_queryspec_payload,
)
from app.agent.strategy import choose_answer_strategy
from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMAdapter
from app.llm.model_router import Complexity
from app.tools import ToolCallRequest, ToolGateway
from app.tools.registry import ToolRegistry, coerce_args_async


class AgentLoop:
    def __init__(
        self,
        *,
        settings: Settings,
        gateway: ToolGateway,
        registry: ToolRegistry,
        llm_adapter: LLMAdapter,
    ):
        self.settings = settings
        self.gateway = gateway
        self.registry = registry
        self.llm_adapter = llm_adapter
        self.tracer = trace.get_tracer(__name__)

    async def run(
        self,
        *,
        question: str,
        context: InternalRequestContext,
        run_id: UUID,
        complexity: Complexity,
        conversation_messages: list[dict[str, str]] | None = None,
        conversation_state: dict[str, Any] | None = None,
        classification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.tracer.start_as_current_span("agent.intent_capability_loop") as span:
            span.set_attribute("waro.agent.engine_version", "intent-capability-v1")
            context_state = conversation_state or {}
            if context_state:
                span.set_attribute("waro.context.used", context_state.get("source") != "none")
                span.set_attribute("waro.context.entity", str(context_state.get("active_entity") or ""))
                active_period = context_state.get("active_period")
                span.set_attribute("waro.context.period", str(active_period or ""))
            snapshot = await self.registry.refresh()
            capabilities = [
                capability_from_spec(
                    spec,
                    arguments_schema=(
                        snapshot.schemas.get(name).arguments_schema
                        if snapshot.schemas.get(name) is not None
                        else None
                    ),
                    response_contract=(
                        snapshot.schemas.get(name).response
                        if snapshot.schemas.get(name) is not None
                        else None
                    ),
                )
                for name, spec in snapshot.tools.items()
            ]
            intent = await parse_question_intent(
                settings=self.settings,
                llm_adapter=self.llm_adapter,
                question=question,
                conversation_messages=conversation_messages,
                conversation_state=context_state,
                capability_hints=[capability.to_dict() for capability in capabilities],
                timezone=context.timezone,
            )
            span.set_attribute("waro.intent.entity", intent.entity)
            span.set_attribute("waro.intent.measures", ",".join(intent.measures))
            span.set_attribute("waro.intent.operations", ",".join(intent.operations))
            conversation_plan = await plan_conversation(
                settings=self.settings,
                llm_adapter=self.llm_adapter,
                question=question,
                intent=intent,
                conversation_state=context_state,
                capability_hints=[capability.to_dict() for capability in capabilities],
                complexity=complexity,
            )
            intent = _intent_from_conversation_plan(intent, conversation_plan, context_state)
            span.set_attribute("waro.conversation.intent_type", conversation_plan.intent_type)
            span.set_attribute("waro.conversation.tool_policy", conversation_plan.tool_policy)
            span.set_attribute("waro.conversation.reuse_previous_artifact", conversation_plan.reuse_previous_artifact)
            if conversation_plan.preserve_dataset:
                span.set_attribute("waro.conversation.preserve_dataset", conversation_plan.preserve_dataset)

            all_matches = match_tools(intent, capabilities, scopes=context.scopes)
            matches = search_capabilities(intent, capabilities, scopes=context.scopes)
            rejected = [
                f"{match.capability.tool_name}:{match.rejected_reason}"
                for match in all_matches
                if not match.accepted and match.rejected_reason
            ][:12]
            span.set_attribute("waro.plan.rejected_tools", ",".join(rejected))
            plan = (
                _previous_artifact_plan(intent)
                if conversation_plan.tool_policy == "reuse_only"
                else build_tool_plan(intent, matches)
            )
            span.set_attribute("waro.plan.valid", plan.valid)
            span.set_attribute("waro.plan.missing_coverage", ",".join(plan.missing_coverage))

            observations: list[dict[str, Any]] = []
            if plan.valid:
                observations = await self._execute_plan(
                    plan_steps=plan.steps,
                    context=context,
                    run_id=run_id,
                    span=span,
                )
            artifact = build_evidence_artifact(
                question=question,
                intent=intent,
                plan=plan,
                observations=observations,
                conversation_messages=conversation_messages,
                conversation_state=context_state,
                classification=classification or {"complexity": complexity},
                conversation_plan=conversation_plan.to_dict(),
            )
            advisor_analysis = await analyze_agent_artifact(
                settings=self.settings,
                llm_adapter=self.llm_adapter,
                artifact=artifact,
                complexity=complexity,
            )
            if advisor_analysis is not None:
                artifact["advisor_analysis"] = advisor_analysis
                update = advisor_analysis.get("advisor_state_update")
                previous = artifact.get("advisor_state") if isinstance(artifact.get("advisor_state"), dict) else {}
                artifact["advisor_state"] = merge_advisor_state(
                    previous=previous,
                    update=update if isinstance(update, dict) else {},
                )
                artifact["advisor_state_update"] = update if isinstance(update, dict) else {}
            strategy = await choose_answer_strategy(
                settings=self.settings,
                llm_adapter=self.llm_adapter,
                question=question,
                intent=intent,
                artifact=artifact,
                conversation_state=context_state,
                complexity=complexity,
            )
            artifact["answer_strategy"] = strategy.to_dict()
            artifact["previous_artifact_used"] = strategy.use_previous_artifact
            span.set_attribute("waro.answer.strategy", strategy.type)
            span.set_attribute("waro.context.uses_previous_artifact", strategy.use_previous_artifact)
            span.set_attribute("waro.answerability.status", str(artifact.get("answerability")))
            span.set_status(Status(StatusCode.OK))
            return artifact

    async def _execute_plan(
        self,
        *,
        plan_steps,
        context: InternalRequestContext,
        run_id: UUID,
        span: Any | None = None,
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for step in plan_steps:
            spec = await self.registry.get_spec(step.tool_name)
            if spec is None or spec.scope not in context.scopes:
                observations.append(
                    {
                        "tool_name": step.tool_name,
                        "status": "failed",
                        "arguments": step.arguments,
                        "fields": list(step.fields),
                        "error": {"message": "unknown_or_forbidden_tool"},
                    }
                )
                continue
            if is_query_tool(step.tool_name):
                query_rules = query_dataset_rules_for_capability(spec)
                query_source = query_dataset_rule_source(spec)
                if span is not None:
                    span.set_attribute("waro.tool.selected", step.tool_name)
                    span.set_attribute("waro.tool.domain", str(spec.domain))
                    span.set_attribute("waro.queries.schema_source", query_source)
                try:
                    validate_queryspec_payload(step.arguments.get("spec"), rules=query_rules)
                except QuerySpecValidationError as exc:
                    if span is not None:
                        for key, value in query_trace_attributes_from_args(
                            step.arguments,
                            valid=False,
                            rejected_reason=exc.reason,
                            rules=query_rules,
                            source=query_source,
                        ).items():
                            span.set_attribute(key, value)
                    observations.append(
                        {
                            "tool_name": step.tool_name,
                            "status": "failed",
                            "arguments": step.arguments,
                            "fields": list(step.fields),
                            "purpose": step.purpose,
                            "expected_evidence": list(step.expected_evidence),
                            "error": {
                                "message": "invalid_queryspec",
                                "kind": "validation",
                                "rejected_reason": exc.reason,
                            },
                        }
                    )
                    continue
                if span is not None:
                    for key, value in query_trace_attributes_from_args(
                        step.arguments,
                        valid=True,
                        rules=query_rules,
                        source=query_source,
                    ).items():
                        span.set_attribute(key, value)
            try:
                args = await coerce_args_async(spec, step.arguments)
            except Exception as exc:
                observations.append(
                    {
                        "tool_name": step.tool_name,
                        "status": "failed",
                        "arguments": step.arguments,
                        "fields": list(step.fields),
                        "error": {"message": str(exc), "kind": "validation"},
                    }
                )
                continue
            if is_query_tool(step.tool_name):
                print(
                    "[agent-api:queryspec] sending "
                    + json.dumps(
                        {
                            "run_id": str(run_id),
                            "tenant_id": context.tenant_id,
                            "tool_name": step.tool_name,
                            "arguments": args.model_dump(
                                by_alias=True,
                                mode="json",
                                exclude_none=True,
                            ),
                            "fields": list(step.fields),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                    flush=True,
                )
            response = await self.gateway.call(
                request=ToolCallRequest(
                    run_id=run_id,
                    tool_name=step.tool_name,
                    arguments=args.model_dump(by_alias=True, mode="json", exclude_none=True),
                    fields=list(step.fields),
                ),
                context=context,
            )
            observations.append(
                {
                    "tool_name": step.tool_name,
                    "status": response.status,
                    "arguments": step.arguments,
                    "fields": list(step.fields),
                    "purpose": step.purpose,
                    "expected_evidence": list(step.expected_evidence),
                    "result_summary": response.result_summary,
                    "result": response.result if isinstance(response.result, dict) else {},
                    "error": response.error,
                }
            )
        return observations


def _previous_artifact_plan(intent: QuestionIntent) -> ToolPlan:
    return ToolPlan(
        steps=(),
        coverage=tuple(sorted({"previous_artifact", *intent.measures, *intent.dimensions})),
        missing_coverage=(),
        valid=True,
        blocked_reason=None,
    )


def _intent_from_conversation_plan(
    intent: QuestionIntent,
    conversation_plan: ConversationPlan,
    conversation_state: dict[str, Any],
) -> QuestionIntent:
    if conversation_plan.subject == "generic_customers" or conversation_plan.preserve_dataset == "customers":
        return replace(
            intent,
            entity="customer",
            grain="customer_period",
            measures=tuple(_dedupe([*intent.measures, "order_count", "total_spent", "avg_ticket"])),
            dimensions=tuple(_dedupe([*intent.dimensions, "customer"])),
            operations=tuple(_dedupe([*intent.operations, "rank", "diagnose"])),
            requires_cross_tool=True,
        )
    if conversation_plan.intent_type not in {"diagnosis", "data_quality_check"}:
        return intent
    active_entity = str(conversation_state.get("active_entity") or "")
    active_grain = str(conversation_state.get("active_grain") or "")
    active_measures = _string_items(conversation_state.get("active_measures"))
    active_dimensions = _string_items(conversation_state.get("active_dimensions"))
    entity = "product" if conversation_plan.preserve_dataset == "product_profitability" else intent.entity
    if entity == "sale" and active_entity:
        entity = active_entity
    measures = list(intent.measures)
    if not measures and active_measures:
        measures = active_measures
    if conversation_plan.preserve_dataset == "product_profitability":
        measures = _dedupe([*measures, "margin", "cost", "quantity_sold", "revenue"])
    operations = _dedupe([*intent.operations, "diagnose"])
    dimensions = list(intent.dimensions)
    if entity == active_entity:
        dimensions = _dedupe([*dimensions, *active_dimensions])
    return replace(
        intent,
        entity=entity,
        grain=active_grain or intent.grain,
        measures=tuple(_dedupe(measures)),
        dimensions=tuple(dimensions),
        operations=tuple(operations),
        requires_cross_tool=intent.requires_cross_tool or entity == "business",
    )


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result
