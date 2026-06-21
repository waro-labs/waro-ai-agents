from __future__ import annotations

from typing import Any
from uuid import UUID

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.agent.capabilities import capability_from_spec, match_tools, search_capabilities
from app.agent.evidence import build_evidence_artifact
from app.agent.intent import parse_question_intent
from app.agent.plan import build_tool_plan
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
            )
            span.set_attribute("waro.intent.entity", intent.entity)
            span.set_attribute("waro.intent.measures", ",".join(intent.measures))
            span.set_attribute("waro.intent.operations", ",".join(intent.operations))

            all_matches = match_tools(intent, capabilities, scopes=context.scopes)
            matches = search_capabilities(intent, capabilities, scopes=context.scopes)
            rejected = [
                f"{match.capability.tool_name}:{match.rejected_reason}"
                for match in all_matches
                if not match.accepted and match.rejected_reason
            ][:12]
            span.set_attribute("waro.plan.rejected_tools", ",".join(rejected))
            plan = build_tool_plan(intent, matches)
            span.set_attribute("waro.plan.valid", plan.valid)
            span.set_attribute("waro.plan.missing_coverage", ",".join(plan.missing_coverage))

            observations: list[dict[str, Any]] = []
            if plan.valid:
                observations = await self._execute_plan(
                    plan_steps=plan.steps,
                    context=context,
                    run_id=run_id,
                )
            artifact = build_evidence_artifact(
                question=question,
                intent=intent,
                plan=plan,
                observations=observations,
                conversation_messages=conversation_messages,
                conversation_state=context_state,
                classification=classification or {"complexity": complexity},
            )
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
