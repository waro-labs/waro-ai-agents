from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.agent.artifact import build_agent_artifact
from app.agent.classifier import classify_complexity
from app.agent.prompts import agent_step_messages, verify_answer_messages
from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMAdapter
from app.llm.model_router import Complexity, max_agent_steps, model_for
from app.tools import ToolCallRequest, ToolGateway
from app.tools.catalog import discover_tools
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
    ) -> dict[str, Any]:
        today = datetime.now(ZoneInfo("America/Bogota")).date()
        catalog = await self.registry.catalog_metadata()
        scoped_catalog = [
            tool
            for tool in catalog
            if str(tool.get("scope")) in context.scopes
        ]
        observations: list[dict[str, Any]] = []
        max_steps = max_agent_steps(self.settings, complexity)
        repeat_tracker: dict[str, int] = {}

        for step in range(1, max_steps + 1):
            decision = await self._next_decision(
                question=question,
                today=today.isoformat(),
                catalog=scoped_catalog,
                observations=observations,
                conversation_messages=conversation_messages,
                complexity=complexity,
                step=step,
                max_steps=max_steps,
            )
            if decision.get("action") == "finish":
                break
            tool_name = str(decision.get("tool_name") or "")
            arguments = decision.get("arguments") if isinstance(decision.get("arguments"), dict) else {}
            if not tool_name:
                break
            repeat_key = json.dumps({"tool": tool_name, "arguments": arguments}, sort_keys=True)
            repeat_tracker[repeat_key] = repeat_tracker.get(repeat_key, 0) + 1
            if repeat_tracker[repeat_key] > 2:
                break
            spec = await self.registry.get_spec(tool_name)
            if spec is None or spec.scope not in context.scopes:
                observations.append(
                    {
                        "tool_name": tool_name,
                        "status": "failed",
                        "error": "unknown_or_forbidden_tool",
                        "arguments": arguments,
                    }
                )
                continue
            try:
                args = await coerce_args_async(spec, arguments)
            except Exception as exc:
                observations.append(
                    {
                        "tool_name": tool_name,
                        "status": "failed",
                        "error": str(exc),
                        "arguments": arguments,
                    }
                )
                continue
            response = await self.gateway.call(
                request=ToolCallRequest(
                    run_id=run_id,
                    tool_name=tool_name,
                    arguments=args.model_dump(by_alias=True, mode="json", exclude_none=True),
                    fields=list(spec.default_fields),
                ),
                context=context,
            )
            observations.append(
                {
                    "tool_name": tool_name,
                    "status": response.status,
                    "arguments": arguments,
                    "result_summary": response.result_summary,
                    "result": response.result if isinstance(response.result, dict) else {},
                    "error": response.error,
                }
            )

        verification = await self._verify(
            question=question,
            observations=observations,
            complexity=complexity,
        )
        return build_agent_artifact(
            question=question,
            observations=observations,
            complexity=complexity,
            verification=verification,
            conversation_messages=conversation_messages,
        )

    async def run_fast_path(
        self,
        *,
        question: str,
        context: InternalRequestContext,
        run_id: UUID,
        complexity: Complexity,
        conversation_messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        discovery = discover_tools(
            question,
            scopes=context.scopes,
            limit=5,
        )
        available = discovery.get("available") or []
        if not available:
            return build_agent_artifact(
                question=question,
                observations=[],
                complexity=complexity,
                verification={"safe_to_answer": False, "missing": "No hay tools disponibles."},
                conversation_messages=conversation_messages,
            )
        tool_name = str(available[0].get("name"))
        spec = await self.registry.get_spec(tool_name)
        if spec is None:
            return build_agent_artifact(
                question=question,
                observations=[],
                complexity=complexity,
                verification={"safe_to_answer": False, "missing": f"Tool {tool_name} no registrada."},
                conversation_messages=conversation_messages,
            )
        response = await self.gateway.call(
            request=ToolCallRequest(
                run_id=run_id,
                tool_name=tool_name,
                arguments={},
                fields=list(spec.default_fields),
            ),
            context=context,
        )
        observations = [
            {
                "tool_name": tool_name,
                "status": response.status,
                "arguments": {},
                "result_summary": response.result_summary,
                "result": response.result if isinstance(response.result, dict) else {},
                "error": response.error,
            }
        ]
        verification = await self._verify(
            question=question,
            observations=observations,
            complexity=complexity,
        )
        return build_agent_artifact(
            question=question,
            observations=observations,
            complexity=complexity,
            verification=verification,
            conversation_messages=conversation_messages,
        )

    async def _next_decision(
        self,
        *,
        question: str,
        today: str,
        catalog: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        conversation_messages: list[dict[str, str]] | None,
        complexity: Complexity,
        step: int,
        max_steps: int,
    ) -> dict[str, Any]:
        if self.settings.llm_provider == "disabled":
            if observations:
                return {"action": "finish", "tool_name": None, "arguments": {}, "reason": "llm_disabled"}
            if catalog:
                return {
                    "action": "call_tool",
                    "tool_name": catalog[0]["name"],
                    "arguments": {},
                    "reason": "heuristic_first_tool",
                }
            return {"action": "finish", "tool_name": None, "arguments": {}, "reason": "no_tools"}

        messages = agent_step_messages(
            question=question,
            today=today,
            timezone="America/Bogota",
            available_tools=catalog,
            observations=observations,
            conversation_messages=conversation_messages,
            step=step,
            max_steps=max_steps,
        )
        with self.tracer.start_as_current_span("llm.agent.step") as span:
            span.set_attribute("agent.step", step)
            span.set_attribute("agent.complexity", complexity)
            response = await self.llm_adapter.complete(
                messages=messages,
                temperature=0,
                model=model_for(self.settings, step="agent_step", complexity=complexity),
            )
            span.set_status(Status(StatusCode.OK))
        try:
            parsed = json.loads(response.content.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"action": "finish", "tool_name": None, "arguments": {}, "reason": "parse_error"}

    async def _verify(
        self,
        *,
        question: str,
        observations: list[dict[str, Any]],
        complexity: Complexity,
    ) -> dict[str, Any]:
        artifact_preview = build_agent_artifact(
            question=question,
            observations=observations,
            complexity=complexity,
        )
        if self.settings.llm_provider == "disabled":
            return {
                "safe_to_answer": bool(observations) and all(
                    obs.get("status") == "succeeded" for obs in observations
                ),
                "missing": "" if observations else "sin datos",
                "needs_more_tools": False,
            }
        messages = verify_answer_messages(question=question, artifact=artifact_preview)
        try:
            response = await self.llm_adapter.complete(
                messages=messages,
                temperature=0,
                model=model_for(self.settings, step="verify", complexity=complexity),
            )
            parsed = json.loads(response.content.strip())
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "safe_to_answer": bool(observations),
            "missing": "",
            "needs_more_tools": False,
        }
