from __future__ import annotations

from typing import Any
from uuid import UUID

from app.agent.classifier import classify_complexity
from app.agent.composer import compose_agent_summary, deterministic_summary
from app.agent.conversation import load_conversation_messages, load_conversation_state
from app.agent.loop import AgentLoop
from app.agent.tool_reasoning import run_tool_reasoning_agent
from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMAdapter
from app.llm.model_router import Complexity
from app.tools import ToolGateway
from app.tools.registry import ToolRegistry


class KaliAgentRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        gateway: ToolGateway,
        registry: ToolRegistry,
        llm_adapter: LLMAdapter,
        connection_factory,
    ):
        self.settings = settings
        self.gateway = gateway
        self.registry = registry
        self.llm_adapter = llm_adapter
        self.connection_factory = connection_factory
        self.loop = AgentLoop(
            settings=settings,
            gateway=gateway,
            registry=registry,
            llm_adapter=llm_adapter,
        )

    async def execute(
        self,
        *,
        question: str,
        context: InternalRequestContext,
        run_id: UUID,
        conversation_id: UUID | None,
    ) -> dict[str, Any]:
        conversation_messages = await load_conversation_messages(
            settings=self.settings,
            connection_factory=self.connection_factory,
            conversation_id=conversation_id,
        )
        conversation_state = await load_conversation_state(
            settings=self.settings,
            connection_factory=self.connection_factory,
            conversation_id=conversation_id,
        )
        classification = await classify_complexity(
            settings=self.settings,
            llm_adapter=self.llm_adapter,
            question=question,
            conversation_messages=conversation_messages,
        )
        complexity: Complexity = classification["complexity"]
        if self.settings.llm_provider != "disabled":
            try:
                artifact = await run_tool_reasoning_agent(
                    settings=self.settings,
                    llm_adapter=self.llm_adapter,
                    gateway=self.gateway,
                    registry=self.registry,
                    question=question,
                    context=context,
                    run_id=run_id,
                    conversation_messages=conversation_messages,
                    complexity=complexity,
                )
                artifact["classification"] = classification
                return artifact
            except Exception as exc:
                print(f"[agent-api:tool-reasoning] failed {type(exc).__name__}: {exc}", flush=True)
                return {
                    "intent": "tool_reasoning",
                    "agent_mode": True,
                    "agent_engine_version": "tool-reasoning-v1",
                    "question": question,
                    "safe_to_answer": False,
                    "answerability": "blocked",
                    "blocked_reason": "tool_reasoning_failed",
                    "summary": (
                        "Tuve un error en el agente conversacional nuevo antes de reunir evidencia suficiente. "
                        "No voy a responder con el flujo anterior para evitar una respuesta repetida o rígida; "
                        "reintenta la pregunta o revisemos el log de tool-reasoning."
                    ),
                    "observations": [],
                    "tool_calls": [],
                    "classification": classification,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
        artifact = await self.loop.run(
            question=question,
            context=context,
            run_id=run_id,
            complexity=complexity,
            conversation_messages=conversation_messages,
            conversation_state=conversation_state.to_dict(),
            classification=classification,
        )
        artifact["classification"] = classification
        fallback = deterministic_summary(artifact)
        summary = await compose_agent_summary(
            settings=self.settings,
            llm_adapter=self.llm_adapter,
            artifact=artifact,
            complexity=complexity,
            fallback=fallback,
        )
        artifact["summary"] = summary
        return artifact
