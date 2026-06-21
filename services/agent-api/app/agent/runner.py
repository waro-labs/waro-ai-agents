from __future__ import annotations

from app.agent.artifact import build_agent_artifact
from app.agent.classifier import classify_complexity
from app.agent.composer import compose_agent_summary, deterministic_summary
from app.agent.conversation import load_conversation_messages
from app.agent.loop import AgentLoop
from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMAdapter
from app.llm.model_router import Complexity
from app.tools import ToolGateway
from app.tools.registry import ToolRegistry
from uuid import UUID


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
        classification = await classify_complexity(
            settings=self.settings,
            llm_adapter=self.llm_adapter,
            question=question,
            conversation_messages=conversation_messages,
        )
        complexity: Complexity = classification["complexity"]
        if complexity == "simple":
            artifact = await self.loop.run_fast_path(
                question=question,
                context=context,
                run_id=run_id,
                complexity=complexity,
                conversation_messages=conversation_messages,
            )
        else:
            artifact = await self.loop.run(
                question=question,
                context=context,
                run_id=run_id,
                complexity=complexity,
                conversation_messages=conversation_messages,
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

    async def execute_shadow(
        self,
        *,
        question: str,
        context: InternalRequestContext,
        run_id: UUID,
        conversation_id: UUID | None,
        legacy_artifact: dict[str, Any],
    ) -> dict[str, Any]:
        agent_artifact = await self.execute(
            question=question,
            context=context,
            run_id=run_id,
            conversation_id=conversation_id,
        )
        agent_artifact["shadow"] = {
            "legacy_safe_to_answer": legacy_artifact.get("response_contract", {}).get(
                "safe_to_answer"
            ),
            "agent_safe_to_answer": agent_artifact.get("safe_to_answer"),
            "legacy_tool_count": len(legacy_artifact.get("tool_calls") or []),
            "agent_observation_count": len(agent_artifact.get("observations") or []),
        }
        merged = dict(legacy_artifact)
        merged["agent_shadow_artifact"] = agent_artifact
        return merged
