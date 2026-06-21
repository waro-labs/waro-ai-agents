import json
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from app.agent.classifier import classify_complexity, heuristic_complexity
from app.agent.loop import AgentLoop
from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMError, LLMMessage, LLMResponse
from app.tools.models import ToolCallResponse
from app.tools.registry import ToolRegistry, set_tool_registry


class FakeLLM:
    provider = "kimi"
    decisions = [
        {"action": "call_tool", "tool_name": "waro.sales.metrics", "arguments": {}, "reason": "metrics"},
        {"action": "finish", "tool_name": None, "arguments": {}, "reason": "done"},
    ]
    verify = {"safe_to_answer": True, "missing": "", "needs_more_tools": False}

    def __init__(self):
        self.calls = 0

    async def complete(self, *, messages, temperature=0.2, model=None):
        self.calls += 1
        system = messages[0].content if messages else ""
        if "Verifica si el artifact" in system:
            return LLMResponse(content=json.dumps(self.verify), model=model or "test", provider="kimi")
        decision = self.decisions[min(self.calls - 1, len(self.decisions) - 1)]
        return LLMResponse(content=json.dumps(decision), model=model or "test", provider="kimi")


class FakeGateway:
    async def call(self, *, request, context):
        return ToolCallResponse(
            tool_call_id=uuid4(),
            tool_name=request.tool_name,
            status="succeeded",
            result={
                "success": True,
                "data": {"totalSales": 1000, "totalOrders": 10, "avgTicket": 100},
            },
            result_summary="metrics ok",
        )


class FailingLLM:
    provider = "kimi"

    async def complete(self, *, messages, temperature=0.2, model=None):
        raise LLMError("Kimi completion request failed.")


@pytest.mark.asyncio
async def test_heuristic_complexity_simple():
    assert heuristic_complexity("dame las ventas de ayer") == "simple"


@pytest.mark.asyncio
async def test_classify_complexity_without_llm():
    result = await classify_complexity(
        settings=Settings(LLM_PROVIDER="disabled"),
        llm_adapter=FakeLLM(),
        question="dame las ventas de ayer",
    )
    assert result["complexity"] == "simple"
    assert result["source"] == "heuristic"


@pytest.mark.asyncio
async def test_classify_complexity_falls_back_when_llm_fails():
    result = await classify_complexity(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test"),
        llm_adapter=FailingLLM(),
        question="dame las ventas de ayer",
    )
    assert result["complexity"] == "simple"
    assert result["source"] == "heuristic"
    assert result["reason"].startswith("heuristic_llm_error")


@pytest.mark.asyncio
async def test_agent_loop_fast_path_builds_artifact():
    settings = Settings(TOOL_CATALOG_SOURCE="static", LLM_PROVIDER="disabled")
    registry = ToolRegistry(settings)
    set_tool_registry(registry)
    await registry.refresh(force=True)
    loop = AgentLoop(
        settings=settings,
        gateway=FakeGateway(),
        registry=registry,
        llm_adapter=FakeLLM(),
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-agent",
        member_id=None,
        scopes=("orders:read", "financial:read", "menu:read", "customers:read", "analytics:read"),
    )
    artifact = await loop.run_fast_path(
        question="dame las ventas de ayer",
        context=context,
        run_id=uuid4(),
        complexity="simple",
    )
    assert artifact["agent_mode"] is True
    assert artifact["observations"]


@pytest.mark.asyncio
async def test_agent_loop_uses_catalog_fallback_when_llm_step_fails():
    settings = Settings(TOOL_CATALOG_SOURCE="static", LLM_PROVIDER="kimi", KIMI_API_KEY="test")
    registry = ToolRegistry(settings)
    set_tool_registry(registry)
    await registry.refresh(force=True)
    loop = AgentLoop(
        settings=settings,
        gateway=FakeGateway(),
        registry=registry,
        llm_adapter=FailingLLM(),
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-agent-fallback",
        member_id=None,
        scopes=("orders:read", "financial:read", "menu:read", "customers:read", "analytics:read"),
    )
    artifact = await loop.run(
        question="dime qué productos venden mucho pero tienen bajo margen",
        context=context,
        run_id=uuid4(),
        complexity="complex",
    )
    assert artifact["agent_mode"] is True
    assert artifact["safe_to_answer"] is True
    assert artifact["observations"]


@pytest.mark.asyncio
async def test_sales_workflow_react_mode_uses_agent_runner():
    from app.workflows.sales import SalesWorkflow

    class RecordingRunner:
        async def execute(self, **kwargs):
            return {
                "agent_mode": True,
                "safe_to_answer": True,
                "summary": "Respuesta agentica",
                "observations": [{"tool_name": "waro.sales.metrics", "status": "succeeded"}],
                "tables": [],
                "complexity": "simple",
            }

        async def execute_shadow(self, **kwargs):
            legacy = kwargs["legacy_artifact"]
            legacy["agent_shadow_artifact"] = {"shadow": True}
            return legacy

    @asynccontextmanager
    async def connection_factory():
        class FakeConnection:
            async def fetchrow(self, query, *args):
                return {"id": uuid4()}

            async def execute(self, *args, **kwargs):
                return None

        yield FakeConnection()

    workflow = SalesWorkflow(
        settings=Settings(AGENT_MODE="react", LLM_PROVIDER="disabled"),
        gateway=FakeGateway(),
        llm_adapter=FakeLLM(),
        connection_factory=connection_factory,
    )
    workflow.kali_runner = RecordingRunner()
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-react",
        member_id=None,
        scopes=("orders:read",),
    )
    state = await workflow._run_kali_agent_node(
        {
            "request": type("Req", (), {"question": "dame ventas de ayer"})(),
            "context": context,
            "run_id": uuid4(),
            "sales_intent": "sales_metrics",
        }
    )
    assert state["summary"] == "Respuesta agentica"
    assert state["artifact"]["agent_mode"] is True


@pytest.mark.asyncio
async def test_sales_workflow_stream_react_mode_uses_graph():
    from app.workflows.sales import SalesWorkflow

    class RecordingRunner:
        async def execute(self, **kwargs):
            return {
                "agent_mode": True,
                "safe_to_answer": True,
                "summary": "Respuesta agentica en stream",
                "observations": [
                    {
                        "tool_name": "waro.sales.metrics",
                        "status": "succeeded",
                        "result_summary": "Returned 1 rows.",
                    }
                ],
                "tables": [],
                "complexity": "simple",
            }

        async def execute_shadow(self, **kwargs):
            legacy = kwargs["legacy_artifact"]
            legacy["agent_shadow_artifact"] = {"shadow": True}
            return legacy

    @asynccontextmanager
    async def connection_factory():
        class FakeConnection:
            async def fetchrow(self, query, *args):
                return {"id": uuid4()}

            async def execute(self, *args, **kwargs):
                return None

        yield FakeConnection()

    workflow = SalesWorkflow(
        settings=Settings(AGENT_MODE="react", LLM_PROVIDER="disabled"),
        gateway=FakeGateway(),
        llm_adapter=FakeLLM(),
        connection_factory=connection_factory,
    )
    workflow.kali_runner = RecordingRunner()
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-react-stream",
        member_id=None,
        scopes=("orders:read",),
    )
    events = [
        event
        async for event in workflow.stream(
            request=type("Req", (), {"question": "dame ventas de ayer", "conversation_id": None})(),
            context=context,
        )
    ]

    assert [event.event for event in events] == [
        "run_started",
        "step_started",
        "tool_started",
        "tool_finished",
        "final",
    ]
    assert events[-1].data["summary"] == "Respuesta agentica en stream"
    assert events[-1].data["artifact_summary"]["agent_mode"] is True
