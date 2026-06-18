import json

import httpx
import pytest

from app.config import Settings
from app.llm.base import LLMError, LLMMessage
from app.llm.factory import get_llm_adapter
from app.llm.kimi import KimiAdapter


@pytest.mark.asyncio
async def test_disabled_llm_adapter_raises_without_network():
    adapter = get_llm_adapter(Settings(LLM_PROVIDER="disabled"))

    with pytest.raises(LLMError):
        await adapter.complete(messages=[LLMMessage(role="user", content="hola")])


@pytest.mark.asyncio
async def test_kimi_adapter_builds_openai_compatible_request():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Resumen desde Kimi."}}]},
        )

    adapter = KimiAdapter(
        Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_BASE_URL="https://api.moonshot.ai/v1",
            KIMI_MODEL="kimi-test",
        ),
        transport=httpx.MockTransport(handler),
    )

    response = await adapter.complete(
        messages=[
            LLMMessage(role="system", content="Responde breve."),
            LLMMessage(role="user", content="Resume esto."),
        ],
        temperature=0.1,
    )

    assert captured["url"] == "https://api.moonshot.ai/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["payload"] == {
        "model": "kimi-test",
        "messages": [
            {"role": "system", "content": "Responde breve."},
            {"role": "user", "content": "Resume esto."},
        ],
        "temperature": 0.1,
    }
    assert response.content == "Resumen desde Kimi."
    assert response.model == "kimi-test"
    assert response.provider == "kimi"


@pytest.mark.asyncio
async def test_kimi_adapter_requires_api_key_at_completion_time():
    adapter = KimiAdapter(Settings(LLM_PROVIDER="kimi", KIMI_API_KEY=None))

    with pytest.raises(LLMError):
        await adapter.complete(messages=[LLMMessage(role="user", content="hola")])


@pytest.mark.asyncio
async def test_kimi_adapter_wraps_http_errors():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "failed"})

    adapter = KimiAdapter(
        Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMError):
        await adapter.complete(messages=[LLMMessage(role="user", content="hola")])
