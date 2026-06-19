import json

import httpx
import pytest

from app.config import Settings
from app.llm.base import LLMError, LLMMessage
from app.llm.factory import get_llm_adapter
from app.llm.kimi import KimiAdapter
from app.llm.pricing import estimate_llm_cost


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
            json={
                "choices": [{"message": {"content": "Resumen desde Kimi."}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 250,
                    "total_tokens": 1250,
                },
            },
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
    assert response.input_tokens == 1000
    assert response.output_tokens == 250
    assert response.total_tokens == 1250
    assert response.estimated_cost_usd is None
    assert response.cost_source == "pricing_unavailable"


@pytest.mark.asyncio
async def test_kimi_adapter_estimates_known_model_cost():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
        )

    adapter = KimiAdapter(
        Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_MODEL="kimi-k2.7-code",
        ),
        transport=httpx.MockTransport(handler),
    )

    response = await adapter.complete(messages=[LLMMessage(role="user", content="hola")])

    assert response.input_tokens == 1000
    assert response.output_tokens == 500
    assert response.total_tokens == 1500
    assert response.estimated_cost_usd == 0.00295
    assert response.prompt_cost_usd == 0.00095
    assert response.completion_cost_usd == 0.002
    assert response.cost_source == "static:official-kimi-pricing-2026-06-18"


@pytest.mark.asyncio
async def test_kimi_adapter_estimates_cheapest_moonshot_model_cost():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
        )

    adapter = KimiAdapter(
        Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_MODEL="moonshot-v1-8k",
        ),
        transport=httpx.MockTransport(handler),
    )

    response = await adapter.complete(messages=[LLMMessage(role="user", content="hola")])

    assert response.estimated_cost_usd == 0.0012
    assert response.prompt_cost_usd == 0.0002
    assert response.completion_cost_usd == 0.001
    assert response.cost_source == "static:official-kimi-pricing-2026-06-18"


@pytest.mark.asyncio
async def test_kimi_adapter_handles_missing_usage_without_cost():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    adapter = KimiAdapter(
        Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_MODEL="kimi-k2.7-code",
        ),
        transport=httpx.MockTransport(handler),
    )

    response = await adapter.complete(messages=[LLMMessage(role="user", content="hola")])

    assert response.input_tokens is None
    assert response.output_tokens is None
    assert response.total_tokens is None
    assert response.estimated_cost_usd is None
    assert response.cost_source == "usage_unavailable"


@pytest.mark.asyncio
async def test_kimi_adapter_streams_openai_compatible_chunks_with_usage():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        chunks = [
            {"choices": [{"delta": {"content": "Resumen"}}]},
            {"choices": [{"delta": {"content": " Kimi."}}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                },
            },
        ]
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        content += "data: [DONE]\n\n"
        return httpx.Response(200, content=content)

    adapter = KimiAdapter(
        Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_MODEL="kimi-k2.7-code",
        ),
        transport=httpx.MockTransport(handler),
    )

    chunks = [
        chunk
        async for chunk in adapter.stream_complete(
            messages=[LLMMessage(role="user", content="hola")]
        )
    ]

    assert captured["payload"]["stream"] is True
    assert [chunk.text for chunk in chunks if chunk.text] == ["Resumen", " Kimi."]
    response = chunks[-1].response
    assert response is not None
    assert response.content == "Resumen Kimi."
    assert response.input_tokens == 1000
    assert response.output_tokens == 500
    assert response.total_tokens == 1500
    assert response.estimated_cost_usd == 0.00295
    assert response.cost_source == "static:official-kimi-pricing-2026-06-18"


@pytest.mark.asyncio
async def test_kimi_adapter_streaming_ignores_malformed_and_irrelevant_chunks():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                ": keepalive\n\n"
                "data: {not-json}\n\n"
                'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    adapter = KimiAdapter(
        Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_MODEL="kimi-k2.7-code",
        ),
        transport=httpx.MockTransport(handler),
    )

    chunks = [
        chunk
        async for chunk in adapter.stream_complete(
            messages=[LLMMessage(role="user", content="hola")]
        )
    ]

    assert [chunk.text for chunk in chunks if chunk.text] == ["OK"]
    response = chunks[-1].response
    assert response is not None
    assert response.content == "OK"
    assert response.input_tokens is None
    assert response.cost_source == "usage_unavailable"


@pytest.mark.asyncio
async def test_kimi_adapter_requires_api_key_at_completion_time():
    adapter = KimiAdapter(Settings(LLM_PROVIDER="kimi", KIMI_API_KEY=None))

    with pytest.raises(LLMError):
        await adapter.complete(messages=[LLMMessage(role="user", content="hola")])


def test_llm_cost_estimate_unknown_model_falls_back():
    estimate = estimate_llm_cost(
        provider="kimi",
        model="unknown-model",
        input_tokens=100,
        output_tokens=50,
    )

    assert estimate.estimated_cost_usd is None
    assert estimate.source == "pricing_unavailable"


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
