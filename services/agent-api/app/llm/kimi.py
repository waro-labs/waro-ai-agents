from typing import Any

import httpx

from app.config import Settings
from app.llm.base import LLMError, LLMMessage, LLMResponse
from app.llm.pricing import estimate_llm_cost


class KimiAdapter:
    provider = "kimi"

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.settings = settings
        self.transport = transport

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        temperature: float = 0.2,
    ) -> LLMResponse:
        if not self.settings.kimi_api_key:
            raise LLMError("KIMI_API_KEY is required when LLM_PROVIDER=kimi.")
        url = self.settings.kimi_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.kimi_model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.kimi_api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(self.settings.llm_timeout_seconds)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self.transport,
            ) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError("Kimi completion request failed.") from exc

        data = response.json()
        content = self._extract_content(data)
        usage = self._extract_usage(data)
        cost = estimate_llm_cost(
            provider=self.provider,
            model=self.settings.kimi_model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )
        return LLMResponse(
            content=content,
            model=self.settings.kimi_model,
            provider=self.provider,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            estimated_cost_usd=cost.estimated_cost_usd,
            cost_source=cost.source,
        )

    def _extract_content(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Kimi completion response did not include content.") from exc
        return content

    def _extract_usage(self, data: dict[str, Any]) -> dict[str, int | None]:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        input_tokens = self._coerce_token_count(usage.get("prompt_tokens"))
        output_tokens = self._coerce_token_count(usage.get("completion_tokens"))
        total_tokens = self._coerce_token_count(usage.get("total_tokens"))
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _coerce_token_count(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value >= 0:
            return value
        return None
