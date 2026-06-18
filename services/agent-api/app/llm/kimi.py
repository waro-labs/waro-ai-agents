from typing import Any

import httpx

from app.config import Settings
from app.llm.base import LLMError, LLMMessage, LLMResponse


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
        return LLMResponse(
            content=content,
            model=self.settings.kimi_model,
            provider=self.provider,
        )

    def _extract_content(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Kimi completion response did not include content.") from exc
        return content
