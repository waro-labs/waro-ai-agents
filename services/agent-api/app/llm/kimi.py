import json
from typing import Any
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.llm.base import LLMError, LLMMessage, LLMResponse, LLMStreamChunk
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
        model: str | None = None,
    ) -> LLMResponse:
        self._require_api_key()
        selected_model = model or self.settings.kimi_model
        url = self._completion_url()
        payload = self._completion_payload(
            messages=messages,
            temperature=temperature,
            model=selected_model,
        )
        headers = self._headers()
        timeout = httpx.Timeout(self.settings.llm_timeout_seconds)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self.transport,
            ) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:1000] if exc.response is not None else ""
            raise LLMError(
                f"Kimi completion request failed: {exc.response.status_code} {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError("Kimi completion request failed.") from exc

        data = response.json()
        content = self._extract_content(data)
        usage = self._extract_usage(data)
        return self._response(content=content, usage=usage, model=selected_model)

    async def stream_complete(
        self,
        *,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        model: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        self._require_api_key()
        selected_model = model or self.settings.kimi_model
        url = self._completion_url()
        payload = {
            **self._completion_payload(
                messages=messages,
                temperature=temperature,
                model=selected_model,
            ),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = self._headers()
        timeout = httpx.Timeout(self.settings.llm_timeout_seconds)
        content_parts: list[str] = []
        usage: dict[str, int | None] = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self.transport,
            ) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        chunk = self._parse_stream_line(line)
                        if chunk is None:
                            continue
                        if chunk == "[DONE]":
                            break
                        chunk_usage = self._extract_usage(chunk)
                        if any(value is not None for value in chunk_usage.values()):
                            usage = chunk_usage
                        text = self._extract_stream_text(chunk)
                        if text:
                            content_parts.append(text)
                            yield LLMStreamChunk(text=text)
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:1000] if exc.response is not None else ""
            raise LLMError(
                f"Kimi streaming completion request failed: {exc.response.status_code} {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError("Kimi streaming completion request failed.") from exc

        yield LLMStreamChunk(
            response=self._response(
                content="".join(content_parts),
                usage=usage,
                model=selected_model,
            )
        )

    def _require_api_key(self) -> None:
        if not self.settings.kimi_api_key:
            raise LLMError("KIMI_API_KEY is required when LLM_PROVIDER=kimi.")

    def _completion_url(self) -> str:
        return self.settings.kimi_base_url.rstrip("/") + "/chat/completions"

    def _completion_payload(
        self,
        *,
        messages: list[LLMMessage],
        temperature: float,
        model: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        }
        if model in {"kimi-k2.5", "kimi-k2.6"}:
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["temperature"] = temperature
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.kimi_api_key}",
            "Content-Type": "application/json",
        }

    def _response(
        self,
        *,
        content: str,
        usage: dict[str, int | None],
        model: str,
    ) -> LLMResponse:
        cost = estimate_llm_cost(
            provider=self.provider,
            model=model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )
        return LLMResponse(
            content=content,
            model=model,
            provider=self.provider,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            estimated_cost_usd=cost.estimated_cost_usd,
            prompt_cost_usd=cost.prompt_cost_usd,
            completion_cost_usd=cost.completion_cost_usd,
            cost_source=cost.source,
        )

    def _parse_stream_line(self, line: str) -> dict[str, Any] | str | None:
        if not line:
            return None
        if line.startswith(":"):
            return None
        if not line.startswith("data:"):
            return None
        payload = line.removeprefix("data:").strip()
        if not payload:
            return None
        if payload == "[DONE]":
            return payload
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return data
        return None

    def _extract_stream_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""
        delta = first_choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
        return ""

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
