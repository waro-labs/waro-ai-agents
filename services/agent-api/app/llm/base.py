from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    prompt_cost_usd: float | None = None
    completion_cost_usd: float | None = None
    cost_source: str = "unavailable"


@dataclass(frozen=True)
class LLMStreamChunk:
    text: str = ""
    response: LLMResponse | None = None


class LLMError(Exception):
    pass


class LLMAdapter(Protocol):
    provider: str

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        temperature: float = 0.2,
    ) -> LLMResponse:
        ...

    def stream_complete(
        self,
        *,
        messages: list[LLMMessage],
        temperature: float = 0.2,
    ) -> AsyncIterator[LLMStreamChunk]:
        ...


class DisabledLLMAdapter:
    provider = "disabled"

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        temperature: float = 0.2,
    ) -> LLMResponse:
        raise LLMError("LLM provider is disabled.")

    async def stream_complete(
        self,
        *,
        messages: list[LLMMessage],
        temperature: float = 0.2,
    ) -> AsyncIterator[LLMStreamChunk]:
        raise LLMError("LLM provider is disabled.")
        if False:
            yield LLMStreamChunk()
