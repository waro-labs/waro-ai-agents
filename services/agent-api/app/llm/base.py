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


class DisabledLLMAdapter:
    provider = "disabled"

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        temperature: float = 0.2,
    ) -> LLMResponse:
        raise LLMError("LLM provider is disabled.")
