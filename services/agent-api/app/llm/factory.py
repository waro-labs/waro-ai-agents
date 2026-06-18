from app.config import Settings
from app.llm.base import DisabledLLMAdapter, LLMAdapter
from app.llm.kimi import KimiAdapter


def get_llm_adapter(settings: Settings) -> LLMAdapter:
    if settings.llm_provider == "kimi":
        return KimiAdapter(settings)
    return DisabledLLMAdapter()
