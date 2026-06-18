import json
from typing import Any

from app.llm.base import LLMMessage
from app.tools.sanitize import sanitize_value


def food_cost_summary_messages(artifact: dict[str, Any]) -> list[LLMMessage]:
    safe_artifact = sanitize_value(
        {
            "question": artifact.get("question"),
            "period": artifact.get("period"),
            "low_margin_products": artifact.get("low_margin_products", []),
            "recommendations": artifact.get("recommendations", []),
            "tool_calls": artifact.get("tool_calls", []),
        }
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres un analista financiero para restaurantes WARO. "
                "Resume hallazgos de food cost en español, con tono claro, "
                "operativo y sin inventar datos que no estén en el JSON."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(safe_artifact, ensure_ascii=False, default=str),
        ),
    ]
