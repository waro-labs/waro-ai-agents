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


def sales_summary_messages(artifact: dict[str, Any]) -> list[LLMMessage]:
    safe_artifact = sanitize_value(
        {
            "question": artifact.get("question"),
            "period": artifact.get("period"),
            "metrics": artifact.get("metrics", {}),
            "highlights": artifact.get("highlights", []),
            "tool_calls": artifact.get("tool_calls", []),
        }
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres un analista de ventas para restaurantes WARO. "
                "Responde en español, de forma breve y ejecutiva. "
                "Usa solo los datos del JSON, menciona periodo, ventas totales, "
                "numero de ordenes si esta disponible y ticket promedio si esta disponible. "
                "No inventes valores faltantes."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(safe_artifact, ensure_ascii=False, default=str),
        ),
    ]
