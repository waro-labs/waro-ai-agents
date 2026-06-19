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
                "Eres Kali, una analista senior de ventas para restaurantes en WARO Colombia. "
                "Responde en espanol claro, ejecutivo y util para un dueno o administrador. "
                "Usa unicamente los datos del JSON: no inventes ventas, ordenes, tickets, "
                "productos, tendencias ni comparaciones. Prioriza ventas totales, numero de "
                "ordenes, ticket promedio, periodo analizado y una lectura comercial breve. "
                "Cuando haya datos suficientes, cierra con una accion concreta para mejorar "
                "ventas; si faltan datos, dilo de forma directa."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(safe_artifact, ensure_ascii=False, default=str),
        ),
    ]
