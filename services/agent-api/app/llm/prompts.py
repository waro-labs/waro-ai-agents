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
            "answer_style": artifact.get("answer_style"),
            "period": artifact.get("period"),
            "metrics": artifact.get("metrics", {}),
            "auxiliary_context": artifact.get("auxiliary_context", {}),
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
                "productos, tendencias ni comparaciones. Obedece answer_style: "
                "si es business_analysis, entrega metricas, lectura comercial breve y una accion; "
                "si es financial_analysis, cruza ventas con margen/costo/productos disponibles en "
                "auxiliary_context y da riesgos y acciones financieras; si es diagnostic, explica "
                "el problema con datos y siguientes verificaciones. Si faltan datos, dilo de forma "
                "directa y no des una lista generica de recomendaciones."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(safe_artifact, ensure_ascii=False, default=str),
        ),
    ]
