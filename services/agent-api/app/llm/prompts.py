import json
from typing import Any

from app.llm.base import LLMMessage
from app.tools.sanitize import sanitize_value


def sales_planner_messages(
    *,
    question: str,
    today: str,
    timezone: str,
    tool_catalog: list[dict[str, Any]],
) -> list[LLMMessage]:
    safe_payload = sanitize_value(
        {
            "question": question,
            "today": today,
            "timezone": timezone,
            "available_tools": [
                {
                    "name": tool.get("name"),
                    "domain": tool.get("domain"),
                    "description": tool.get("description"),
                    "default_fields": tool.get("default_fields"),
                    "arguments_schema": tool.get("arguments_schema"),
                }
                for tool in tool_catalog
            ],
            "allowed_intents": ["small_talk", "sales_metrics"],
            "allowed_answer_styles": [
                "direct_metric",
                "business_analysis",
                "financial_analysis",
                "diagnostic",
            ],
            "allowed_group_by": [None, "date", "weekday", "hour", "product", "payment", "ticket"],
        }
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres el planner semantico de Kali para ventas WARO. "
                "No redactes respuesta al usuario. Devuelve SOLO JSON valido, sin markdown. "
                "Tu trabajo es resolver intencion, periodo, estilo de respuesta y llamadas a tools. "
                "Usa la fecha actual y zona horaria entregadas. Reglas criticas: "
                "'mes pasado' es el mes calendario anterior completo; 'este mes' o 'del mes' "
                "es desde el dia 1 del mes actual hasta today; 'ayer' es today menos un dia. "
                "Si el usuario pide promedio por dia, tendencia diaria, dia a dia o ultimos N dias, "
                "usa group_by='date'. Para preguntas concretas como 'ventas de ayer', "
                "answer_style debe ser direct_metric. Para analisis financiero, usa financial_analysis. "
                "Formato exacto: {\"intent\":\"small_talk|sales_metrics\","
                "\"date_from\":\"YYYY-MM-DD|null\",\"date_to\":\"YYYY-MM-DD|null\","
                "\"group_by\":\"date|weekday|hour|product|payment|ticket|null\","
                "\"answer_style\":\"direct_metric|business_analysis|financial_analysis|diagnostic\","
                "\"tools\":[{\"name\":\"waro.sales.metrics\",\"reason\":\"...\"}],"
                "\"confidence\":0.0,\"reason\":\"...\"}."
            ),
        ),
        LLMMessage(
            role="user",
            content=json.dumps(safe_payload, ensure_ascii=False, default=str),
        ),
    ]


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
                "si es direct_metric, responde maximo en 2 frases, sin lectura comercial ni accion; "
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
