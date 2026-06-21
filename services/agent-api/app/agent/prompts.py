from __future__ import annotations

import json
from typing import Any

from app.llm.base import LLMMessage
from app.tools.sanitize import sanitize_value


def agent_step_messages(
    *,
    question: str,
    today: str,
    timezone: str,
    available_tools: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    conversation_messages: list[dict[str, str]] | None = None,
    step: int,
    max_steps: int,
) -> list[LLMMessage]:
    safe_payload = sanitize_value(
        {
            "question": question,
            "today": today,
            "timezone": timezone,
            "step": step,
            "max_steps": max_steps,
            "conversation_messages": conversation_messages or [],
            "available_tools": available_tools,
            "observations": observations,
        }
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres Kali, agente analitico de WARO. Elige la siguiente accion leyendo "
                "available_tools.capabilities (entity, measures, dimensions, supported_operations, "
                "supports_period). No dependas de nombres fijos de tools: si aparece una tool nueva "
                "con capabilities relevantes, puedela usar. "
                "Si observations ya responden la pregunta, termina. "
                "Si falta informacion, llama otra tool con argumentos validos segun arguments_schema. "
                "Devuelve SOLO JSON valido: "
                '{"action":"call_tool|finish","tool_name":"waro....|null",'
                '"arguments":{},"reason":"..."}. '
                "Para finish, tool_name debe ser null y arguments {}."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(safe_payload, ensure_ascii=False, default=str)),
    ]


def verify_answer_messages(
    *,
    question: str,
    artifact: dict[str, Any],
) -> list[LLMMessage]:
    safe_payload = sanitize_value({"question": question, "artifact": artifact})
    return [
        LLMMessage(
            role="system",
            content=(
                "Verifica si el artifact contiene datos suficientes para responder la pregunta. "
                "Devuelve SOLO JSON: "
                '{"safe_to_answer":true|false,"missing":"...","needs_more_tools":true|false}.'
            ),
        ),
        LLMMessage(role="user", content=json.dumps(safe_payload, ensure_ascii=False, default=str)),
    ]


def compose_summary_messages(*, artifact: dict[str, Any]) -> list[LLMMessage]:
    safe_payload = sanitize_value({"artifact": artifact})
    return [
        LLMMessage(
            role="system",
            content=(
                "Redacta la respuesta final en espanol para el usuario de WARO. "
                "Usa unicamente artifact.observations y artifact.tables. "
                "No inventes ventas, ordenes, productos ni tendencias. "
                "Si safe_to_answer es false, explica brevemente que falta usando error_message."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(safe_payload, ensure_ascii=False, default=str)),
    ]
