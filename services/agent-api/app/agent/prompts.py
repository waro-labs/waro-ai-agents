from __future__ import annotations

import json
from typing import Any

from app.llm.base import LLMMessage
from app.tools.sanitize import sanitize_value


def compose_summary_messages(*, artifact: dict[str, Any]) -> list[LLMMessage]:
    safe_payload = sanitize_value({"artifact": artifact})
    return [
        LLMMessage(
            role="system",
            content=(
                "Redacta la respuesta final en espanol para el usuario de WARO. "
                "Usa unicamente artifact.metrics, artifact.ranked_rows, artifact.evidence, "
                "artifact.analysis, artifact.agent_profile, artifact.limitations, "
                "artifact.conversation_state, artifact.answer_strategy y artifact.tool_results. "
                "No inventes ventas, ordenes, productos ni tendencias. "
                "Si safe_to_answer es false, responde solo el error_message. "
                "La estructura de la respuesta debe obedecer artifact.answer_strategy.type. "
                "Para preguntas abiertas de negocio, prioriza artifact.analysis.facts, "
                "patterns, risks, opportunities y recommended_actions. "
                "Para recomendaciones, convierte la evidencia en acciones concretas y explica por que. "
                "Para rankings, usa solo las metricas pedidas; no agregues margen si el intent no lo pidio. "
                "Si answer_strategy.avoid_repeating es true, no repitas la respuesta anterior: aporta un angulo nuevo. "
                "Usa artifact.agent_profile como rol analitico si existe. "
                "No respondas 'Encontre datos' sin datos concretos."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(safe_payload, ensure_ascii=False, default=str)),
    ]


def choose_answer_strategy_messages(
    *,
    question: str,
    intent: dict[str, Any],
    artifact: dict[str, Any],
    conversation_state: dict[str, Any],
    fallback: dict[str, Any],
) -> list[LLMMessage]:
    payload = sanitize_value(
        {
            "question": question,
            "intent": intent,
            "artifact_summary": {
                "answerability": artifact.get("answerability"),
                "metrics_keys": sorted((artifact.get("metrics") or {}).keys())
                if isinstance(artifact.get("metrics"), dict)
                else [],
                "ranked_row_count": len(artifact.get("ranked_rows") or [])
                if isinstance(artifact.get("ranked_rows"), list)
                else 0,
                "analysis": artifact.get("analysis"),
                "limitations": artifact.get("limitations"),
            },
            "conversation_state": conversation_state,
            "fallback": fallback,
            "allowed_types": [
                "direct_metric",
                "ranking",
                "diagnosis",
                "recommendation",
                "comparison",
                "explanation",
                "follow_up",
                "blocked",
            ],
        }
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Decide la estrategia de respuesta para una consulta analitica WARO. "
                "No elijas tools y no redactes la respuesta final. Devuelve SOLO JSON valido con: "
                "type, objective, use_previous_artifact, avoid_repeating, reasoning_focus, confidence. "
                "Usa recommendation cuando el usuario pregunte que hacer con datos previos. "
                "Usa follow_up cuando la pregunta dependa principalmente del contexto anterior. "
                "Usa ranking solo si el usuario pide ordenar/listar mejores/mas/menos. "
                "Usa diagnosis para diagnosticos, comportamientos, riesgos u oportunidades."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]
