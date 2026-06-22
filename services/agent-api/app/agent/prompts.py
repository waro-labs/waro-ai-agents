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
                "artifact.query_metadata, artifact.analysis, artifact.advisor_analysis, "
                "artifact.agent_profile, artifact.limitations, artifact.conversation_state, "
                "artifact.advisor_state, artifact.conversation_plan, artifact.answer_strategy, "
                "artifact.deterministic_summary y artifact.tool_results. "
                "No inventes ventas, ordenes, productos ni tendencias. "
                "Si safe_to_answer es false, responde solo el error_message. "
                "Si artifact.conversation_plan.subject es kali_capabilities, responde como Kali explicando "
                "brevemente que puedes conversar, mantener contexto y analizar datos con evidencia; no cites "
                "metricas ni digas que consultaste datos. "
                "La estructura de la respuesta debe obedecer artifact.answer_strategy.type. "
                "Cuando artifact.advisor_analysis exista, usalo como lectura experta principal, "
                "pero las cifras deben salir de ranked_rows, metrics, evidence o query_metadata. "
                "Si artifact.deterministic_summary existe, conserva sus cifras principales y mejora "
                "la explicacion alrededor de ellas. "
                "Responde como una analista senior conversando: respuesta directa, lectura breve, "
                "alerta/limitacion si aplica y siguiente paso natural. "
                "Cuando artifact.query_metadata exista, cita dataset, measures, dimensions, filters, "
                "ordenamiento y limitations solo si aparecen ahi o en artifact.evidence. "
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


def analyze_artifact_messages(*, artifact: dict[str, Any]) -> list[LLMMessage]:
    safe_payload = sanitize_value(
        {
            "question": artifact.get("question"),
            "question_intent": artifact.get("question_intent"),
            "answerability": artifact.get("answerability"),
            "metrics": artifact.get("metrics"),
            "ranked_rows": artifact.get("ranked_rows"),
            "query_metadata": artifact.get("query_metadata"),
            "evidence": artifact.get("evidence"),
            "limitations": artifact.get("limitations"),
            "conversation_state": artifact.get("conversation_state"),
            "advisor_state": artifact.get("advisor_state"),
            "prior_analysis": artifact.get("analysis"),
        }
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Eres Kali, analista senior de restaurantes para WARO. "
                "Analiza SOLO la evidencia JSON. No redactes la respuesta final. "
                "No inventes cifras, productos, clientes, tablas ni conclusiones fuera de la evidencia. "
                "Devuelve SOLO JSON valido con estas llaves: "
                "direct_answer, diagnosis, data_quality_notes, business_risks, "
                "recommended_actions, follow_up_questions, advisor_state_update, confidence. "
                "Cada lista debe contener strings cortos. direct_answer debe ser una frase corta. "
                "advisor_state_update debe ser un objeto con active_topic, hypotheses, "
                "known_data_issues y next_steps. Si no hay base suficiente, usa listas vacias "
                "y confidence bajo. Interpreta anomalias de negocio cuando la evidencia las sugiera, "
                "pero marca incertidumbre como nota de calidad de datos."
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
