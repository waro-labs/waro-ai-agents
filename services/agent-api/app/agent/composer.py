from __future__ import annotations

import json
from typing import Any

from app.agent.evidence import deterministic_evidence_summary
from app.agent.prompts import compose_summary_messages
from app.config import Settings
from app.llm.base import LLMAdapter
from app.llm.model_router import Complexity, model_for


async def compose_agent_summary(
    *,
    settings: Settings,
    llm_adapter: LLMAdapter,
    artifact: dict[str, Any],
    complexity: Complexity,
    fallback: str,
) -> str:
    if settings.llm_provider == "disabled" or not artifact.get("safe_to_answer"):
        if not artifact.get("safe_to_answer"):
            return str(artifact.get("error_message") or fallback)
        return fallback
    prompt_artifact = {**artifact, "deterministic_summary": fallback}
    messages = compose_summary_messages(artifact=prompt_artifact)
    try:
        response = await llm_adapter.complete(
            messages=messages,
            temperature=0.2,
            model=model_for(settings, step="compose", complexity=complexity),
        )
        content = response.content.strip()
        if not content:
            _log_composer_fallback(artifact=artifact, reason="empty_content")
            return fallback
        if _prefer_factual_fallback(artifact=artifact, content=content, fallback=fallback):
            _log_composer_fallback(artifact=artifact, reason="factual_fallback_hybrid")
            return _hybrid_summary(artifact=artifact, content=content, fallback=fallback)
        return content
    except Exception as exc:
        _log_composer_fallback(artifact=artifact, reason="exception", error_type=type(exc).__name__)
        return fallback


def deterministic_summary(artifact: dict[str, Any]) -> str:
    if artifact.get("agent_engine_version") == "intent-capability-v1":
        return deterministic_evidence_summary(artifact)
    if not artifact.get("safe_to_answer"):
        return str(artifact.get("error_message") or "No pude responder con los datos disponibles.")
    tables = artifact.get("tables") if isinstance(artifact.get("tables"), list) else []
    parts = ["Encontre datos para tu consulta."]
    for table in tables[:3]:
        tool = table.get("tool")
        metrics = table.get("metrics") if isinstance(table.get("metrics"), dict) else {}
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        if metrics:
            metric_bits = ", ".join(f"{key}={value}" for key, value in list(metrics.items())[:4])
            parts.append(f"{tool}: {metric_bits}.")
        elif rows:
            parts.append(f"{tool}: {len(rows)} filas.")
    return " ".join(parts)


def _prefer_factual_fallback(*, artifact: dict[str, Any], content: str, fallback: str) -> bool:
    strategy = artifact.get("answer_strategy") if isinstance(artifact.get("answer_strategy"), dict) else {}
    if strategy.get("type") != "ranking":
        return False
    fallback_has_values = any(token in fallback for token in ("$", "%", " unidades", " ordenes"))
    content_has_values = any(token in content for token in ("$", "%", " unidades", " ordenes"))
    return fallback_has_values and not content_has_values


def _hybrid_summary(*, artifact: dict[str, Any], content: str, fallback: str) -> str:
    advisor = artifact.get("advisor_analysis") if isinstance(artifact.get("advisor_analysis"), dict) else {}
    sections = [content.strip()]
    notes = [
        *_string_items(advisor.get("diagnosis")),
        *_string_items(advisor.get("data_quality_notes")),
        *_string_items(advisor.get("business_risks")),
    ]
    if notes and not any(note in content for note in notes[:2]):
        sections.append("Lectura: " + " ".join(notes[:2]))
    sections.append(fallback)
    actions = _string_items(advisor.get("recommended_actions"))
    if actions:
        sections.append("Siguiente paso: " + actions[0])
    return "\n\n".join(section for section in sections if section)


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _log_composer_fallback(*, artifact: dict[str, Any], reason: str, error_type: str | None = None) -> None:
    strategy = artifact.get("answer_strategy") if isinstance(artifact.get("answer_strategy"), dict) else {}
    plan = artifact.get("conversation_plan") if isinstance(artifact.get("conversation_plan"), dict) else {}
    payload = {
        "reason": reason,
        "error_type": error_type,
        "strategy": strategy.get("type"),
        "conversation_intent_type": plan.get("intent_type"),
        "conversation_subject": plan.get("subject"),
        "safe_to_answer": bool(artifact.get("safe_to_answer")),
        "row_count": len(artifact.get("ranked_rows") or []) if isinstance(artifact.get("ranked_rows"), list) else 0,
    }
    print("[agent-api:composer] fallback " + json.dumps(payload, ensure_ascii=False), flush=True)
