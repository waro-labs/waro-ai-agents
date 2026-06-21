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
    messages = compose_summary_messages(artifact=artifact)
    try:
        response = await llm_adapter.complete(
            messages=messages,
            temperature=0.2,
            model=model_for(settings, step="compose", complexity=complexity),
        )
        content = response.content.strip()
        if not content or _prefer_factual_fallback(artifact=artifact, content=content, fallback=fallback):
            return fallback
        return content
    except Exception:
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
