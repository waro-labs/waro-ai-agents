from __future__ import annotations

import json
from typing import Any

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
        return content or fallback
    except Exception:
        return fallback


def deterministic_summary(artifact: dict[str, Any]) -> str:
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
