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
                "artifact.limitations y artifact.tool_results. "
                "No inventes ventas, ordenes, productos ni tendencias. "
                "Si safe_to_answer es false, responde solo el error_message. "
                "No respondas 'Encontre datos' sin datos concretos."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(safe_payload, ensure_ascii=False, default=str)),
    ]
