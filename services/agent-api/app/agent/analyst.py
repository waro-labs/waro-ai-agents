from __future__ import annotations

import json
from typing import Any

from app.agent.prompts import analyze_artifact_messages
from app.config import Settings
from app.llm.base import LLMAdapter
from app.llm.model_router import Complexity, model_for
from app.tools.sanitize import sanitize_value


ANALYSIS_KEYS = (
    "direct_answer",
    "diagnosis",
    "data_quality_notes",
    "business_risks",
    "recommended_actions",
    "follow_up_questions",
    "advisor_state_update",
    "confidence",
)


async def analyze_agent_artifact(
    *,
    settings: Settings,
    llm_adapter: LLMAdapter,
    artifact: dict[str, Any],
    complexity: Complexity,
) -> dict[str, Any] | None:
    if settings.llm_provider == "disabled" or not artifact.get("safe_to_answer"):
        return None
    conversation_plan = artifact.get("conversation_plan") if isinstance(artifact.get("conversation_plan"), dict) else {}
    if conversation_plan.get("subject") == "kali_capabilities":
        return None
    try:
        response = await llm_adapter.complete(
            messages=analyze_artifact_messages(artifact=artifact),
            temperature=0.1,
            model=model_for(settings, step="verify", complexity=complexity),
        )
        parsed = json.loads(response.content.strip())
    except Exception as exc:
        print(
            "[agent-api:analyst] fallback "
            + json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "conversation_subject": conversation_plan.get("subject"),
                    "row_count": len(artifact.get("ranked_rows") or [])
                    if isinstance(artifact.get("ranked_rows"), list)
                    else 0,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return None
    if not isinstance(parsed, dict):
        return None
    normalized = _normalize_analysis(parsed)
    return sanitize_value(normalized)


def _normalize_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("advisor_state_update")
    state_payload = state if isinstance(state, dict) else {}
    result = {
        "direct_answer": _text(payload.get("direct_answer")),
        "diagnosis": _string_list(payload.get("diagnosis")),
        "data_quality_notes": _string_list(payload.get("data_quality_notes")),
        "business_risks": _string_list(payload.get("business_risks")),
        "recommended_actions": _string_list(payload.get("recommended_actions")),
        "follow_up_questions": _string_list(payload.get("follow_up_questions")),
        "advisor_state_update": {
            "active_topic": _text(state_payload.get("active_topic")),
            "hypotheses": _string_list(state_payload.get("hypotheses")),
            "known_data_issues": _string_list(state_payload.get("known_data_issues")),
            "next_steps": _string_list(state_payload.get("next_steps")),
        },
        "confidence": _confidence(payload.get("confidence")),
        "source": "llm",
    }
    return {key: result[key] for key in ANALYSIS_KEYS if key in result} | {"source": "llm"}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:8]


def _confidence(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    return "medium"
