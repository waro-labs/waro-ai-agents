from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from app.config import Settings


@dataclass(frozen=True)
class ConversationState:
    active_entity: str | None = None
    active_grain: str | None = None
    active_period: dict[str, Any] | None = None
    active_measures: tuple[str, ...] = ()
    active_dimensions: tuple[str, ...] = ()
    last_question: str | None = None
    last_summary: str | None = None
    last_answer_strategy: str | None = None
    last_artifact: dict[str, Any] | None = None
    prior_insights: tuple[str, ...] = ()
    prior_actions: tuple[str, ...] = ()
    prior_limitations: tuple[str, ...] = ()
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_measures"] = list(self.active_measures)
        payload["active_dimensions"] = list(self.active_dimensions)
        payload["prior_insights"] = list(self.prior_insights)
        payload["prior_actions"] = list(self.prior_actions)
        payload["prior_limitations"] = list(self.prior_limitations)
        return payload


async def load_conversation_messages(
    *,
    settings: Settings,
    connection_factory: Any,
    conversation_id: UUID | None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    if conversation_id is None:
        return []
    message_limit = limit or settings.agent_conversation_message_limit
    async with connection_factory() as connection:
        rows = await connection.fetch(
            """
            SELECT role, content
            FROM ai.messages
            WHERE conversation_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            conversation_id,
            message_limit,
        )
    messages = [
        {"role": str(row["role"]), "content": str(row["content"])}
        for row in reversed(rows)
        if row.get("content")
    ]
    return messages


async def load_conversation_state(
    *,
    settings: Settings,
    connection_factory: Any,
    conversation_id: UUID | None,
) -> ConversationState:
    if conversation_id is None:
        return ConversationState()
    message_limit = max(3, settings.agent_conversation_message_limit)
    async with connection_factory() as connection:
        rows = await connection.fetch(
            """
            SELECT role, content, content_sanitized, metadata
            FROM ai.messages
            WHERE conversation_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            conversation_id,
            message_limit,
        )
    last_artifact: dict[str, Any] | None = None
    last_summary: str | None = None
    last_question: str | None = None
    for row in rows:
        role = str(row["role"])
        content = str(row["content"] or "")
        if role == "assistant" and last_artifact is None:
            parsed = _parse_artifact(content)
            if parsed is not None:
                last_artifact = parsed
                sanitized = row.get("content_sanitized")
                last_summary = str(sanitized) if sanitized else str(parsed.get("summary") or "")
        elif role == "user" and last_question is None:
            last_question = content
        if last_artifact is not None and last_question is not None:
            break
    if last_artifact is None:
        return ConversationState(last_question=last_question, source="messages")
    return _state_from_artifact(
        artifact=last_artifact,
        last_question=last_question,
        last_summary=last_summary,
    )


def _parse_artifact(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) and parsed.get("agent_mode") else None


def _state_from_artifact(
    *,
    artifact: dict[str, Any],
    last_question: str | None,
    last_summary: str | None,
) -> ConversationState:
    intent = artifact.get("question_intent") if isinstance(artifact.get("question_intent"), dict) else {}
    analysis = artifact.get("analysis") if isinstance(artifact.get("analysis"), dict) else {}
    strategy = artifact.get("answer_strategy")
    if isinstance(strategy, dict):
        strategy_name = str(strategy.get("type") or strategy.get("strategy") or "")
    else:
        strategy_name = str(strategy or "")
    return ConversationState(
        active_entity=_string_or_none(intent.get("entity")),
        active_grain=_string_or_none(intent.get("grain")),
        active_period=intent.get("time_range") if isinstance(intent.get("time_range"), dict) else None,
        active_measures=tuple(_string_items(intent.get("measures"))),
        active_dimensions=tuple(_string_items(intent.get("dimensions"))),
        last_question=last_question or _string_or_none(artifact.get("question")),
        last_summary=last_summary or _string_or_none(artifact.get("summary")),
        last_answer_strategy=strategy_name or None,
        last_artifact=_compact_artifact(artifact),
        prior_insights=tuple(
            [
                *_string_items(analysis.get("facts")),
                *_string_items(analysis.get("patterns")),
                *_string_items(artifact.get("insights")),
            ][:8]
        ),
        prior_actions=tuple(
            [
                *_string_items(analysis.get("recommended_actions")),
                *_string_items(artifact.get("recommended_actions")),
            ][:6]
        ),
        prior_limitations=tuple(
            [
                *_string_items(analysis.get("limitations")),
                *_string_items(artifact.get("limitations")),
            ][:5]
        ),
        source="artifact",
    )


def _compact_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": artifact.get("question"),
        "question_intent": artifact.get("question_intent"),
        "metrics": artifact.get("metrics"),
        "ranked_rows": (artifact.get("ranked_rows") or [])[:10]
        if isinstance(artifact.get("ranked_rows"), list)
        else [],
        "analysis": artifact.get("analysis"),
        "answer_strategy": artifact.get("answer_strategy"),
        "summary": artifact.get("summary"),
    }


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
