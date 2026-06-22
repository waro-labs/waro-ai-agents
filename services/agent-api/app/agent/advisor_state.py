from __future__ import annotations

from typing import Any

from app.tools.sanitize import sanitize_value


def merge_advisor_state(
    *,
    previous: dict[str, Any] | None,
    update: dict[str, Any] | None,
) -> dict[str, Any]:
    previous = previous if isinstance(previous, dict) else {}
    update = update if isinstance(update, dict) else {}
    active_topic = _text(update.get("active_topic")) or _text(previous.get("active_topic"))
    merged = {
        "active_topic": active_topic,
        "hypotheses": _merge_lists(previous.get("hypotheses"), update.get("hypotheses")),
        "known_data_issues": _merge_lists(previous.get("known_data_issues"), update.get("known_data_issues")),
        "next_steps": _merge_lists(previous.get("next_steps"), update.get("next_steps")),
    }
    return sanitize_value(merged)


def _merge_lists(previous: Any, update: Any) -> list[str]:
    values: list[str] = []
    for source in (previous, update):
        if not isinstance(source, list):
            continue
        for item in source:
            text = _text(item)
            if text and text not in values:
                values.append(text)
    return values[:8]


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""
