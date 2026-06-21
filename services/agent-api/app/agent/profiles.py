from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.agent.intent import QuestionIntent, normalize_text


@dataclass(frozen=True)
class AgentProfile:
    id: str
    description: str
    match: dict[str, Any]
    planning: dict[str, Any]
    analysis: dict[str, Any]

    @property
    def tool_priorities(self) -> tuple[str, ...]:
        values = self.planning.get("tool_priorities")
        if not isinstance(values, list):
            return ()
        return tuple(str(item) for item in values if item)

    @property
    def max_steps(self) -> int:
        value = self.planning.get("max_steps")
        return int(value) if isinstance(value, int) and value > 0 else 5

    def tool_group(self, name: str) -> set[str]:
        groups = self.analysis.get("tool_groups")
        if not isinstance(groups, dict):
            return set()
        values = groups.get(name)
        if not isinstance(values, list):
            return set()
        return {str(item) for item in values if item}

    @property
    def signals(self) -> tuple[dict[str, Any], ...]:
        values = self.analysis.get("signals")
        if not isinstance(values, list):
            return ()
        return tuple(item for item in values if isinstance(item, dict))

    @property
    def missing_evidence_limitations(self) -> tuple[dict[str, Any], ...]:
        values = self.analysis.get("missing_evidence_limitations")
        if not isinstance(values, list):
            return ()
        return tuple(item for item in values if isinstance(item, dict))


@lru_cache(maxsize=1)
def load_agent_profiles() -> tuple[AgentProfile, ...]:
    path = Path(__file__).with_name("analysis_profiles.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles") if isinstance(payload, dict) else []
    result: list[AgentProfile] = []
    for item in profiles if isinstance(profiles, list) else []:
        if not isinstance(item, dict):
            continue
        result.append(
            AgentProfile(
                id=str(item.get("id") or "unknown"),
                description=str(item.get("description") or ""),
                match=item.get("match") if isinstance(item.get("match"), dict) else {},
                planning=item.get("planning") if isinstance(item.get("planning"), dict) else {},
                analysis=item.get("analysis") if isinstance(item.get("analysis"), dict) else {},
            )
        )
    return tuple(result)


def profile_for_intent(intent: QuestionIntent) -> AgentProfile | None:
    for profile in load_agent_profiles():
        if _matches_intent(profile, intent):
            return profile
    return None


def _matches_intent(profile: AgentProfile, intent: QuestionIntent) -> bool:
    entities = _set(profile.match.get("entities"))
    grains = _set(profile.match.get("grains"))
    operations = _set(profile.match.get("operations"))
    if entities and intent.entity not in entities:
        return False
    if grains and intent.grain not in grains:
        return False
    if operations and not operations.intersection(intent.operations):
        return False
    return True


def rows_for_group(
    *,
    tables: list[dict[str, Any]],
    profile: AgentProfile | None,
    group: str,
) -> list[dict[str, Any]]:
    if profile is None:
        return []
    tool_names = profile.tool_group(group)
    rows: list[dict[str, Any]] = []
    for table in tables:
        if table.get("tool") not in tool_names:
            continue
        for row in table.get("rows") or []:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def first_value(row: dict[str, Any], fields: list[Any] | tuple[Any, ...]) -> Any:
    for field in fields:
        value = row.get(str(field))
        if value is not None:
            return value
    return None


def normalized_any(value: Any) -> str:
    return normalize_text(str(value or "").strip())


def _set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if item}
