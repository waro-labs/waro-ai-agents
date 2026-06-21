from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.agent.intent import QuestionIntent, normalize_text
from app.agent.prompts import choose_answer_strategy_messages
from app.config import Settings
from app.llm.base import LLMAdapter
from app.llm.model_router import Complexity, model_for

AnswerStrategyType = Literal[
    "direct_metric",
    "ranking",
    "diagnosis",
    "recommendation",
    "comparison",
    "explanation",
    "follow_up",
    "blocked",
]


@dataclass(frozen=True)
class AnswerStrategy:
    type: AnswerStrategyType
    objective: str
    use_previous_artifact: bool = False
    avoid_repeating: bool = False
    reasoning_focus: tuple[str, ...] = ()
    confidence: float = 0.6
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasoning_focus"] = list(self.reasoning_focus)
        return payload


async def choose_answer_strategy(
    *,
    settings: Settings,
    llm_adapter: LLMAdapter,
    question: str,
    intent: QuestionIntent,
    artifact: dict[str, Any],
    conversation_state: dict[str, Any] | None = None,
    complexity: Complexity = "moderate",
) -> AnswerStrategy:
    fallback = heuristic_answer_strategy(
        question=question,
        intent=intent,
        artifact=artifact,
        conversation_state=conversation_state,
    )
    if settings.llm_provider == "disabled":
        return fallback
    try:
        response = await llm_adapter.complete(
            messages=choose_answer_strategy_messages(
                question=question,
                intent=intent.to_dict(),
                artifact=artifact,
                conversation_state=conversation_state or {},
                fallback=fallback.to_dict(),
            ),
            temperature=0,
            model=model_for(settings, step="verify", complexity=complexity),
        )
        parsed = json.loads(response.content.strip())
        if isinstance(parsed, dict):
            return coerce_answer_strategy(parsed, fallback=fallback, source="llm")
    except Exception:
        return fallback
    return fallback


def heuristic_answer_strategy(
    *,
    question: str,
    intent: QuestionIntent,
    artifact: dict[str, Any] | None = None,
    conversation_state: dict[str, Any] | None = None,
) -> AnswerStrategy:
    artifact = artifact or {}
    normalized = normalize_text(question)
    safe = bool(artifact.get("safe_to_answer", True))
    uses_context = _uses_context(normalized, conversation_state)
    avoid_repeating = bool(re.search(r"\b(que mas|otro|otra|profundiza|mas detalle|algo mas)\b", normalized))
    if not safe:
        return AnswerStrategy(
            type="blocked",
            objective="Explicar por que no hay evidencia suficiente.",
            use_previous_artifact=uses_context,
            avoid_repeating=avoid_repeating,
        )
    if re.search(r"\b(que puedo hacer|acciones?|recomendaciones?|siguiente paso|como mejoro|que hago)\b", normalized):
        return AnswerStrategy(
            type="recommendation",
            objective="Convertir la evidencia disponible en acciones concretas.",
            use_previous_artifact=uses_context,
            avoid_repeating=avoid_repeating,
            reasoning_focus=("actions", "risks", "opportunities"),
        )
    explicit_compare = bool(re.search(r"\b(compara|contra|vs|versus)\b", normalized))
    if explicit_compare:
        return AnswerStrategy(
            type="comparison",
            objective="Comparar los segmentos o metricas solicitadas con evidencia.",
            use_previous_artifact=uses_context,
            avoid_repeating=avoid_repeating,
            reasoning_focus=("differences", "tradeoffs"),
        )
    if "diagnose" in intent.operations or "summarize" in intent.operations:
        return AnswerStrategy(
            type="diagnosis",
            objective="Explicar comportamientos, riesgos, oportunidades y acciones.",
            use_previous_artifact=uses_context,
            avoid_repeating=avoid_repeating,
            reasoning_focus=("facts", "patterns", "risks", "actions"),
        )
    if "rank" in intent.operations:
        return AnswerStrategy(
            type="ranking",
            objective="Ordenar los resultados segun las metricas pedidas.",
            use_previous_artifact=uses_context,
            avoid_repeating=avoid_repeating,
            reasoning_focus=tuple(intent.measures),
        )
    if uses_context:
        return AnswerStrategy(
            type="follow_up",
            objective="Responder usando el contexto analitico anterior.",
            use_previous_artifact=True,
            avoid_repeating=avoid_repeating,
            reasoning_focus=tuple(intent.measures),
        )
    if intent.entity == "sale" and len(intent.measures) <= 2:
        return AnswerStrategy(
            type="direct_metric",
            objective="Responder metricas directas con cifras concretas.",
            reasoning_focus=tuple(intent.measures),
        )
    return AnswerStrategy(
        type="explanation",
        objective="Explicar los datos disponibles sin inventar informacion.",
        use_previous_artifact=uses_context,
        avoid_repeating=avoid_repeating,
        reasoning_focus=tuple(intent.measures),
    )


def coerce_answer_strategy(
    payload: dict[str, Any],
    *,
    fallback: AnswerStrategy,
    source: str,
) -> AnswerStrategy:
    allowed: set[AnswerStrategyType] = {
        "direct_metric",
        "ranking",
        "diagnosis",
        "recommendation",
        "comparison",
        "explanation",
        "follow_up",
        "blocked",
    }
    strategy = str(payload.get("type") or payload.get("strategy") or fallback.type)
    if strategy not in allowed:
        strategy = fallback.type
    focus = payload.get("reasoning_focus")
    if isinstance(focus, str):
        focus_items = [focus]
    elif isinstance(focus, list):
        focus_items = [str(item) for item in focus if item]
    else:
        focus_items = list(fallback.reasoning_focus)
    return AnswerStrategy(
        type=strategy,  # type: ignore[arg-type]
        objective=str(payload.get("objective") or fallback.objective),
        use_previous_artifact=bool(payload.get("use_previous_artifact", fallback.use_previous_artifact)),
        avoid_repeating=bool(payload.get("avoid_repeating", fallback.avoid_repeating)),
        reasoning_focus=tuple(focus_items),
        confidence=float(payload.get("confidence", fallback.confidence) or fallback.confidence),
        source=source,
    )


def _uses_context(normalized_question: str, conversation_state: dict[str, Any] | None) -> bool:
    if not conversation_state or conversation_state.get("source") == "none":
        return False
    return bool(
        re.search(
            r"\b(esto|eso|ese|esa|esos|esas|anterior|lo anterior|que mas|profundiza|tienen|tiene|hacer con|acciones?|recomendaciones?)\b",
            normalized_question,
        )
    )
