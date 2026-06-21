from __future__ import annotations

import json
import re
from typing import Any, Literal

from app.llm.base import LLMAdapter, LLMMessage
from app.llm.model_router import Complexity, model_for
from app.config import Settings

ComplexityLevel = Complexity


def heuristic_complexity(question: str) -> ComplexityLevel:
    normalized = question.lower().strip()
    if len(normalized) < 60 and not re.search(
        r"\b(por que|porque|recomienda|compara|analiza|diagnostico|explica|combina|"
        r"cruza|relacion|impacto|estrategia|optimiza)\b",
        normalized,
    ):
        entity_hits = len(
            re.findall(
                r"\b(ventas?|clientes?|productos?|margen|food\s*cost|recetas?|inventario|"
                r"proveedores?|ordenes?|menu)\b",
                normalized,
            )
        )
        if entity_hits <= 1:
            return "simple"
    if re.search(
        r"\b(por que|porque|recomienda|compara|analiza|diagnostico|explica|combina|"
        r"cruza|relacion|impacto|estrategia|optimiza|que deberia)\b",
        normalized,
    ) or len(normalized) > 180:
        return "complex"
    return "moderate"


async def classify_complexity(
    *,
    settings: Settings,
    llm_adapter: LLMAdapter,
    question: str,
    conversation_messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    fallback = heuristic_complexity(question)
    if settings.llm_provider == "disabled":
        return {
            "complexity": fallback,
            "reason": "heuristic_llm_disabled",
            "estimated_tool_calls": 1 if fallback == "simple" else 3,
            "source": "heuristic",
        }

    payload = {
        "question": question,
        "conversation_messages": conversation_messages or [],
    }
    messages = [
        LLMMessage(
            role="system",
            content=(
                "Clasifica la complejidad de la pregunta del usuario para un agente de datos WARO. "
                "Devuelve SOLO JSON valido: "
                '{"complexity":"simple|moderate|complex","reason":"...",'
                '"estimated_tool_calls":1}. '
                "simple = una metrica o ranking directo; moderate = un dominio con filtros; "
                "complex = multi-dominio, diagnostico, recomendaciones o comparaciones."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
    ]
    try:
        response = await llm_adapter.complete(
            messages=messages,
            temperature=0,
            model=model_for(settings, step="classify", complexity=fallback),
        )
        parsed = json.loads(response.content.strip())
        complexity = str(parsed.get("complexity", fallback))
        if complexity not in {"simple", "moderate", "complex"}:
            complexity = fallback
        return {
            "complexity": complexity,
            "reason": str(parsed.get("reason", "")),
            "estimated_tool_calls": int(parsed.get("estimated_tool_calls", 1) or 1),
            "source": "llm",
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "complexity": fallback,
            "reason": "heuristic_parse_fallback",
            "estimated_tool_calls": 1 if fallback == "simple" else 3,
            "source": "heuristic",
        }
