import re
import unicodedata
from typing import Any

from app.tools.allowlist import TOOL_SPECS, ToolSpec


def normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def tool_metadata(spec: ToolSpec) -> dict[str, Any]:
    schema = spec.args_model.model_json_schema(by_alias=True)
    return {
        "name": spec.name,
        "command": list(spec.command),
        "domain": spec.domain,
        "scope": spec.scope,
        "description": spec.description,
        "tags": list(spec.tags),
        "examples": list(spec.examples),
        "default_fields": list(spec.default_fields),
        "allowed_fields": sorted(spec.allowed_fields),
        "arguments_schema": schema,
    }


def tool_catalog() -> list[dict[str, Any]]:
    return [tool_metadata(spec) for spec in TOOL_SPECS.values()]


def candidate_tools(
    question: str,
    *,
    preferred_domain: str | None = None,
    scopes: tuple[str, ...] = (),
    limit: int = 5,
) -> list[ToolSpec]:
    normalized = normalize_query(question)
    scope_set = set(scopes)
    scored: list[tuple[int, str, ToolSpec]] = []
    for spec in TOOL_SPECS.values():
        if scope_set and spec.scope not in scope_set:
            continue
        score = 0
        if preferred_domain and spec.domain == preferred_domain:
            score += 4
        haystack = " ".join((spec.name, spec.description, *spec.tags, *spec.examples))
        for token in re.findall(r"[a-z0-9_]+", normalize_query(haystack)):
            if len(token) >= 3 and re.search(rf"\b{re.escape(token)}\b", normalized):
                score += 1
        if spec.name == "waro.sales.metrics" and re.search(
            r"\b(ventas?|ingresos?|ordenes?|ticket|vendimos|vendido)\b",
            normalized,
        ):
            score += 8
        if spec.name == "waro.financial.products" and re.search(
            r"\b(productos?|margen|rentabilidad|ingresos?|costo|cantidad)\b",
            normalized,
        ):
            score += 6
        if spec.name == "waro.menu.products" and re.search(
            r"\b(menu|productos?|disponibles?|precio)\b",
            normalized,
        ):
            score += 4
        if spec.name == "waro.analytics.food_cost" and re.search(
            r"\b(food cost|margen|rentabilidad|costo)\b",
            normalized,
        ):
            score += 6
        if score > 0:
            scored.append((score, spec.name, spec))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [spec for _, _, spec in scored[:limit]]
