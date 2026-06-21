import re
import unicodedata
from dataclasses import dataclass
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
        "capabilities": dict(spec.capabilities),
        "arguments_schema": schema,
    }


def tool_catalog() -> list[dict[str, Any]]:
    return [tool_metadata(spec) for spec in TOOL_SPECS.values()]


@dataclass(frozen=True)
class ToolDiscoveryMatch:
    spec: ToolSpec
    score: int
    reasons: tuple[str, ...]
    available: bool
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.spec.name,
            "domain": self.spec.domain,
            "scope": self.spec.scope,
            "score": self.score,
            "reasons": list(self.reasons),
            "description": self.spec.description,
            "default_fields": list(self.spec.default_fields),
        }
        if self.rejected_reason:
            payload["rejected_reason"] = self.rejected_reason
        return payload


def discover_tools(
    question: str,
    *,
    preferred_domain: str | None = None,
    scopes: tuple[str, ...] = (),
    limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Return auditable tool discovery results for the current request.

    This is intentionally lightweight: it exposes enough signal for traces and
    persisted planner steps without putting every full schema into the run log.
    """

    matches = _rank_tools(
        question,
        preferred_domain=preferred_domain,
        scopes=scopes,
    )
    available = [match.to_dict() for match in matches if match.available][:limit]
    rejected = [match.to_dict() for match in matches if not match.available][:limit]
    return {"available": available, "rejected": rejected}


def candidate_tools(
    question: str,
    *,
    preferred_domain: str | None = None,
    scopes: tuple[str, ...] = (),
    limit: int = 5,
) -> list[ToolSpec]:
    matches = _rank_tools(
        question,
        preferred_domain=preferred_domain,
        scopes=scopes,
    )
    return [match.spec for match in matches if match.available][:limit]


def _rank_tools(
    question: str,
    *,
    preferred_domain: str | None,
    scopes: tuple[str, ...],
) -> list[ToolDiscoveryMatch]:
    normalized = normalize_query(question)
    scope_set = set(scopes)
    scored: list[tuple[int, str, ToolDiscoveryMatch]] = []
    for spec in TOOL_SPECS.values():
        score, reasons = _score_tool(spec, normalized, preferred_domain=preferred_domain)
        if score <= 0:
            continue
        available = not scope_set or spec.scope in scope_set
        rejected_reason = None if available else f"missing_scope:{spec.scope}"
        scored.append(
            (
                score,
                spec.name,
                ToolDiscoveryMatch(
                    spec=spec,
                    score=score,
                    reasons=tuple(reasons),
                    available=available,
                    rejected_reason=rejected_reason,
                ),
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [match for _, _, match in scored]


def _score_tool(
    spec: ToolSpec,
    normalized_question: str,
    *,
    preferred_domain: str | None,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if preferred_domain and spec.domain == preferred_domain:
        score += 4
        reasons.append(f"preferred_domain:{preferred_domain}")
    capability_terms = _capability_terms(spec.capabilities)
    haystack = " ".join(
        (
            spec.name,
            spec.description,
            *spec.tags,
            *spec.examples,
            *capability_terms,
        )
    )
    matched_tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9_]+", normalize_query(haystack)):
        if (
            len(token) >= 3
            and token not in matched_tokens
            and re.search(rf"\b{re.escape(token)}\b", normalized_question)
        ):
            matched_tokens.add(token)
            score += 1
    if matched_tokens:
        reasons.append("keyword:" + ",".join(sorted(matched_tokens)[:8]))
    if spec.name == "waro.sales.metrics" and re.search(
        r"\b(ventas?|ingresos?|ordenes?|ticket|vendimos|vendido)\b",
        normalized_question,
    ):
        score += 8
        reasons.append("sales_metrics_signal")
    if spec.name == "waro.financial.products" and re.search(
        r"\b(productos?|margen|rentabilidad|ingresos?|costo|cantidad|financier[ao]s?|utilidad|ganancia)\b",
        normalized_question,
    ):
        score += 6
        reasons.append("financial_product_signal")
    if spec.name == "waro.menu.products" and re.search(
        r"\b(menu|productos?|platos?|disponibles?|precio|precios?)\b",
        normalized_question,
    ):
        score += 4
        reasons.append("menu_product_signal")
    if spec.name == "waro.analytics.food_cost" and re.search(
        r"\b(food cost|margen|rentabilidad|costo)\b",
        normalized_question,
    ):
        score += 6
        reasons.append("food_cost_signal")
    if spec.name == "waro.analytics.menu" and re.search(
        r"\b(menu|portafolio|estrella|bajo rendimiento|performance|productos?)\b",
        normalized_question,
    ):
        score += 5
        reasons.append("menu_analytics_signal")
    if spec.name == "waro.analytics.alerts" and re.search(
        r"\b(alertas?|advertencias?|inventario|agotad[oa]s?|riesgo)\b",
        normalized_question,
    ):
        score += 5
        reasons.append("analytics_alerts_signal")
    if spec.name == "waro.analytics.data_quality" and re.search(
        r"\b(calidad de datos|datos|anomalias?|inconsistencias?|validacion)\b",
        normalized_question,
    ):
        score += 5
        reasons.append("data_quality_signal")
    if spec.name.startswith("waro.customers") and re.search(
        r"\b(clientes?|compradores?|retencion|frecuencia|fidelidad|churn|recencia)\b",
        normalized_question,
    ):
        score += 6
        reasons.append("customer_signal")
    return score, reasons


def _capability_terms(value: Any) -> list[str]:
    terms: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            terms.append(str(key))
            terms.extend(_capability_terms(item))
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            terms.extend(_capability_terms(item))
    elif value is not None:
        terms.append(str(value))
    return terms
