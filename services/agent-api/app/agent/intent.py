from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from app.config import Settings
from app.llm.base import LLMAdapter, LLMMessage
from app.llm.model_router import model_for

Entity = Literal["sale", "order", "product", "customer", "menu_item", "financial", "unknown"]
Grain = Literal["period", "product_period", "customer_period", "order", "daily_series", "unknown"]
Answerability = Literal["answerable", "partial", "blocked"]

KNOWN_ENTITIES = {"sale", "order", "product", "customer", "menu_item", "financial", "unknown"}
KNOWN_GRAINS = {"period", "product_period", "customer_period", "order", "daily_series", "unknown"}
KNOWN_OPERATIONS = {"aggregate", "rank", "filter", "compare", "diagnose", "summarize", "group", "sort", "limit"}


@dataclass(frozen=True)
class TimeRange:
    date_from: str | None
    date_to: str | None
    timezone: str = "America/Bogota"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionIntent:
    entity: Entity
    grain: Grain
    measures: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    time_range: TimeRange = field(default_factory=lambda: TimeRange(None, None))
    answer_goal: str = ""
    requires_cross_tool: bool = False
    confidence: float = 0.5
    ambiguities: tuple[str, ...] = ()
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["time_range"] = self.time_range.to_dict()
        return payload


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


async def parse_question_intent(
    *,
    settings: Settings,
    llm_adapter: LLMAdapter,
    question: str,
    conversation_messages: list[dict[str, str]] | None = None,
) -> QuestionIntent:
    fallback = heuristic_intent(question)
    if settings.llm_provider == "disabled":
        return fallback

    payload = {
        "question": question,
        "conversation_messages": conversation_messages or [],
        "fallback_intent": fallback.to_dict(),
        "allowed": {
            "entity": sorted(KNOWN_ENTITIES),
            "grain": sorted(KNOWN_GRAINS),
            "operations": sorted(KNOWN_OPERATIONS),
        },
    }
    messages = [
        LLMMessage(
            role="system",
            content=(
                "Extrae la intencion analitica de una pregunta WARO. No elijas tools. "
                "Devuelve SOLO JSON valido con: entity, grain, measures, dimensions, "
                "operations, time_range{date_from,date_to,timezone,label}, answer_goal, "
                "requires_cross_tool, confidence, ambiguities. Usa nombres canonicos en ingles."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]
    try:
        response = await llm_adapter.complete(
            messages=messages,
            temperature=0,
            model=model_for(settings, step="classify", complexity="moderate"),
        )
        parsed = json.loads(response.content.strip())
        if isinstance(parsed, dict):
            return coerce_intent(parsed, fallback=fallback, source="llm")
    except Exception:
        pass
    return fallback


def heuristic_intent(question: str) -> QuestionIntent:
    normalized = normalize_text(question)
    today = datetime.now(ZoneInfo("America/Bogota")).date()
    time_range = infer_time_range(normalized, today=today)
    entity: Entity = "sale"
    grain: Grain = "period"
    measures: list[str] = []
    dimensions: list[str] = []
    operations: list[str] = []

    if re.search(r"\b(clientes?|customers?)\b", normalized):
        entity = "customer"
        grain = "customer_period"
        dimensions.append("customer")
    if re.search(r"\b(productos?|items?|platos?|menu)\b", normalized):
        entity = "product"
        grain = "product_period"
        dimensions.append("product")
    if re.search(r"\b(ordenes?|pedidos?)\b", normalized) and entity == "sale":
        entity = "order"
        grain = "order"

    if re.search(r"\b(vendi|vendio|ventas?|ingresos?|factur|compraron|comprado)\b", normalized):
        if entity == "customer":
            measures.append("total_spent")
        else:
            measures.append("total_sales" if entity in {"sale", "order"} else "revenue")
    if re.search(r"\b(ticket promedio|promedio|avg ticket)\b", normalized):
        measures.append("avg_ticket")
    if re.search(r"\b(frecuentes?|frecuencia|ordenes?|pedidos?)\b", normalized):
        measures.append("order_count")
    if re.search(r"\b(mas vendidos?|vendieron mucho|cantidad|unidades|quantity)\b", normalized):
        measures.append("quantity_sold")
    if re.search(r"\b(margen|margin|rentabilidad|profit|utilidad)\b", normalized):
        measures.append("margin")
    if re.search(r"\b(costo|cost|food cost)\b", normalized):
        measures.append("cost")
    if re.search(r"\b(mayor valor|mas dinero|compraron|comprado|gasto|spent)\b", normalized):
        measures.append("total_spent" if entity == "customer" else "revenue")

    if re.search(r"\b(por hora|hora)\b", normalized):
        dimensions.append("hour")
        operations.append("group")
    if re.search(r"\b(por dia|diario|fecha|tendencia)\b", normalized):
        dimensions.append("date")
        operations.append("group")

    if re.search(r"\b(top|ranking|mejores?|peores?|mas|mayor|menor|bajo|alto)\b", normalized):
        operations.append("rank")
    if re.search(r"\b(bajo|menor|filtra|con)\b", normalized):
        operations.append("filter")
    if re.search(r"\b(compara|contra|vs|versus)\b", normalized):
        operations.append("compare")
    if not operations:
        operations.append("aggregate")

    if entity == "product" and "margin" in measures and any(
        measure in measures for measure in ("quantity_sold", "revenue", "total_sales")
    ):
        requires_cross_tool = True
    elif entity == "customer" and len(set(measures).intersection({"order_count", "total_spent", "avg_ticket"})) > 1:
        requires_cross_tool = True
    else:
        requires_cross_tool = False

    if entity == "sale" and not measures:
        measures.append("total_sales")
    if entity == "product" and not measures:
        measures.extend(["quantity_sold", "revenue"])
    if entity == "customer" and not measures:
        measures.extend(["total_spent", "order_count"])

    return QuestionIntent(
        entity=entity,
        grain=grain,
        measures=tuple(normalize_measures_for_entity(entity, _dedupe(measures))),
        dimensions=tuple(_dedupe(dimensions)),
        operations=tuple(_dedupe(operations)),
        time_range=time_range,
        answer_goal=question.strip(),
        requires_cross_tool=requires_cross_tool,
        confidence=0.72,
        ambiguities=(),
        source="heuristic",
    )


def infer_time_range(normalized: str, *, today: date) -> TimeRange:
    if "ayer" in normalized:
        day = today - timedelta(days=1)
        return TimeRange(day.isoformat(), day.isoformat(), label="ayer")
    if re.search(r"\b(este mes|presente mes|mes actual|del mes)\b", normalized):
        first = today.replace(day=1)
        return TimeRange(first.isoformat(), today.isoformat(), label="este mes")
    if re.search(r"\b(hoy)\b", normalized):
        return TimeRange(today.isoformat(), today.isoformat(), label="hoy")
    return TimeRange(None, None, label="")


def coerce_intent(payload: dict[str, Any], *, fallback: QuestionIntent, source: str) -> QuestionIntent:
    entity = str(payload.get("entity") or fallback.entity)
    grain = str(payload.get("grain") or fallback.grain)
    if entity not in KNOWN_ENTITIES:
        entity = fallback.entity
    if grain not in KNOWN_GRAINS:
        grain = fallback.grain

    time_payload = payload.get("time_range") if isinstance(payload.get("time_range"), dict) else {}
    time_range = TimeRange(
        date_from=str(time_payload.get("date_from") or fallback.time_range.date_from or "") or None,
        date_to=str(time_payload.get("date_to") or fallback.time_range.date_to or "") or None,
        timezone=str(time_payload.get("timezone") or fallback.time_range.timezone),
        label=str(time_payload.get("label") or fallback.time_range.label),
    )
    measures = tuple(normalize_measures_for_entity(entity, _dedupe([*fallback.measures, *_string_list(payload.get("measures"))])))
    dimensions = tuple(_dedupe([*fallback.dimensions, *_string_list(payload.get("dimensions"))]))
    operations = tuple(
        item for item in _dedupe([*fallback.operations, *_string_list(payload.get("operations"))]) if item in KNOWN_OPERATIONS
    )
    return QuestionIntent(
        entity=entity,  # type: ignore[arg-type]
        grain=grain,  # type: ignore[arg-type]
        measures=measures,
        dimensions=dimensions,
        operations=operations or fallback.operations,
        time_range=time_range,
        answer_goal=str(payload.get("answer_goal") or fallback.answer_goal),
        requires_cross_tool=bool(payload.get("requires_cross_tool", fallback.requires_cross_tool)),
        confidence=float(payload.get("confidence", fallback.confidence) or fallback.confidence),
        ambiguities=tuple(_string_list(payload.get("ambiguities"))),
        source=source,
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = normalize_measure(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def normalize_measure(value: str) -> str:
    normalized = normalize_text(str(value)).replace("-", "_").replace(" ", "_")
    aliases = {
        "totalsales": "total_sales",
        "total_sales": "total_sales",
        "totalamount": "total_sales",
        "sales": "total_sales",
        "revenue": "revenue",
        "total_revenue": "revenue",
        "ingresos": "revenue",
        "avg_ticket": "avg_ticket",
        "avgticket": "avg_ticket",
        "ticket": "avg_ticket",
        "totalorders": "order_count",
        "ordercount": "order_count",
        "order_count": "order_count",
        "orders": "order_count",
        "quantity": "quantity_sold",
        "quantity_sold": "quantity_sold",
        "units": "quantity_sold",
        "margin": "margin",
        "profit_margin_pct": "margin",
        "profit": "margin",
        "total_profit": "margin",
        "cost": "cost",
        "estimated_cost": "cost",
        "total_spent": "total_spent",
        "spent": "total_spent",
    }
    return aliases.get(normalized, normalized)


def normalize_measures_for_entity(entity: str, measures: list[str]) -> list[str]:
    normalized: list[str] = []
    for measure in measures:
        value = measure
        if entity == "customer" and value in {"revenue", "total_sales"}:
            value = "total_spent"
        if entity == "product" and value == "total_sales":
            value = "revenue"
        if value not in normalized:
            normalized.append(value)
    return normalized
