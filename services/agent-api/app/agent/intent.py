from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.config import Settings
from app.llm.base import LLMAdapter, LLMMessage
from app.llm.model_router import model_for

Entity = str
Grain = str
Answerability = str

KNOWN_ENTITIES = {
    "sale",
    "order",
    "product",
    "customer",
    "menu_item",
    "financial",
    "business",
    "loyalty_transaction",
    "loyalty_balance",
    "loyalty_customer",
    "ingredient",
    "inventory",
    "purchase",
    "supplier",
    "procurement",
    "unknown",
}
KNOWN_GRAINS = {
    "period",
    "period_or_group",
    "period_or_customer",
    "product_period",
    "customer_period",
    "customer_period_summary",
    "customer_period_segment",
    "customer_risk",
    "cohort_period",
    "order",
    "daily_series",
    "business_period",
    "inventory_snapshot",
    "inventory_movement",
    "purchase_period",
    "supplier_period",
    "unknown",
}
KNOWN_OPERATIONS = {
    "aggregate",
    "rank",
    "filter",
    "compare",
    "diagnose",
    "summarize",
    "group",
    "sort",
    "limit",
    "segment",
    "list",
    "lookup",
    "calculate",
}


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
    conversation_state: dict[str, Any] | None = None,
    capability_hints: list[dict[str, Any]] | None = None,
) -> QuestionIntent:
    vocabulary = vocabulary_from_capabilities(capability_hints or [])
    fallback = resolve_contextual_intent(
        heuristic_intent(question, capability_hints=capability_hints),
        question=question,
        conversation_state=conversation_state,
        source="heuristic_context",
    )
    if settings.llm_provider == "disabled":
        return fallback

    payload = {
        "question": question,
        "conversation_messages": conversation_messages or [],
        "conversation_state": conversation_state or {},
        "fallback_intent": fallback.to_dict(),
        "allowed": {
            "entity": sorted(vocabulary["entities"]),
            "grain": sorted(vocabulary["grains"]),
            "measures": sorted(vocabulary["measures"]),
            "dimensions": sorted(vocabulary["dimensions"]),
            "operations": sorted(vocabulary["operations"]),
        },
        "capabilities": capability_hints or [],
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
            intent = coerce_intent(
                parsed,
                fallback=fallback,
                source="llm",
                allowed=vocabulary,
            )
            return resolve_contextual_intent(
                intent,
                question=question,
                conversation_state=conversation_state,
                source="llm_context",
            )
    except Exception:
        pass
    return fallback


def heuristic_intent(
    question: str,
    capability_hints: list[dict[str, Any]] | None = None,
) -> QuestionIntent:
    _ = capability_hints
    normalized = normalize_text(question)
    today = datetime.now(ZoneInfo("America/Bogota")).date()
    time_range = infer_time_range(normalized, today=today)
    entity: Entity = "sale"
    grain: Grain = "period"
    measures: list[str] = []
    dimensions: list[str] = []
    operations: list[str] = []

    if re.search(
        r"\b(negocio|comportamientos?|analisis profundo|diagnostico|salud del negocio|que puedo identificar)\b",
        normalized,
    ):
        entity = "business"
        grain = "business_period"
        measures.extend(
            [
                "total_sales",
                "avg_ticket",
                "order_count",
                "quantity_sold",
                "revenue",
                "margin",
                "total_spent",
                "retention_pct",
            ]
        )
        dimensions.extend(["date", "product", "customer", "cohort"])
        operations.extend(["summarize", "diagnose", "compare", "rank"])
    elif re.search(
        r"\b(inventario|stock|existencias?|insumos?|ingredientes?|bajo stock|vencimientos?)\b",
        normalized,
    ):
        entity = "ingredient"
        grain = "inventory_snapshot"
        dimensions.append("ingredient")
        measures.extend(["current_stock", "minimum_stock"])
        operations.extend(["rank", "filter"])
        if re.search(r"\b(movimientos?|consumo|consumio|consumieron|salidas?|entradas?)\b", normalized):
            grain = "inventory_movement"
            measures.extend(["net_quantity", "movement_count"])
    elif re.search(r"\b(abastecimiento|compras?|comprar|proveedor(?:es)?|supplier|purchase)\b", normalized):
        if re.search(r"\b(proveedor(?:es)?|supplier)\b", normalized):
            entity = "supplier"
            grain = "supplier_period"
            dimensions.append("supplier")
        else:
            entity = "purchase"
            grain = "purchase_period"
        dimensions.append("ingredient")
        measures.extend(["quantity_purchased", "total_cost", "avg_unit_cost", "purchase_count"])
        operations.extend(["rank", "filter"])
    elif re.search(r"\b(waros?|puntos?|redencion|redenciones|fidelidad|loyalty)\b", normalized):
        entity = "loyalty_transaction"
        grain = "period_or_customer"
        measures.append("total_issued")
        if re.search(r"\b(redim|redencion|redenciones|usados?)\b", normalized):
            measures.append("total_redeemed")
        if re.search(r"\b(tasa|porcentaje|rate)\b", normalized):
            measures.append("redemption_rate_pct")
        if re.search(r"\b(clientes?|customers?)\b", normalized):
            dimensions.append("customer")
            operations.append("rank")
    elif re.search(r"\b(cohortes?|cohorts?|retencion por cohortes?)\b", normalized):
        entity = "customer"
        grain = "cohort_period"
        dimensions.append("cohort")
        measures.extend(["retention_pct", "cohort_size"])
        operations.extend(["aggregate", "summarize"])
    elif re.search(r"\b(rfm|segmentacion|segmenta|champions?|loyal|hibernating|lost)\b", normalized):
        entity = "customer"
        grain = "customer_period_segment"
        dimensions.append("segment")
        measures.extend(["r_score", "f_score", "m_score", "total_spent", "order_count"])
        operations.extend(["segment", "summarize"])
    elif re.search(r"\b(churn|abandono|riesgo|no han vuelto|no ha vuelto|silenciosos?)\b", normalized):
        entity = "customer"
        grain = "customer_risk"
        dimensions.append("customer")
        measures.extend(["risk_score", "days_since_last_order", "lifetime_value"])
        operations.extend(["rank", "diagnose"])
    elif re.search(r"\b(clientes?|customers?)\b", normalized):
        entity = "customer"
        grain = "customer_period"
        dimensions.append("customer")
    if entity not in {"business", "ingredient", "inventory", "purchase", "supplier", "procurement"} and re.search(r"\b(productos?|items?|platos?|menu)\b", normalized):
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
    if entity in {"ingredient", "inventory"} and re.search(r"\b(stock|existencias?|inventario|bajo)\b", normalized):
        measures.extend(["current_stock", "minimum_stock"])
    if entity in {"purchase", "supplier", "procurement"} and re.search(r"\b(precio|costo|costos?|subieron?|compras?|comprar)\b", normalized):
        measures.extend(["avg_unit_cost", "total_cost"])
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
    if entity in {"ingredient", "inventory"} and not measures:
        measures.extend(["current_stock", "minimum_stock"])
    if entity in {"purchase", "supplier", "procurement"} and not measures:
        measures.extend(["total_cost", "purchase_count"])
    if entity == "business":
        requires_cross_tool = True

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
    if re.search(r"\b(ultimo ano|ultimos 365 dias|ultimo año|ultimos doce meses|ultimos 12 meses|del ultimo ano|del ultimo año)\b", normalized):
        start = today - timedelta(days=365)
        return TimeRange(start.isoformat(), today.isoformat(), label="ultimo año")
    if re.search(r"\b(este mes|presente mes|mes actual|del mes)\b", normalized):
        first = today.replace(day=1)
        return TimeRange(first.isoformat(), today.isoformat(), label="este mes")
    if re.search(r"\b(hoy)\b", normalized):
        return TimeRange(today.isoformat(), today.isoformat(), label="hoy")
    return TimeRange(None, None, label="")


def resolve_contextual_intent(
    intent: QuestionIntent,
    *,
    question: str,
    conversation_state: dict[str, Any] | None,
    source: str | None = None,
) -> QuestionIntent:
    if not isinstance(conversation_state, dict) or not conversation_state:
        return intent
    if not _is_contextual_question(question):
        return intent

    active_entity = _string_or_none(conversation_state.get("active_entity"))
    active_grain = _string_or_none(conversation_state.get("active_grain"))
    active_period = conversation_state.get("active_period")
    active_measures = _string_list(conversation_state.get("active_measures"))
    active_dimensions = _string_list(conversation_state.get("active_dimensions"))

    entity = intent.entity
    grain = intent.grain
    if active_entity and _intent_has_weak_entity_signal(intent, question):
        entity = active_entity
        grain = active_grain or intent.grain

    time_range = intent.time_range
    if not time_range.date_from and isinstance(active_period, dict):
        date_from = _string_or_none(active_period.get("date_from"))
        date_to = _string_or_none(active_period.get("date_to"))
        if date_from or date_to:
            time_range = TimeRange(
                date_from=date_from,
                date_to=date_to,
                timezone=str(active_period.get("timezone") or time_range.timezone),
                label=str(active_period.get("label") or time_range.label or "periodo anterior"),
            )

    measures = list(intent.measures)
    if not measures and active_measures:
        measures = active_measures
    if entity == active_entity:
        dimensions = _dedupe([*intent.dimensions, *active_dimensions])
    else:
        dimensions = list(intent.dimensions)

    return QuestionIntent(
        entity=entity,
        grain=grain,
        measures=tuple(normalize_measures_for_entity(entity, _dedupe(measures))),
        dimensions=tuple(dimensions),
        operations=intent.operations,
        time_range=time_range,
        answer_goal=intent.answer_goal,
        requires_cross_tool=intent.requires_cross_tool or entity == "business",
        confidence=intent.confidence,
        ambiguities=intent.ambiguities,
        source=source or intent.source,
    )


def coerce_intent(
    payload: dict[str, Any],
    *,
    fallback: QuestionIntent,
    source: str,
    allowed: dict[str, set[str]] | None = None,
) -> QuestionIntent:
    allowed = allowed or vocabulary_from_capabilities([])
    entity = str(payload.get("entity") or fallback.entity)
    grain = str(payload.get("grain") or fallback.grain)
    if _is_specific_domain_intent(fallback) and (
        entity != fallback.entity or grain != fallback.grain
    ):
        entity = fallback.entity
        grain = fallback.grain
    if entity not in allowed["entities"]:
        entity = fallback.entity
    if grain not in allowed["grains"]:
        grain = fallback.grain

    time_payload = payload.get("time_range") if isinstance(payload.get("time_range"), dict) else {}
    time_range = TimeRange(
        date_from=str(time_payload.get("date_from") or fallback.time_range.date_from or "") or None,
        date_to=str(time_payload.get("date_to") or fallback.time_range.date_to or "") or None,
        timezone=str(time_payload.get("timezone") or fallback.time_range.timezone),
        label=str(time_payload.get("label") or fallback.time_range.label),
    )
    payload_measures = _string_list(payload.get("measures"))
    payload_dimensions = _string_list(payload.get("dimensions"))
    payload_operations = _string_list(payload.get("operations"))
    if _is_specific_domain_intent(fallback):
        raw_measures = [*fallback.measures, *payload_measures]
        raw_dimensions = [*fallback.dimensions, *payload_dimensions]
        raw_operations = [*fallback.operations, *payload_operations]
    else:
        raw_measures = payload_measures or list(fallback.measures)
        raw_dimensions = payload_dimensions or list(fallback.dimensions)
        raw_operations = payload_operations or list(fallback.operations)
    measures = tuple(normalize_measures_for_entity(entity, _dedupe(raw_measures)))
    dimensions = tuple(_dedupe(raw_dimensions))
    operations = tuple(
        item
        for item in _dedupe(raw_operations)
        if item in allowed["operations"]
    )
    return QuestionIntent(
        entity=entity,
        grain=grain,
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


def _is_specific_domain_intent(intent: QuestionIntent) -> bool:
    return intent.entity in {"loyalty_transaction", "loyalty_balance", "loyalty_customer"} or intent.grain in {
        "cohort_period",
        "customer_period_segment",
        "customer_risk",
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_contextual_question(question: str) -> bool:
    normalized = normalize_text(question)
    return bool(
        re.search(
            r"\b(esto|eso|ese|esa|esos|esas|anterior|lo anterior|que mas|profundiza|detall|tienen|tiene|hacer con|que puedo hacer|acciones?|recomendaciones?)\b",
            normalized,
        )
    )


def _intent_has_weak_entity_signal(intent: QuestionIntent, question: str) -> bool:
    normalized = normalize_text(question)
    explicit_domains = (
        "cliente",
        "clientes",
        "producto",
        "productos",
        "venta",
        "ventas",
        "waros",
        "puntos",
        "cohorte",
        "cohortes",
        "negocio",
        "inventario",
        "stock",
        "insumo",
        "ingrediente",
        "compra",
        "proveedor",
        "abastecimiento",
    )
    return intent.entity in {"sale", "unknown"} and not any(token in normalized for token in explicit_domains)


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
        "total_issued": "total_issued",
        "issued": "total_issued",
        "earned": "total_issued",
        "total_earned": "total_issued",
        "total_redeemed": "total_redeemed",
        "redeemed": "total_redeemed",
        "redemption_rate_pct": "redemption_rate_pct",
        "retention_pct": "retention_pct",
        "retention": "retention_pct",
        "cohort_size": "cohort_size",
        "risk_score": "risk_score",
        "days_since_last_order": "days_since_last_order",
        "lifetime_value": "lifetime_value",
        "r_score": "r_score",
        "f_score": "f_score",
        "m_score": "m_score",
        "stock": "current_stock",
        "current_stock": "current_stock",
        "minimum_stock": "minimum_stock",
        "low_stock": "minimum_stock",
        "quantity_purchased": "quantity_purchased",
        "purchase_count": "purchase_count",
        "avg_unit_cost": "avg_unit_cost",
        "unit_cost": "avg_unit_cost",
        "total_cost": "total_cost",
        "movement_count": "movement_count",
        "net_quantity": "net_quantity",
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


def vocabulary_from_capabilities(capabilities: list[dict[str, Any]]) -> dict[str, set[str]]:
    entities = set(KNOWN_ENTITIES)
    grains = set(KNOWN_GRAINS)
    measures: set[str] = set()
    dimensions: set[str] = set()
    operations = set(KNOWN_OPERATIONS)
    for capability in capabilities:
        if not isinstance(capability, dict):
            continue
        entity = capability.get("entity")
        grain = capability.get("grain")
        if entity:
            entities.add(str(entity))
        if grain:
            grains.add(str(grain))
        measures.update(normalize_measure(str(item)) for item in _string_list(capability.get("measures")))
        dimensions.update(normalize_measure(str(item)) for item in _string_list(capability.get("dimensions")))
        operations.update(normalize_measure(str(item)) for item in _string_list(capability.get("supported_operations")))
    return {
        "entities": entities,
        "grains": grains,
        "measures": measures,
        "dimensions": dimensions,
        "operations": operations,
    }
