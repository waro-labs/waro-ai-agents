from __future__ import annotations

import json
from typing import Any

from app.agent.intent import QuestionIntent
from app.agent.plan import ToolPlan
from app.agent.profiles import first_value, normalized_any, profile_for_intent, rows_for_group
from app.tools.sanitize import sanitize_value


def build_evidence_artifact(
    *,
    question: str,
    intent: QuestionIntent,
    plan: ToolPlan,
    observations: list[dict[str, Any]],
    conversation_messages: list[dict[str, str]] | None = None,
    conversation_state: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    conversation_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = profile_for_intent(intent)
    tables = [_table_from_observation(observation) for observation in observations]
    if conversation_plan and (
        conversation_plan.get("reuse_previous_artifact")
        or conversation_plan.get("tool_policy") == "merge"
    ):
        previous_table = _table_from_previous_artifact(conversation_state)
        if previous_table is not None:
            tables.insert(0, previous_table)
    metrics = _merge_metrics(tables)
    ranked_rows = _ranked_rows(intent, tables)
    analysis = _analysis_from_evidence(intent, tables, metrics, ranked_rows)
    failed = [obs for obs in observations if obs.get("status") != "succeeded"]
    query_metadata = _query_metadata_from_tables(tables)
    if not plan.valid:
        answerability = "blocked"
        blocked_reason = plan.blocked_reason or "invalid_plan"
    elif _is_meta_conversation_plan(conversation_plan):
        answerability = "answerable"
        blocked_reason = None
    elif not tables:
        answerability = "blocked"
        blocked_reason = "no_tool_results"
    elif observations and len(failed) == len(observations):
        answerability = "blocked"
        blocked_reason = "all_tools_failed"
    elif failed:
        answerability = "partial"
        blocked_reason = None
    else:
        answerability = "answerable"
        blocked_reason = None

    safe_to_answer = answerability in {"answerable", "partial"}
    artifact = {
        "intent": "agent_query",
        "agent_mode": True,
        "agent_engine_version": "intent-capability-v1",
        "agent_profile": (
            {"id": profile.id, "description": profile.description}
            if profile is not None
            else None
        ),
        "question": question,
        "question_intent": intent.to_dict(),
        "conversation_state": conversation_state or {},
        "advisor_state": (
            conversation_state.get("advisor_state")
            if isinstance(conversation_state, dict)
            and isinstance(conversation_state.get("advisor_state"), dict)
            else {}
        ),
        "context_usage": _context_usage(question=question, conversation_state=conversation_state),
        "conversation_plan": conversation_plan or {},
        "classification": classification or {},
        "plan": plan.to_dict(),
        "tool_results": observations,
        "observations": observations,
        "tables": tables,
        "metrics": metrics,
        "query_metadata": query_metadata,
        "ranked_rows": ranked_rows,
        "analysis": analysis,
        "insights": [
            *_string_items(analysis.get("facts")),
            *_string_items(analysis.get("patterns")),
            *_string_items(analysis.get("risks")),
            *_string_items(analysis.get("opportunities")),
        ],
        "recommended_actions": _string_items(analysis.get("recommended_actions")),
        "open_questions": _open_questions(intent, analysis),
        "evidence": _evidence(tables),
        "limitations": _limitations(plan, failed, tables),
        "answerability": answerability,
        "blocked_reason": blocked_reason,
        "safe_to_answer": safe_to_answer,
        "error_message": None if safe_to_answer else _blocked_message(blocked_reason, plan),
        "conversation_messages": conversation_messages or [],
        "response_contract": {
            "safe_to_answer": safe_to_answer,
            "data_status": "ok" if answerability == "answerable" else answerability,
            "error_message": None if safe_to_answer else _blocked_message(blocked_reason, plan),
            "missing_required_data": list(plan.missing_coverage),
            "request_kind": f"{intent.entity}_{'_'.join(intent.operations) or 'query'}",
        },
    }
    return sanitize_value(artifact)


def deterministic_evidence_summary(artifact: dict[str, Any]) -> str:
    if not artifact.get("safe_to_answer"):
        return str(artifact.get("error_message") or "No pude responder con los datos disponibles.")
    intent = artifact.get("question_intent") if isinstance(artifact.get("question_intent"), dict) else {}
    entity = intent.get("entity")
    measures = set(intent.get("measures") or [])
    metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
    rows = artifact.get("ranked_rows") if isinstance(artifact.get("ranked_rows"), list) else []
    period = intent.get("time_range", {}).get("label") if isinstance(intent.get("time_range"), dict) else ""
    strategy_payload = artifact.get("answer_strategy") if isinstance(artifact.get("answer_strategy"), dict) else {}
    strategy = str(strategy_payload.get("type") or "")
    conversation_plan = artifact.get("conversation_plan") if isinstance(artifact.get("conversation_plan"), dict) else {}
    if _is_meta_conversation_plan(conversation_plan):
        return _meta_conversation_summary(artifact)
    if conversation_plan.get("subject") == "generic_customers":
        return _generic_customer_impact_summary(artifact, rows, period)
    product_context = entity == "product" or conversation_plan.get("preserve_dataset") == "product_profitability"
    if strategy in {"follow_up", "recommendation"} and not rows:
        rows = _previous_ranked_rows(artifact)

    if strategy == "recommendation":
        return _recommendation_summary(artifact, period)
    if strategy == "diagnosis" and product_context and rows:
        contract = conversation_plan.get("answer_contract") if isinstance(conversation_plan.get("answer_contract"), dict) else {}
        must_explain = _string_items(contract.get("must_explain"))
        if "price_signal" in must_explain or "cost_signal" in must_explain:
            return _product_cause_classification_summary(artifact, rows, period)
        return _product_diagnosis_summary(artifact, rows, period)
    if strategy == "follow_up" and product_context and rows:
        return _product_follow_up_summary(artifact, rows, period)
    if strategy == "diagnosis" and entity == "business":
        if bool(strategy_payload.get("avoid_repeating")):
            return _business_follow_up_summary(artifact, period)
        return _business_analysis_summary(artifact, period)
    if strategy == "comparison" and entity == "customer" and rows:
        return _customer_comparison_summary(rows, period)

    if entity == "business":
        return _business_analysis_summary(artifact, period)
    if entity == "sale" and ("total_sales" in measures or "avg_ticket" in measures):
        total = _fmt_money(metrics.get("total_sales"))
        avg = _fmt_money(metrics.get("avg_ticket"))
        if total and avg:
            return f"{_period_prefix(period)} vendiste {total} y el ticket promedio fue {avg}."
        if total:
            return f"{_period_prefix(period)} vendiste {total}."
    if entity == "product":
        if not rows:
            return (
                "Puedo analizar productos, pero esta consulta no trajo filas de productos para ordenar. "
                "Revisa el periodo o pideme ventas de productos, margen o rentabilidad con un rango especifico."
            )
        lines = [_product_title(measures, period)]
        for index, row in enumerate(rows[:10], start=1):
            name = row.get("name") or row.get("product_name") or row.get("id") or "Producto"
            quantity = row.get("quantity") or row.get("quantity_sold") or row.get("total_units_sold")
            revenue = _fmt_money(row.get("revenue") or row.get("total_revenue"))
            margin = row.get("margin") or row.get("margin_pct") or row.get("profit_margin_pct")
            bits = []
            if quantity is not None:
                bits.append(f"{quantity} unidades")
            if revenue:
                bits.append(f"{revenue} vendido")
            if "margin" in measures and margin is not None:
                bits.append(f"margen {_fmt_percent(margin)}")
            cost_source = row.get("cost_source")
            if cost_source:
                bits.append(f"costo {cost_source}")
            lines.append(f"{index}. {name}" + (f" ({', '.join(bits)})" if bits else ""))
        return "\n".join(lines)
    if entity == "customer":
        if intent.get("grain") == "cohort_period":
            if not rows:
                return (
                    "Puedo analizar retencion, pero esta consulta no trajo cohortes suficientes. "
                    "Prueba con un periodo mas amplio o una pregunta de clientes recurrentes."
                )
            lines = [f"Retencion por cohortes para {_period_label(period)}:"]
            for index, row in enumerate(rows[:10], start=1):
                label = row.get("cohort_label") or row.get("cohort_date") or "Cohorte"
                size = row.get("cohort_size")
                retention = row.get("retention")
                bits = []
                if size is not None:
                    bits.append(f"{size} clientes iniciales")
                if isinstance(retention, list) and retention:
                    first = retention[0] if isinstance(retention[0], dict) else {}
                    pct = first.get("pct") or first.get("retention_pct")
                    if pct is not None:
                        bits.append(f"primer periodo {pct}%")
                lines.append(f"{index}. {label}" + (f" ({', '.join(bits)})" if bits else ""))
            return "\n".join(lines)
        if intent.get("grain") == "customer_period_segment":
            if not rows:
                return (
                    "Puedo segmentar clientes, pero esta consulta no trajo clientes suficientes para RFM. "
                    "Prueba con un periodo mas amplio o con clientes por frecuencia y valor."
                )
            lines = [f"Segmentacion RFM para {_period_label(period)}:"]
            for index, row in enumerate(rows[:20], start=1):
                name = row.get("customer_name") or row.get("customer_id") or "Cliente"
                segment = row.get("segment") or "segmento sin clasificar"
                spent = _fmt_money(row.get("total_spent"))
                orders = row.get("order_count")
                bits = [str(segment)]
                if orders is not None:
                    bits.append(f"{orders} ordenes")
                if spent:
                    bits.append(f"{spent} comprado")
                lines.append(f"{index}. {name} ({', '.join(bits)})")
            return "\n".join(lines)
        if intent.get("grain") == "customer_risk":
            if not rows:
                return (
                    "Puedo revisar riesgo de abandono, pero esta consulta no trajo clientes que cumplan esos criterios. "
                    "Podemos ampliar el periodo o bajar el umbral de dias sin compra."
                )
            lines = ["Clientes en riesgo de no volver:"]
            for index, row in enumerate(rows[:20], start=1):
                name = row.get("name") or row.get("customer_id") or "Cliente"
                days = row.get("days_since_last_order")
                ltv = _fmt_money(row.get("lifetime_value"))
                risk = row.get("risk_score")
                bits = []
                if days is not None:
                    bits.append(f"{days} dias sin comprar")
                if ltv:
                    bits.append(f"LTV {ltv}")
                if risk is not None:
                    bits.append(f"riesgo {risk}")
                lines.append(f"{index}. {name}" + (f" ({', '.join(bits)})" if bits else ""))
            return "\n".join(lines)
        if not rows:
            return (
                "Puedo analizar clientes, pero esta consulta no trajo filas suficientes para construir un ranking. "
                "Prueba con clientes por ventas, frecuencia, ticket promedio o WAROS."
            )
        if "compare" in intent.get("operations", []):
            label = "Comparacion de clientes frecuentes contra mayor valor comprado"
        else:
            label = "Clientes"
        lines = [f"{label} para {_period_label(period)}:"]
        for index, row in enumerate(rows[:20], start=1):
            name = row.get("name") or row.get("customer_name") or row.get("customer_id") or "Cliente"
            spent = _fmt_money(row.get("total_spent"))
            orders = row.get("order_count")
            avg = _fmt_money(row.get("avg_ticket"))
            bits = []
            if orders is not None:
                bits.append(f"{orders} ordenes")
            if spent:
                bits.append(f"{spent} comprado")
            if avg:
                bits.append(f"ticket {avg}")
            lines.append(f"{index}. {name}" + (f" ({', '.join(bits)})" if bits else ""))
        return "\n".join(lines)
    if entity in {"ingredient", "inventory"}:
        return _inventory_summary(artifact, rows, period)
    if entity in {"purchase", "supplier", "procurement"}:
        if strategy == "recommendation" or _asks_for_purchase_recommendation(str(artifact.get("question") or "")):
            return _procurement_recommendation_summary(artifact, rows, period)
        return _purchase_summary(artifact, rows, period)
    if entity == "loyalty_transaction":
        if not rows and metrics:
            issued = metrics.get("total_issued") or metrics.get("total_earned")
            redeemed = metrics.get("total_redeemed")
            rate = metrics.get("redemption_rate_pct")
            bits = []
            if issued is not None:
                bits.append(f"WAROS emitidos: {issued}")
            if redeemed is not None:
                bits.append(f"WAROS redimidos: {redeemed}")
            if rate is not None:
                bits.append(f"tasa de redencion: {rate}%")
            return f"Analitica WAROS para {_period_label(period)}: " + ", ".join(bits)
        if not rows:
            return (
                "Puedo analizar WAROS, pero esta consulta no trajo movimientos suficientes. "
                "Prueba con WAROS generados, redimidos o balance por cliente en un periodo especifico."
            )
        lines = [f"Clientes que mas generaron WAROS para {_period_label(period)}:"]
        for index, row in enumerate(rows[:20], start=1):
            name = row.get("name") or row.get("customer_name") or row.get("customer_id") or row.get("period") or "Cliente"
            earned = row.get("total_earned") or row.get("total_issued")
            redeemed = row.get("total_redeemed")
            txs = row.get("transaction_count")
            bits = []
            if earned is not None:
                bits.append(f"{earned} WAROS generados")
            if redeemed is not None:
                bits.append(f"{redeemed} redimidos")
            if txs is not None:
                bits.append(f"{txs} transacciones")
            lines.append(f"{index}. {name}" + (f" ({', '.join(bits)})" if bits else ""))
        return "\n".join(lines)
    if rows:
        lines = [f"Resultados para {_period_label(period)}:"]
        for index, row in enumerate(rows[:10], start=1):
            name = _row_name(row, fallback=f"Fila {index}")
            bits = _row_summary_bits(row)
            lines.append(f"{index}. {name}" + (f" ({', '.join(bits)})" if bits else ""))
        return "\n".join(lines)
    return _no_evidence_summary(artifact)


def _is_meta_conversation_plan(conversation_plan: dict[str, Any] | None) -> bool:
    if not isinstance(conversation_plan, dict):
        return False
    return (
        conversation_plan.get("intent_type") in {"definition", "clarification"}
        and conversation_plan.get("subject") == "kali_capabilities"
    )


def _meta_conversation_summary(artifact: dict[str, Any]) -> str:
    question = str(artifact.get("question") or "").lower()
    if "convers" in question or "hablar" in question:
        return (
            "Si, puedo conversar contigo. Tambien puedo mantener el contexto entre preguntas, "
            "explicar resultados, revisar posibles problemas de datos y sugerir siguientes pasos. "
            "Cuando hablemos de cifras, voy a apoyarme en evidencia consultada para no inventar metricas."
        )
    return (
        "Puedo ayudarte a explorar ventas, productos, margenes, clientes y calidad de datos en forma conversacional. "
        "Puedes preguntarme una metrica y luego pedirme causas, riesgos o acciones siguientes."
    )


def _no_evidence_summary(artifact: dict[str, Any]) -> str:
    plan = artifact.get("conversation_plan") if isinstance(artifact.get("conversation_plan"), dict) else {}
    context = artifact.get("conversation_state") if isinstance(artifact.get("conversation_state"), dict) else {}
    question = str(artifact.get("question") or "").strip()
    if plan.get("context_dependency") == "required" or plan.get("reuse_previous_artifact"):
        return (
            "Necesito el contexto analitico anterior para responder bien esta pregunta. "
            "Haz primero una consulta con datos, por ejemplo productos vendidos con margen, y luego puedo explicar causas o riesgos."
        )
    if context.get("source") in {None, "none", "messages"}:
        return (
            "Puedo ayudarte, pero esta pregunta no produjo evidencia de datos para analizar. "
            "Preguntame por ventas, productos, margenes, clientes o calidad de datos y continuo desde ahi."
        )
    if question:
        return (
            "No tengo evidencia suficiente para responder esa pregunta sin inventar. "
            "Puedo reformularla como consulta de datos o usar el contexto anterior si aplica."
        )
    return "No tengo evidencia suficiente para responder sin inventar."


def _inventory_summary(artifact: dict[str, Any], rows: list[dict[str, Any]], period: str | None) -> str:
    if not rows:
        return (
            "Puedo analizar inventario, pero esta consulta no trajo insumos o movimientos suficientes. "
            "Prueba con bajo stock, consumo de ingredientes o movimientos de inventario en un periodo."
        )
    intent = artifact.get("question_intent") if isinstance(artifact.get("question_intent"), dict) else {}
    movement_view = intent.get("grain") == "inventory_movement" or any(
        row.get("net_quantity") is not None or row.get("quantity_out") is not None for row in rows
    )
    title = "Consumo de ingredientes" if movement_view else "Inventario de insumos"
    lines = [f"{title} para {_period_label(period)}:"]
    for index, row in enumerate(rows[:10], start=1):
        name = row.get("ingredient") or row.get("ingredient_name") or row.get("name") or "Ingrediente"
        bits = []
        if row.get("current_stock") is not None:
            bits.append(f"stock {row.get('current_stock')}")
        if row.get("minimum_stock") is not None:
            bits.append(f"minimo {row.get('minimum_stock')}")
        if row.get("net_quantity") is not None:
            bits.append(f"consumo neto {row.get('net_quantity')}")
        if row.get("quantity_out") is not None:
            bits.append(f"salidas {row.get('quantity_out')}")
        if row.get("movement_count") is not None:
            bits.append(f"{row.get('movement_count')} movimientos")
        if row.get("unit"):
            bits.append(str(row.get("unit")))
        lines.append(f"{index}. {name}" + (f" ({', '.join(bits)})" if bits else ""))
    limitations = _procurement_limitations(artifact)
    if limitations:
        lines.append("")
        lines.append("Limitaciones:")
        lines.extend(f"- {item}" for item in limitations[:3])
    return "\n".join(lines)


def _purchase_summary(artifact: dict[str, Any], rows: list[dict[str, Any]], period: str | None) -> str:
    if not rows:
        return (
            "Puedo analizar compras y proveedores, pero esta consulta no trajo compras suficientes. "
            "Prueba con compras recientes, proveedor, ingrediente o costo unitario en un periodo."
        )
    lines = [f"Compras y proveedores para {_period_label(period)}:"]
    for index, row in enumerate(rows[:10], start=1):
        supplier = row.get("supplier") or row.get("supplier_name") or "Proveedor"
        ingredient = row.get("ingredient") or row.get("ingredient_name")
        label = f"{supplier}" + (f" - {ingredient}" if ingredient else "")
        bits = []
        if row.get("avg_unit_cost") is not None:
            bits.append(f"costo unitario {_fmt_money(row.get('avg_unit_cost')) or row.get('avg_unit_cost')}")
        if row.get("total_cost") is not None:
            bits.append(f"total {_fmt_money(row.get('total_cost')) or row.get('total_cost')}")
        if row.get("quantity_purchased") is not None:
            bits.append(f"cantidad {row.get('quantity_purchased')}")
        if row.get("purchase_count") is not None:
            bits.append(f"{row.get('purchase_count')} compras")
        lines.append(f"{index}. {label}" + (f" ({', '.join(bits)})" if bits else ""))
    limitations = _procurement_limitations(artifact)
    if limitations:
        lines.append("")
        lines.append("Limitaciones:")
        lines.extend(f"- {item}" for item in limitations[:3])
    return "\n".join(lines)


def _procurement_recommendation_summary(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    period: str | None,
) -> str:
    lines = [f"Compra sugerida para {_period_label(period)}:"]
    if rows:
        lines.append("")
        lines.append("Evidencia disponible:")
        for row in rows[:5]:
            name = row.get("ingredient") or row.get("ingredient_name") or row.get("name") or "Ingrediente"
            bits = []
            if row.get("current_stock") is not None:
                bits.append(f"stock {row.get('current_stock')}")
            if row.get("minimum_stock") is not None:
                bits.append(f"minimo {row.get('minimum_stock')}")
            if row.get("avg_unit_cost") is not None:
                bits.append(f"costo unitario {_fmt_money(row.get('avg_unit_cost')) or row.get('avg_unit_cost')}")
            lines.append(f"- {name}" + (f": {', '.join(bits)}." if bits else "."))
    lines.append("")
    lines.append(
        "No emitiria una orden de compra cerrada solo con esta evidencia. "
        "La usaria como alerta inicial y validaria los datos faltantes antes de comprar."
    )
    limitations = _dedupe_text_items(
        [
            *_procurement_limitations(artifact),
            "Falta lead time de proveedores para calcular urgencia.",
            "Faltan compras pendientes para evitar duplicar abastecimiento.",
            "Falta demanda esperada o consumo proyectado para estimar cantidad.",
        ]
    )
    lines.append("")
    lines.append("Limitaciones:")
    lines.extend(f"- {item}" for item in limitations[:5])
    return "\n".join(lines)


def _asks_for_purchase_recommendation(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in ("deberia comprar", "debería comprar", "que comprar", "qué comprar"))


def _procurement_limitations(artifact: dict[str, Any]) -> list[str]:
    analysis = artifact.get("analysis") if isinstance(artifact.get("analysis"), dict) else {}
    return _dedupe_text_items(
        [
            *_string_items(analysis.get("limitations")),
            *_string_items(artifact.get("limitations")),
        ]
    )


def _context_usage(*, question: str, conversation_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(conversation_state, dict) or conversation_state.get("source") in {None, "none"}:
        return {"used": False, "reason": "no_context"}
    normalized = question.lower()
    refers_to_previous = any(
        token in normalized
        for token in ("esto", "eso", "anterior", "qué más", "que mas", "profundiza", "ese", "esa")
    )
    return {
        "used": refers_to_previous,
        "source": conversation_state.get("source"),
        "active_entity": conversation_state.get("active_entity"),
        "active_period": conversation_state.get("active_period"),
    }


def _previous_ranked_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    context = artifact.get("conversation_state") if isinstance(artifact.get("conversation_state"), dict) else {}
    previous = context.get("last_artifact") if isinstance(context.get("last_artifact"), dict) else {}
    rows = previous.get("ranked_rows") if isinstance(previous.get("ranked_rows"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _table_from_previous_artifact(conversation_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(conversation_state, dict):
        return None
    previous = conversation_state.get("last_artifact")
    if not isinstance(previous, dict):
        return None
    rows = previous.get("ranked_rows") if isinstance(previous.get("ranked_rows"), list) else []
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return None
    metadata_items = previous.get("query_metadata") if isinstance(previous.get("query_metadata"), list) else []
    query_metadata = next((item for item in metadata_items if isinstance(item, dict)), {})
    return {
        "tool": "previous_artifact",
        "status": "succeeded",
        "expected_evidence": ["previous_artifact", "ranked_rows"],
        "rows": [_normalize_query_row(row, query_metadata) for row in rows],
        "metrics": previous.get("metrics") if isinstance(previous.get("metrics"), dict) else {},
        "query_metadata": query_metadata,
        "limitations": _query_limitations(query_metadata),
        "summary": previous.get("summary"),
    }


def _query_metadata_from_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        metadata
        for table in tables
        if isinstance((metadata := table.get("query_metadata")), dict) and metadata
    ]


def _query_metadata_from_observation(observation: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    arguments = observation.get("arguments") if isinstance(observation.get("arguments"), dict) else {}
    spec = _parse_spec(arguments.get("spec"))
    meta = _metadata_from_result(result)
    if not spec and not meta:
        return {}
    metadata: dict[str, Any] = {
        "dataset": spec.get("dataset") or meta.get("dataset"),
        "measures": _string_items(spec.get("measures") if "measures" in spec else meta.get("measures")),
        "dimensions": _string_items(spec.get("dimensions") if "dimensions" in spec else meta.get("dimensions")),
        "filters": spec.get("filters") or meta.get("filters") or {},
        "order_by": spec.get("order_by") or meta.get("order_by") or meta.get("sort") or [],
        "limit": spec.get("limit") or meta.get("limit"),
        "row_count": meta.get("row_count") or meta.get("rowCount") or _safe_len(result.get("rows")),
        "limitations": _string_items(
            meta.get("limitations")
            or meta.get("warnings")
            or result.get("limitations")
            or result.get("warnings")
        ),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "", [], {})}


def _metadata_from_result(result: dict[str, Any]) -> dict[str, Any]:
    for key in ("meta", "metadata"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    data = result.get("data")
    if isinstance(data, dict):
        for key in ("meta", "metadata"):
            value = data.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _parse_spec(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _query_limitations(metadata: dict[str, Any]) -> list[str]:
    limitations = _string_items(metadata.get("limitations"))
    if metadata.get("dataset") and not metadata.get("measures"):
        limitations.append("La evidencia de QuerySpec no declaro measures concretas.")
    if metadata.get("dataset") and not metadata.get("dimensions"):
        limitations.append("La evidencia de QuerySpec no declaro dimensions concretas.")
    return _dedupe_text_items(limitations)


def _normalize_query_row(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    aliases = {
        "product": "name",
        "product_name": "name",
        "product_id": "id",
        "customer": "name",
        "customer_name": "name",
        "customer_id": "id",
        "ingredient": "name",
        "ingredient_name": "name",
        "quantity_sold": "quantity",
        "total_units_sold": "quantity",
        "total_revenue": "revenue",
        "profit_margin_pct": "margin",
        "profit_margin_real_pct": "margin",
        "margin_pct": "margin",
    }
    for source, target in aliases.items():
        if source in row and target not in normalized:
            normalized[target] = row[source]
    if metadata.get("dataset") and "_query_dataset" not in normalized:
        normalized["_query_dataset"] = metadata.get("dataset")
    return normalized


def _open_questions(intent: QuestionIntent, analysis: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    limitations = _string_items(analysis.get("limitations"))
    if limitations:
        questions.append("Validar las limitaciones de datos antes de tomar decisiones.")
    if intent.entity == "business":
        questions.append("Comparar el periodo contra un periodo anterior equivalente.")
    return questions[:3]


def _table_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
    query_metadata = _query_metadata_from_observation(observation, result)
    rows = [_normalize_query_row(row, query_metadata) for row in _rows_from_result(result)]
    return {
        "tool": observation.get("tool_name"),
        "status": observation.get("status"),
        "expected_evidence": [
            str(item)
            for item in observation.get("expected_evidence") or []
            if item is not None
        ],
        "rows": rows,
        "metrics": _metrics_from_result(result),
        "query_metadata": query_metadata,
        "limitations": _query_limitations(query_metadata),
        "summary": observation.get("result_summary"),
    }


def _rows_from_result(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    rows = result.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    data = result.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("rows", "products", "series", "top_customers", "items", "customers"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    for key in ("products", "series", "top_customers", "customers", "items"):
        value = result.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _metrics_from_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    data = result.get("data")
    source = data if isinstance(data, dict) else result
    metrics: dict[str, Any] = {}
    aliases = {
        "totalSales": "total_sales",
        "totalAmount": "total_sales",
        "totalOrders": "order_count",
        "orderCount": "order_count",
        "avgTicket": "avg_ticket",
        "revenue": "revenue",
        "total_revenue": "revenue",
        "margin": "margin",
        "profit_margin_pct": "margin",
        "total_issued": "total_issued",
        "total_earned": "total_issued",
        "total_redeemed": "total_redeemed",
        "redemption_rate_pct": "redemption_rate_pct",
        "cohort_size": "cohort_size",
        "risk_score": "risk_score",
        "lifetime_value": "lifetime_value",
    }
    for key, normalized in aliases.items():
        if key in source:
            metrics[normalized] = source[key]
    for key in ("metrics", "summary", "insights"):
        value = source.get(key)
        if isinstance(value, dict):
            metrics.update(value)
    return metrics


def _merge_metrics(tables: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for table in tables:
        metrics = table.get("metrics") if isinstance(table.get("metrics"), dict) else {}
        merged.update(metrics)
    return merged


def _ranked_rows(intent: QuestionIntent, tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in tables:
        for row in table.get("rows") or []:
            if isinstance(row, dict):
                rows.append(row)
    if intent.entity == "product":
        rows = _merge_product_rows(rows)
        return sorted(
            rows,
            key=lambda row: (
                -(float(row.get("quantity") or row.get("quantity_sold") or row.get("total_units_sold") or 0)),
                float(row.get("margin") or row.get("margin_pct") or row.get("profit_margin_pct") or 999999),
            ),
        )
    if intent.entity == "customer":
        if intent.grain in {"cohort_period", "customer_period_segment", "customer_risk"}:
            return rows
        key = "order_count" if "order_count" in intent.measures else "total_spent"
        rows = [
            row
            for row in rows
            if _number(row.get("order_count")) > 0 or _number(row.get("total_spent")) > 0
        ]
        if "compare" in intent.operations and {"order_count", "total_spent"}.issubset(set(intent.measures)):
            return sorted(
                rows,
                key=lambda row: (-_number(row.get("order_count")), -_number(row.get("total_spent"))),
            )
        return sorted(rows, key=lambda row: -_number(row.get(key)))
    if intent.entity == "loyalty_transaction":
        return sorted(
            rows,
            key=lambda row: -_number(row.get("total_earned") or row.get("total_issued")),
        )
    return rows


def _merge_product_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("id") or row.get("product_id") or row.get("name") or row.get("product_name") or "").strip().lower()
        if not key:
            continue
        current = merged.setdefault(key, {})
        for source, target in (
            ("id", "id"),
            ("product_id", "id"),
            ("name", "name"),
            ("product_name", "name"),
            ("quantity", "quantity"),
            ("quantity_sold", "quantity"),
            ("total_units_sold", "quantity"),
            ("revenue", "revenue"),
            ("total_revenue", "revenue"),
            ("margin", "margin"),
            ("margin_pct", "margin"),
            ("profit_margin_pct", "margin"),
            ("profit_margin_real_pct", "margin"),
            ("profit_margin_operativo_pct", "margin_operativo"),
            ("cost_source", "cost_source"),
            ("total_profit", "total_profit"),
            ("profit_per_unit", "profit_per_unit"),
            ("category", "category"),
            ("classification", "classification"),
            ("cost", "cost"),
            ("estimated_cost", "cost"),
        ):
            value = row.get(source)
            if value is not None and current.get(target) is None:
                current[target] = value
    return list(merged.values())


def _evidence(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "tool": table.get("tool"),
            "row_count": len(table.get("rows") or []),
            "metrics": table.get("metrics") or {},
            "query_metadata": table.get("query_metadata") or {},
            "limitations": table.get("limitations") or [],
        }
        for table in tables
    ]


def _analysis_from_evidence(
    intent: QuestionIntent,
    tables: list[dict[str, Any]],
    metrics: dict[str, Any],
    ranked_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    facts: list[str] = []
    patterns: list[str] = []
    risks: list[str] = []
    opportunities: list[str] = []
    recommended_actions: list[str] = []
    limitations: list[str] = []
    profile = profile_for_intent(intent)

    total_sales = metrics.get("total_sales") or metrics.get("totalSales")
    avg_ticket = metrics.get("avg_ticket") or metrics.get("avgTicket")
    order_count = metrics.get("order_count") or metrics.get("totalOrders")
    if total_sales is not None:
        facts.append(f"Ventas totales observadas: {_fmt_money(total_sales)}.")
    if avg_ticket is not None:
        facts.append(f"Ticket promedio observado: {_fmt_money(avg_ticket)}.")
    if order_count is not None:
        facts.append(f"Ordenes observadas: {order_count}.")

    if profile is not None:
        grouped_rows = {
            group: rows_for_group(tables=tables, profile=profile, group=group)
            for group in _profile_groups(profile)
        }
        if "products" in grouped_rows:
            grouped_rows["products"] = _merge_product_rows(grouped_rows["products"])

        for signal in profile.signals:
            _apply_signal(
                signal=signal,
                grouped_rows=grouped_rows,
                patterns=patterns,
                risks=risks,
                opportunities=opportunities,
                recommended_actions=recommended_actions,
            )
        for item in profile.missing_evidence_limitations:
            group = str(item.get("group") or "")
            message = str(item.get("message") or "")
            if group and message and not grouped_rows.get(group):
                limitations.append(message)

    _apply_query_analysis(
        intent=intent,
        tables=tables,
        ranked_rows=ranked_rows,
        facts=facts,
        patterns=patterns,
        risks=risks,
        opportunities=opportunities,
        recommended_actions=recommended_actions,
        limitations=limitations,
    )

    return {
        "facts": facts,
        "patterns": patterns,
        "risks": risks,
        "opportunities": opportunities,
        "recommended_actions": recommended_actions,
        "limitations": limitations,
        "confidence": "medium" if patterns or facts else "low",
        "source_tools": [str(table.get("tool")) for table in tables if table.get("tool")],
    }


def _apply_query_analysis(
    *,
    intent: QuestionIntent,
    tables: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
    facts: list[str],
    patterns: list[str],
    risks: list[str],
    opportunities: list[str],
    recommended_actions: list[str],
    limitations: list[str],
) -> None:
    query_tables = [
        table
        for table in tables
        if isinstance(table.get("query_metadata"), dict) and table.get("query_metadata")
    ]
    if not query_tables:
        return
    for table in query_tables:
        metadata = table.get("query_metadata") if isinstance(table.get("query_metadata"), dict) else {}
        dataset = str(metadata.get("dataset") or "query")
        row_count = metadata.get("row_count") or len(table.get("rows") or [])
        measures = ", ".join(_string_items(metadata.get("measures")))
        dimensions = ", ".join(_string_items(metadata.get("dimensions")))
        facts.append(
            f"QuerySpec {dataset} devolvio {row_count} filas"
            + (f" con measures {measures}" if measures else "")
            + (f" por dimensions {dimensions}" if dimensions else "")
            + "."
        )
        limitations.extend(_query_limitations(metadata))
    if intent.entity == "product":
        _apply_product_query_analysis(
            rows=ranked_rows,
            facts=facts,
            patterns=patterns,
            risks=risks,
            opportunities=opportunities,
            recommended_actions=recommended_actions,
            limitations=limitations,
        )
    elif intent.entity == "customer":
        _apply_customer_query_analysis(
            rows=ranked_rows,
            facts=facts,
            patterns=patterns,
            opportunities=opportunities,
            recommended_actions=recommended_actions,
        )


def _apply_product_query_analysis(
    *,
    rows: list[dict[str, Any]],
    facts: list[str],
    patterns: list[str],
    risks: list[str],
    opportunities: list[str],
    recommended_actions: list[str],
    limitations: list[str],
) -> None:
    if not rows:
        limitations.append("No hubo filas de productos para analizar.")
        return
    top = rows[0]
    top_name = _row_name(top, fallback="Producto")
    top_bits = _product_bits(top, include_margin=True, include_profit=True)
    facts.append(f"{top_name} lidera la evidencia" + (f" ({', '.join(top_bits)})" if top_bits else "") + ".")
    with_margin = [row for row in rows if _has_value(row, "margin")]
    if with_margin:
        lower_margin = sorted(with_margin, key=lambda row: _number(row.get("margin")))[0]
        low_name = _row_name(lower_margin, fallback="Producto")
        risks.append(
            f"{low_name} tiene el margen mas bajo entre las filas consultadas ({lower_margin.get('margin')}%)."
        )
    volume_rows = [row for row in rows if _number(row.get("quantity")) > 0]
    if volume_rows and with_margin:
        mixed = sorted(
            volume_rows,
            key=lambda row: (-_number(row.get("quantity")), _number(row.get("margin") or 999999)),
        )[0]
        mixed_name = _row_name(mixed, fallback="Producto")
        if _has_value(mixed, "margin"):
            patterns.append(
                f"{mixed_name} combina volumen alto con margen de {mixed.get('margin')}% en la evidencia consultada."
            )
            opportunities.append(f"Revisar precio, costo o receta de {mixed_name} antes de empujar mas volumen.")
            recommended_actions.append(f"Priorizar auditoria de margen para {mixed_name} usando costo, precio y ventas recientes.")
    if not with_margin:
        limitations.append("La evidencia de productos no incluyo margen; no se puede evaluar rentabilidad.")


def _apply_customer_query_analysis(
    *,
    rows: list[dict[str, Any]],
    facts: list[str],
    patterns: list[str],
    opportunities: list[str],
    recommended_actions: list[str],
) -> None:
    if not rows:
        return
    with_orders = [row for row in rows if _number(row.get("order_count")) > 0]
    with_spend = [row for row in rows if _number(row.get("total_spent")) > 0]
    if with_orders:
        frequent = sorted(with_orders, key=lambda row: -_number(row.get("order_count")))[0]
        facts.append(f"{_row_name(frequent, fallback='Cliente')} lidera frecuencia con {frequent.get('order_count')} ordenes.")
    if with_spend:
        valuable = sorted(with_spend, key=lambda row: -_number(row.get("total_spent")))[0]
        spent = _fmt_money(valuable.get("total_spent"))
        facts.append(f"{_row_name(valuable, fallback='Cliente')} lidera valor comprado" + (f" con {spent}" if spent else "") + ".")
    if with_orders and with_spend:
        patterns.append("La comparacion separa frecuencia, valor comprado y ticket promedio cuando esas cifras vienen en la evidencia.")
        opportunities.append("Tratar clientes frecuentes y clientes de mayor valor como segmentos distintos.")
        recommended_actions.append("Crear acciones separadas: retencion para frecuentes y upsell para clientes de mayor valor.")


def _business_analysis_summary(artifact: dict[str, Any], period: str | None) -> str:
    analysis = artifact.get("analysis") if isinstance(artifact.get("analysis"), dict) else {}
    facts = _string_items(analysis.get("facts"))
    patterns = _string_items(analysis.get("patterns"))
    risks = _string_items(analysis.get("risks"))
    opportunities = _string_items(analysis.get("opportunities"))
    actions = _string_items(analysis.get("recommended_actions"))
    limitations = _string_items(analysis.get("limitations"))

    lines = [f"Analisis del negocio para {_period_label(period)}:"]
    if facts:
        lines.append("")
        lines.append("Datos base:")
        lines.extend(f"- {item}" for item in facts[:4])
    if patterns:
        lines.append("")
        lines.append("Comportamientos detectados:")
        lines.extend(f"- {item}" for item in patterns[:5])
    if risks:
        lines.append("")
        lines.append("Riesgos:")
        lines.extend(f"- {item}" for item in risks[:4])
    if opportunities:
        lines.append("")
        lines.append("Oportunidades:")
        lines.extend(f"- {item}" for item in opportunities[:4])
    if actions:
        lines.append("")
        lines.append("Acciones recomendadas:")
        lines.extend(f"- {item}" for item in actions[:5])
    if limitations:
        lines.append("")
        lines.append("Limitaciones:")
        lines.extend(f"- {item}" for item in limitations[:3])
    if len(lines) == 1:
        lines.append("Todavia no tengo evidencia suficiente para separar patrones profundos sin inventar.")
    return "\n".join(lines)


def _recommendation_summary(artifact: dict[str, Any], period: str | None) -> str:
    analysis = artifact.get("analysis") if isinstance(artifact.get("analysis"), dict) else {}
    context = artifact.get("conversation_state") if isinstance(artifact.get("conversation_state"), dict) else {}
    facts = _string_items(analysis.get("facts"))[:3]
    risks = _string_items(analysis.get("risks"))[:3]
    opportunities = _string_items(analysis.get("opportunities"))[:3]
    actions = _string_items(analysis.get("recommended_actions"))[:5]
    limitations = _dedupe_text_items(
        [
            *_string_items(analysis.get("limitations")),
            *_string_items(artifact.get("limitations")),
            *_string_items(context.get("prior_limitations")),
        ]
    )[:3]
    if not actions:
        actions = _string_items(context.get("prior_actions"))[:5]
    actions = _dedupe_text_items(actions)
    if not actions and artifact.get("ranked_rows"):
        actions = [
            "Prioriza acciones sobre los segmentos o items con mayor concentracion en el ranking.",
            "Separa casos atipicos antes de tomar decisiones comerciales.",
        ]
    lines = [f"Acciones recomendadas para {_period_label(period)}:"]
    if facts:
        lines.append("")
        lines.append("Base de la recomendacion:")
        lines.extend(f"- {item}" for item in facts)
    if risks:
        lines.append("")
        lines.append("Riesgos a controlar:")
        lines.extend(f"- {item}" for item in risks)
    if opportunities:
        lines.append("")
        lines.append("Oportunidades:")
        lines.extend(f"- {item}" for item in opportunities)
    if actions:
        lines.append("")
        lines.append("Que haria:")
        lines.extend(f"- {item}" for item in actions)
    if limitations:
        lines.append("")
        lines.append("Limitaciones:")
        lines.extend(f"- {item}" for item in limitations)
    if len(lines) == 1:
        lines.append("Todavia no tengo acciones suficientemente respaldadas por la evidencia disponible.")
    return "\n".join(lines)


def _business_follow_up_summary(artifact: dict[str, Any], period: str | None) -> str:
    analysis = artifact.get("analysis") if isinstance(artifact.get("analysis"), dict) else {}
    context = artifact.get("conversation_state") if isinstance(artifact.get("conversation_state"), dict) else {}
    previous = context.get("last_artifact") if isinstance(context.get("last_artifact"), dict) else {}
    previous_summary = str(context.get("last_summary") or previous.get("summary") or "")
    facts = _dedupe_text_items(_string_items(analysis.get("facts")))
    patterns = _dedupe_text_items(_string_items(analysis.get("patterns")))
    risks = _dedupe_text_items(_string_items(analysis.get("risks")))
    opportunities = _dedupe_text_items(_string_items(analysis.get("opportunities")))
    actions = _dedupe_text_items(_string_items(analysis.get("recommended_actions")))

    lines = [f"Otro angulo para {_period_label(period)}:"]
    if patterns:
        lines.append("")
        lines.append("Lecturas adicionales:")
        lines.extend(f"- {item}" for item in _exclude_seen(patterns, previous_summary)[:4])
    if opportunities:
        lines.append("")
        lines.append("Oportunidades no obvias:")
        lines.extend(f"- {item}" for item in _exclude_seen(opportunities, previous_summary)[:3])
    if risks:
        lines.append("")
        lines.append("Riesgos a revisar:")
        lines.extend(f"- {item}" for item in _exclude_seen(risks, previous_summary)[:3])
    if actions:
        lines.append("")
        lines.append("Siguiente analisis recomendado:")
        lines.extend(f"- {item}" for item in _exclude_seen(actions, previous_summary)[:3])
    if len(lines) == 1 and facts:
        lines.append("")
        lines.append("Base disponible:")
        lines.extend(f"- {item}" for item in facts[:3])
        lines.append("- Para avanzar, compara este periodo contra uno anterior equivalente y separa ventas genericas de clientes identificados.")
    if len(lines) == 1:
        lines.append("Con la evidencia actual no separo un angulo nuevo sin repetir o inventar; pideme comparar periodos, segmentos o clientes identificados.")
    return "\n".join(lines)


def _product_follow_up_summary(artifact: dict[str, Any], rows: list[dict[str, Any]], period: str | None) -> str:
    analysis = artifact.get("analysis") if isinstance(artifact.get("analysis"), dict) else {}
    context = artifact.get("conversation_state") if isinstance(artifact.get("conversation_state"), dict) else {}
    previous = context.get("last_artifact") if isinstance(context.get("last_artifact"), dict) else {}
    previous_summary = str(context.get("last_summary") or previous.get("summary") or "")
    patterns = _exclude_seen(_dedupe_text_items(_string_items(analysis.get("patterns"))), previous_summary)
    risks = _exclude_seen(_dedupe_text_items(_string_items(analysis.get("risks"))), previous_summary)
    opportunities = _exclude_seen(_dedupe_text_items(_string_items(analysis.get("opportunities"))), previous_summary)
    actions = _exclude_seen(_dedupe_text_items(_string_items(analysis.get("recommended_actions"))), previous_summary)
    limitations = _dedupe_text_items(
        [
            *_string_items(analysis.get("limitations")),
            *_string_items(artifact.get("limitations")),
            *_string_items(context.get("prior_limitations")),
        ]
    )
    lines = [f"Otro angulo sobre esos productos para {_period_label(period)}:"]
    for label, items, limit in (
        ("Lecturas adicionales", patterns, 3),
        ("Riesgos a revisar", risks, 3),
        ("Oportunidades", opportunities, 3),
        ("Siguiente accion", actions, 3),
    ):
        if items:
            lines.append("")
            lines.append(f"{label}:")
            lines.extend(f"- {item}" for item in items[:limit])
    if len(lines) == 1:
        sample = rows[:3]
        lines.append("")
        lines.append("Base disponible:")
        for row in sample:
            bits = _product_bits(row, include_margin=True, include_profit=True)
            lines.append(f"- {_row_name(row, fallback='Producto')}" + (f": {', '.join(bits)}." if bits else "."))
    if limitations:
        lines.append("")
        lines.append("Limitaciones:")
        lines.extend(f"- {item}" for item in limitations[:3])
    return "\n".join(lines)


def _product_diagnosis_summary(artifact: dict[str, Any], rows: list[dict[str, Any]], period: str | None) -> str:
    analysis = artifact.get("analysis") if isinstance(artifact.get("analysis"), dict) else {}
    plan = artifact.get("conversation_plan") if isinstance(artifact.get("conversation_plan"), dict) else {}
    risks = _dedupe_text_items(_string_items(analysis.get("risks")))
    patterns = _dedupe_text_items(_string_items(analysis.get("patterns")))
    actions = _dedupe_text_items(_string_items(analysis.get("recommended_actions")))
    limitations = _dedupe_text_items(
        [
            *_string_items(analysis.get("limitations")),
            *_string_items(artifact.get("limitations")),
        ]
    )
    negative_rows = [row for row in rows if _number(row.get("margin")) < 0]
    cost_sources = _dedupe_text_items([str(row.get("cost_source")) for row in rows if row.get("cost_source")])
    lines = [f"Mi lectura para {_period_label(period)}:"]
    if negative_rows:
        sample = negative_rows[:3]
        lines.append(
            "Hay margen negativo en productos de alto volumen; eso normalmente indica que el costo usado supera el precio/base de margen registrado."
        )
        lines.append("")
        lines.append("Evidencia:")
        for row in sample:
            bits = _product_bits(row, include_margin=True, include_profit=False)
            lines.append(f"- {_row_name(row, fallback='Producto')}" + (f": {', '.join(bits)}." if bits else "."))
    elif patterns:
        lines.append(patterns[0])
    else:
        lines.append("No veo una senal suficiente para explicar la causa solo con la evidencia actual.")
    if cost_sources:
        lines.append("")
        lines.append("Fuente de costo:")
        lines.append("- La consulta viene marcada como " + ", ".join(cost_sources[:3]) + ".")
    if risks:
        lines.append("")
        lines.append("Riesgo:")
        lines.extend(f"- {item}" for item in risks[:3])
    if actions:
        lines.append("")
        lines.append("Que revisaria primero:")
        lines.extend(f"- {item}" for item in actions[:3])
    elif negative_rows:
        lines.append("")
        lines.append("Que revisaria primero:")
        lines.append("- Comparar precio de venta, costo usado para margen y receta/costo cargado de los productos negativos.")
        lines.append("- Validar si el costo real representa costo unitario correcto o si esta mezclando insumos, presentaciones o conversiones.")
    if limitations:
        lines.append("")
        lines.append("Limitacion:")
        lines.extend(f"- {item}" for item in limitations[:2])
    if "repeat_full_ranking" in _string_items((plan.get("answer_contract") or {}).get("must_not")):
        return "\n".join(lines)
    return "\n".join(lines)


def _product_cause_classification_summary(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    period: str | None,
) -> str:
    negative_rows = [row for row in rows if _number(row.get("margin")) < 0]
    positive_rows = [row for row in rows if _number(row.get("margin")) > 0]
    cost_sources = _dedupe_text_items([str(row.get("cost_source")) for row in rows if row.get("cost_source")])
    lines = [f"Mi lectura para {_period_label(period)}:"]

    if negative_rows:
        lines.append(
            "No lo leeria primero como problema comercial de precio. La senal mas fuerte es costo o datos: "
            "hay productos con volumen alto donde el margen queda negativo, lo que sugiere que el costo usado "
            "esta por encima de la base de venta registrada."
        )
        lines.append("")
        lines.append("Evidencia que pesa:")
        for row in negative_rows[:3]:
            bits = _product_bits(row, include_margin=True, include_profit=False)
            lines.append(f"- {_row_name(row, fallback='Producto')}" + (f": {', '.join(bits)}." if bits else "."))
    else:
        lines.append(
            "Con las filas actuales no veo margen negativo suficiente para culpar precio, costo o datos sin otra consulta."
        )

    if positive_rows:
        lines.append("")
        lines.append("Contraste util:")
        for row in positive_rows[:2]:
            bits = _product_bits(row, include_margin=True, include_profit=False)
            lines.append(f"- {_row_name(row, fallback='Producto')}" + (f": {', '.join(bits)}." if bits else "."))

    lines.append("")
    lines.append("Como lo separaria:")
    lines.append("- Precio: revisaria si el precio promedio vendido esta demasiado bajo frente al menu o descuentos.")
    lines.append("- Costo: revisaria receta, conversiones y costo unitario de los productos con margen negativo.")
    lines.append("- Datos: revisaria si el costo real esta mezclando presentaciones, insumos o unidades incompatibles.")
    if cost_sources:
        lines.append(f"- Fuente actual de costo en la evidencia: {', '.join(cost_sources[:3])}.")
    return "\n".join(lines)


def _generic_customer_impact_summary(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    period: str | None,
) -> str:
    generic_rows = [
        row
        for row in rows
        if "generic" in normalized_any(_row_name(row, fallback=""))
        or "generico" in normalized_any(_row_name(row, fallback=""))
        or "generica" in normalized_any(_row_name(row, fallback=""))
    ]
    total_orders = sum(_number(row.get("order_count") or row.get("orders_count")) for row in rows)
    total_spent = sum(_number(row.get("total_spent") or row.get("revenue")) for row in rows)
    generic_orders = sum(_number(row.get("order_count") or row.get("orders_count")) for row in generic_rows)
    generic_spent = sum(_number(row.get("total_spent") or row.get("revenue")) for row in generic_rows)

    lines = [f"Sobre clientes genericos para {_period_label(period)}:"]
    if generic_rows:
        lines.append(
            "Si los genericos aparecen en ventas, deben incluirse: son evidencia de que una parte del negocio "
            "no esta identificando al cliente real."
        )
        lines.append("")
        lines.append("Impacto:")
        if total_orders and generic_orders:
            lines.append(f"- Concentracion de ordenes genericas: {_fmt_percent(generic_orders * 100 / total_orders)}.")
        if total_spent and generic_spent:
            lines.append(f"- Concentracion de venta generica: {_fmt_percent(generic_spent * 100 / total_spent)}.")
        for row in generic_rows[:3]:
            bits = _row_summary_bits(row)
            lines.append(f"- {_row_name(row, fallback='Cliente generico')}" + (f": {', '.join(bits)}." if bits else "."))
    else:
        lines.append(
            "La evidencia actual no trae filas claramente marcadas como cliente generico. Para cuantificar el impacto "
            "necesito una consulta por cliente con ordenes, venta y ticket."
        )

    lines.append("")
    lines.append("Por que importa:")
    lines.append("- Distorsiona recompra: muchas compras parecen venir de un solo cliente falso o sin identificar.")
    lines.append("- Distorsiona frecuencia: no sabemos quien realmente vuelve.")
    lines.append("- Distorsiona segmentacion: RFM, fidelizacion y campanas pierden precision.")
    return "\n".join(lines)


def _customer_comparison_summary(rows: list[dict[str, Any]], period: str | None) -> str:
    lines = [f"Comparacion de clientes para {_period_label(period)}:"]
    for index, row in enumerate(rows[:20], start=1):
        name = row.get("name") or row.get("customer_name") or row.get("customer_id") or "Cliente"
        spent = _fmt_money(row.get("total_spent"))
        orders = row.get("order_count")
        avg = _fmt_money(row.get("avg_ticket"))
        bits = []
        if orders is not None:
            bits.append(f"{orders} ordenes")
        if spent:
            bits.append(f"{spent} comprado")
        if avg:
            bits.append(f"ticket {avg}")
        lines.append(f"{index}. {name}" + (f" ({', '.join(bits)})" if bits else ""))
    return "\n".join(lines)


def _product_title(measures: set[str], period: str | None) -> str:
    label = _period_label(period)
    criteria = _measure_labels(
        measures,
        {
            "quantity_sold": "unidades vendidas",
            "quantity": "unidades vendidas",
            "margin": "margen",
            "profit_margin_pct": "margen",
            "revenue": "valor vendido",
            "total_sales": "valor vendido",
            "cost": "costo",
            "profit": "utilidad",
        },
    )
    if criteria:
        return f"Productos por {', '.join(criteria)} para {label}:"
    return f"Productos para {label}:"


def _measure_labels(measures: set[str], labels: dict[str, str]) -> list[str]:
    result: list[str] = []
    for measure, label in labels.items():
        if measure in measures and label not in result:
            result.append(label)
    return result


def _dedupe_text_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = normalized_any(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    return result


def _exclude_seen(items: list[str], previous_summary: str) -> list[str]:
    previous = normalized_any(previous_summary)
    filtered = [item for item in items if normalized_any(item) not in previous]
    return filtered or items


def _profile_groups(profile) -> set[str]:
    result: set[str] = set()
    groups = profile.analysis.get("tool_groups")
    if isinstance(groups, dict):
        result.update(str(group) for group in groups if group)
    selectors = profile.analysis.get("tool_group_selectors")
    if isinstance(selectors, dict):
        result.update(str(group) for group in selectors if group)
    return result


def _apply_signal(
    *,
    signal: dict[str, Any],
    grouped_rows: dict[str, list[dict[str, Any]]],
    patterns: list[str],
    risks: list[str],
    opportunities: list[str],
    recommended_actions: list[str],
) -> None:
    rows = grouped_rows.get(str(signal.get("group") or ""), [])
    if not rows:
        return
    kind = str(signal.get("kind") or "")
    matched: list[dict[str, Any]] = []
    if kind == "threshold_rows":
        positive_fields = _list(signal.get("positive_field_any"))
        max_fields = _list(signal.get("max_field_any"))
        max_value = _number(signal.get("max_value"))
        for row in rows:
            threshold_value = first_value(row, max_fields)
            if threshold_value is None:
                continue
            if _number(first_value(row, positive_fields)) > 0 and _number(threshold_value) <= max_value:
                matched.append(row)
    elif kind == "name_match":
        name_fields = _list(signal.get("name_field_any"))
        values = {normalized_any(item) for item in _list(signal.get("match_values"))}
        matched = [
            row
            for row in rows
            if normalized_any(first_value(row, name_fields)) in values
        ]
    elif kind == "active_rows":
        positive_fields = _list(signal.get("positive_field_any"))
        matched = [row for row in rows if _number(first_value(row, positive_fields)) > 0]
    elif kind == "cohort_retention_threshold":
        max_value = _number(signal.get("max_value"))
        for row in rows:
            retention = row.get("retention")
            if not isinstance(retention, list) or not retention:
                continue
            first = retention[0] if isinstance(retention[0], dict) else {}
            pct = _number(first.get("pct") or first.get("retention_pct"))
            if pct and pct < max_value:
                matched.append(row)
    if not matched:
        return

    sample = _sample_text(matched, _list(signal.get("sample_name_any")))
    context = {"sample": sample}
    for key, target in (
        ("pattern", patterns),
        ("risk", risks),
        ("opportunity", opportunities),
        ("recommended_action", recommended_actions),
    ):
        message = signal.get(key)
        if isinstance(message, str) and message:
            target.append(message.format(**context))


def _sample_text(rows: list[dict[str, Any]], fields: list[Any]) -> str:
    samples = []
    for row in rows[:3]:
        value = first_value(row, fields)
        if value is not None:
            samples.append(str(value))
    return ", ".join(samples) if samples else "varias filas"


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item]


def _limitations(plan: ToolPlan, failed: list[dict[str, Any]], tables: list[dict[str, Any]]) -> list[str]:
    limitations = []
    if plan.missing_coverage:
        limitations.append("Falta cobertura: " + ", ".join(plan.missing_coverage))
    if failed:
        limitations.append("Fallaron tools: " + ", ".join(str(item.get("tool_name")) for item in failed))
    for table in tables:
        limitations.extend(_string_items(table.get("limitations")))
    return _dedupe_text_items(limitations)


def _blocked_message(reason: str | None, plan: ToolPlan) -> str:
    if reason == "missing_coverage" and plan.missing_coverage:
        return (
            "Para responder con evidencia necesito datos que aun no estan cubiertos por las herramientas: "
            + ", ".join(plan.missing_coverage)
            + ". Puedo intentar una pregunta mas acotada o usar el ultimo contexto disponible."
        )
    if reason == "no_compatible_tools":
        return (
            "Esa pregunta no tiene una herramienta de datos compatible todavia. "
            "Puedo ayudarte reformulandola hacia ventas, productos, margenes, clientes, WAROS o calidad de datos."
        )
    if reason == "all_tools_failed":
        tool_names = [str(step.tool_name) for step in plan.steps if getattr(step, "tool_name", None)]
        suffix = f" ({', '.join(tool_names[:3])})" if tool_names else ""
        return (
            "La consulta de datos necesaria fallo"
            + suffix
            + ". No voy a inventar la respuesta; prueba de nuevo o acota periodo, producto, cliente o metrica."
        )
    return "No tengo evidencia suficiente para responder sin inventar; puedo intentarlo con una consulta mas acotada."


def _fmt_money(value: Any) -> str:
    if value is None:
        return ""
    try:
        return "$" + f"{float(value):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return str(value)


def _fmt_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{value}%"
    if number.is_integer():
        return f"{int(number)}%"
    return f"{number:.1f}%"


def _row_summary_bits(row: dict[str, Any]) -> list[str]:
    bits: list[str] = []
    quantity = row.get("quantity") or row.get("quantity_sold") or row.get("total_units_sold")
    revenue = _fmt_money(row.get("revenue") or row.get("total_revenue") or row.get("total_spent"))
    orders = row.get("order_count") or row.get("orders_count")
    margin = row.get("margin") or row.get("margin_pct") or row.get("profit_margin_pct")
    if quantity is not None:
        bits.append(f"{quantity} unidades")
    if orders is not None:
        bits.append(f"{orders} ordenes")
    if revenue:
        bits.append(f"{revenue}")
    if margin is not None:
        bits.append(f"margen {_fmt_percent(margin)}")
    if row.get("current_stock") is not None:
        bits.append(f"stock {row.get('current_stock')}")
    if row.get("minimum_stock") is not None:
        bits.append(f"minimo {row.get('minimum_stock')}")
    if row.get("net_quantity") is not None:
        bits.append(f"consumo neto {row.get('net_quantity')}")
    if row.get("avg_unit_cost") is not None:
        bits.append(f"costo unitario {_fmt_money(row.get('avg_unit_cost')) or row.get('avg_unit_cost')}")
    if row.get("total_cost") is not None:
        bits.append(f"total {_fmt_money(row.get('total_cost')) or row.get('total_cost')}")
    if row.get("cost_source"):
        bits.append(f"costo {row['cost_source']}")
    return bits


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _safe_len(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


def _has_value(row: dict[str, Any], key: str) -> bool:
    return row.get(key) is not None and str(row.get(key)).strip() != ""


def _row_name(row: dict[str, Any], *, fallback: str) -> str:
    return str(
        row.get("name")
        or row.get("product")
        or row.get("product_name")
        or row.get("customer")
        or row.get("customer_name")
        or row.get("ingredient")
        or row.get("ingredient_name")
        or row.get("supplier")
        or row.get("supplier_name")
        or row.get("id")
        or row.get("product_id")
        or row.get("customer_id")
        or fallback
    )


def _product_bits(row: dict[str, Any], *, include_margin: bool, include_profit: bool) -> list[str]:
    bits: list[str] = []
    quantity = row.get("quantity") or row.get("quantity_sold") or row.get("total_units_sold")
    revenue = _fmt_money(row.get("revenue") or row.get("total_revenue"))
    margin = row.get("margin") or row.get("profit_margin_pct") or row.get("margin_pct")
    total_profit = _fmt_money(row.get("total_profit"))
    category = row.get("category")
    if quantity is not None:
        bits.append(f"{quantity} unidades")
    if revenue:
        bits.append(f"{revenue} vendido")
    if include_margin and margin is not None:
        bits.append(f"margen {margin}%")
    if include_profit and total_profit:
        bits.append(f"utilidad {total_profit}")
    if category:
        bits.append(f"categoria {category}")
    return bits


def _period_prefix(label: str | None) -> str:
    return f"Para {label}," if label else "En el periodo consultado,"


def _period_label(label: str | None) -> str:
    return label or "el periodo consultado"
