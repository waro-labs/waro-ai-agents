from __future__ import annotations

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
) -> dict[str, Any]:
    profile = profile_for_intent(intent)
    tables = [_table_from_observation(observation) for observation in observations]
    metrics = _merge_metrics(tables)
    ranked_rows = _ranked_rows(intent, tables)
    analysis = _analysis_from_evidence(intent, tables, metrics, ranked_rows)
    failed = [obs for obs in observations if obs.get("status") != "succeeded"]
    if not plan.valid:
        answerability = "blocked"
        blocked_reason = plan.blocked_reason or "invalid_plan"
    elif not observations:
        answerability = "blocked"
        blocked_reason = "no_tool_results"
    elif len(failed) == len(observations):
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
        "context_usage": _context_usage(question=question, conversation_state=conversation_state),
        "classification": classification or {},
        "plan": plan.to_dict(),
        "tool_results": observations,
        "observations": observations,
        "tables": tables,
        "metrics": metrics,
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
        "limitations": _limitations(plan, failed),
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
    if strategy in {"follow_up", "recommendation"} and not rows:
        rows = _previous_ranked_rows(artifact)

    if strategy == "recommendation":
        return _recommendation_summary(artifact, period)
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
            return "No encontre filas de productos suficientes para responder con ranking."
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
                bits.append(f"margen {margin}%")
            lines.append(f"{index}. {name}" + (f" ({', '.join(bits)})" if bits else ""))
        return "\n".join(lines)
    if entity == "customer":
        if intent.get("grain") == "cohort_period":
            if not rows:
                return "No encontre cohortes suficientes para analizar retencion."
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
                return "No encontre clientes suficientes para segmentacion RFM."
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
                return "No encontre clientes en riesgo con los criterios consultados."
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
            return "No encontre clientes suficientes para responder con ranking."
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
            return "No encontre datos de WAROS suficientes para responder."
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
    return "Tengo datos suficientes, pero no pude generar un resumen especifico para esta consulta."


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
    return {
        "tool": observation.get("tool_name"),
        "status": observation.get("status"),
        "expected_evidence": [
            str(item)
            for item in observation.get("expected_evidence") or []
            if item is not None
        ],
        "rows": _rows_from_result(result),
        "metrics": _metrics_from_result(result),
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
        for key in ("products", "series", "top_customers", "items", "customers"):
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
            ("cost", "cost"),
            ("estimated_cost", "cost"),
        ):
            value = row.get(source)
            if value is not None and current.get(target) is None:
                current[target] = value
    return list(merged.values())


def _evidence(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"tool": table.get("tool"), "row_count": len(table.get("rows") or []), "metrics": table.get("metrics") or {}}
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
        lines.append("No encontre evidencia suficiente para identificar patrones profundos.")
    return "\n".join(lines)


def _recommendation_summary(artifact: dict[str, Any], period: str | None) -> str:
    analysis = artifact.get("analysis") if isinstance(artifact.get("analysis"), dict) else {}
    context = artifact.get("conversation_state") if isinstance(artifact.get("conversation_state"), dict) else {}
    facts = _string_items(analysis.get("facts"))[:3]
    risks = _string_items(analysis.get("risks"))[:3]
    opportunities = _string_items(analysis.get("opportunities"))[:3]
    actions = _string_items(analysis.get("recommended_actions"))[:5]
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
    if len(lines) == 1:
        lines.append("No encontre acciones suficientemente respaldadas por la evidencia disponible.")
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
        lines.append("No encontre un angulo nuevo suficientemente respaldado por la evidencia actual.")
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
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _limitations(plan: ToolPlan, failed: list[dict[str, Any]]) -> list[str]:
    limitations = []
    if plan.missing_coverage:
        limitations.append("Falta cobertura: " + ", ".join(plan.missing_coverage))
    if failed:
        limitations.append("Fallaron tools: " + ", ".join(str(item.get("tool_name")) for item in failed))
    return limitations


def _blocked_message(reason: str | None, plan: ToolPlan) -> str:
    if reason == "missing_coverage" and plan.missing_coverage:
        return "No pude completar esta respuesta porque faltan datos requeridos: " + ", ".join(plan.missing_coverage)
    if reason == "no_compatible_tools":
        return "No encontre herramientas compatibles para responder esa pregunta."
    if reason == "all_tools_failed":
        return "No pude completar esta respuesta porque fallaron las herramientas requeridas."
    return "No pude completar esta respuesta."


def _fmt_money(value: Any) -> str:
    if value is None:
        return ""
    try:
        return "$" + f"{float(value):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return str(value)


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _period_prefix(label: str | None) -> str:
    return f"Para {label}," if label else "En el periodo consultado,"


def _period_label(label: str | None) -> str:
    return label or "el periodo consultado"
