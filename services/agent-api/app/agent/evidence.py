from __future__ import annotations

from typing import Any

from app.agent.intent import QuestionIntent
from app.agent.plan import ToolPlan
from app.tools.sanitize import sanitize_value


def build_evidence_artifact(
    *,
    question: str,
    intent: QuestionIntent,
    plan: ToolPlan,
    observations: list[dict[str, Any]],
    conversation_messages: list[dict[str, str]] | None = None,
    classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tables = [_table_from_observation(observation) for observation in observations]
    metrics = _merge_metrics(tables)
    ranked_rows = _ranked_rows(intent, tables)
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
        "question": question,
        "question_intent": intent.to_dict(),
        "classification": classification or {},
        "plan": plan.to_dict(),
        "tool_results": observations,
        "observations": observations,
        "tables": tables,
        "metrics": metrics,
        "ranked_rows": ranked_rows,
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
        lines = ["Productos con alta venta y bajo margen:"]
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
            if margin is not None:
                bits.append(f"margen {margin}%")
            lines.append(f"{index}. {name}" + (f" ({', '.join(bits)})" if bits else ""))
        return "\n".join(lines)
    if entity == "customer":
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
    return "Tengo datos suficientes, pero no pude generar un resumen especifico para esta consulta."


def _table_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
    return {
        "tool": observation.get("tool_name"),
        "status": observation.get("status"),
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
