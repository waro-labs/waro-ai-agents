from __future__ import annotations

from typing import Any


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
        for key in ("products", "series", "top_customers", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    for key in ("products", "series", "top_customers"):
        value = result.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _metrics_from_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    data = result.get("data")
    if isinstance(data, dict):
        metrics: dict[str, Any] = {}
        for key in (
            "totalSales",
            "totalOrders",
            "orderCount",
            "avgTicket",
            "metrics",
            "summary",
            "insights",
        ):
            if key in data:
                metrics[key] = data[key]
        if metrics:
            return metrics
    for key in ("totalSales", "totalOrders", "orderCount", "avgTicket", "metrics", "summary"):
        if key in result:
            return {key: result[key]}
    return {}


def build_agent_artifact(
    *,
    question: str,
    observations: list[dict[str, Any]],
    complexity: str,
    verification: dict[str, Any] | None = None,
    conversation_messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    for observation in observations:
        result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
        tables.append(
            {
                "tool": observation.get("tool_name"),
                "status": observation.get("status"),
                "rows": _rows_from_result(result),
                "metrics": _metrics_from_result(result),
                "summary": observation.get("result_summary"),
            }
        )
    safe_to_answer = bool((verification or {}).get("safe_to_answer", bool(observations)))
    error_message = str((verification or {}).get("missing") or "")
    if not observations:
        safe_to_answer = False
        error_message = error_message or "No se ejecutaron herramientas para esta pregunta."
    return {
        "intent": "agent_query",
        "question": question,
        "complexity": complexity,
        "observations": observations,
        "tables": tables,
        "conversation_messages": conversation_messages or [],
        "safe_to_answer": safe_to_answer,
        "error_message": error_message if not safe_to_answer else None,
        "response_contract": {
            "safe_to_answer": safe_to_answer,
            "data_status": "ok" if safe_to_answer else "missing",
            "error_message": error_message if not safe_to_answer else None,
        },
        "agent_mode": True,
    }
