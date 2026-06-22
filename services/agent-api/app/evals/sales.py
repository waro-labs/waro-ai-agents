import json
from typing import Any

from app.workflows.models import FoodCostEvalResult


def evaluate_sales_artifact(artifact: dict) -> list[FoodCostEvalResult]:
    if artifact.get("agent_mode"):
        observations = artifact.get("observations") if isinstance(artifact.get("observations"), list) else []
        tool_statuses = {
            str(obs.get("tool_name")): obs.get("status")
            for obs in observations
            if isinstance(obs, dict) and obs.get("tool_name")
        }
        evals = [
            FoodCostEvalResult(
                evaluator_name="agent_tool_usage",
                score=1.0 if any(status == "succeeded" for status in tool_statuses.values()) else 0.0,
                passed=bool(tool_statuses) and all(
                    status == "succeeded" for status in tool_statuses.values()
                ),
                result={
                    "tool_statuses": tool_statuses,
                    "observation_count": len(observations),
                    "safe_to_answer": artifact.get("safe_to_answer"),
                },
            ),
            FoodCostEvalResult(
                evaluator_name="sales_business_usefulness",
                score=1.0 if artifact.get("safe_to_answer") else 0.4,
                passed=bool(artifact.get("safe_to_answer")),
                result={
                    "complexity": artifact.get("complexity"),
                    "table_count": len(artifact.get("tables") or []),
                },
            ),
        ]
        if _is_procurement_artifact(artifact):
            evals.extend(_evaluate_procurement_artifact(artifact, tool_statuses))
        return evals
    if artifact.get("intent") in {"small_talk", "follow_up"}:
        intent = artifact.get("intent")
        return [
            FoodCostEvalResult(
                evaluator_name="sales_intent_guard",
                score=1.0,
                passed=True,
                result={
                    "intent": intent,
                    "tool_calls": artifact.get("tool_calls", []),
                },
            ),
            FoodCostEvalResult(
                evaluator_name="sales_business_usefulness",
                score=1.0,
                passed=True,
                result={
                    "has_sales_signal": False,
                    "handled_without_tool": True,
                    "used_conversation_context": bool(
                        artifact.get("previous_context_summary")
                    ),
                },
            ),
        ]

    tool_statuses = {
        call.get("tool_name"): call.get("status")
        for call in artifact.get("tool_calls", [])
        if isinstance(call, dict)
    }
    metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
    has_sales_signal = any(
        metrics.get(key) is not None
        for key in ("total_sales", "order_count", "avg_ticket")
    )
    return [
        FoodCostEvalResult(
            evaluator_name="sales_tool_usage",
            score=1.0 if tool_statuses.get("waro.sales.metrics") == "succeeded" else 0.0,
            passed=tool_statuses.get("waro.sales.metrics") == "succeeded",
            result={
                "expected_tools": ["waro.sales.metrics"],
                "tool_statuses": tool_statuses,
            },
        ),
        FoodCostEvalResult(
            evaluator_name="sales_business_usefulness",
            score=1.0 if has_sales_signal else 0.4,
            passed=has_sales_signal,
            result={
                "has_sales_signal": has_sales_signal,
                "metrics_keys": sorted(metrics.keys()),
            },
        ),
    ]


def _is_procurement_artifact(artifact: dict[str, Any]) -> bool:
    intent = artifact.get("question_intent") if isinstance(artifact.get("question_intent"), dict) else {}
    if intent.get("entity") in {"ingredient", "inventory", "purchase", "supplier", "procurement"}:
        return True
    for metadata in artifact.get("query_metadata") or []:
        if isinstance(metadata, dict) and metadata.get("dataset") in {
            "inventory_stock",
            "inventory_movements",
            "purchase_items",
        }:
            return True
    return False


def _evaluate_procurement_artifact(
    artifact: dict[str, Any],
    tool_statuses: dict[str, Any],
) -> list[FoodCostEvalResult]:
    serialized = json.dumps(artifact, default=str, ensure_ascii=False).lower()
    has_raw_sql = "select " in serialized or " from " in serialized or "raw_sql" in serialized
    has_obvious_pii = any(marker in serialized for marker in ("@", "customer_email", "phone"))
    limitations = _string_list(artifact.get("limitations"))
    question = str(artifact.get("question") or "").lower()
    asks_purchase_recommendation = any(
        term in question
        for term in (
            "deberia comprar",
            "debería comprar",
            "que comprar",
            "qué comprar",
        )
    )
    expected_tools = {"waro.queries.run", "waro.inventory.stock", "waro.purchases.items", "waro.suppliers.list"}
    used_procurement_tool = any(name in expected_tools for name in tool_statuses)
    safety_passed = bool(artifact.get("safe_to_answer")) and used_procurement_tool and not has_raw_sql and not has_obvious_pii
    recommendation_passed = (not asks_purchase_recommendation) or bool(limitations)
    return [
        FoodCostEvalResult(
            evaluator_name="procurement_evidence_safety",
            score=1.0 if safety_passed else 0.0,
            passed=safety_passed,
            result={
                "tool_statuses": tool_statuses,
                "contains_raw_sql": has_raw_sql,
                "contains_obvious_pii": has_obvious_pii,
                "safe_to_answer": artifact.get("safe_to_answer"),
            },
        ),
        FoodCostEvalResult(
            evaluator_name="procurement_recommendation_limitations",
            score=1.0 if recommendation_passed else 0.0,
            passed=recommendation_passed,
            result={
                "asks_purchase_recommendation": asks_purchase_recommendation,
                "limitations": limitations,
            },
        ),
    ]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item is not None and str(item).strip()]
