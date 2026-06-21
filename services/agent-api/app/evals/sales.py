from app.workflows.models import FoodCostEvalResult


def evaluate_sales_artifact(artifact: dict) -> list[FoodCostEvalResult]:
    if artifact.get("agent_mode"):
        observations = artifact.get("observations") if isinstance(artifact.get("observations"), list) else []
        tool_statuses = {
            str(obs.get("tool_name")): obs.get("status")
            for obs in observations
            if isinstance(obs, dict) and obs.get("tool_name")
        }
        return [
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
