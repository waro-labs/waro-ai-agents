from app.workflows.models import FoodCostEvalResult


def evaluate_sales_artifact(artifact: dict) -> list[FoodCostEvalResult]:
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
