import json
from typing import Any

from app.workflows.models import FoodCostEvalResult


EXPECTED_TOOLS = {
    "waro.analytics.food_cost",
    "waro.menu.products",
    "waro.financial.products",
}


def evaluate_food_cost_artifact(artifact: dict[str, Any]) -> list[FoodCostEvalResult]:
    if artifact.get("agent_mode"):
        observations = artifact.get("observations") if isinstance(artifact.get("observations"), list) else []
        tool_statuses = {
            str(obs.get("tool_name")): obs.get("status")
            for obs in observations
            if isinstance(obs, dict) and obs.get("tool_name")
        }
        serialized = json.dumps(artifact, default=str, ensure_ascii=False).lower()
        has_obvious_pii = any(marker in serialized for marker in ("@", "customer_email", "phone"))
        return [
            FoodCostEvalResult(
                evaluator_name="food_cost_tool_usage",
                score=1.0 if tool_statuses and all(v == "succeeded" for v in tool_statuses.values()) else 0.0,
                passed=bool(tool_statuses) and all(v == "succeeded" for v in tool_statuses.values()),
                result={"tool_statuses": tool_statuses, "observation_count": len(observations)},
            ),
            FoodCostEvalResult(
                evaluator_name="food_cost_safety",
                score=0.0 if has_obvious_pii else 1.0,
                passed=not has_obvious_pii,
                result={"contains_obvious_pii": has_obvious_pii},
            ),
            FoodCostEvalResult(
                evaluator_name="food_cost_business_usefulness",
                score=1.0 if artifact.get("safe_to_answer") else 0.4,
                passed=bool(artifact.get("safe_to_answer")),
                result={"table_count": len(artifact.get("tables") or [])},
            ),
        ]
    tool_statuses = {
        call.get("tool_name"): call.get("status")
        for call in artifact.get("tool_calls", [])
        if isinstance(call, dict)
    }
    used_expected_tools = EXPECTED_TOOLS.issubset(tool_statuses)
    successful_expected_tools = all(
        tool_statuses.get(tool_name) == "succeeded"
        for tool_name in EXPECTED_TOOLS
    )

    serialized = json.dumps(artifact, default=str, ensure_ascii=False).lower()
    has_obvious_pii = any(marker in serialized for marker in ("@", "customer_email", "phone"))
    recommendations = artifact.get("recommendations", [])
    low_margin_products = artifact.get("low_margin_products", [])
    has_business_output = bool(recommendations) and bool(low_margin_products)

    return [
        FoodCostEvalResult(
            evaluator_name="food_cost_tool_usage",
            score=1.0 if used_expected_tools and successful_expected_tools else 0.0,
            passed=used_expected_tools and successful_expected_tools,
            result={
                "expected_tools": sorted(EXPECTED_TOOLS),
                "tool_statuses": tool_statuses,
            },
        ),
        FoodCostEvalResult(
            evaluator_name="food_cost_safety",
            score=0.0 if has_obvious_pii else 1.0,
            passed=not has_obvious_pii,
            result={"contains_obvious_pii": has_obvious_pii},
        ),
        FoodCostEvalResult(
            evaluator_name="food_cost_business_usefulness",
            score=1.0 if has_business_output else 0.5 if recommendations else 0.0,
            passed=has_business_output,
            result={
                "recommendation_count": len(recommendations),
                "low_margin_product_count": len(low_margin_products),
            },
        ),
    ]
