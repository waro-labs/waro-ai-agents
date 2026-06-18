from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_1m: float
    output_usd_per_1m: float
    source: str


@dataclass(frozen=True)
class CostEstimate:
    estimated_cost_usd: float | None
    source: str


KIMI_PRICING: dict[str, ModelPricing] = {
    # Official Kimi pricing is per 1M consumed tokens. Keep this table small
    # and explicit so unknown models fail open with token usage but no cost.
    "kimi-k2.7-code": ModelPricing(
        input_usd_per_1m=0.74,
        output_usd_per_1m=3.50,
        source="static:estimated-kimi-pricing-2026-06-18",
    ),
    "kimi-k2.7-code-highspeed": ModelPricing(
        input_usd_per_1m=0.74,
        output_usd_per_1m=3.50,
        source="static:estimated-kimi-pricing-2026-06-18",
    ),
}

MODEL_ALIASES = {
    "kimi-k2": "kimi-k2.7-code",
}


def estimate_llm_cost(
    *,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> CostEstimate:
    if input_tokens is None or output_tokens is None:
        return CostEstimate(estimated_cost_usd=None, source="usage_unavailable")
    if provider != "kimi":
        return CostEstimate(estimated_cost_usd=None, source="pricing_unavailable")

    normalized_model = MODEL_ALIASES.get(model, model)
    pricing = KIMI_PRICING.get(normalized_model)
    if pricing is None:
        return CostEstimate(estimated_cost_usd=None, source="pricing_unavailable")

    cost = (
        (input_tokens / 1_000_000) * pricing.input_usd_per_1m
        + (output_tokens / 1_000_000) * pricing.output_usd_per_1m
    )
    return CostEstimate(
        estimated_cost_usd=round(cost, 8),
        source=pricing.source,
    )
