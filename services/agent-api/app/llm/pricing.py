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
    prompt_cost_usd: float | None = None
    completion_cost_usd: float | None = None


KIMI_PRICING: dict[str, ModelPricing] = {
    # Official Kimi pricing is per 1M consumed tokens. Keep this table small
    # and explicit so unknown models fail open with token usage but no cost.
    "moonshot-v1-8k": ModelPricing(
        input_usd_per_1m=0.20,
        output_usd_per_1m=2.00,
        source="static:official-kimi-pricing-2026-06-18",
    ),
    "kimi-k2.5": ModelPricing(
        input_usd_per_1m=0.60,
        output_usd_per_1m=3.00,
        source="static:official-kimi-pricing-2026-06-20",
    ),
    "kimi-k2.6": ModelPricing(
        input_usd_per_1m=0.95,
        output_usd_per_1m=4.00,
        source="static:official-kimi-pricing-2026-06-20",
    ),
    "kimi-k2.7-code": ModelPricing(
        input_usd_per_1m=0.95,
        output_usd_per_1m=4.00,
        source="static:official-kimi-pricing-2026-06-18",
    ),
    "kimi-k2.7-code-highspeed": ModelPricing(
        input_usd_per_1m=1.90,
        output_usd_per_1m=8.00,
        source="static:official-kimi-pricing-2026-06-18",
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

    prompt_cost = (input_tokens / 1_000_000) * pricing.input_usd_per_1m
    completion_cost = (output_tokens / 1_000_000) * pricing.output_usd_per_1m
    cost = prompt_cost + completion_cost
    return CostEstimate(
        estimated_cost_usd=round(cost, 8),
        source=pricing.source,
        prompt_cost_usd=round(prompt_cost, 8),
        completion_cost_usd=round(completion_cost, 8),
    )
