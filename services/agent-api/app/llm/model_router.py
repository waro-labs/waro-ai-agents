from __future__ import annotations

from typing import Literal

from app.config import Settings

AgentStep = Literal["classify", "agent_step", "verify", "compose"]
Complexity = Literal["simple", "moderate", "complex"]


def model_for(settings: Settings, *, step: AgentStep, complexity: Complexity) -> str:
    if step == "classify":
        return settings.llm_router_model
    if step == "verify" or (step == "agent_step" and complexity == "complex"):
        return settings.llm_analysis_model
    if step == "compose" and complexity == "simple":
        return settings.llm_composer_model
    if step == "compose" and complexity == "complex":
        return settings.llm_analysis_model
    return settings.llm_planner_model


def max_agent_steps(settings: Settings, complexity: Complexity) -> int:
    if complexity == "complex":
        return settings.agent_max_steps_complex
    return settings.agent_max_steps_simple
