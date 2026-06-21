from app.config import Settings
from app.llm.model_router import max_agent_steps, model_for


def test_model_router_selects_roles_by_complexity():
    settings = Settings(
        KIMI_MODEL="default",
        KIMI_ROUTER_MODEL="router",
        KIMI_PLANNER_MODEL="planner",
        KIMI_ANALYSIS_MODEL="analysis",
        KIMI_COMPOSER_MODEL="composer",
    )
    assert model_for(settings, step="classify", complexity="simple") == "router"
    assert model_for(settings, step="agent_step", complexity="complex") == "analysis"
    assert model_for(settings, step="compose", complexity="simple") == "composer"
    assert model_for(settings, step="compose", complexity="complex") == "analysis"
    assert max_agent_steps(settings, "complex") == settings.agent_max_steps_complex
