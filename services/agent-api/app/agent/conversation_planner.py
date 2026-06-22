from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.intent import QuestionIntent, normalize_text
from app.config import Settings
from app.llm.base import LLMAdapter, LLMMessage
from app.llm.model_router import Complexity, model_for


IntentType = Literal[
    "metric_lookup",
    "ranking",
    "diagnosis",
    "comparison",
    "drilldown",
    "data_quality_check",
    "recommendation",
    "definition",
    "scope_change",
    "clarification",
]
ContextDependency = Literal["none", "optional", "required"]
ToolPolicy = Literal["auto", "reuse_only", "run_tools", "merge"]


class AnswerContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    must_explain: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    style: str = "conversational_analyst"

    @field_validator("must_explain", "must_not")
    @classmethod
    def clean_items(cls, value: list[str]) -> list[str]:
        return _clean_list(value)[:8]


class ConversationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_type: IntentType
    context_dependency: ContextDependency = "optional"
    subject: str | None = None
    reuse_previous_artifact: bool = False
    preserve_dataset: str | None = None
    required_evidence: list[str] = Field(default_factory=list)
    tool_policy: ToolPolicy = "auto"
    answer_contract: AnswerContract = Field(default_factory=AnswerContract)
    confidence: float = Field(default=0.6, ge=0, le=1)
    source: str = "heuristic"

    @field_validator("required_evidence")
    @classmethod
    def clean_required_evidence(cls, value: list[str]) -> list[str]:
        return _clean_list(value)[:12]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


async def plan_conversation(
    *,
    settings: Settings,
    llm_adapter: LLMAdapter,
    question: str,
    intent: QuestionIntent,
    conversation_state: dict[str, Any] | None,
    capability_hints: list[dict[str, Any]] | None = None,
    complexity: Complexity = "moderate",
) -> ConversationPlan:
    fallback = heuristic_conversation_plan(
        question=question,
        intent=intent,
        conversation_state=conversation_state,
    )
    if settings.llm_provider == "disabled":
        return fallback
    payload = {
        "question": question,
        "intent": intent.to_dict(),
        "conversation_state": _compact_conversation_state(conversation_state or {}),
        "capabilities": capability_hints or [],
        "fallback_plan": fallback.to_dict(),
        "allowed_intent_types": list(IntentType.__args__),
        "allowed_context_dependency": list(ContextDependency.__args__),
        "allowed_tool_policy": list(ToolPolicy.__args__),
    }
    messages = [
        LLMMessage(
            role="system",
            content=(
                "Eres el conversation planner de Kali, un agente de analytics para restaurantes. "
                "No redactes la respuesta final y no elijas SQL. Devuelve SOLO JSON valido con: "
                "intent_type, context_dependency, subject, reuse_previous_artifact, preserve_dataset, "
                "required_evidence, tool_policy, answer_contract{must_explain,must_not,style}, confidence. "
                "Decide si la pregunta actual debe reutilizar el artifact anterior, correr tools nuevas "
                "o mezclar ambas cosas. Para preguntas como por que, explica, que significa, parece problema "
                "de precios/costos/datos, prioriza diagnosis o data_quality_check usando el contexto previo "
                "si existe. Si el contexto previo viene de product_profitability y la pregunta sigue hablando "
                "de margen, costo, precio o datos, conserva preserve_dataset='product_profitability'. "
                "Para preguntas meta sobre si puedes conversar, que puedes hacer o como ayudas, usa "
                "intent_type='definition', tool_policy='reuse_only' y no pidas tools. "
                "No inventes evidencia requerida; enumera solo campos o conceptos necesarios."
            ),
        ),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]
    try:
        response = await llm_adapter.complete(
            messages=messages,
            temperature=0,
            model=model_for(settings, step="verify", complexity=complexity),
        )
        parsed = json.loads(response.content.strip())
        if isinstance(parsed, dict):
            plan = ConversationPlan.model_validate(parsed)
            return plan.model_copy(update={"source": "llm"})
    except Exception as exc:
        print(
            "[agent-api:conversation-planner] fallback "
            + json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "fallback_intent_type": fallback.intent_type,
                    "fallback_tool_policy": fallback.tool_policy,
                    "question_length": len(question),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return fallback
    return fallback


def heuristic_conversation_plan(
    *,
    question: str,
    intent: QuestionIntent,
    conversation_state: dict[str, Any] | None,
) -> ConversationPlan:
    state = conversation_state if isinstance(conversation_state, dict) else {}
    previous = state.get("last_artifact") if isinstance(state.get("last_artifact"), dict) else {}
    previous_metadata = previous.get("query_metadata") if isinstance(previous.get("query_metadata"), list) else []
    previous_rows = previous.get("ranked_rows") if isinstance(previous.get("ranked_rows"), list) else []
    previous_dataset = _first_dataset(previous_metadata)
    normalized = normalize_text(question)
    has_context = bool(previous_rows or previous_metadata)
    asks_why = bool(
        any(
            phrase in normalized
            for phrase in (
                "por que",
                "porque",
                "explica",
                "explicame",
                "que significa",
                "parece problema",
                "datos mal",
                "mal carg",
            )
        )
    )
    mentions_margin_context = bool(
        any(token in normalized for token in ("margen", "costo", "costos", "precio", "precios", "rentabilidad"))
    )
    asks_generic_customers = bool(
        any(token in normalized for token in ("generico", "genericos", "genérico", "genéricos"))
        and any(token in normalized for token in ("cliente", "clientes", "impacto", "incluye"))
    )
    asks_cause_classification = bool(
        any(phrase in normalized for phrase in ("precios", "costos", "datos mal", "mal carg", "cargados"))
        and any(phrase in normalized for phrase in ("parece", "problema", "eso es", "es de"))
    )
    if _is_agent_capability_question(normalized):
        return ConversationPlan(
            intent_type="definition",
            context_dependency="none",
            subject="kali_capabilities",
            reuse_previous_artifact=False,
            required_evidence=[],
            tool_policy="reuse_only",
            answer_contract=AnswerContract(
                must_explain=["conversation", "analytics_capabilities", "ask_follow_ups"],
                must_not=["query_database", "invent_metrics"],
            ),
            confidence=0.78,
        )
    if asks_generic_customers:
        return ConversationPlan(
            intent_type="data_quality_check",
            context_dependency="optional" if has_context else "none",
            subject="generic_customers",
            reuse_previous_artifact=has_context,
            preserve_dataset="customers",
            required_evidence=["customers", "order_count", "total_spent", "generic_customer_concentration"],
            tool_policy="merge" if has_context else "run_tools",
            answer_contract=AnswerContract(
                must_explain=["generic_customer_impact", "identified_vs_generic_customers"],
                must_not=["repeat_product_ranking", "invent_customer_counts"],
            ),
            confidence=0.8,
        )
    if has_context and asks_cause_classification:
        return ConversationPlan(
            intent_type="data_quality_check",
            context_dependency="required",
            subject=str(previous_dataset or state.get("active_entity") or intent.entity),
            reuse_previous_artifact=True,
            preserve_dataset=(
                "product_profitability"
                if previous_dataset == "product_profitability" or mentions_margin_context
                else previous_dataset
            ),
            required_evidence=["price_vs_cost_vs_data_quality", "margin", "cost_source"],
            tool_policy="reuse_only",
            answer_contract=AnswerContract(
                must_explain=["price_signal", "cost_signal", "data_quality_signal", "next_verification"],
                must_not=["repeat_full_ranking", "repeat_same_diagnosis"],
            ),
            confidence=0.76,
        )
    if has_context and asks_why:
        return ConversationPlan(
            intent_type="data_quality_check" if "datos" in normalized or "mal carg" in normalized else "diagnosis",
            context_dependency="required",
            subject=str(previous_dataset or state.get("active_entity") or intent.entity),
            reuse_previous_artifact=True,
            preserve_dataset=(
                "product_profitability"
                if previous_dataset == "product_profitability" and mentions_margin_context
                else previous_dataset
            ),
            required_evidence=["ranked_rows", "query_metadata", "margin", "cost_source"],
            tool_policy="reuse_only",
            answer_contract=AnswerContract(
                must_explain=["negative_margin", "cost_source", "data_quality_uncertainty"],
                must_not=["repeat_full_ranking", "invent_costs"],
            ),
            confidence=0.72,
        )
    if "rank" in intent.operations:
        intent_type: IntentType = "ranking"
    elif "compare" in intent.operations:
        intent_type = "comparison"
    elif "diagnose" in intent.operations or "summarize" in intent.operations:
        intent_type = "diagnosis"
    else:
        intent_type = "metric_lookup"
    return ConversationPlan(
        intent_type=intent_type,
        context_dependency="optional" if has_context else "none",
        subject=intent.entity,
        reuse_previous_artifact=False,
        preserve_dataset=None,
        required_evidence=list(intent.measures),
        tool_policy="auto",
        answer_contract=AnswerContract(),
        confidence=0.55,
    )


def _is_agent_capability_question(normalized: str) -> bool:
    conversational_terms = (
        "conversar",
        "hablar",
        "dialogar",
        "charlar",
        "entiendes contexto",
        "mantener contexto",
        "que puedes hacer",
        "como ayudas",
        "como me ayudas",
        "quien eres",
    )
    if not any(term in normalized for term in conversational_terms):
        return False
    data_terms = (
        "venta",
        "ventas",
        "producto",
        "productos",
        "margen",
        "cliente",
        "clientes",
        "waros",
        "food cost",
        "costo",
        "costos",
    )
    return not any(term in normalized for term in data_terms)
def _compact_conversation_state(state: dict[str, Any]) -> dict[str, Any]:
    previous = state.get("last_artifact") if isinstance(state.get("last_artifact"), dict) else {}
    return {
        "source": state.get("source"),
        "active_entity": state.get("active_entity"),
        "active_grain": state.get("active_grain"),
        "active_period": state.get("active_period"),
        "active_measures": state.get("active_measures"),
        "active_dimensions": state.get("active_dimensions"),
        "advisor_state": state.get("advisor_state"),
        "last_artifact": {
            "question_intent": previous.get("question_intent"),
            "query_metadata": previous.get("query_metadata"),
            "ranked_rows": (previous.get("ranked_rows") or [])[:5]
            if isinstance(previous.get("ranked_rows"), list)
            else [],
            "analysis": previous.get("analysis"),
            "limitations": previous.get("limitations"),
            "summary": previous.get("summary"),
        }
        if previous
        else {},
    }


def _first_dataset(metadata: list[Any]) -> str | None:
    for item in metadata:
        if isinstance(item, dict) and item.get("dataset"):
            return str(item["dataset"])
    return None


def _clean_list(value: list[str]) -> list[str]:
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result
