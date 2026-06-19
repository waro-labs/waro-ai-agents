from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FoodCostQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    conversation_id: UUID | None = None
    date_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    compare_to: str | None = Field(default=None, max_length=80)


class FoodCostEvalResult(BaseModel):
    evaluator_name: str
    score: float
    passed: bool
    result: dict[str, Any]


class FoodCostWorkflowResponse(BaseModel):
    conversation_id: UUID
    run_id: UUID
    input_message_id: UUID
    output_message_id: UUID
    status: Literal["completed", "failed"]
    artifact: dict[str, Any]
    summary: str
    evals: list[FoodCostEvalResult]


class SalesQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    conversation_id: UUID | None = None
    date_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    group_by: Literal["date", "weekday", "hour", "product", "payment", "ticket"] | None = None


class SalesWorkflowResponse(BaseModel):
    conversation_id: UUID
    run_id: UUID
    input_message_id: UUID
    output_message_id: UUID
    status: Literal["completed", "failed"]
    artifact: dict[str, Any]
    summary: str
    evals: list[FoodCostEvalResult]


class AgentQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    conversation_id: UUID | None = None
    workflow: Literal["sales", "food_cost"] | None = None
    date_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    compare_to: str | None = Field(default=None, max_length=80)
    group_by: Literal["date", "weekday", "hour", "product", "payment", "ticket"] | None = None


class AgentRoute(BaseModel):
    workflow: Literal["sales", "food_cost"]
    confidence: float = Field(ge=0, le=1)
    reason: str


class AgentWorkflowResponse(BaseModel):
    conversation_id: UUID
    run_id: UUID
    input_message_id: UUID
    output_message_id: UUID
    status: Literal["completed", "failed"]
    workflow: Literal["sales", "food_cost"]
    route: AgentRoute
    artifact: dict[str, Any]
    summary: str
    evals: list[FoodCostEvalResult]
