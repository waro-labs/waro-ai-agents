from contextlib import asynccontextmanager
from datetime import datetime as real_datetime
import json
from uuid import UUID, uuid4

import pytest

from app.config import Settings
from app.dependencies.internal_auth import InternalRequestContext
from app.llm.base import LLMMessage, LLMResponse, LLMStreamChunk
from app.tools.models import ToolCallResponse
from app.workflows.models import SalesQuestionRequest
from app.workflows.sales import SalesWorkflow


def parse_sse_frame(frame: str) -> tuple[str, dict]:
    lines = frame.splitlines()
    event_line = next(line for line in lines if line.startswith("event: "))
    data_line = next(line for line in lines if line.startswith("data: "))
    return event_line.removeprefix("event: "), json.loads(
        data_line.removeprefix("data: ")
    )


class FakeConnection:
    def __init__(self):
        self.fetches = []
        self.executes = []
        self.ids = [uuid4() for _ in range(20)]
        self.latest_context_summary: str | None = None

    async def fetchrow(self, query, *args):
        self.fetches.append((query, args))
        if "FROM ai.context_summaries" in query:
            if self.latest_context_summary is None:
                return None
            return {"summary": self.latest_context_summary}
        return {"id": self.ids.pop(0)}

    async def execute(self, query, *args):
        self.executes.append((query, args))


class FakeGateway:
    def __init__(self):
        self.calls = []

    async def call(self, *, request, context):
        self.calls.append(request)
        if request.tool_name == "waro.financial.products":
            return ToolCallResponse(
                tool_call_id=uuid4(),
                tool_name=request.tool_name,
                status="succeeded",
                result={
                    "metrics": {
                        "total_profit": 435200,
                        "total_revenue": 1088000,
                    },
                    "insights": {
                        "optimization_needed": {"count": 1},
                    },
                    "products": [
                        {
                            "id": "p1",
                            "name": "Burger",
                            "margin": 0.24,
                            "total_revenue": 180000,
                            "cost": 136800,
                            "sales": 8,
                            "profit": 43200,
                            "classification": "Low Performance",
                        }
                    ],
                    "meta": {"period": request.arguments.get("period")},
                    "success": True,
                },
                result_summary="Returned product financial rows.",
            )
        return ToolCallResponse(
            tool_call_id=uuid4(),
            tool_name=request.tool_name,
            status="succeeded",
            result={
                "data": {
                    "totalSales": 431500.0,
                    "totalOrders": 14,
                    "avgTicket": 30821.428571428572,
                    "series": [{"date": "2026-06-17", "totalSales": 431500.0}],
                },
                "meta": {"dateFrom": "2026-06-17", "dateTo": "2026-06-17"},
                "success": True,
            },
            result_summary="Returned sales metrics.",
        )


class FakeLLMAdapter:
    provider = "fake"

    def __init__(self, content: str = "Resumen de ventas generado por LLM."):
        self.content = content
        self.calls: list[list[LLMMessage]] = []
        self.call_kwargs: list[dict] = []

    async def complete(self, *, messages, temperature=0.2, model=None):
        self.calls.append(messages)
        self.call_kwargs.append({"temperature": temperature, "model": model})
        return LLMResponse(
            content=self.content,
            model=model or "fake-model",
            provider=self.provider,
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            estimated_cost_usd=0.0001,
            prompt_cost_usd=0.00004,
            completion_cost_usd=0.00006,
            cost_source="test",
        )


class FakeStreamingLLMAdapter(FakeLLMAdapter):
    async def stream_complete(self, *, messages, temperature=0.2, model=None):
        self.calls.append(messages)
        self.call_kwargs.append({"temperature": temperature, "model": model, "stream": True})
        yield LLMStreamChunk(text="Ayer vendiste ")
        yield LLMStreamChunk(text="$431.500.")
        yield LLMStreamChunk(
            response=LLMResponse(
                content="Ayer vendiste $431.500.",
                model=model or "fake-model",
                provider=self.provider,
                input_tokens=100,
                output_tokens=40,
                total_tokens=140,
                estimated_cost_usd=0.0001,
                cost_source="test",
            )
        )


class FixedDateTime(real_datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 6, 19, 10, 30, tzinfo=tz)


def test_sales_workflow_resolves_relative_periods_from_question(monkeypatch):
    monkeypatch.setattr("app.workflows.sales.datetime", FixedDateTime)
    workflow = SalesWorkflow(settings=Settings())

    assert workflow._resolve_period(
        SalesQuestionRequest(question="dame las ventas de ayer")
    ) == {"date_from": "2026-06-18", "date_to": "2026-06-18"}
    assert workflow._resolve_period(
        SalesQuestionRequest(question="dame las ventas del mes")
    ) == {"date_from": "2026-06-01", "date_to": "2026-06-19"}
    assert workflow._resolve_period(
        SalesQuestionRequest(question="dime los 20 productos mas vendidos dle presnete mes")
    ) == {"date_from": "2026-06-01", "date_to": "2026-06-19"}
    assert workflow._resolve_period(
        SalesQuestionRequest(question="cuales son las ventas del mes pasado?")
    ) == {"date_from": "2026-05-01", "date_to": "2026-05-31"}
    assert workflow._resolve_period(
        SalesQuestionRequest(question="dame las ventas de los ultimos 7 dias")
    ) == {"date_from": "2026-06-13", "date_to": "2026-06-19"}
    assert workflow._resolve_period(
        SalesQuestionRequest(question="dime las ventas de los ultimos 15 dias")
    ) == {"date_from": "2026-06-05", "date_to": "2026-06-19"}
    assert workflow._resolve_period(
        SalesQuestionRequest(question="dime las ventas de los ultimos quince dias")
    ) == {"date_from": "2026-06-05", "date_to": "2026-06-19"}
    assert workflow._resolve_period(
        SalesQuestionRequest(question="cuales son las ventas del dia 11 de junio del 2026")
    ) == {"date_from": "2026-06-11", "date_to": "2026-06-11"}
    assert workflow._resolve_period(
        SalesQuestionRequest(question="11 no 18")
    ) == {"date_from": "2026-06-11", "date_to": "2026-06-11"}
    assert workflow._resolve_period(
        SalesQuestionRequest(question="ventas 2026-06-11")
    ) == {"date_from": "2026-06-11", "date_to": "2026-06-11"}


def test_sales_workflow_routes_small_talk_without_sales_tool():
    workflow = SalesWorkflow(settings=Settings())

    assert workflow._resolve_sales_intent("hola") == "small_talk"
    assert workflow._resolve_sales_intent("como estas") == "small_talk"
    assert workflow._resolve_sales_intent("funcionas") == "small_talk"
    assert workflow._resolve_sales_intent("hola, dame las ventas de ayer") == "sales_metrics"
    assert workflow._resolve_sales_intent("dime las ventas de los ultimos 15 dias") == "sales_metrics"
    assert (
        workflow._resolve_sales_intent("como asi", has_conversation_context=True)
        == "follow_up"
    )
    assert workflow._resolve_sales_intent("como asi") == "sales_metrics"


def test_sales_workflow_resolves_answer_style():
    workflow = SalesWorkflow(settings=Settings())

    assert workflow._resolve_answer_style("cuales son las ventas de ayer") == "direct_metric"
    assert workflow._resolve_answer_style("analiza las ventas del mes") == "business_analysis"
    assert (
        workflow._resolve_answer_style("dime los productos mas vendidos del mes")
        == "business_analysis"
    )
    assert (
        workflow._resolve_answer_style("hazme un analisis financiero del mes")
        == "financial_analysis"
    )


def test_sales_workflow_product_fallback_does_not_return_generic_sales_summary():
    workflow = SalesWorkflow(settings=Settings())

    summary = workflow._build_summary(
        {
            "intent": "sales_metrics",
            "answer_style": "business_analysis",
            "period": {"date_from": "2026-06-01", "date_to": "2026-06-20"},
            "metrics": {
                "total_sales": 10107000,
                "order_count": 393,
                "avg_ticket": 27024,
            },
            "analysis_request": {
                "objective": "rank_entities",
                "dimensions": ["overall", "product"],
                "requested_metrics": ["sales", "product_ranking"],
            },
            "auxiliary_context": {
                "financial_products": [
                    {
                        "name": "Burger",
                        "quantity": 12,
                        "revenue": 240000,
                        "profit": 96000,
                    }
                ]
            },
            "tool_calls": [],
        }
    )

    assert "Burger" in summary
    assert "vendiste" not in summary


def test_sales_workflow_product_fallback_respects_requested_limit():
    workflow = SalesWorkflow(settings=Settings())
    products = [
        {"name": f"Producto {index}", "quantity": 30 - index, "profit": index * 1000}
        for index in range(1, 9)
    ]

    summary = workflow._build_summary(
        {
            "intent": "sales_metrics",
            "answer_style": "business_analysis",
            "period": {"date_from": "2026-06-01", "date_to": "2026-06-20"},
            "metrics": {"total_sales": 10107000, "order_count": 393},
            "analysis_request": {
                "request_kind": "product_ranking",
                "objective": "rank_entities",
                "dimensions": ["overall", "product"],
                "requested_metrics": ["quantity_sold", "product_ranking"],
                "limit": 7,
            },
            "auxiliary_context": {"financial_products": products},
            "tool_calls": [],
        }
    )

    assert summary.count("\n") == 7
    assert "Producto 7" in summary
    assert "Producto 8" not in summary
    assert "No pude generar" not in summary
    assert "vendiste" not in summary


def test_sales_workflow_product_fallback_respects_profitability_sort_field():
    workflow = SalesWorkflow(settings=Settings())

    summary = workflow._build_summary(
        {
            "intent": "sales_metrics",
            "answer_style": "business_analysis",
            "period": {"date_from": "2026-06-01", "date_to": "2026-06-20"},
            "metrics": {"total_sales": 10107000, "order_count": 393},
            "analysis_request": {
                "request_kind": "product_ranking",
                "objective": "rank_entities",
                "dimensions": ["overall", "product"],
                "requested_metrics": ["quantity_sold", "product_ranking"],
                "sort_field": "margin",
                "limit": 2,
            },
            "auxiliary_context": {
                "financial_products": [
                    {"name": "Muchas unidades", "quantity": 100, "profit": 10000},
                    {"name": "Mas rentable", "quantity": 3, "profit": 90000},
                    {"name": "Rentable medio", "quantity": 5, "profit": 50000},
                ]
            },
            "tool_calls": [],
        }
    )

    assert summary.splitlines()[1].startswith("1. Mas rentable")
    assert summary.splitlines()[2].startswith("2. Rentable medio")
    assert "Muchas unidades" not in summary


def test_sales_workflow_customer_fallback_does_not_return_generic_sales_summary():
    workflow = SalesWorkflow(settings=Settings())

    summary = workflow._build_summary(
        {
            "intent": "sales_metrics",
            "answer_style": "business_analysis",
            "period": {"date_from": "2026-06-01", "date_to": "2026-06-20"},
            "metrics": {
                "total_sales": 10107000,
                "order_count": 393,
                "avg_ticket": 27024,
            },
            "analysis_request": {
                "request_kind": "customer_ranking",
                "area": "customers",
                "objective": "rank_entities",
                "dimensions": ["overall", "customer"],
                "requested_metrics": ["frequency", "customer_activity"],
                "limit": 2,
            },
            "response_contract": {
                "request_kind": "customer_ranking",
                "safe_to_answer": True,
            },
            "auxiliary_context": {
                "customers": [
                    {"name": "Cliente A", "order_count": 12, "total_spent": 300000},
                    {"name": "Cliente B", "order_count": 9, "total_spent": 250000},
                    {"name": "Cliente C", "order_count": 4, "total_spent": 100000},
                ]
            },
            "tool_calls": [],
        }
    )

    assert "Cliente A" in summary
    assert "Cliente B" in summary
    assert "Cliente C" not in summary
    assert "vendiste" not in summary


def test_sales_workflow_product_fallback_reports_product_failure():
    workflow = SalesWorkflow(settings=Settings())

    summary = workflow._build_summary(
        {
            "intent": "sales_metrics",
            "answer_style": "business_analysis",
            "period": {"date_from": "2026-06-01", "date_to": "2026-06-20"},
            "metrics": {
                "total_sales": 10107000,
                "order_count": 393,
                "avg_ticket": 27024,
            },
            "analysis_request": {
                "objective": "rank_entities",
                "dimensions": ["overall", "product"],
                "requested_metrics": ["sales", "product_ranking"],
            },
            "auxiliary_context": {},
            "tool_calls": [
                {
                    "tool_name": "waro.financial.products",
                    "status": "failed",
                }
            ],
        }
    )

    assert "Fallo la consulta de datos de producto" in summary
    assert "vendiste" not in summary


def test_sales_workflow_response_contract_blocks_wrong_fallback():
    workflow = SalesWorkflow(settings=Settings())

    summary = workflow._build_summary(
        {
            "intent": "sales_metrics",
            "answer_style": "business_analysis",
            "period": {"date_from": "2026-06-01", "date_to": "2026-06-20"},
            "metrics": {
                "total_sales": 10107000,
                "order_count": 393,
                "avg_ticket": 27024,
            },
            "response_contract": {
                "request_kind": "product_ranking",
                "safe_to_answer": False,
                "error_message": (
                    "No pude completar el ranking de productos para Del 2026-06-01 al 2026-06-20: "
                    "faltan datos requeridos (product_financials)."
                ),
            },
        }
    )

    assert "ranking de productos" in summary
    assert "vendiste" not in summary


def test_sales_workflow_aggregates_grouped_metric_rows():
    workflow = SalesWorkflow(settings=Settings())
    data = workflow._metrics_data_for(
        [
            {
                "tool_name": "waro.sales.metrics",
                "status": "succeeded",
                "result": {
                    "data": [
                        {"date": "2026-06-01", "totalSales": 100000, "totalOrders": 4},
                        {"date": "2026-06-02", "totalSales": 200000, "totalOrders": 6},
                    ]
                },
            }
        ],
        "waro.sales.metrics",
    )

    assert data["totalSales"] == 300000
    assert data["totalOrders"] == 10
    assert data["avgTicket"] == 30000
    assert len(data["series"]) == 2


def test_sales_workflow_daily_fallback_uses_series_not_generic_summary():
    workflow = SalesWorkflow(settings=Settings())

    summary = workflow._build_summary(
        {
            "intent": "sales_metrics",
            "answer_style": "business_analysis",
            "period": {"date_from": "2026-06-01", "date_to": "2026-06-20"},
            "metrics": {
                "total_sales": 300000,
                "order_count": 10,
                "avg_ticket": 30000,
                "series": [{"date": "2026-06-01"}, {"date": "2026-06-02"}],
            },
            "response_contract": {
                "request_kind": "daily_analysis",
                "safe_to_answer": True,
            },
        }
    )

    assert "encontre 2 dias con datos" in summary
    assert "Sales workflow completed" not in summary


@pytest.mark.asyncio
async def test_sales_semantic_planner_guardrails_previous_month(monkeypatch):
    monkeypatch.setattr("app.workflows.sales.datetime", FixedDateTime)
    connection = FakeConnection()
    llm = FakeLLMAdapter(
        content=json.dumps(
            {
                "intent": "sales_metrics",
                "date_from": "2026-06-01",
                "date_to": "2026-06-19",
                "group_by": None,
                "answer_style": "direct_metric",
                "tools": [{"name": "waro.sales.metrics", "reason": "wrong month"}],
                "confidence": 0.9,
                "reason": "misread current month",
            }
        )
    )

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-planner",
        member_id=None,
        scopes=("orders:read",),
    )

    plan = await workflow._plan_sales_tool_calls(
        request=SalesQuestionRequest(question="cuales son las ventas del mes pasado?"),
        context=context,
        run_id=uuid4(),
    )

    assert plan.steps[0].arguments == {
        "date-from": "2026-05-01",
        "date-to": "2026-05-31",
    }
    assert plan.semantic_plan
    assert plan.semantic_plan["period_source"] == "guardrail_previous_month"
    assert plan.semantic_plan["source"] == "guardrail_previous_month"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_sales_semantic_planner_groups_daily_average(monkeypatch):
    monkeypatch.setattr("app.workflows.sales.datetime", FixedDateTime)
    connection = FakeConnection()
    llm = FakeLLMAdapter(
        content=json.dumps(
            {
                "intent": "sales_metrics",
                "date_from": "2026-06-01",
                "date_to": "2026-06-19",
                "group_by": None,
                "answer_style": "direct_metric",
                "tools": [{"name": "waro.sales.metrics", "reason": "daily average"}],
                "confidence": 0.8,
                "reason": "needs daily average",
            }
        )
    )

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-planner-daily",
        member_id=None,
        scopes=("orders:read",),
    )

    plan = await workflow._plan_sales_tool_calls(
        request=SalesQuestionRequest(question="cuanto es el promedio de ventas por dia del mes"),
        context=context,
        run_id=uuid4(),
    )

    assert plan.steps[0].arguments == {
        "date-from": "2026-06-01",
        "date-to": "2026-06-19",
        "group-by": "date",
    }
    assert plan.semantic_plan
    assert plan.semantic_plan["group_by"] == "date"


@pytest.mark.asyncio
async def test_sales_semantic_planner_clamps_future_current_month(monkeypatch):
    monkeypatch.setattr("app.workflows.sales.datetime", FixedDateTime)
    connection = FakeConnection()
    llm = FakeLLMAdapter(
        content=json.dumps(
            {
                "intent": "sales_metrics",
                "date_from": "2026-06-01",
                "date_to": "2026-06-30",
                "group_by": "date",
                "answer_style": "direct_metric",
                "tools": [{"name": "waro.sales.metrics", "reason": "month average"}],
                "confidence": 0.8,
                "reason": "full calendar month",
            }
        )
    )

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-planner-future-clamp",
        member_id=None,
        scopes=("orders:read",),
    )

    plan = await workflow._plan_sales_tool_calls(
        request=SalesQuestionRequest(question="cuál fue el promedio de ventas por día del mes"),
        context=context,
        run_id=uuid4(),
    )

    assert plan.steps[0].arguments == {
        "date-from": "2026-06-01",
        "date-to": "2026-06-19",
        "group-by": "date",
    }
    assert plan.semantic_plan
    assert plan.semantic_plan["period_source"] == "guardrail_future_clamped"
    assert plan.semantic_plan["answer_style"] == "business_analysis"


def test_sales_workflow_keeps_explicit_dates_ahead_of_question(monkeypatch):
    monkeypatch.setattr("app.workflows.sales.datetime", FixedDateTime)
    workflow = SalesWorkflow(settings=Settings())

    assert workflow._resolve_period(
        SalesQuestionRequest(
            question="dame las ventas del mes",
            date_from="2026-05-01",
            date_to="2026-05-31",
        )
    ) == {"date_from": "2026-05-01", "date_to": "2026-05-31"}


@pytest.mark.asyncio
async def test_sales_workflow_persists_run_tools_message_summary_and_evals():
    connection = FakeConnection()
    gateway = FakeGateway()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(),
        gateway=gateway,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales",
        member_id=None,
        scopes=("orders:read",),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(
            question="Cuanto vendi ayer?",
            date_from="2026-06-17",
            date_to="2026-06-17",
        ),
        context=context,
    )

    assert response.status == "completed"
    assert isinstance(response.run_id, UUID)
    assert [call.tool_name for call in gateway.calls] == ["waro.sales.metrics"]
    assert gateway.calls[0].arguments == {
        "date-from": "2026-06-17",
        "date-to": "2026-06-17",
    }
    assert gateway.calls[0].fields == ["data", "meta", "success"]
    assert response.artifact["metrics"]["total_sales"] == 431500.0
    assert response.artifact["metrics"]["order_count"] == 14
    assert response.artifact["metrics"]["avg_ticket"] == 30821.428571428572
    assert response.artifact["answer_style"] == "direct_metric"
    assert response.summary == (
        "El 2026-06-17: vendiste $431.500 en 14 ordenes. "
        "Ticket promedio: $30.821."
    )
    assert "question" not in response.artifact
    assert response.evals[0].evaluator_name == "sales_tool_usage"
    assert all(eval_result.passed for eval_result in response.evals)

    executed_sql = "\n".join(query for query, _ in connection.executes)
    fetched_sql = "\n".join(query for query, _ in connection.fetches)
    assert "INSERT INTO ai.conversations" in fetched_sql
    assert "INSERT INTO ai.messages" in fetched_sql
    assert "INSERT INTO ai.runs" in fetched_sql
    assert "INSERT INTO ai.steps" in fetched_sql
    assert "INSERT INTO ai.context_summaries" in executed_sql
    assert executed_sql.count("INSERT INTO ai.eval_results") == 2
    assert "status = 'completed'" in executed_sql
    step_payloads = " ".join(
        str(args)
        for query, args in connection.fetches
        if "INSERT INTO ai.steps" in query
    )
    assert "Cuanto vendi ayer?" not in step_payloads
    assert "question_length" in step_payloads


@pytest.mark.asyncio
async def test_sales_workflow_plans_auxiliary_financial_tool_for_product_context():
    connection = FakeConnection()
    gateway = FakeGateway()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(),
        gateway=gateway,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-products",
        member_id=None,
        scopes=("orders:read", "financial:read"),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(
            question="Dame ventas y productos con peor margen",
            date_from="2026-06-01",
            date_to="2026-06-19",
        ),
        context=context,
    )

    assert response.status == "completed"
    assert [call.tool_name for call in gateway.calls] == [
        "waro.sales.metrics",
        "waro.financial.products",
    ]
    assert "group-by" not in gateway.calls[0].arguments
    assert gateway.calls[1].arguments == {"sort-by": "margin", "period": 19}
    assert gateway.calls[1].fields == ["products", "metrics", "insights"]
    assert response.artifact["tool_plan"]["strategy"] == "catalog_sales_planner_v1"
    assert response.artifact["tool_plan"]["steps"][1]["tool_name"] == (
        "waro.financial.products"
    )
    assert response.artifact["auxiliary_context"]["financial_products"][0]["name"] == (
        "Burger"
    )
    assert response.artifact["auxiliary_context"]["financial_products"][0]["revenue"] == 180000
    assert response.artifact["auxiliary_context"]["financial_products"][0]["quantity"] == 8

    step_payloads = " ".join(
        str(args)
        for query, args in connection.fetches
        if "INSERT INTO ai.steps" in query
    )
    assert "sales_tool_planner" in step_payloads
    assert "Product financial context can explain sales performance." in step_payloads


@pytest.mark.asyncio
async def test_sales_workflow_plans_financial_tool_for_financial_analysis():
    connection = FakeConnection()
    gateway = FakeGateway()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(),
        gateway=gateway,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-financial-analysis",
        member_id=None,
        scopes=("orders:read", "financial:read"),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(
            question="realiza un analisis financiero del mes de junio",
            date_from="2026-06-01",
            date_to="2026-06-19",
        ),
        context=context,
    )

    assert response.artifact["answer_style"] == "financial_analysis"
    assert [call.tool_name for call in gateway.calls] == [
        "waro.sales.metrics",
        "waro.financial.products",
    ]
    assert gateway.calls[1].arguments == {"sort-by": "margin", "period": 19}
    assert gateway.calls[1].fields == ["products", "metrics", "insights"]
    assert response.artifact["auxiliary_context"]["financial_products"][0]["name"] == (
        "Burger"
    )
    assert response.artifact["auxiliary_context"]["financial_metrics"]["total_profit"] == 435200
    assert response.artifact["analysis_request"]["area"] == "finance"
    assert response.artifact["analysis_request"]["objective"] == "evaluate_gross_profitability"
    assert response.artifact["analysis_request"]["dimensions"] == ["overall", "product"]
    assert response.artifact["analysis_request"]["data_available"]["sales"] is True
    assert response.artifact["analysis_request"]["data_available"]["product_financials"] is True
    assert response.artifact["analysis_request"]["data_available"]["labor_cost"] is False
    financial_analysis = response.artifact["financial_analysis"]
    assert financial_analysis["analysis_quality"] == "partial"
    assert financial_analysis["analysis_scope"] == "commercial_finance_gross_margin"
    assert "net_profit" in financial_analysis["missing_metrics"]
    assert "prime_cost" in financial_analysis["missing_metrics"]
    assert financial_analysis["supported_metrics"]["product_gross_profit"] == 435200
    assert financial_analysis["top_products_by_profit"][0]["name"] == "Burger"
    assert financial_analysis["top_products_by_profit"][0]["profit"] == 43200


@pytest.mark.asyncio
async def test_sales_workflow_handles_small_talk_without_tool_call():
    connection = FakeConnection()
    gateway = FakeGateway()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(),
        gateway=gateway,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-small-talk",
        member_id=None,
        scopes=("orders:read",),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(question="hola"),
        context=context,
    )

    assert response.status == "completed"
    assert gateway.calls == []
    assert response.artifact["intent"] == "small_talk"
    assert response.artifact["period"] is None
    assert response.artifact["tool_calls"] == []
    assert response.summary.startswith("Hola, estoy funcionando.")
    assert {eval_result.evaluator_name for eval_result in response.evals} == {
        "sales_intent_guard",
        "sales_business_usefulness",
    }
    assert all(eval_result.passed for eval_result in response.evals)
    step_payloads = " ".join(
        str(args)
        for query, args in connection.fetches
        if "INSERT INTO ai.steps" in query
    )
    assert "small_talk" in step_payloads
    assert "tool_planned" in step_payloads


@pytest.mark.asyncio
async def test_sales_workflow_handles_follow_up_with_conversation_context():
    connection = FakeConnection()
    connection.latest_context_summary = (
        "El 2026-06-18 vendiste $423.000 en 17 ordenes. "
        "Ticket promedio: $24.882."
    )
    gateway = FakeGateway()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(),
        gateway=gateway,
        connection_factory=connection_factory,
    )
    conversation_id = uuid4()
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-follow-up",
        member_id=None,
        scopes=("orders:read",),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(
            question="como asi?",
            conversation_id=conversation_id,
        ),
        context=context,
    )

    assert response.artifact["intent"] == "follow_up"
    assert response.artifact["previous_context_summary"] == connection.latest_context_summary
    assert gateway.calls == []
    assert response.summary.startswith("Claro. Me referia a esto:")
    assert all(eval_result.passed for eval_result in response.evals)
    assert any(
        "FROM ai.context_summaries" in query
        for query, _ in connection.fetches
    )


@pytest.mark.asyncio
async def test_sales_workflow_inherits_context_for_product_profit_drilldown():
    connection = FakeConnection()
    connection.latest_context_summary = (
        "Durante el periodo del 1 de junio al 19 de junio de 2026, "
        "las ventas totales fueron de 10,107,000 COP, con un margen promedio "
        "del 40% en los productos. El producto con el mejor margen fue el "
        "\"COMBO 4 PAREJA BURGUERS POLLO/RES/CHORIZO\", que genero una "
        "ganancia real de 448,800 COP."
    )
    gateway = FakeGateway()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(),
        gateway=gateway,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-product-drilldown",
        member_id=None,
        scopes=("orders:read", "financial:read"),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(
            question="cuales son los productos con mayor ganancia?",
            conversation_id=uuid4(),
        ),
        context=context,
    )

    assert response.artifact["answer_style"] == "financial_analysis"
    assert response.artifact["semantic_plan"]["period"] == {
        "date_from": "2026-06-01",
        "date_to": "2026-06-19",
    }
    assert response.artifact["semantic_plan"]["period_source"] == "conversation_context"
    assert response.artifact["semantic_plan"]["context_resolution"]["inherited_period"] is True
    assert [call.tool_name for call in gateway.calls] == [
        "waro.sales.metrics",
        "waro.financial.products",
    ]
    assert gateway.calls[0].arguments == {
        "date-from": "2026-06-01",
        "date-to": "2026-06-19",
    }
    assert gateway.calls[1].arguments == {"sort-by": "revenue", "period": 19}
    assert any(
        "sales_context_resolver" in str(args)
        for query, args in connection.fetches
        if "INSERT INTO ai.steps" in query
    )


@pytest.mark.asyncio
async def test_sales_workflow_uses_llm_summary_when_enabled():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(content="Ayer vendiste $431.500 con ticket promedio de $30.821.")

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_MODEL="moonshot-v1-8k",
            KIMI_PLANNER_MODEL=None,
            KIMI_COMPOSER_MODEL=None,
        ),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-llm",
        member_id=None,
        scopes=("orders:read",),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(
            question="Analiza cuanto vendi ayer?",
            date_from="2026-06-17",
            date_to="2026-06-17",
        ),
        context=context,
    )

    assert response.summary == "Ayer vendiste $431.500 con ticket promedio de $30.821."
    assert len(llm.calls) == 2
    assert llm.call_kwargs[0]["model"] == "moonshot-v1-8k"
    assert llm.call_kwargs[1]["model"] == "moonshot-v1-8k"
    assert llm.calls[0][0].role == "system"
    assert "planner semantico" in llm.calls[0][0].content
    assert "analista senior de ventas" in llm.calls[1][0].content
    assert "answer_style" in llm.calls[1][1].content
    fetched_sql = "\n".join(query for query, _ in connection.fetches)
    assert "sales_summary" in str(connection.fetches)
    assert "INSERT INTO ai.steps" in fetched_sql


@pytest.mark.asyncio
async def test_sales_workflow_uses_role_models_for_planner_and_composer():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(content="Resumen comercial.")

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_MODEL="kimi-default",
            KIMI_PLANNER_MODEL="kimi-cheap-planner",
            KIMI_COMPOSER_MODEL="kimi-composer",
        ),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-role-models",
        member_id=None,
        scopes=("orders:read",),
    )

    response = await workflow.run(
        request=SalesQuestionRequest(
            question="Analiza cuanto vendi ayer?",
            date_from="2026-06-17",
            date_to="2026-06-17",
        ),
        context=context,
    )

    assert response.summary == "Resumen comercial."
    assert llm.call_kwargs[0]["model"] == "kimi-cheap-planner"
    assert llm.call_kwargs[1]["model"] == "kimi-composer"


@pytest.mark.asyncio
async def test_sales_workflow_uses_analysis_model_for_financial_summary():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(
        content=json.dumps(
            {
                "intent": "sales_metrics",
                "date_from": "2026-06-01",
                "date_to": "2026-06-19",
                "group_by": None,
                "answer_style": "financial_analysis",
                "tools": [{"name": "waro.financial.products", "reason": "analysis"}],
                "confidence": 0.9,
                "reason": "financial analysis",
            }
        )
    )

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(
            LLM_PROVIDER="kimi",
            KIMI_API_KEY="test-key",
            KIMI_MODEL="kimi-default",
            KIMI_PLANNER_MODEL="kimi-cheap-planner",
            KIMI_ANALYSIS_MODEL="kimi-strong-analysis",
        ),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-analysis-model",
        member_id=None,
        scopes=("orders:read", "financial:read"),
    )

    await workflow.run(
        request=SalesQuestionRequest(question="dame un analisis financiero del mes"),
        context=context,
    )

    assert llm.call_kwargs[0]["model"] == "kimi-cheap-planner"
    assert llm.call_kwargs[1]["model"] == "kimi-strong-analysis"


@pytest.mark.asyncio
async def test_sales_workflow_streams_event_order_and_final_payload():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(content="Ayer vendiste $431.500 con ticket promedio de $30.821.")

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-stream",
        member_id=None,
        scopes=("orders:read",),
    )

    events = [
        event
        async for event in workflow.stream(
            request=SalesQuestionRequest(
                question="Cuanto vendi ayer?",
                date_from="2026-06-17",
                date_to="2026-06-17",
            ),
            context=context,
        )
    ]

    assert [event.event for event in events] == [
        "run_started",
        "step_started",
        "tool_started",
        "tool_finished",
        "final",
    ]
    assert [call.tool_name for call in gateway.calls] == ["waro.sales.metrics"]

    final_event_name, final_payload = parse_sse_frame(events[-1].to_sse())
    assert final_event_name == "final"
    assert final_payload["status"] == "completed"
    assert final_payload["summary"] == (
        "El 2026-06-17: vendiste $431.500 en 14 ordenes. "
        "Ticket promedio: $30.821."
    )
    assert final_payload["artifact_summary"]["answer_style"] == "direct_metric"
    assert final_payload["artifact_summary"]["metrics"] == {
        "total_sales": 431500.0,
        "order_count": 14,
        "avg_ticket": 30821.428571428572,
    }
    assert final_payload["artifact_summary"]["tool_calls"][0]["tool_name"] == "waro.sales.metrics"
    assert "question" not in final_payload["artifact_summary"]

    executed_sql = "\n".join(query for query, _ in connection.executes)
    fetched_sql = "\n".join(query for query, _ in connection.fetches)
    assert "INSERT INTO ai.runs" in fetched_sql
    assert "INSERT INTO ai.steps" in fetched_sql
    assert executed_sql.count("INSERT INTO ai.eval_results") == 2
    assert "status = 'completed'" in executed_sql


@pytest.mark.asyncio
async def test_sales_workflow_streams_small_talk_without_tool_or_llm():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeLLMAdapter(content="No deberia llamarse.")

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-small-talk-stream",
        member_id=None,
        scopes=("orders:read",),
    )

    events = [
        event
        async for event in workflow.stream(
            request=SalesQuestionRequest(question="como estas"),
            context=context,
        )
    ]

    assert [event.event for event in events] == ["run_started", "step_started", "final"]
    assert gateway.calls == []
    assert llm.calls == []
    final_event_name, final_payload = parse_sse_frame(events[-1].to_sse())
    assert final_event_name == "final"
    assert final_payload["summary"].startswith("Hola, estoy funcionando.")
    assert final_payload["artifact_summary"]["intent"] == "small_talk"


@pytest.mark.asyncio
async def test_sales_workflow_streams_llm_token_events_before_final_payload():
    connection = FakeConnection()
    gateway = FakeGateway()
    llm = FakeStreamingLLMAdapter()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    workflow = SalesWorkflow(
        settings=Settings(LLM_PROVIDER="kimi", KIMI_API_KEY="test-key"),
        gateway=gateway,
        llm_adapter=llm,
        connection_factory=connection_factory,
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-sales-token-stream",
        member_id=None,
        scopes=("orders:read",),
    )

    events = [
        event
        async for event in workflow.stream(
            request=SalesQuestionRequest(question="Analiza cuanto vendi ayer?"),
            context=context,
        )
    ]

    assert [event.event for event in events][-4:] == [
        "llm_started",
        "token",
        "token",
        "final",
    ]
    token_payloads = [
        parse_sse_frame(event.to_sse())[1]
        for event in events
        if event.event == "token"
    ]
    assert [payload["text"] for payload in token_payloads] == [
        "Ayer vendiste ",
        "$431.500.",
    ]
    assert all("content" not in payload for payload in token_payloads)
    assert parse_sse_frame(events[-1].to_sse())[1]["summary"] == "Ayer vendiste $431.500."
    assert len(llm.calls) == 2
    assert "planner semantico" in llm.calls[0][0].content
