import hashlib
import hmac
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.config import Settings
from app.dependencies import internal_auth
from app.dependencies.internal_auth import InternalRequestContext, require_internal_request
from app.tools.allowlist import coerce_args, get_tool_spec, resolve_fields
from app.tools.audit import ToolCallAudit
from app.tools.catalog import candidate_tools, discover_tools, tool_catalog
from app.tools.gateway import ToolGateway
from app.tools.models import ToolCallRequest
from app.tools.planner import ToolPlanner
from app.tools.runner import ToolRunError, ToolRunResult, WaroCliRunner


class FakeConnection:
    def __init__(self, tool_call_id: UUID | None = None):
        self.tool_call_id = tool_call_id or uuid4()
        self.fetches = []
        self.executes = []

    async def fetchrow(self, query, *args):
        self.fetches.append((query, args))
        return {"id": self.tool_call_id}

    async def execute(self, query, *args):
        self.executes.append((query, args))


class FakeRunner:
    def __init__(self, result=None, error: ToolRunError | None = None):
        self.result = result or {"data": [{"id": "p1", "name": "Arepa"}]}
        self.error = error
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return ToolRunResult(result=self.result, stderr="", argv=("waro",))


def test_unknown_tool_is_not_allowlisted():
    assert get_tool_spec("shell.rm") is None


def test_fields_are_required_or_defaulted_and_limited():
    spec = get_tool_spec("waro.menu.products")

    assert resolve_fields(spec, None) == ("id", "name", "price", "is_available")

    with pytest.raises(ValueError):
        resolve_fields(spec, ["id", "customer_email"])


def test_sales_metrics_defaults_to_cli_envelope_fields():
    spec = get_tool_spec("waro.sales.metrics")

    assert resolve_fields(spec, None) == ("data", "meta", "success")
    assert resolve_fields(spec, ["totalSales", "totalOrders", "avgTicket"]) == (
        "totalSales",
        "totalOrders",
        "avgTicket",
    )


def test_financial_products_defaults_to_top_level_sections():
    spec = get_tool_spec("waro.financial.products")

    assert resolve_fields(spec, None) == ("products", "metrics", "insights")
    assert resolve_fields(spec, ["products", "metrics"]) == ("products", "metrics")


def test_tool_catalog_exposes_auditable_tool_metadata():
    catalog = tool_catalog()
    sales_metrics = next(tool for tool in catalog if tool["name"] == "waro.sales.metrics")
    tool_names = {tool["name"] for tool in catalog}

    assert sales_metrics["domain"] == "sales"
    assert sales_metrics["scope"] == "orders:read"
    assert "description" in sales_metrics
    assert "arguments_schema" in sales_metrics
    assert "data" in sales_metrics["default_fields"]
    assert "waro.analytics.menu" in tool_names
    assert "waro.customers.metrics" in tool_names


def test_candidate_tools_rank_relevant_scoped_tools():
    candidates = candidate_tools(
        "dame ventas y productos con peor margen",
        preferred_domain="sales",
        scopes=("orders:read", "financial:read"),
    )

    assert [spec.name for spec in candidates][:2] == [
        "waro.sales.metrics",
        "waro.financial.products",
    ]


def test_tool_discovery_records_relevant_tools_rejected_by_scope():
    discovery = discover_tools(
        "realiza un analisis financiero de ventas con margen",
        preferred_domain="sales",
        scopes=("orders:read",),
    )

    assert discovery["available"][0]["name"] == "waro.sales.metrics"
    rejected_by_name = {tool["name"]: tool for tool in discovery["rejected"]}
    assert rejected_by_name["waro.financial.products"]["rejected_reason"] == (
        "missing_scope:financial:read"
    )


def test_sales_tool_planner_adds_financial_context_when_available():
    plan = ToolPlanner().plan_sales(
        question="dame ventas y productos con peor margen",
        period={"date_from": "2026-06-01", "date_to": "2026-06-19"},
        scopes=("orders:read", "financial:read"),
    )

    assert plan.strategy == "catalog_sales_planner_v1"
    assert [step.tool_name for step in plan.steps] == [
        "waro.sales.metrics",
        "waro.financial.products",
    ]
    assert "group-by" not in plan.steps[0].arguments
    assert plan.steps[1].arguments == {"sort-by": "margin", "period": 19}
    assert plan.semantic_plan
    assert plan.semantic_plan["group_by"] == "product"
    assert plan.semantic_plan["sales_metrics_group_by"] is None
    assert plan.available_tools[0]["name"] == "waro.sales.metrics"
    assert "waro.financial.products" not in {
        str(tool["name"]) for tool in plan.rejected_tools
    }


def test_sales_tool_planner_maps_profit_language_to_revenue_sort():
    plan = ToolPlanner().plan_sales(
        question="cuales son los productos con mayor ganancia?",
        period={"date_from": "2026-06-01", "date_to": "2026-06-19"},
        scopes=("orders:read", "financial:read"),
    )

    assert [step.tool_name for step in plan.steps] == [
        "waro.sales.metrics",
        "waro.financial.products",
    ]
    assert plan.steps[1].arguments == {"sort-by": "revenue", "period": 19}


def test_sales_tool_planner_exposes_financial_tool_rejection_without_scope():
    plan = ToolPlanner().plan_sales(
        question="realiza un analisis financiero de ventas con margen",
        period={"date_from": "2026-06-01", "date_to": "2026-06-19"},
        scopes=("orders:read",),
        answer_style="financial_analysis",
    )

    assert [step.tool_name for step in plan.steps] == ["waro.sales.metrics"]
    rejected_by_name = {tool["name"]: tool for tool in plan.rejected_tools}
    assert rejected_by_name["waro.financial.products"]["rejected_reason"] == (
        "missing_scope:financial:read"
    )


def test_sales_tool_planner_does_not_group_single_day_language_by_date():
    plan = ToolPlanner().plan_sales(
        question="dime las ventas del dia de ayer",
        period={"date_from": "2026-06-18", "date_to": "2026-06-18"},
        scopes=("orders:read",),
    )

    assert plan.steps[0].arguments == {
        "date-from": "2026-06-18",
        "date-to": "2026-06-18",
    }


def test_sales_tool_planner_groups_when_user_asks_by_day():
    plan = ToolPlanner().plan_sales(
        question="dame las ventas por dia",
        period={"date_from": "2026-06-01", "date_to": "2026-06-18"},
        scopes=("orders:read",),
    )

    assert plan.steps[0].arguments["group-by"] == "date"


def test_sales_tool_planner_can_plan_multiple_context_tools_when_scopes_allow():
    plan = ToolPlanner().plan_sales(
        question=(
            "haz un analisis de ventas, food cost, menu y clientes del mes "
            "para entender productos y retencion"
        ),
        period={"date_from": "2026-06-01", "date_to": "2026-06-19"},
        scopes=(
            "orders:read",
            "financial:read",
            "analytics:read",
            "menu:read",
            "customers:read",
        ),
        answer_style="financial_analysis",
    )

    assert [step.tool_name for step in plan.steps] == [
        "waro.sales.metrics",
        "waro.financial.products",
        "waro.analytics.food_cost",
        "waro.analytics.menu",
        "waro.menu.products",
        "waro.customers.metrics",
    ]
    assert len(plan.steps) == 6


def test_sales_tool_planner_ranks_customers_when_scope_allows():
    plan = ToolPlanner().plan_sales(
        question="quienes son los mejores clientes del mes",
        period={"date_from": "2026-06-01", "date_to": "2026-06-20"},
        scopes=("orders:read", "customers:read"),
        answer_style="business_analysis",
    )

    assert [step.tool_name for step in plan.steps] == [
        "waro.sales.metrics",
        "waro.customers.metrics",
        "waro.customers.list",
    ]
    assert plan.steps[2].arguments == {
        "date-from": "2026-06-01",
        "date-to": "2026-06-20",
        "sort-field": "total_spent",
        "sort-direction": "desc",
        "limit": 20,
    }


def test_args_reject_extra_flags_and_build_typed_cli_args():
    spec = get_tool_spec("waro.menu.products")

    with pytest.raises(Exception):
        coerce_args(spec, {"limit": 20, "free-form": "--help"})

    args = coerce_args(
        spec,
        {"limit": 20, "is-available": True, "include-ingredients": False},
    )

    assert args.cli_args() == [
        "--limit",
        "20",
        "--offset",
        "0",
        "--is-available",
        "true",
        "--include-recipe-bases",
        "--include-modifiers",
    ]


def test_runner_builds_argv_without_shell_strings():
    settings = Settings(WARO_CLI_BINARY="/tmp/agent-api/.local/bin/waro")
    spec = get_tool_spec("waro.analytics.food_cost")
    args = coerce_args(spec, {"date-from": "2026-06-01"})
    runner = WaroCliRunner(settings)

    argv = runner.build_argv(
        spec=spec,
        args=args,
        fields=("product_id", "product_name"),
        dry_run=True,
    )

    assert argv[:7] == (
        "/tmp/agent-api/.local/bin/waro",
        "--output",
        "json",
        "--no-color",
        "--fields",
        "product_id,product_name",
        "analytics",
    )
    assert argv[7:] == ("food-cost", "--date-from", "2026-06-01", "--dry-run")


@pytest.mark.asyncio
async def test_internal_request_verifies_hmac_signature(monkeypatch):
    secret = "test-secret"
    body = b'{"tool_name":"waro.menu.products"}'
    request_id = "req-123"
    tenant_id = str(uuid4())
    profile_id = str(uuid4())
    scopes = "menu:read,orders:read"
    digest = hashlib.sha256(body).hexdigest()
    canonical = "\n".join(
        [
            "POST",
            "/internal/tools/call",
            request_id,
            tenant_id,
            profile_id,
            "",
            scopes,
            digest,
        ]
    )
    signature = hmac.new(secret.encode(), canonical.encode(), hashlib.sha256).hexdigest()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/internal/tools/call",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
        },
        receive,
    )
    monkeypatch.setattr(
        internal_auth,
        "get_settings",
        lambda: SimpleNamespace(
            is_signature_verification_enabled=True,
            internal_signature_secret=secret,
        ),
    )

    context = await require_internal_request(
        request,
        x_waro_tenant_id=tenant_id,
        x_waro_profile_id=profile_id,
        x_waro_scopes=scopes,
        x_waro_request_id=request_id,
        x_waro_internal_signature=signature,
    )

    assert context.tenant_id == tenant_id
    assert context.scopes == ("menu:read", "orders:read")


@pytest.mark.asyncio
async def test_gateway_rejects_missing_scope_before_persistence():
    connection = FakeConnection()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    gateway = ToolGateway(
        settings=Settings(),
        runner=FakeRunner(),
        connection_factory=connection_factory,
    )
    request = ToolCallRequest(
        run_id=uuid4(),
        tool_name="waro.financial.products",
        arguments={},
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-1",
        member_id=None,
        scopes=("menu:read",),
    )

    with pytest.raises(HTTPException) as exc:
        await gateway.call(request=request, context=context)

    assert exc.value.status_code == 403
    assert connection.fetches == []


@pytest.mark.asyncio
async def test_gateway_persists_sanitized_success():
    connection = FakeConnection()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    gateway = ToolGateway(
        settings=Settings(TOOL_RESULT_MAX_BYTES=10_000),
        runner=FakeRunner(),
        connection_factory=connection_factory,
    )
    request = ToolCallRequest(
        run_id=uuid4(),
        tool_name="waro.menu.products",
        arguments={"limit": 5},
        fields=["id", "name"],
        idempotency_key="idem-1",
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-1",
        member_id=None,
        scopes=("menu:read",),
    )

    response = await gateway.call(request=request, context=context)

    assert response.status == "succeeded"
    assert response.tool_call_id == connection.tool_call_id
    assert len(connection.fetches) == 1
    executed_sql = "\n".join(query for query, _ in connection.executes)
    assert "INSERT INTO audit.ai_action_events" in executed_sql
    insert_args = connection.fetches[0][1]
    assert "WARO_API_KEY" not in str(insert_args)
    assert '"fields": ["id", "name"]' in insert_args[4]


@pytest.mark.asyncio
async def test_gateway_persists_sanitized_error():
    connection = FakeConnection()

    @asynccontextmanager
    async def connection_factory():
        yield connection

    gateway = ToolGateway(
        settings=Settings(WARO_API_KEY="super-secret-key"),
        runner=FakeRunner(
            error=ToolRunError(
                message="Tool execution failed.",
                returncode=1,
                stderr="bad key super-secret-key",
                argv=("waro", "menu", "products"),
            )
        ),
        connection_factory=connection_factory,
    )
    request = ToolCallRequest(
        run_id=uuid4(),
        tool_name="waro.menu.products",
        arguments={"limit": 5},
    )
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-1",
        member_id=None,
        scopes=("menu:read",),
    )

    response = await gateway.call(request=request, context=context)

    assert response.status == "failed"
    assert "super-secret-key" not in str(response.error)
    executed_sql = "\n".join(query for query, _ in connection.executes)
    assert "INSERT INTO audit.ai_action_events" in executed_sql


@pytest.mark.asyncio
async def test_audit_persists_trace_and_span_references():
    connection = FakeConnection()
    audit = ToolCallAudit(connection)
    run_id = uuid4()
    step_id = uuid4()
    context = InternalRequestContext(
        tenant_id=str(uuid4()),
        profile_id=str(uuid4()),
        request_id="req-1",
        member_id=None,
        scopes=("menu:read",),
    )

    tool_call_id = await audit.start(
        context=context,
        run_id=run_id,
        step_id=step_id,
        tool_name="waro.menu.products",
        arguments={"arguments": {}, "fields": ["id"]},
        dry_run=False,
        idempotency_key=None,
        trace_id="0" * 32,
        span_id="1" * 16,
    )

    assert tool_call_id == connection.tool_call_id
    assert len(connection.fetches) == 1
    assert len(connection.executes) == 3
    assert "UPDATE ai.runs" in connection.executes[0][0]
    assert connection.executes[0][1] == ("0" * 32, run_id)
    assert "UPDATE ai.steps" in connection.executes[1][0]
    assert connection.executes[1][1] == ("1" * 16, step_id)
