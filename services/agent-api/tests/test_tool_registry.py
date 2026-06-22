import json

import pytest

from app.config import Settings
from app.tools.registry import ToolRegistry, set_tool_registry
from app.tools.schema_v2 import (
    AGENT_SCHEMA_V2,
    parse_agent_catalog_payload,
    static_tool_to_v2_dict,
    tool_schema_from_dict,
)
from app.tools.allowlist import TOOL_SPECS


def test_parse_agent_catalog_v2_envelope():
    payload = {
        "schema_version": AGENT_SCHEMA_V2,
        "tools": [
            {
                "name": "waro.inventory.stock",
                "command": ["inventory", "stock"],
                "scope": "inventory:read",
                "domain": "inventory",
                "description": "Stock levels by ingredient",
                "tags": ["inventory", "stock"],
                "examples": ["stock bajo de insumos"],
                "capabilities": {
                    "entity": "ingredient",
                    "measures": ["quantity"],
                    "dimensions": ["sku", "name"],
                    "supported_operations": ["filter", "rank", "limit"],
                    "supports_period": False,
                },
                "arguments": {"type": "object", "properties": {"limit": {"type": "integer"}}},
                "response": {
                    "shape": "rows",
                    "row_path": "items",
                    "fields": ["sku", "name", "quantity"],
                    "default_fields": ["sku", "name", "quantity"],
                    "top_level_keys": [],
                },
            }
        ],
    }
    version, tools = parse_agent_catalog_payload(payload)
    assert version == AGENT_SCHEMA_V2
    assert len(tools) == 1
    assert tools[0].name == "waro.inventory.stock"
    assert tools[0].capabilities["entity"] == "ingredient"


def test_static_tool_exports_v2_metadata():
    spec = TOOL_SPECS["waro.sales.metrics"]
    exported = static_tool_to_v2_dict(spec)
    schema = tool_schema_from_dict(exported)
    assert schema is not None
    assert schema.name == "waro.sales.metrics"
    assert "totalSales" in schema.capabilities.get("measures", [])


@pytest.mark.asyncio
async def test_tool_registry_loads_static_fallback():
    settings = Settings(TOOL_CATALOG_SOURCE="static")
    registry = ToolRegistry(settings)
    set_tool_registry(registry)
    snapshot = await registry.refresh(force=True)
    assert snapshot.source == "static"
    assert "waro.sales.metrics" in snapshot.tools
    catalog = await registry.catalog_metadata()
    assert any(item["name"] == "waro.sales.metrics" for item in catalog)


@pytest.mark.asyncio
async def test_tool_registry_merges_dynamic_cli_tool(monkeypatch):
    dynamic_tools = [
        ("waro.inventory.stock", ["inventory", "stock"], "inventory", "inventory:read"),
        ("waro.purchases.items", ["purchases", "items"], "purchases", "purchases:read"),
        ("waro.suppliers.list", ["suppliers", "list"], "suppliers", "suppliers:read"),
        (
            "waro.procurement.recommendations",
            ["procurement", "recommendations"],
            "procurement",
            "procurement:read",
        ),
    ]
    payload = {
        "schema_version": AGENT_SCHEMA_V2,
        "tools": [
            {
                "name": name,
                "command": command,
                "scope": scope,
                "domain": domain,
                "description": "Stock levels",
                "capabilities": {"entity": "ingredient", "measures": ["quantity"]},
                "arguments": {"type": "object", "properties": {}},
                "response": {
                    "shape": "rows",
                    "row_path": "items",
                    "fields": ["sku"],
                    "default_fields": ["sku"],
                    "top_level_keys": [],
                },
            }
            for name, command, domain, scope in dynamic_tools
        ],
    }

    async def fake_load(self):
        return payload

    settings = Settings(TOOL_CATALOG_SOURCE="cli")
    registry = ToolRegistry(settings)
    monkeypatch.setattr(ToolRegistry, "_load_cli_schema", fake_load)
    snapshot = await registry.refresh(force=True)
    assert snapshot.source == "cli"
    for name, _command, domain, _scope in dynamic_tools:
        assert name in snapshot.tools
        assert snapshot.tools[name].domain == domain
    assert "waro.sales.metrics" in snapshot.tools


@pytest.mark.asyncio
async def test_tool_registry_exposes_dynamic_queries_run(monkeypatch):
    payload = {
        "schema_version": AGENT_SCHEMA_V2,
        "tools": [
            {
                "name": "waro.queries.run",
                "command": ["queries", "run"],
                "scope": "analytics:read",
                "domain": "queries",
                "description": "Run a safe QuerySpec",
                "capabilities": {
                    "entity": "query_row",
                    "grain": "dynamic_dataset_row",
                    "measures": ["quantity_sold", "revenue", "total_spent", "avg_ticket", "profit_margin_pct"],
                    "dimensions": ["product", "customer", "category", "day"],
                    "supported_operations": ["filter", "aggregate", "group", "rank", "sort", "limit", "compare"],
                    "supports_period": True,
                    "default_rank": ["revenue", "quantity_sold", "total_spent"],
                },
                "arguments": {
                    "type": "object",
                    "properties": {
                        "spec": {"type": "string"},
                        "dry-run": {"type": "boolean"},
                    },
                    "required": ["spec"],
                },
                "response": {
                    "shape": "rows",
                    "row_path": "rows",
                    "fields": ["rows", "meta"],
                    "default_fields": ["rows", "meta"],
                    "top_level_keys": ["rows", "meta"],
                },
            }
        ],
    }

    async def fake_load(self):
        return payload

    settings = Settings(TOOL_CATALOG_SOURCE="cli")
    registry = ToolRegistry(settings)
    monkeypatch.setattr(ToolRegistry, "_load_cli_schema", fake_load)
    snapshot = await registry.refresh(force=True)
    queries = snapshot.tools["waro.queries.run"]
    metadata = await registry.catalog_metadata()
    query_metadata = next(item for item in metadata if item["name"] == "waro.queries.run")

    assert queries.command == ("queries", "run")
    assert queries.scope == "analytics:read"
    assert query_metadata["arguments_schema"]["properties"]["spec"]["type"] == "string"
    assert "profit_margin_pct" in queries.capabilities["measures"]
