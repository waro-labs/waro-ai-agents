import json

import pytest

from app.config import Settings
from app.tools.registry import ToolRegistry, set_tool_registry
from app.tools.schema_v2 import (
    AGENT_SCHEMA_V2,
    legacy_tool_to_v2_dict,
    parse_agent_catalog_payload,
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


def test_legacy_tool_exports_v2_metadata():
    spec = TOOL_SPECS["waro.sales.metrics"]
    exported = legacy_tool_to_v2_dict(spec)
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
    payload = {
        "schema_version": AGENT_SCHEMA_V2,
        "tools": [
            {
                "name": "waro.inventory.stock",
                "command": ["inventory", "stock"],
                "scope": "inventory:read",
                "domain": "inventory",
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
        ],
    }

    async def fake_load(self):
        return payload

    settings = Settings(TOOL_CATALOG_SOURCE="cli")
    registry = ToolRegistry(settings)
    monkeypatch.setattr(ToolRegistry, "_load_cli_schema", fake_load)
    snapshot = await registry.refresh(force=True)
    assert snapshot.source == "cli"
    assert "waro.inventory.stock" in snapshot.tools
    assert "waro.sales.metrics" in snapshot.tools
