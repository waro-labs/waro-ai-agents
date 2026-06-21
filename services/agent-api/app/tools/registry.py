"""Dynamic tool catalog loaded from waro CLI schema with legacy fallback."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.tools.allowlist import TOOL_SPECS, ToolArgs, ToolSpec, coerce_args as legacy_coerce_args
from app.tools.schema_v2 import (
    AGENT_SCHEMA_V2,
    AgentToolSchema,
    legacy_tool_to_v2_dict,
    parse_agent_catalog_payload,
    tool_schema_from_dict,
)
from app.tools.sanitize import sanitize_text


DomainName = Literal["sales", "food_cost", "menu", "financial", "analytics", "customers"]


class DynamicToolArgs(ToolArgs):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


_registry_singleton: ToolRegistry | None = None


def _build_dynamic_args_model(arguments_schema: dict[str, Any]) -> type[ToolArgs]:
    _ = arguments_schema
    return DynamicToolArgs


def agent_schema_to_tool_spec(schema: AgentToolSchema, *, legacy: ToolSpec | None = None) -> ToolSpec:
    if legacy is not None:
        return legacy
    args_model = _build_dynamic_args_model(schema.arguments_schema)
    domain: DomainName = schema.domain if schema.domain in {
        "sales",
        "food_cost",
        "menu",
        "financial",
        "analytics",
        "customers",
    } else "analytics"
    allowed = schema.allowed_fields or frozenset(schema.default_fields) or frozenset({"data", "meta", "success"})
    return ToolSpec(
        name=schema.name,
        command=schema.command,
        scope=schema.scope,
        args_model=args_model,
        default_fields=schema.default_fields or ("data", "meta", "success"),
        allowed_fields=allowed,
        domain=domain,
        description=schema.description,
        tags=schema.tags,
        examples=schema.examples,
        capabilities=schema.capabilities,
    )


@dataclass
class CatalogSnapshot:
    version: str
    source: Literal["cli", "static"]
    loaded_at: float
    tools: dict[str, ToolSpec]
    schemas: dict[str, AgentToolSchema]


class ToolRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._snapshot: CatalogSnapshot | None = None
        self._lock = asyncio.Lock()

    async def refresh(self, *, force: bool = False) -> CatalogSnapshot:
        async with self._lock:
            if (
                not force
                and self._snapshot is not None
                and time.monotonic() - self._snapshot.loaded_at
                < self.settings.tool_catalog_refresh_seconds
            ):
                return self._snapshot
            self._snapshot = await self._load()
            return self._snapshot

    async def get_spec(self, tool_name: str) -> ToolSpec | None:
        snapshot = await self.refresh()
        return snapshot.tools.get(tool_name)

    async def all_specs(self) -> dict[str, ToolSpec]:
        snapshot = await self.refresh()
        return dict(snapshot.tools)

    async def catalog_metadata(self) -> list[dict[str, Any]]:
        snapshot = await self.refresh()
        return [self._tool_metadata(spec, snapshot.schemas.get(name)) for name, spec in snapshot.tools.items()]

    def _tool_metadata(self, spec: ToolSpec, schema: AgentToolSchema | None) -> dict[str, Any]:
        arguments_schema = (
            schema.arguments_schema
            if schema is not None and schema.arguments_schema
            else spec.args_model.model_json_schema(by_alias=True)
        )
        return {
            "name": spec.name,
            "command": list(spec.command),
            "domain": spec.domain,
            "scope": spec.scope,
            "description": spec.description,
            "tags": list(spec.tags),
            "examples": list(spec.examples),
            "default_fields": list(spec.default_fields),
            "allowed_fields": sorted(spec.allowed_fields),
            "capabilities": dict(spec.capabilities),
            "arguments_schema": arguments_schema,
        }

    async def _load(self) -> CatalogSnapshot:
        if self.settings.tool_catalog_source == "static":
            return self._load_static()
        cli_payload = await self._load_cli_schema()
        if cli_payload is None:
            return self._load_static()
        version, agent_schemas = parse_agent_catalog_payload(cli_payload)
        tools: dict[str, ToolSpec] = {}
        schemas: dict[str, AgentToolSchema] = {}
        for agent_schema in agent_schemas:
            legacy = TOOL_SPECS.get(agent_schema.name)
            spec = agent_schema_to_tool_spec(agent_schema, legacy=legacy)
            tools[spec.name] = spec
            schemas[spec.name] = agent_schema
        if not tools:
            return self._load_static()
        for name, legacy_spec in TOOL_SPECS.items():
            if name not in tools:
                tools[name] = legacy_spec
                schemas[name] = tool_schema_from_dict(legacy_tool_to_v2_dict(legacy_spec))  # type: ignore[arg-type]
        return CatalogSnapshot(
            version=version or AGENT_SCHEMA_V2,
            source="cli",
            loaded_at=time.monotonic(),
            tools=tools,
            schemas={k: v for k, v in schemas.items() if v is not None},
        )

    def _load_static(self) -> CatalogSnapshot:
        schemas = {
            name: tool_schema_from_dict(legacy_tool_to_v2_dict(spec))
            for name, spec in TOOL_SPECS.items()
        }
        return CatalogSnapshot(
            version="static-legacy",
            source="static",
            loaded_at=time.monotonic(),
            tools=dict(TOOL_SPECS),
            schemas={k: v for k, v in schemas.items() if v is not None},
        )

    async def _load_cli_schema(self) -> Any | None:
        argv = (self.settings.waro_cli_binary, "schema")
        env = {"PATH": os.environ.get("PATH", ""), "NO_COLOR": "1"}
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_bytes, _stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.settings.tool_timeout_seconds,
            )
        except (asyncio.TimeoutError, FileNotFoundError):
            return None
        stdout = sanitize_text(
            stdout_bytes.decode("utf-8", errors="replace"),
            secrets=[self.settings.waro_api_key or ""],
        )
        if proc.returncode != 0:
            return None
        try:
            return json.loads(stdout.strip() or "[]")
        except json.JSONDecodeError:
            return None


def get_tool_registry(settings: Settings | None = None) -> ToolRegistry:
    global _registry_singleton
    if settings is not None:
        return ToolRegistry(settings)
    if _registry_singleton is None:
        from app.config import get_settings

        _registry_singleton = ToolRegistry(get_settings())
    return _registry_singleton


def set_tool_registry(registry: ToolRegistry | None) -> None:
    global _registry_singleton
    _registry_singleton = registry


async def get_tool_spec_async(tool_name: str, *, settings: Settings | None = None) -> ToolSpec | None:
    registry = get_tool_registry(settings)
    return await registry.get_spec(tool_name)


def get_tool_spec(tool_name: str) -> ToolSpec | None:
    """Sync lookup — uses static legacy catalog; prefer async registry in hot paths."""
    return TOOL_SPECS.get(tool_name)


async def coerce_args_async(spec: ToolSpec, arguments: dict[str, Any]) -> ToolArgs:
    if spec.name in TOOL_SPECS and TOOL_SPECS[spec.name].args_model is spec.args_model:
        return legacy_coerce_args(spec, arguments)
    return spec.args_model.model_validate(arguments)
