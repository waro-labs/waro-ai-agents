"""Parse waro CLI agent catalog schema (v1 list or v2 envelope)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.tools.response_contract import ResponseContract, contract_from_schema, contract_tool_name


AGENT_SCHEMA_V2 = "waro.agent.v2"


@dataclass(frozen=True)
class AgentToolSchema:
    name: str
    command: tuple[str, str]
    scope: str
    domain: str
    description: str
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    capabilities: dict[str, Any] = field(default_factory=dict)
    arguments_schema: dict[str, Any] = field(default_factory=dict)
    default_fields: tuple[str, ...] = ()
    allowed_fields: frozenset[str] = frozenset()
    response: ResponseContract | None = None


def _command_from_value(value: Any) -> tuple[str, str] | None:
    if isinstance(value, list) and len(value) == 2:
        return str(value[0]), str(value[1])
    if isinstance(value, str):
        parts = value.split()
        if len(parts) == 2:
            return parts[0], parts[1]
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if item is not None)


def tool_schema_from_dict(item: dict[str, Any]) -> AgentToolSchema | None:
    command = _command_from_value(item.get("command"))
    if command is None:
        return None

    name = str(item.get("name") or contract_tool_name(command))
    scope = str(item.get("scope") or "")
    if not scope:
        return None

    response_payload = item.get("response")
    response = None
    if isinstance(response_payload, dict):
        response = contract_from_schema({"command": f"{command[0]} {command[1]}", "response": response_payload})
    elif "response" not in item:
        response = contract_from_schema(item)

    arguments = item.get("arguments")
    arguments_schema = arguments if isinstance(arguments, dict) else {}

    default_fields: tuple[str, ...] = ()
    allowed_fields: frozenset[str] = frozenset()
    if response is not None:
        default_fields = response.default_fields
        allowed_fields = frozenset(response.fields) | frozenset(response.default_fields)
    elif isinstance(response_payload, dict):
        default_fields = _string_tuple(response_payload.get("default_fields"))
        allowed_fields = frozenset(_string_tuple(response_payload.get("fields"))) | frozenset(default_fields)

    capabilities = item.get("capabilities")
    return AgentToolSchema(
        name=name,
        command=command,
        scope=scope,
        domain=str(item.get("domain") or command[0]),
        description=str(item.get("description") or ""),
        tags=_string_tuple(item.get("tags")),
        examples=_string_tuple(item.get("examples")),
        capabilities=dict(capabilities) if isinstance(capabilities, dict) else {},
        arguments_schema=arguments_schema,
        default_fields=default_fields,
        allowed_fields=allowed_fields,
        response=response,
    )


def parse_agent_catalog_payload(parsed: Any) -> tuple[str, list[AgentToolSchema]]:
    if isinstance(parsed, dict):
        version = str(parsed.get("schema_version") or AGENT_SCHEMA_V2)
        tools_payload = parsed.get("tools")
        if isinstance(tools_payload, list):
            tools = [
                schema
                for item in tools_payload
                if isinstance(item, dict) and (schema := tool_schema_from_dict(item)) is not None
            ]
            return version, tools
    if isinstance(parsed, list):
        tools: list[AgentToolSchema] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            schema = tool_schema_from_dict(item)
            if schema is not None:
                tools.append(schema)
        return "waro.agent.v1", tools
    return "unknown", []


def static_tool_to_v2_dict(spec: Any) -> dict[str, Any]:
    """Export a static ToolSpec as a v2-compatible tool dict for the static catalog."""
    arguments_schema = spec.args_model.model_json_schema(by_alias=True)
    return {
        "name": spec.name,
        "command": list(spec.command),
        "scope": spec.scope,
        "domain": spec.domain,
        "description": spec.description,
        "tags": list(spec.tags),
        "examples": list(spec.examples),
        "capabilities": dict(spec.capabilities),
        "arguments": arguments_schema,
        "response": {
            "shape": "static",
            "row_path": "",
            "fields": sorted(spec.allowed_fields),
            "default_fields": list(spec.default_fields),
            "top_level_keys": [],
        },
    }
