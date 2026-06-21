from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResponseContract:
    command: tuple[str, str]
    shape: str
    row_path: str
    fields: tuple[str, ...]
    default_fields: tuple[str, ...]
    top_level_keys: tuple[str, ...]


def contract_tool_name(command: tuple[str, str]) -> str:
    group, subcommand = command
    return f"waro.{group}.{subcommand.replace('-', '_')}"


def contract_from_schema(schema: dict[str, Any]) -> ResponseContract | None:
    command_text = schema.get("command")
    response = schema.get("response")
    if not isinstance(command_text, str) or not isinstance(response, dict):
        return None
    parts = command_text.split()
    if len(parts) != 2:
        return None
    fields = response.get("fields")
    default_fields = response.get("default_fields")
    top_level_keys = response.get("top_level_keys")
    if not isinstance(fields, list) or not isinstance(default_fields, list):
        return None
    return ResponseContract(
        command=(parts[0], parts[1]),
        shape=str(response.get("shape") or ""),
        row_path=str(response.get("row_path") or ""),
        fields=tuple(str(field) for field in fields),
        default_fields=tuple(str(field) for field in default_fields),
        top_level_keys=tuple(str(field) for field in top_level_keys or []),
    )
