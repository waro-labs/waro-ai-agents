import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.tools.sanitize import sanitize_text


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


class WaroContractRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._contracts: dict[str, ResponseContract] | None = None
        self._lock = asyncio.Lock()

    async def get(self, tool_name: str) -> ResponseContract | None:
        contracts = await self.all()
        return contracts.get(tool_name)

    async def all(self) -> dict[str, ResponseContract]:
        if self._contracts is not None:
            return self._contracts
        async with self._lock:
            if self._contracts is None:
                self._contracts = await self._load()
        return self._contracts

    async def _load(self) -> dict[str, ResponseContract]:
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
            return {}
        stdout = sanitize_text(
            stdout_bytes.decode("utf-8", errors="replace"),
            secrets=[self.settings.waro_api_key or ""],
        )
        if proc.returncode != 0:
            return {}
        try:
            parsed = json.loads(stdout.strip() or "[]")
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, list):
            return {}
        contracts: dict[str, ResponseContract] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            contract = contract_from_schema(item)
            if contract is None:
                continue
            contracts[contract_tool_name(contract.command)] = contract
        return contracts
