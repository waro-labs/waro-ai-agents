import asyncio
import json
import os
from typing import Any

from app.config import Settings
from app.tools.response_contract import ResponseContract, contract_from_schema, contract_tool_name
from app.tools.schema_v2 import parse_agent_catalog_payload
from app.tools.sanitize import sanitize_text


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
        version, agent_schemas = parse_agent_catalog_payload(parsed)
        contracts: dict[str, ResponseContract] = {}
        for agent_schema in agent_schemas:
            if agent_schema.response is not None:
                contracts[agent_schema.name] = agent_schema.response
        if contracts:
            return contracts
        if not isinstance(parsed, list):
            return {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            contract = contract_from_schema(item)
            if contract is None:
                continue
            contracts[contract_tool_name(contract.command)] = contract
        return contracts
