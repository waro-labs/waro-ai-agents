import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.tools.allowlist import ToolArgs, ToolSpec
from app.tools.sanitize import sanitize_text, truncate_text


@dataclass(frozen=True)
class ToolRunResult:
    result: Any
    stderr: str
    argv: tuple[str, ...]


class ToolRunError(Exception):
    def __init__(
        self,
        *,
        message: str,
        returncode: int | None = None,
        stderr: str = "",
        stdout: str = "",
        argv: tuple[str, ...] = (),
    ):
        super().__init__(message)
        self.message = message
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout
        self.argv = argv

    def to_context(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "returncode": self.returncode,
            "stderr": truncate_text(self.stderr),
            "stdout": truncate_text(self.stdout),
            "argv": list(self.argv),
        }


class WaroCliRunner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_argv(
        self,
        *,
        spec: ToolSpec,
        args: ToolArgs,
        fields: tuple[str, ...],
        dry_run: bool,
    ) -> tuple[str, ...]:
        argv = [
            self.settings.waro_cli_binary,
            "--output",
            "json",
            "--no-color",
            "--fields",
            ",".join(fields),
            *spec.command,
            *args.cli_args(),
        ]
        if dry_run:
            argv.append("--dry-run")
        return tuple(argv)

    async def run(
        self,
        *,
        spec: ToolSpec,
        args: ToolArgs,
        fields: tuple[str, ...],
        dry_run: bool,
    ) -> ToolRunResult:
        argv = self.build_argv(spec=spec, args=args, fields=fields, dry_run=dry_run)
        env = self._env()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.settings.tool_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ToolRunError(message="Tool execution timed out.", argv=argv) from exc
        except FileNotFoundError as exc:
            raise ToolRunError(message="waro CLI binary was not found.", argv=argv) from exc

        stdout = sanitize_text(
            stdout_bytes.decode("utf-8", errors="replace"),
            secrets=self._secret_values(),
        )
        stderr = sanitize_text(
            stderr_bytes.decode("utf-8", errors="replace"),
            secrets=self._secret_values(),
        )

        if len(stdout_bytes) > self.settings.tool_result_max_bytes:
            raise ToolRunError(
                message="Tool result exceeded size limit.",
                returncode=proc.returncode,
                stderr=stderr,
                stdout="",
                argv=argv,
            )

        if proc.returncode != 0:
            raise ToolRunError(
                message="Tool execution failed.",
                returncode=proc.returncode,
                stderr=stderr,
                stdout=stdout,
                argv=argv,
            )

        result = self._parse_stdout(stdout)
        return ToolRunResult(result=result, stderr=truncate_text(stderr), argv=argv)

    def _parse_stdout(self, stdout: str) -> Any:
        stripped = stdout.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in stripped.splitlines() if line.strip()]
            return rows

    def _env(self) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "NO_COLOR": "1",
        }
        if self.settings.waro_api_url:
            env["WARO_API_URL"] = self.settings.waro_api_url
        if self.settings.waro_api_key:
            env["WARO_API_KEY"] = self.settings.waro_api_key
        return env

    def _secret_values(self) -> list[str]:
        return [value for value in [self.settings.waro_api_key] if value]
