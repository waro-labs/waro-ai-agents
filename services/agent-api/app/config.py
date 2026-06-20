from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: PostgresDsn = Field(
        default="postgresql://saifer:change-me@postgres:5432/postresWaroLabs",
        alias="DATABASE_URL",
    )
    redis_url: RedisDsn = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    phoenix_collector_endpoint: str = Field(
        default="http://phoenix:4317",
        alias="PHOENIX_COLLECTOR_ENDPOINT",
    )
    phoenix_collector_protocol: Literal["grpc", "http/protobuf"] = Field(
        default="grpc",
        alias="PHOENIX_COLLECTOR_PROTOCOL",
    )
    phoenix_api_key: str | None = Field(default=None, alias="PHOENIX_API_KEY")
    otel_service_name: str = Field(default="waro-ai-agents", alias="OTEL_SERVICE_NAME")
    otel_enabled: bool = Field(default=True, alias="OTEL_ENABLED")
    otel_export_timeout_seconds: int = Field(
        default=5,
        alias="OTEL_EXPORT_TIMEOUT_SECONDS",
    )
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        alias="ENVIRONMENT",
    )
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8100, alias="PORT")
    internal_signature_secret: str | None = Field(
        default=None,
        alias="INTERNAL_SIGNATURE_SECRET",
    )
    waro_cli_binary: str = Field(default="waro", alias="WARO_CLI_BINARY")
    waro_api_url: str | None = Field(default=None, alias="WARO_API_URL")
    waro_api_key: str | None = Field(default=None, alias="WARO_API_KEY")
    tool_timeout_seconds: int = Field(default=30, alias="TOOL_TIMEOUT_SECONDS")
    tool_result_max_bytes: int = Field(default=200_000, alias="TOOL_RESULT_MAX_BYTES")
    llm_provider: Literal["disabled", "kimi"] = Field(
        default="disabled",
        alias="LLM_PROVIDER",
    )
    kimi_api_key: str | None = Field(default=None, alias="KIMI_API_KEY")
    kimi_base_url: str = Field(
        default="https://api.moonshot.ai/v1",
        alias="KIMI_BASE_URL",
    )
    kimi_model: str = Field(default="moonshot-v1-8k", alias="KIMI_MODEL")
    kimi_router_model: str | None = Field(default=None, alias="KIMI_ROUTER_MODEL")
    kimi_planner_model: str | None = Field(default=None, alias="KIMI_PLANNER_MODEL")
    kimi_analysis_model: str | None = Field(default=None, alias="KIMI_ANALYSIS_MODEL")
    kimi_composer_model: str | None = Field(default=None, alias="KIMI_COMPOSER_MODEL")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def is_signature_verification_enabled(self) -> bool:
        return bool(self.internal_signature_secret)

    @property
    def llm_router_model(self) -> str:
        return self.kimi_router_model or self.kimi_model

    @property
    def llm_planner_model(self) -> str:
        return self.kimi_planner_model or self.kimi_model

    @property
    def llm_analysis_model(self) -> str:
        return self.kimi_analysis_model or self.kimi_model

    @property
    def llm_composer_model(self) -> str:
        return self.kimi_composer_model or self.kimi_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
