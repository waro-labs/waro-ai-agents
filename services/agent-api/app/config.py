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
    otel_service_name: str = Field(default="waro-ai-agents", alias="OTEL_SERVICE_NAME")
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def is_signature_verification_enabled(self) -> bool:
        return bool(self.internal_signature_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
