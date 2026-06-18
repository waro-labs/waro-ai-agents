from typing import Any

SECRET_KEY_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "token",
    "secret",
    "signature",
    "password",
)
PII_KEY_PARTS = ("email", "phone", "document", "identification")
REDACTED = "[redacted]"


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in SECRET_KEY_PARTS + PII_KEY_PARTS):
                sanitized[key] = REDACTED
            else:
                sanitized[key] = sanitize_value(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value]
    return value


def sanitize_text(value: str, secrets: list[str] | None = None) -> str:
    sanitized = value
    for secret in secrets or []:
        if secret:
            sanitized = sanitized.replace(secret, REDACTED)
    return sanitized


def truncate_text(value: str, max_chars: int = 4000) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}... [truncated]"
