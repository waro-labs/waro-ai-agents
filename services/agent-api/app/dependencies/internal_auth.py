from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from app.config import get_settings


@dataclass(frozen=True)
class InternalRequestContext:
    tenant_id: str
    profile_id: str
    request_id: str
    member_id: str | None
    scopes: tuple[str, ...]


async def require_internal_request(
    request: Request,
    x_waro_tenant_id: str | None = Header(default=None),
    x_waro_profile_id: str | None = Header(default=None),
    x_waro_member_id: str | None = Header(default=None),
    x_waro_scopes: str | None = Header(default=""),
    x_waro_request_id: str | None = Header(default=None),
    x_waro_internal_signature: str | None = Header(default=None),
) -> InternalRequestContext:
    """Placeholder for signed WARO FastAPI -> agent-api requests.

    Future batches should verify the signature over method, path, body digest,
    timestamp, and request id before any internal AI route is enabled.
    """
    settings = get_settings()
    missing_headers = [
        name
        for name, value in {
            "x-waro-tenant-id": x_waro_tenant_id,
            "x-waro-profile-id": x_waro_profile_id,
            "x-waro-request-id": x_waro_request_id,
            "x-waro-internal-signature": x_waro_internal_signature,
        }.items()
        if not value
    ]

    if missing_headers:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_internal_headers",
                "headers": missing_headers,
            },
        )

    if not settings.is_signature_verification_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal signature verification is not configured.",
        )

    # Keep the request object in the signature dependency boundary so the
    # future verifier can consume body/path/method without changing route code.
    _ = request
    _ = x_waro_internal_signature

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Internal signature verification is not implemented yet.",
    )
