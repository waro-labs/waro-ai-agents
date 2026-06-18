from dataclasses import dataclass
import hashlib
import hmac

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
    """Verify signed WARO FastAPI -> agent-api requests.

    The public API boundary signs the method, path, request id, tenant/profile
    context, and request body digest. Keeping this small contract here lets
    route code depend on InternalRequestContext without knowing signature
    details.
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

    body = await request.body()
    body_digest = hashlib.sha256(body).hexdigest()
    member_id = x_waro_member_id if isinstance(x_waro_member_id, str) else None
    canonical = "\n".join(
        [
            request.method.upper(),
            request.url.path,
            x_waro_request_id or "",
            x_waro_tenant_id or "",
            x_waro_profile_id or "",
            member_id or "",
            x_waro_scopes or "",
            body_digest,
        ]
    )
    expected = hmac.new(
        settings.internal_signature_secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, x_waro_internal_signature or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal signature.",
        )

    scopes = tuple(
        scope.strip()
        for scope in (x_waro_scopes or "").split(",")
        if scope.strip()
    )
    return InternalRequestContext(
        tenant_id=x_waro_tenant_id or "",
        profile_id=x_waro_profile_id or "",
        request_id=x_waro_request_id or "",
        member_id=member_id,
        scopes=scopes,
    )
