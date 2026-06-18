# Agent API Service

FastAPI service for LangGraph workflows.

This service should expose internal endpoints only:

- `POST /internal/ai/conversations`
- `POST /internal/ai/conversations/{id}/messages`
- `GET /internal/ai/runs/{id}/events`
- `POST /internal/ai/approvals/{id}/decision`
- `GET /health`

The public WARO API should validate user/session/tenant permissions before calling this service.

## Local run

```bash
cd services/agent-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8100
curl http://127.0.0.1:8100/health
```

Docker compose from the repository root:

```bash
docker compose -f infra/docker-compose.yml up --build agent-api
curl http://127.0.0.1:8100/health
```

## Internal boundary

`GET /health` is the only unauthenticated runtime endpoint in this skeleton.
Future `/internal/ai/*` routes must depend on `require_internal_request` from
`app.dependencies.internal_auth`.

Expected signed context headers from the public WARO FastAPI boundary:

- `x-waro-tenant-id`
- `x-waro-profile-id`
- `x-waro-member-id` when available
- `x-waro-scopes`
- `x-waro-request-id`
- `x-waro-internal-signature`

Signature verification is intentionally not enabled until
`INTERNAL_SIGNATURE_SECRET` is configured and the verifier is completed in a
follow-up batch.
