# Agent API Service

FastAPI service for LangGraph workflows.

This service should expose internal endpoints only:

- `POST /internal/ai/conversations`
- `POST /internal/ai/conversations/{id}/messages`
- `GET /internal/ai/runs/{id}/events`
- `POST /internal/ai/approvals/{id}/decision`
- `GET /health`

The public WARO API should validate user/session/tenant permissions before calling this service.
