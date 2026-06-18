# WARO AI Agents

Internal AI agent runtime for WARO Colombia.

`waro-ai-agents` is the service boundary for WARO's agent workflows. It sits
beside the main WARO FastAPI backend instead of inside it, so agent workloads
can evolve with their own runtime, tracing, memory, approval, and tool-safety
contracts.

- Repository: <https://github.com/waro-labs/waro-ai-agents>
- Editable architecture diagram: <https://www.figma.com/board/iDPBndrqLp0UUieI2ifqQs?utm_source=other&utm_content=edit_in_figjam&oai_id=&request_id=d7474638-b2df-4cfe-8d8c-b6b0403857e3&architecture=true>
- Status: early internal skeleton
- License: source-available proprietary, not open source

## Why this exists

The main WARO API already owns public authentication, tenant resolution,
permissions, operations, billing, inventory, POS, analytics, and public API
access. AI workloads have different operational needs:

- long-running and resumable workflows;
- streaming responses and event buffers;
- model/tool tracing;
- embeddings and RAG storage;
- eval datasets and run review;
- approval-gated mutations.

Keeping the agent runtime separate reduces blast radius while preserving the
existing WARO FastAPI backend as the public/auth boundary.

## Current architecture

![WARO AI Agents architecture](docs/assets/waro-ai-agents-architecture.png)

```text
Nuxt UI
  -> WARO FastAPI
    -> waro-ai-agents internal FastAPI service
      -> Postgres schemas: ai, rag, audit
      -> Redis for ephemeral runtime state
      -> Phoenix via OpenTelemetry
      -> approved WARO tool surface
```

The agent service should not be exposed publicly. WARO FastAPI validates the
session, tenant, member/profile, and permissions, then forwards signed internal
context to this service.

## What is implemented now

- FastAPI app skeleton in `services/agent-api`.
- Settings for Postgres, Redis, Phoenix/OTLP, environment, host/port, and
  internal signature secret.
- `GET /health` endpoint for local startup validation.
- Lazy asyncpg pool helper for future DB work.
- Fail-closed internal request dependency placeholder.
- Dockerfile and local compose wiring.
- Draft Postgres schemas for `ai`, `rag`, and `audit`.
- Database and architecture notes in `docs/`.

## Planned runtime

The first production-oriented runtime will include:

- LangGraph workflows for stateful specialist agents.
- A Waro Tool Gateway with typed, allowlisted tool calls.
- OpenTelemetry/OpenLLMetry traces exported to Phoenix.
- Postgres-backed conversations, runs, steps, tool calls, approvals, evals, and
  RAG chunks.
- Redis for locks, short-lived cache, active run coordination, and stream
  buffers.

## First agents

1. Food Cost Agent
   - Data: food-cost analytics, financial products, menu products, recipes.
   - Output: low-margin products, cost movement, suggested next actions.

2. Purchasing Agent
   - Data: inventory, purchases, suppliers, sales forecast.
   - Output: purchase order drafts.
   - Mutations require human approval.

3. Marketing Agent
   - Data: sales metrics, RFM, churn risk, campaign context.
   - Output: campaign drafts and segments.
   - PII minimization by default.

4. Finance Agent
   - Data: sales, expenses, journal entries, payment methods.
   - Output: explainable summaries and alerts.
   - Critical accounting mutations require approval.

## Local quick start

Run the service directly:

```bash
cd services/agent-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8100
```

Then verify:

```bash
curl http://127.0.0.1:8100/health
```

Expected shape:

```json
{
  "status": "ok",
  "service": "waro-ai-agents",
  "environment": "development",
  "dependencies": {
    "postgres": "configured",
    "redis": "configured",
    "phoenix": "configured"
  },
  "internal_auth": {
    "signature_secret": "not_configured",
    "signature_verification": "not_implemented"
  }
}
```

Run with compose from the repository root:

```bash
docker compose -f infra/docker-compose.yml up --build agent-api
```

If your local Docker install uses the legacy binary:

```bash
docker-compose -f infra/docker-compose.yml up --build agent-api
```

## Configuration

See `.env.example`.

```env
DATABASE_URL=postgresql://saifer:change-me@postgres:5432/postresWaroLabs
REDIS_URL=redis://redis:6379/0
PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:4317
OTEL_SERVICE_NAME=waro-ai-agents
ENVIRONMENT=development
INTERNAL_SIGNATURE_SECRET=
```

## Internal API boundary

`GET /health` is intentionally unauthenticated for local and container health
checks. Future `/internal/ai/*` routes must use
`app.dependencies.internal_auth.require_internal_request`.

Expected signed context headers from WARO FastAPI:

- `x-waro-tenant-id`
- `x-waro-profile-id`
- `x-waro-member-id` when available
- `x-waro-scopes`
- `x-waro-request-id`
- `x-waro-internal-signature`

The current dependency fails closed until signature verification is implemented.

## Storage boundaries

Postgres is the source of truth for:

- conversations and messages;
- run summaries and step metadata;
- tool call audit;
- human approvals;
- eval datasets/results;
- RAG documents and chunks.

Redis is ephemeral only:

- locks;
- short-lived stream buffers;
- cache;
- active run coordination.

Phoenix is for trace inspection, not business source of truth.

## Useful docs

- [Architecture](docs/architecture.md)
- [Database analysis](docs/db-analysis.md)
- [Agent API service](services/agent-api/README.md)
- [Draft schema migration](migrations/001_ai_rag_audit_schemas.sql)
- [License](LICENSE.md)
- [Notice](NOTICE.md)
- [Contributing](CONTRIBUTING.md)

## Validation commands

```bash
cd services/agent-api && python -m compileall app
cd services/agent-api && uvicorn app.main:app --host 127.0.0.1 --port 8100
curl http://127.0.0.1:8100/health
docker compose -f infra/docker-compose.yml config
```

## License

This repository is publicly visible as source-available proprietary software.
It is not licensed as open source. See [LICENSE.md](LICENSE.md) and
[NOTICE.md](NOTICE.md).
