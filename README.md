# WARO AI Agents

Internal multi-agent runtime for WARO Colombia.

## Purpose

This project hosts the AI agent service that will sit beside the existing WARO FastAPI backend. The main API remains the public/auth boundary; this service runs LangGraph workflows, traces every run, stores agent memory/evals, and calls approved WARO tools.

## Initial Stack

- FastAPI for the internal agent API.
- LangGraph for stateful agent workflows.
- Pydantic for typed tool inputs/outputs.
- OpenTelemetry + OpenLLMetry for traces.
- Phoenix self-hosted for trace/eval inspection.
- Postgres + pgvector for product truth, traces summaries, evals, and RAG chunks.
- Redis for ephemeral runtime state, locks, cache, and streaming buffers.
- `waro-cli` as the first safe tool surface.

## Deployment Shape

```text
Nuxt
  -> WARO FastAPI
    -> waro-ai-agents internal service
      -> LangGraph workflows
      -> Waro Tool Gateway
      -> Postgres schemas: ai, rag, audit
      -> Redis
      -> OpenTelemetry/Phoenix
```

The agent service should not be exposed publicly at first. WARO FastAPI validates session, tenant, permissions, and forwards a signed internal request to this service.

## Current Findings

- Local Postgres connection works via `api_warocol.com/.env`.
- DB is PostgreSQL 16.13.
- `vector` extension is installed at version 0.8.1.
- No active columns of type `vector(...)` were found, so RAG tables still need to be created.
- Main operational data is in `public`, with strong `tenant_id` usage.
- Existing public API/token model already supports scoped API access and should inform agent tool permissions.

See [docs/db-analysis.md](docs/db-analysis.md) and [docs/architecture.md](docs/architecture.md).
