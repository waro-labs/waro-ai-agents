# Architecture

## Decision

Create a separate internal AI agent service instead of embedding LangGraph directly into the existing WARO FastAPI monolith.

## Why

The current FastAPI service is already the core product API: auth, tenant detection, permissions, POS, operations, billing, analytics, finance, inventory, and public API. Agent workloads have different behavior:

- long-running workflows
- streaming
- retries and resumability
- model/tool tracing
- embeddings
- eval execution
- higher memory and latency variance

Keeping them in a separate container reduces blast radius while still allowing shared Postgres access.

## Request Flow

```text
Nuxt UI
  -> WARO FastAPI
    1. Validates session cookie.
    2. Resolves tenant/member/profile.
    3. Checks module/tool permissions.
    4. Creates or resumes ai.conversation.
    5. Calls internal agent service.

Internal Agent Service
  1. Starts OpenTelemetry trace.
  2. Loads LangGraph state/checkpoint.
  3. Routes to a specialist workflow.
  4. Calls approved tools.
  5. Persists messages, steps, summaries, evals.
  6. Streams events back to FastAPI/Nuxt.
```

## First Agents

1. Food Cost Agent
   - Data: `analytics food-cost`, `financial products`, menu/recipes.
   - Output: low-margin products, cost movement, suggested next actions.

2. Purchasing Agent
   - Data: inventory, purchases, suppliers, sales forecast.
   - Output: purchase order drafts.
   - Requires human approval before mutations.

3. Marketing Agent
   - Data: sales metrics, RFM, churn risk, campaigns.
   - Output: campaign drafts and target segments.
   - PII minimization by default.

4. Finance Agent
   - Data: sales, expenses, journal entries, payment methods.
   - Output: explainable summaries and alerts.
   - No critical accounting mutation without approval.

## Tool Gateway

The model must not run arbitrary shell commands.

Use allowlisted tools:

- `waro.sales.metrics`
- `waro.analytics.food_cost`
- `waro.financial.products`
- `waro.menu.products`
- `waro.menu.recipes`
- `waro.customers.metrics`
- `waro.analytics.rfm`
- `waro.analytics.churn_risk`

Each tool stores:

- typed arguments
- sanitized output
- `tenant_id`
- `run_id`
- `step_id`
- `trace_id`
- `span_id`
- latency
- error if any

Mutating tools need:

- dry-run first
- approval row
- audit event
- idempotency key

## Tracing

OpenTelemetry is the contract.

Postgres stores product/audit summaries:

- `ai.runs.trace_id`
- `ai.steps.span_id`
- `ai.tool_calls`
- eval scores

Phoenix stores visual traces via OTLP.

## Storage Boundary

Postgres is the source of truth for:

- conversation history
- messages
- run summaries
- tool call audit
- eval datasets/results
- RAG documents/chunks

Phoenix is the inspection UI, not the business source of truth.

Redis is ephemeral only:

- locks
- short-lived stream buffers
- cache
- active run coordination

## GitHub Org Note

GitHub CLI can create repositories and projects, but it does not expose an org creation command for normal GitHub.com accounts. Use GitHub web to create org login `waro-colombia` with display name `Waro Colombia`, then transfer this repo and project into it.
