# Production Deployment Contract

This document defines the production layout for running `services/agent-api`
beside the main WARO FastAPI backend. It is a contract for the deployment,
proxy, networking, and validation batches that follow.

## Server Layout

Keep the agent runtime and public API as separate sibling checkouts on the same
server:

```text
/home/saifer/
  api_warocol.com/    # public WARO FastAPI backend
  waro-ai-agents/     # internal agent runtime
```

The exact parent directory can vary by server, but the ownership boundary should
not. `api_warocol.com` remains the public API service. `waro-ai-agents` owns the
internal `agent-api` service under `services/agent-api`.

## Repository And Deploy Ownership

Each repository deploys independently:

- `api_warocol.com` owns public authentication, session and tenant resolution,
  member/module permission checks, request id generation, and signed proxying to
  `agent-api`.
- `waro-ai-agents` owns agent workflows, the tool gateway, SSE event semantics,
  agent database writes, Redis-backed runtime state, and optional telemetry.
- Do not merge agent code into `api_warocol.com`.
- Do not copy real secrets, production `.env` files, or server-local paths into
  either repository.

Rollouts should update one service at a time unless a later batch explicitly
requires a coordinated deploy. When both services change, deploy `agent-api`
first, verify its health endpoint, then deploy `api_warocol.com` and validate the
public proxy path.

## Network Exposure

`api_warocol.com` is the only public boundary for agent features. It validates
the WARO user context before forwarding signed internal requests.

`agent-api` must stay private:

- Prefer Docker service DNS or a shared private Docker network for
  `api_warocol.com -> agent-api`.
- On `warolabs`, use the shared external Docker network `waro-network` and set
  `AGENT_API_URL=http://agent-api:8100` in `api_warocol.com`.
- A localhost binding such as `127.0.0.1:8100` is acceptable only as a host-side
  bind for operator health checks or a host reverse proxy. Do not use
  `localhost` or `127.0.0.1` as the URL from inside `api_warocol.com`; inside a
  container it points back to that same container.
- Do not expose `/internal/ai/*` routes directly to the internet.
- `GET /health` may remain unauthenticated for local and container health
  checks, but it should still be reachable only through the selected private
  Docker network or host-local bind.

`infra/docker-compose.yml` remains local/dev oriented.
`infra/docker-compose.server.yml` is the production-adjacent operator compose for
`agent-api`; it binds the API to localhost by default and keeps Redis on the
private compose network. It also joins the external runtime network configured
by `WARO_RUNTIME_NETWORK`, defaulting to `waro-network`, so `api_warocol.com` can
reach `http://agent-api:8100` by Docker DNS.

## Environment Contract

### `agent-api`

Configure these values in the `waro-ai-agents` runtime environment:

| Variable | Required | Owner | Notes |
|---|---:|---|---|
| `DATABASE_URL` | yes | `agent-api` | Postgres connection for `ai`, `rag`, and `audit` data. |
| `REDIS_URL` | yes | `agent-api` | Runtime locks, cache, and active stream state. Server compose defaults to its owned Redis service. |
| `INTERNAL_SIGNATURE_SECRET` | yes | shared | Same value as `api_warocol.com`; never commit it. |
| `WARO_CLI_BINARY` | yes | `agent-api` | Path to the installed `waro` CLI inside the runtime. |
| `WARO_API_URL` | yes | `agent-api` | Base URL used by the CLI/tool gateway. |
| `WARO_API_KEY` | yes | `agent-api` | Runtime credential for tool gateway API calls. |
| `LLM_PROVIDER` | yes | `agent-api` | `disabled` or `kimi`. |
| `KIMI_API_KEY` | when Kimi enabled | `agent-api` | Required only when `LLM_PROVIDER=kimi`. |
| `KIMI_BASE_URL` | when Kimi enabled | `agent-api` | Defaults to Moonshot/OpenAI-compatible endpoint. |
| `KIMI_MODEL` | when Kimi enabled | `agent-api` | Model used for workflow summaries. |
| `LLM_TIMEOUT_SECONDS` | yes | `agent-api` | LLM request timeout. |
| `AGENT_MODE` | yes | `agent-api` | `legacy`, `shadow`, or `react`. Default `legacy`. |
| `TOOL_CATALOG_SOURCE` | yes | `agent-api` | `cli` (from `waro schema`) or `static` (embedded fallback). |
| `TOOL_CATALOG_REFRESH_SECONDS` | optional | `agent-api` | Catalog cache TTL; default `300`. |
| `AGENT_MAX_STEPS_SIMPLE` | optional | `agent-api` | ReAct step limit for simple queries; default `6`. |
| `AGENT_MAX_STEPS_COMPLEX` | optional | `agent-api` | ReAct step limit for complex queries; default `10`. |
| `AGENT_CONVERSATION_MESSAGE_LIMIT` | optional | `agent-api` | Messages loaded from `ai.messages`; default `8`. |
| `KIMI_ROUTER_MODEL` | optional | `agent-api` | Complexity classifier / domain router; falls back to `KIMI_MODEL`. |
| `KIMI_PLANNER_MODEL` | optional | `agent-api` | Agent loop steps (moderate); falls back to `KIMI_MODEL`. |
| `KIMI_ANALYSIS_MODEL` | optional | `agent-api` | Loop + verify + compose for complex queries. |
| `KIMI_COMPOSER_MODEL` | optional | `agent-api` | Final NLG for simple queries. |
| `OTEL_ENABLED` | optional | `agent-api` | Set `false` in production if no collector is deployed. |
| `AGENT_API_PORT` | optional | `agent-api` | Host localhost port used by `infra/docker-compose.server.yml`; defaults to `8100`. |
| `WARO_RUNTIME_NETWORK` | optional | shared | External Docker network shared with `api_warocol.com`; defaults to `waro-network`. |
| `PHOENIX_COLLECTOR_ENDPOINT` | local/dev | `agent-api` | Local trace receiver endpoint. Not a production dependency. |
| `OTEL_SERVICE_NAME` | optional | `agent-api` | Defaults to `waro-ai-agents`. |
| `OTEL_EXPORT_TIMEOUT_SECONDS` | optional | `agent-api` | Export timeout for telemetry. |

Phoenix receives OpenTelemetry traces when configured, but for this rollout it
is local/dev observability only. Production must not depend on Phoenix to serve
agent requests. Runtime health should be validated through `/health`, SSE final
events, service logs, and persisted `ai.runs` rows.

### `api_warocol.com`

Configure these values in the public API runtime:

| Variable | Required | Owner | Notes |
|---|---:|---|---|
| `AGENT_API_URL` | yes | `api_warocol.com` | Private URL for `agent-api`; on `warolabs` use `http://agent-api:8100`. |
| `INTERNAL_SIGNATURE_SECRET` | yes | shared | Same value as `agent-api`; used to sign internal requests. |
| `AGENT_API_CONNECT_TIMEOUT_SECONDS` | yes | `api_warocol.com` | Short connect timeout for upstream setup. |
| `AGENT_API_READ_TIMEOUT_SECONDS` | yes | `api_warocol.com` | SSE-compatible read timeout; must allow long-lived streams. |
| `WARO_RUNTIME_NETWORK` | optional | shared | External Docker network shared with `agent-api`; defaults to `waro-network`. |

`api_warocol.com` should sign the exact JSON body and WARO context headers using
the canonical order implemented in
`services/agent-api/app/dependencies/internal_auth.py`.

## Manual Validation

The final operator checklist lives in
[`docs/production-validation.md`](production-validation.md). It includes the
approval boundary for production deploys, the `/home/saifer/bin` deploy-script
pattern, and the evidence expected for issue #35.

For direct `agent-api` validation, start from the server checkout:

```bash
cd /home/saifer/waro-ai-agents
cp .env.example .env
$EDITOR .env
cd services/agent-api
./scripts/install-local-waro-cli.sh --from-source ../../../waro-cli
cd ../..
docker network inspect "${WARO_RUNTIME_NETWORK:-waro-network}"
docker compose -f infra/docker-compose.server.yml config
docker compose -f infra/docker-compose.server.yml up -d --build agent-api
curl http://127.0.0.1:8100/health
docker compose -f infra/docker-compose.server.yml logs --tail=100 agent-api
```

After `api_warocol.com` is attached to the same runtime network, validate the
container-to-container path from the API runtime context:

```bash
docker exec api-warocolcom-web-1 python - <<'PY'
import urllib.request
print(urllib.request.urlopen("http://agent-api:8100/health", timeout=5).read().decode())
PY
```

Operational commands for the server compose:

```bash
docker compose -f infra/docker-compose.server.yml ps
docker compose -f infra/docker-compose.server.yml logs -f agent-api
docker compose -f infra/docker-compose.server.yml restart agent-api
docker compose -f infra/docker-compose.server.yml down
```

If the server has the legacy Compose binary instead of the Docker Compose v2
plugin, use `docker-compose` with the same flags.

After the proxy batch lands, validate both public SSE routes through
`api_warocol.com` with a real tenant/profile/member and confirm:

- token events stream before the final event when an LLM provider is enabled;
- the response ends with `event: final` and `status=completed`;
- `x-waro-request-id` correlates public API logs with `agent-api` runs.

## Out Of Scope

- Implementing the public `api_warocol.com` proxy.
- Changing production server state.
- Deploying Phoenix as a required production service.
