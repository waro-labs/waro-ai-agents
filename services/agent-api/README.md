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
./scripts/install-local-waro-cli.sh --from-source ../../../waro-cli
export WARO_CLI_BINARY=$PWD/.local/bin/waro
uvicorn app.main:app --host 127.0.0.1 --port 8100
curl http://127.0.0.1:8100/health
```

`WARO_API_URL` and `WARO_API_KEY` are passed through to the CLI subprocess when
the tool gateway executes a WARO tool. Keep API keys in your local `.env`; do
not commit them.

LLM summaries are disabled by default. To enable Kimi/Moonshot for workflow
summaries, set these values in `services/agent-api/.env`:

```bash
LLM_PROVIDER=kimi
KIMI_API_KEY=moonshot-or-kimi-key
KIMI_BASE_URL=https://api.moonshot.ai/v1
KIMI_MODEL=moonshot-v1-8k
LLM_TIMEOUT_SECONDS=30
```

Kimi summarizes validated workflow artifacts only. Tool execution still goes
through the allowlisted `ToolGateway`.

LLM token usage and cost metadata are best-effort estimates. `agent-api` uses
the token counts returned by Kimi when present, then applies a local pricing
table for known models and writes estimated usage/cost metadata to Phoenix
spans and safe workflow audit metadata. The Kimi dashboard remains the source
of truth for billing.

## Streaming contract

AI workflow streaming uses Server-Sent Events with `text/event-stream` frames.
Each frame uses the SSE `event:` field and a single JSON `data:` object:

```text
event: tool_finished
data: {"event":"tool_finished","emitted_at":"2026-06-18T12:00:00+00:00","run_id":"<run-id>","tool_name":"waro.menu.products","status":"succeeded","result_summary":"Returned 1 row."}
```

Initial event names are:

- `run_started`
- `step_started`
- `tool_started`
- `tool_finished`
- `llm_started`
- `token`
- `final`
- `error`

Every JSON payload includes `event`, `emitted_at`, and `run_id` when a run has
been created. Workflow events should include ids, workflow names, status,
provider/model, tool names, safe argument summaries, tool call ids, result
summaries, usage/cost metadata, and sanitized errors as needed.

Prompt text, user message content, assistant completion content, and raw LLM
messages must not be duplicated into event metadata or Phoenix traces. The
`token` event is reserved for explicit Kimi token streaming work; until that is
implemented, LLM events should only expose provider/model and safe metadata.

Tool failures should use `tool_finished` with `status=failed` when the workflow
can continue or report a tool-level result. Terminal workflow failures should
emit `error` with `status=failed` and a sanitized `error` object. `final` should
return final ids plus the user-facing summary and a deliberate artifact summary
or artifact payload chosen by the endpoint contract.

The local CLI binary lives at `services/agent-api/.local/bin/waro`. That path is
ignored by git so local builds and copied binaries do not enter the repository.
You can also copy an already installed binary into the service-owned path:

```bash
./scripts/install-local-waro-cli.sh --from-path "$(command -v waro)"
```

Docker compose from the repository root:

```bash
cd services/agent-api
./scripts/install-local-waro-cli.sh --from-source ../../../waro-cli
cd ../..
docker compose -f infra/docker-compose.yml up --build agent-api
curl http://127.0.0.1:8100/health
```

The Docker image copies `services/agent-api/.local/bin/waro` into
`/usr/local/bin/waro` and sets `WARO_CLI_BINARY=/usr/local/bin/waro`. Provision
the local binary before building the image. Runtime API credentials should be
provided through environment variables, not baked into the image.

Run with local Phoenix trace inspection:

```bash
docker compose -f infra/docker-compose.yml up --build agent-api phoenix
```

Open <http://127.0.0.1:6006>, send a signed internal tool request, then verify
the matching Postgres run row:

```sql
SELECT id, trace_id
FROM ai.runs
WHERE id = '<run-id>';
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

Signature verification fails closed until `INTERNAL_SIGNATURE_SECRET` is
configured, then validates the request body digest and signed WARO context
headers.
