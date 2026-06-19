# Production Validation Runbook

This runbook closes the deployment epic for `agent-api` and the
`api_warocol.com` signed SSE proxy. It separates read-only preflight checks from
operator-approved production changes.

## Current Server State

Read-only checks on `warolabs` showed:

- `/home/saifer/api-warocol.com` exists.
- `/home/saifer/waro-ai-agents` does not exist yet.
- `/home/saifer/bin/deploy-api-prod.sh` is the existing API deploy/rollback
  entrypoint.
- `waro-network` and `postgresqlwarolabs_default` exist.
- `api-warocolcom-web-1` is not attached to `waro-network` yet.
- No `agent-api` container is running yet.
- `api-warocolcom-web-1` does not yet expose `AGENT_API_URL`,
  `INTERNAL_SIGNATURE_SECRET`, or agent SSE timeout env vars.

Never print classic GitHub tokens, runtime `.env` contents, or real
`INTERNAL_SIGNATURE_SECRET` values in PRs, comments, logs, or docs.

## Approval Boundary

Ask the operator before any command that changes production state:

- cloning or pulling server repos;
- editing runtime `.env` files;
- creating or changing `/home/saifer/bin/*` scripts;
- attaching containers to networks;
- running `docker compose up`, `restart`, `down`, or rollback;
- running either API or agent deploy script.

Read-only inspection commands are safe to run without a deploy approval.

## Read-Only Preflight

```bash
ssh warolabs 'ls -ld /home/saifer/api-warocol.com /home/saifer/waro-ai-agents 2>/dev/null || true'
ssh warolabs 'docker network inspect waro-network >/dev/null && echo waro-network=present'
ssh warolabs 'docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Networks}}"'
ssh warolabs 'git -C /home/saifer/api-warocol.com remote -v | sed -E "s#https://[^/@]+@github.com/#https://<classic-token>@github.com/#g"'
```

## api-warocol.com Prep

`api-warocol.com` already has a production deploy script. For this batch,
Codex should only prepare the env/pull instructions; the operator runs the
deploy script.

Set or confirm these runtime values in `/home/saifer/api-warocol.com/.env`:

```dotenv
AGENT_API_URL=http://agent-api:8100
INTERNAL_SIGNATURE_SECRET=<same-value-as-agent-api>
AGENT_API_CONNECT_TIMEOUT_SECONDS=5
AGENT_API_READ_TIMEOUT_SECONDS=300
WARO_RUNTIME_NETWORK=waro-network
```

Then pull the repo. The operator deploys with:

```bash
/home/saifer/bin/deploy-api-prod.sh deploy
```

Rollback remains:

```bash
/home/saifer/bin/deploy-api-prod.sh rollback
```

## agent-api Provisioning

Provision `waro-ai-agents` beside the existing API checkout:

```bash
cd /home/saifer
git clone https://<classic-token>@github.com/waro-labs/waro-ai-agents.git waro-ai-agents
cd /home/saifer/waro-ai-agents
cp .env.example .env
```

Fill `/home/saifer/waro-ai-agents/.env` with real runtime values:

```dotenv
DATABASE_URL=<shared-postgres-url>
REDIS_URL=redis://redis:6379/0
ENVIRONMENT=production
AGENT_API_PORT=8100
WARO_RUNTIME_NETWORK=waro-network
INTERNAL_SIGNATURE_SECRET=<same-value-as-api-warocol>
WARO_CLI_BINARY=/usr/local/bin/waro
WARO_API_URL=<api-warocol-base-url>
WARO_API_KEY=<runtime-tool-gateway-key>
LLM_PROVIDER=disabled
OTEL_ENABLED=false
```

If Kimi is enabled later, add `KIMI_API_KEY`, `KIMI_BASE_URL`, `KIMI_MODEL`, and
an appropriate `LLM_TIMEOUT_SECONDS`.

Provision the `waro` binary before image build. Prefer the sibling source
checkout when available:

```bash
cd /home/saifer/waro-ai-agents/services/agent-api
./scripts/install-local-waro-cli.sh --from-source ../../../waro-cli
```

Or copy an already trusted binary:

```bash
./scripts/install-local-waro-cli.sh --from-path /path/to/waro
```

## agent-api Deploy Script

Create `/home/saifer/bin/deploy-agent-api-prod.sh` with the same deploy,
rollback, and tags shape as the existing API deploy script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="/home/saifer/waro-ai-agents"
IMAGE="waro-ai-agents-agent-api"
CONTAINER="waro-ai-agents-agent-api-1"
COMPOSE="docker compose -f infra/docker-compose.server.yml"
TAG_FILE="/home/saifer/.rollback-agent-api-prod.tag"

usage() {
  echo "Uso: $0 {deploy|rollback|tags}"
}

cmd_deploy() {
  cd "$REPO"
  git pull
  cd services/agent-api
  ./scripts/install-local-waro-cli.sh --from-source ../../../waro-cli
  cd ../..
  ROLLBACK_TAG="${IMAGE}:rollback-$(date +%Y%m%d-%H%M)"
  docker image inspect "${IMAGE}:latest" >/dev/null 2>&1 && {
    docker tag "${IMAGE}:latest" "$ROLLBACK_TAG"
    echo "$ROLLBACK_TAG" > "$TAG_FILE"
  }
  $COMPOSE config
  $COMPOSE build agent-api
  $COMPOSE up -d agent-api
  echo "OK - $CONTAINER ($(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo 'unknown'))"
}

cmd_rollback() {
  cd "$REPO"
  [[ -f "$TAG_FILE" ]] || { echo "No rollback tag saved"; exit 1; }
  ROLLBACK_TAG="$(cat "$TAG_FILE")"
  docker image inspect "$ROLLBACK_TAG" >/dev/null
  docker tag "$ROLLBACK_TAG" "${IMAGE}:latest"
  $COMPOSE up -d --force-recreate agent-api
  echo "OK - rollback to $ROLLBACK_TAG"
}

cmd_tags() {
  docker images "${IMAGE}" --format '{{.Repository}}:{{.Tag}}\t{{.CreatedSince}}' | grep rollback || true
  [[ -f "$TAG_FILE" ]] && echo "Last deploy: $(cat "$TAG_FILE")"
}

case "${1:-}" in
  deploy) cmd_deploy ;;
  rollback) cmd_rollback ;;
  tags) cmd_tags ;;
  *) usage; exit 1 ;;
esac
```

Make it executable only after operator approval:

```bash
chmod +x /home/saifer/bin/deploy-agent-api-prod.sh
```

The operator deploys with:

```bash
/home/saifer/bin/deploy-agent-api-prod.sh deploy
```

## Health Validation

Host-local bind check:

```bash
curl http://127.0.0.1:8100/health
```

Container-to-container DNS check from API runtime:

```bash
docker exec api-warocolcom-web-1 python - <<'PY'
import urllib.request
print(urllib.request.urlopen("http://agent-api:8100/health", timeout=5).read().decode())
PY
```

The health response should show signature verification configured once
`INTERNAL_SIGNATURE_SECRET` is set.

## SSE Validation

Run the public API routes with a real authenticated browser/session or copied
operator cookie for a tenant/member that has the required module access:

- `POST /ai/sales/messages/stream` requires `Module.VENTAS`.
- `POST /ai/food-cost/messages/stream` requires `Module.ANALITICA`.

For each stream, capture:

- request id sent or returned in `x-waro-request-id`;
- first token/data frame when an LLM provider is enabled;
- final frame with `event: final` and `status=completed`;
- absence of raw prompt duplication in trace metadata or logs.

If `LLM_PROVIDER=disabled`, document that token generation is intentionally not
validated and only routing/auth/signature behavior can be smoke-tested.

## Logs And Correlation

```bash
docker compose -f /home/saifer/waro-ai-agents/infra/docker-compose.server.yml logs --tail=100 agent-api
docker logs api-warocolcom-web-1 --tail=100
```

Use the same `x-waro-request-id` to correlate API logs with agent logs and
persisted `ai.runs` rows.

## Phoenix And OTel

Phoenix is optional. Production can keep:

```dotenv
OTEL_ENABLED=false
```

If the operator explicitly enables Phoenix, start the profile and validate trace
correlation separately:

```bash
docker compose -f /home/saifer/waro-ai-agents/infra/docker-compose.server.yml --profile observability up -d phoenix
```

Do not require Phoenix for production request serving.

## Evidence For PR

In the PR, separate evidence into:

- local/static validation;
- read-only `warolabs` preflight;
- operator-approved deploy steps, if performed;
- pending operator actions, if deploy was deferred;
- known limitations, such as `LLM_PROVIDER=disabled` or Phoenix not deployed.
