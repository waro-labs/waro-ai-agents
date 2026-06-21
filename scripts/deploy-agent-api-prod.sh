#!/usr/bin/env bash
set -euo pipefail

REPO="/home/saifer/waro-ai-agents"
IMAGE="infra-agent-api"
CONTAINER="infra-agent-api-1"
PHOENIX_CONTAINER="infra-phoenix-1"
COMPOSE="docker compose --env-file .env -f infra/docker-compose.server.yml"
OBS_COMPOSE="docker compose --env-file .env -f infra/docker-compose.server.yml --profile observability"
TAG_FILE="/home/saifer/.rollback-agent-api-prod.tag"

usage() {
  echo "Uso: $0 {deploy|rollback|tags|validate|logs}"
  echo "  deploy   - git pull, valida CLI/env Phoenix, tag rollback, build, up -d phoenix + agent-api"
  echo "  rollback - vuelve al ultimo tag guardado y recrea agent-api"
  echo "  tags     - lista tags rollback de agent-api"
  echo "  validate - healthchecks, Phoenix/Postgres/env y errores recientes de trazas"
  echo "  logs     - muestra logs recientes de agent-api y phoenix"
}

require_file() {
  [[ -f "$1" ]] || { echo "Falta archivo requerido: $1"; exit 1; }
}

ensure_waro_cli() {
  local cli="$REPO/services/agent-api/.local/bin/waro"
  local source_dir="$REPO/../waro-cli"
  local marker="$REPO/services/agent-api/.local/bin/waro.gitrev"
  local source_rev=""
  local installed_rev=""
  local needs_build="false"
  local build_reason=""
  if [[ -f "$source_dir/Cargo.toml" ]]; then
    (cd "$source_dir" && git pull origin main >/dev/null)
    source_rev="$(cd "$source_dir" && git rev-parse HEAD)"
    installed_rev="$(cat "$marker" 2>/dev/null || true)"
  fi
  if [[ ! -x "$cli" ]]; then
    needs_build="true"
    build_reason="missing_binary"
  elif ! "$cli" --help 2>/dev/null | grep -q 'agent-json'; then
    needs_build="true"
    build_reason="missing_agent_json_output"
  elif ! "$cli" schema customers list 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); fields=((d.get("response") or {}).get("fields") or []); raise SystemExit(0 if fields else 1)'; then
    needs_build="true"
    build_reason="missing_response_contracts"
  elif ! "$cli" schema analytics waros >/dev/null 2>&1; then
    needs_build="true"
    build_reason="missing_analytics_waros_schema"
  elif ! "$cli" schema analytics cohort >/dev/null 2>&1; then
    needs_build="true"
    build_reason="missing_analytics_cohort_schema"
  elif ! "$cli" schema analytics rfm >/dev/null 2>&1; then
    needs_build="true"
    build_reason="missing_analytics_rfm_schema"
  elif ! "$cli" schema analytics churn-risk >/dev/null 2>&1; then
    needs_build="true"
    build_reason="missing_analytics_churn_risk_schema"
  elif [[ -n "$source_rev" && "$installed_rev" != "$source_rev" ]]; then
    needs_build="true"
    build_reason="source_rev_changed"
  fi
  echo "WARO CLI source rev: ${source_rev:-unknown}"
  echo "WARO CLI installed rev: ${installed_rev:-unknown}"
  echo "WARO CLI rebuild: $needs_build${build_reason:+ ($build_reason)}"
  if [[ "$needs_build" == "true" ]]; then
    if [[ -f "$source_dir/Cargo.toml" ]]; then
      echo "WARO CLI no tiene contrato agent-json actualizado; reconstruyendo desde $source_dir..."
      (cd "$REPO/services/agent-api" && ./scripts/build-linux-waro-cli.sh --source "$source_dir")
      [[ -n "$source_rev" ]] && echo "$source_rev" > "$marker"
      installed_rev="$source_rev"
    else
      echo "No se encontro source de WARO CLI en $source_dir; no se puede reconstruir." >&2
    fi
  fi
  [[ -x "$cli" ]] || { echo "Falta WARO CLI ejecutable: $cli"; exit 1; }
  "$cli" --help 2>/dev/null | grep -q 'agent-json' || {
    echo "WARO CLI en $cli no soporta --output agent-json. Reconstruye con: cd $REPO/services/agent-api && ./scripts/build-linux-waro-cli.sh --source $source_dir"
    exit 1
  }
  "$cli" schema customers list 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); fields=((d.get("response") or {}).get("fields") or []); raise SystemExit(0 if fields else 1)' || {
    echo "WARO CLI en $cli no expone response contracts en waro schema."
    exit 1
  }
  "$cli" schema analytics waros >/dev/null 2>&1 || {
    echo "WARO CLI en $cli no expone analytics waros. Reconstruye con: cd $REPO/services/agent-api && ./scripts/build-linux-waro-cli.sh --source $source_dir"
    exit 1
  }
  "$cli" schema analytics cohort >/dev/null 2>&1 || {
    echo "WARO CLI en $cli no expone analytics cohort. Reconstruye con: cd $REPO/services/agent-api && ./scripts/build-linux-waro-cli.sh --source $source_dir"
    exit 1
  }
  "$cli" schema analytics rfm >/dev/null 2>&1 || {
    echo "WARO CLI en $cli no expone analytics rfm. Reconstruye con: cd $REPO/services/agent-api && ./scripts/build-linux-waro-cli.sh --source $source_dir"
    exit 1
  }
  "$cli" schema analytics churn-risk >/dev/null 2>&1 || {
    echo "WARO CLI en $cli no expone analytics churn-risk. Reconstruye con: cd $REPO/services/agent-api && ./scripts/build-linux-waro-cli.sh --source $source_dir"
    exit 1
  }
  echo "WARO CLI: $("$cli" --version 2>/dev/null || echo ok)"
  echo "WARO CLI installed rev final: ${installed_rev:-$(cat "$marker" 2>/dev/null || echo unknown)}"
}

upsert_env_from_python() {
  python3 - <<'PY'
from pathlib import Path

env = Path(".env")
lines = env.read_text().splitlines()

def get(key):
    for line in lines:
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    return None

def set_value(key, value):
    global lines
    out = []
    updated = False
    for line in lines:
        if line.startswith(key + "="):
            out.append(f"{key}={value}")
            updated = True
        else:
            out.append(line)
    if not updated:
        out.append(f"{key}={value}")
    lines = out

database_url = get("DATABASE_URL")
if not database_url:
    raise SystemExit("DATABASE_URL no existe en .env")

# Phoenix debe usar la misma DB Postgres que local: postresWaroLabs, esquema phoenix.
set_value("PHOENIX_SQL_DATABASE_URL", database_url)
set_value("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
set_value("PHOENIX_COLLECTOR_PROTOCOL", "http/protobuf")
set_value("OTEL_ENABLED", "true")
set_value("PHOENIX_ROOT_URL", "https://phoenix.warocol.com")
set_value("PHOENIX_CSRF_TRUSTED_ORIGINS", "https://phoenix.warocol.com")
set_value("PHOENIX_USE_SECURE_COOKIES", "true")
set_value("PHOENIX_ALLOW_EXTERNAL_RESOURCES", "false")

env.write_text("\n".join(lines) + "\n")
PY
}

ensure_phoenix_env() {
  require_file ".env"
  if ! grep -q '^PHOENIX_SECRET=' .env || ! grep -q '^PHOENIX_ADMIN_SECRET=' .env; then
    echo "Phoenix secrets no existen; generando setup inicial..."
    PHOENIX_ROOT_URL=https://phoenix.warocol.com ./scripts/setup-phoenix-env.sh .env
  fi
  upsert_env_from_python
}

save_rollback_tag() {
  local rollback_tag="${IMAGE}:rollback-$(date +%Y%m%d-%H%M)"
  if docker image inspect "${IMAGE}:latest" >/dev/null 2>&1; then
    docker tag "${IMAGE}:latest" "$rollback_tag"
    echo "$rollback_tag" > "$TAG_FILE"
    echo "Rollback guardado: $rollback_tag"
  else
    echo "No existe ${IMAGE}:latest todavia; se omite tag rollback inicial."
  fi
}

cmd_deploy() {
  cd "$REPO"
  git pull origin main
  ensure_waro_cli
  ensure_phoenix_env
  save_rollback_tag
  $OBS_COMPOSE up -d phoenix
  $COMPOSE build agent-api
  $COMPOSE up -d agent-api
  cmd_validate
}

cmd_rollback() {
  cd "$REPO"
  [[ -f "$TAG_FILE" ]] || { echo "No hay tag guardado. Usa: $0 tags"; exit 1; }
  local rollback_tag
  rollback_tag="$(cat "$TAG_FILE")"
  docker image inspect "$rollback_tag" >/dev/null 2>&1 || { echo "Imagen no existe: $rollback_tag"; exit 1; }
  docker tag "$rollback_tag" "${IMAGE}:latest"
  $COMPOSE up -d --force-recreate agent-api
  echo "OK - rollback a $rollback_tag"
  cmd_validate
}

cmd_tags() {
  docker images "$IMAGE" --format '{{.Repository}}:{{.Tag}}\t{{.CreatedSince}}' | grep rollback || true
  [[ -f "$TAG_FILE" ]] && echo "Ultimo deploy: $(cat "$TAG_FILE")"
}

cmd_validate() {
  cd "$REPO"
  echo "== compose ps =="
  $COMPOSE ps
  echo "== health =="
  curl -fsS http://127.0.0.1:8100/health || true
  echo
  curl -fsSI http://127.0.0.1:6006/healthz | head -n 12 || true
  echo "== agent settings =="
  docker exec "$CONTAINER" python -c 'from app.config import get_settings; s=get_settings(); print("phoenix_endpoint=", s.phoenix_collector_endpoint); print("phoenix_protocol=", s.phoenix_collector_protocol); print("phoenix_api_key=", bool(s.phoenix_api_key)); print("otel_enabled=", s.otel_enabled)'
  echo "== phoenix env =="
  docker inspect "$PHOENIX_CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | grep -E '^PHOENIX_SQL_DATABASE_URL=|^PHOENIX_ROOT_URL=|^PHOENIX_ENABLE_AUTH=' || true
  echo "== postgres phoenix =="
  docker exec saifer-postgres-1 psql -U saifer -d postresWaroLabs -Atc "select schema_name from information_schema.schemata where schema_name='phoenix';" || true
  docker exec saifer-postgres-1 psql -U saifer -d postresWaroLabs -Atc "select count(*) from phoenix.traces;" || true
  echo "== recent agent trace export errors =="
  $COMPOSE logs --since=5m agent-api | grep -iE 'phoenix|trace|export|unavailable|error|unauthorized|forbidden' || true
  echo "== recent phoenix trace ingestion =="
  $COMPOSE logs --since=5m phoenix | grep -iE 'v1/traces|error|exception' || true
}

cmd_logs() {
  cd "$REPO"
  $COMPOSE logs --tail=160 agent-api
  $OBS_COMPOSE logs --tail=160 phoenix
}

case "${1:-}" in
  deploy)   cmd_deploy ;;
  rollback) cmd_rollback ;;
  tags)     cmd_tags ;;
  validate) cmd_validate ;;
  logs)     cmd_logs ;;
  *)        usage; exit 1 ;;
esac
