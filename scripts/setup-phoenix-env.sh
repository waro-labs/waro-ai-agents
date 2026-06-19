#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"
ROOT_URL="${PHOENIX_ROOT_URL:-https://phoenix.warocol.com}"

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required to generate Phoenix secrets." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  touch "$ENV_FILE"
  chmod 600 "$ENV_FILE"
fi

random_hex() {
  openssl rand -hex 32
}

random_password() {
  printf 'Aa1!%s' "$(openssl rand -base64 24 | tr -d '\n' | tr '/+' '_-' | cut -c1-24)"
}

env_value() {
  local key="$1"
  local value="${!key:-}"
  if [[ -z "$value" && -f "$ENV_FILE" ]]; then
    value="$(awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$ENV_FILE")"
  fi
  printf '%s' "$value"
}

default_phoenix_database_url() {
  local base_url
  base_url="$(env_value "DATABASE_URL")"
  if [[ -z "$base_url" ]]; then
    echo "DATABASE_URL must be set before setting PHOENIX_SQL_DATABASE_URL." >&2
    exit 1
  fi
  printf '%s' "$base_url"
}

upsert_env() {
  local key="$1"
  local value="$2"
  local tmp
  tmp="$(mktemp)"
  if grep -q "^${key}=" "$ENV_FILE"; then
    awk -v key="$key" -v value="$value" '
      BEGIN { prefix = key "=" }
      index($0, prefix) == 1 { print key "=" value; next }
      { print }
    ' "$ENV_FILE" > "$tmp"
  else
    cp "$ENV_FILE" "$tmp"
    printf '%s=%s\n' "$key" "$value" >> "$tmp"
  fi
  mv "$tmp" "$ENV_FILE"
}

phoenix_secret="$(random_hex)"
phoenix_admin_secret="$(random_hex)"
phoenix_admin_password="$(random_password)"
phoenix_database_url="${PHOENIX_SQL_DATABASE_URL:-$(default_phoenix_database_url)}"

while [[ "$phoenix_admin_secret" == "$phoenix_secret" ]]; do
  phoenix_admin_secret="$(random_hex)"
done

upsert_env "PHOENIX_ENABLE_AUTH" "true"
upsert_env "PHOENIX_SECRET" "$phoenix_secret"
upsert_env "PHOENIX_ADMIN_SECRET" "$phoenix_admin_secret"
upsert_env "PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD" "$phoenix_admin_password"
upsert_env "PHOENIX_ENABLE_STRONG_PASSWORD_POLICY" "true"
upsert_env "PHOENIX_USE_SECURE_COOKIES" "true"
upsert_env "PHOENIX_ROOT_URL" "$ROOT_URL"
upsert_env "PHOENIX_CSRF_TRUSTED_ORIGINS" "$ROOT_URL"
upsert_env "PHOENIX_SQL_DATABASE_URL" "$phoenix_database_url"
upsert_env "OTEL_ENABLED" "true"
upsert_env "PHOENIX_COLLECTOR_ENDPOINT" "http://phoenix:6006/v1/traces"
upsert_env "PHOENIX_COLLECTOR_PROTOCOL" "http/protobuf"

# The admin secret can authenticate ingestion immediately. Replace this later
# with a Phoenix System API Key created from the UI.
upsert_env "PHOENIX_API_KEY" "$phoenix_admin_secret"

chmod 600 "$ENV_FILE"

cat <<EOF
Phoenix environment updated in: $ENV_FILE

Initial UI login:
  email: admin@localhost
  password: $phoenix_admin_password

Next:
  1. Start Phoenix.
  2. Log in and change the admin password.
  3. Create a System API Key in Phoenix settings.
  4. Replace PHOENIX_API_KEY in $ENV_FILE with that System API Key.

EOF
