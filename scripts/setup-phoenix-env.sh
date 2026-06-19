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
upsert_env "PHOENIX_SQL_DATABASE_URL" "sqlite:////data/phoenix.db"
upsert_env "OTEL_ENABLED" "true"
upsert_env "PHOENIX_COLLECTOR_ENDPOINT" "http://phoenix:4317"

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
