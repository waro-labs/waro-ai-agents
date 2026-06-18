#!/usr/bin/env sh
set -eu

usage() {
  cat <<'EOF'
Usage:
  ./scripts/install-local-waro-cli.sh --from-source ../../../waro-cli
  ./scripts/install-local-waro-cli.sh --from-path /path/to/waro
  ./scripts/install-local-waro-cli.sh

Without arguments, the script copies the first waro binary found on PATH.
EOF
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
call_dir=$(pwd)
dest="$service_dir/.local/bin/waro"

mkdir -p "$(dirname "$dest")"

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$call_dir/$1" ;;
  esac
}

case "${1:-}" in
  --from-source)
    source_dir="${2:-}"
    if [ -z "$source_dir" ]; then
      usage
      exit 2
    fi
    source_dir=$(resolve_path "$source_dir")
    (
      cd "$source_dir"
      cargo build --release
    )
    cp "$source_dir/target/release/waro" "$dest"
    ;;
  --from-path)
    source_binary="${2:-}"
    if [ -z "$source_binary" ]; then
      usage
      exit 2
    fi
    cp "$source_binary" "$dest"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  "")
    source_binary=$(command -v waro || true)
    if [ -z "$source_binary" ]; then
      echo "waro was not found on PATH. Use --from-source or --from-path." >&2
      exit 1
    fi
    cp "$source_binary" "$dest"
    ;;
  *)
    usage
    exit 2
    ;;
esac

chmod +x "$dest"
"$dest" --version
echo "Installed waro CLI at $dest"
