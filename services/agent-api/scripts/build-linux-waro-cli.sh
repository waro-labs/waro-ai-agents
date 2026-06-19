#!/usr/bin/env sh
set -eu

usage() {
  cat <<'EOF'
Usage:
  ./scripts/build-linux-waro-cli.sh [--source ../../../waro-cli] [--image rust:1.88-bookworm] [--platform linux/arm64]

Builds the WARO CLI inside a Linux Rust container and installs the resulting
ELF binary at services/agent-api/.local/bin/waro for Docker image builds.
EOF
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
service_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
call_dir=$(pwd)

source_dir="$service_dir/../../../waro-cli"
image="${WARO_CLI_BUILD_IMAGE:-rust:1.88-bookworm}"
platform="${WARO_CLI_BUILD_PLATFORM:-}"
dest="$service_dir/.local/bin/waro"
out_dir="$service_dir/.local/build/linux-waro-cli"

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$call_dir/$1" ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source)
      source_dir="${2:-}"
      [ -n "$source_dir" ] || { usage; exit 2; }
      shift 2
      ;;
    --image)
      image="${2:-}"
      [ -n "$image" ] || { usage; exit 2; }
      shift 2
      ;;
    --platform)
      platform="${2:-}"
      [ -n "$platform" ] || { usage; exit 2; }
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

source_dir=$(resolve_path "$source_dir")

if [ ! -f "$source_dir/Cargo.toml" ]; then
  echo "Cargo.toml not found in source dir: $source_dir" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to build the Linux WARO CLI." >&2
  exit 1
fi

mkdir -p "$(dirname "$dest")" "$out_dir"

docker_platform_args=""
if [ -n "$platform" ]; then
  docker_platform_args="--platform $platform"
fi

# shellcheck disable=SC2086
docker run --rm $docker_platform_args \
  -v "$source_dir:/src:ro" \
  -v "$out_dir:/out" \
  -w /src \
  "$image" \
  sh -eu -c 'cargo build --release --locked --target-dir /tmp/waro-target && cp /tmp/waro-target/release/waro /out/waro'

cp "$out_dir/waro" "$dest"
chmod +x "$dest"

case "$(uname -s)" in
  Linux)
    "$dest" --version
    ;;
  *)
    if command -v file >/dev/null 2>&1; then
      file "$dest"
    fi
    ;;
esac

echo "Installed Linux waro CLI at $dest"
