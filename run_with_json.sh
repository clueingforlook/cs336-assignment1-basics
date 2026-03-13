#!/usr/bin/env bash
set -euo pipefail

script=""
config=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --script)
            script="${2:-}"
            shift 2
            ;;
        --config)
            config="${2:-}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: ./run_with_json.sh --script <script.py> --config <config.json>" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$script" || -z "$config" ]]; then
    echo "Usage: ./run_with_json.sh --script <script.py> --config <config.json>" >&2
    exit 1
fi

mapfile -d '' args < <(
    python3 - "$config" <<'PY'
import json
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

for key, value in config.items():
    if value is None:
        continue

    sys.stdout.write(f"--{key}\0")
    if isinstance(value, list):
        for item in value:
            sys.stdout.write(f"{item}\0")
    else:
        sys.stdout.write(f"{value}\0")
PY
)

uv run "$script" "${args[@]}"
