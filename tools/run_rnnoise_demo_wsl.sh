#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: run_rnnoise_demo_wsl.sh <input.raw> <output.raw>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEMO_PATH="$(/bin/bash "${SCRIPT_DIR}/build_rnnoise_wsl.sh")"
"${DEMO_PATH}" "$1" "$2"
