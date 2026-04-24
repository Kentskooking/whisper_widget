#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BINARY_PATH="$("${SCRIPT_DIR}/build_webrtc_apm_wsl.sh")"

exec "${BINARY_PATH}" "$@"
