#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_ROOT="${WEBRTC_APM_BUILD_ROOT:-${REPO_ROOT}/sidecache/webrtc_apm_build}"
BINARY_PATH="${BUILD_ROOT}/webrtc_apm_wav"
SOURCE_PATH="${REPO_ROOT}/tools/webrtc_apm_wav.cpp"

mkdir -p "${BUILD_ROOT}"

if ! command -v pkg-config >/dev/null 2>&1; then
  echo "pkg-config not found. Install: sudo apt-get install -y pkg-config" >&2
  exit 1
fi

if ! pkg-config --exists webrtc-audio-processing sndfile; then
  cat >&2 <<'EOF'
Missing native build dependencies for the WebRTC APM helper.
Install them in WSL with:
  sudo apt-get update
  sudo apt-get install -y libwebrtc-audio-processing-dev libsndfile1-dev g++ pkg-config cmake
EOF
  exit 1
fi

if [[ ! -x "${BINARY_PATH}" || "${SOURCE_PATH}" -nt "${BINARY_PATH}" || "${BASH_SOURCE[0]}" -nt "${BINARY_PATH}" ]]; then
  g++ \
    -std=c++17 \
    -O3 \
    -Wall \
    -Wextra \
    -pedantic \
    "${SOURCE_PATH}" \
    -o "${BINARY_PATH}" \
    $(pkg-config --cflags --libs webrtc-audio-processing sndfile)
fi

printf '%s\n' "${BINARY_PATH}"
