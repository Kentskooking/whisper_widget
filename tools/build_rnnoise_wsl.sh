#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_ROOT="${RNNOISE_BUILD_ROOT:-${REPO_ROOT}/sidecache/rnnoise_build}"
TARBALL="${BUILD_ROOT}/rnnoise-0.2.tar.gz"
SOURCE_DIR="${BUILD_ROOT}/rnnoise-0.2"
DEMO_PATH="${SOURCE_DIR}/examples/rnnoise_demo"

mkdir -p "${BUILD_ROOT}"

if [[ ! -f "${TARBALL}" ]]; then
  python3 - <<PY
import urllib.request
urllib.request.urlretrieve(
    "https://github.com/xiph/rnnoise/releases/download/v0.2/rnnoise-0.2.tar.gz",
    r"${TARBALL}",
)
PY
fi

if [[ ! -x "${DEMO_PATH}" ]]; then
  rm -rf "${SOURCE_DIR}"
  tar -xzf "${TARBALL}" -C "${BUILD_ROOT}"
  cd "${SOURCE_DIR}"
  ./configure
  make -j"${RNNOISE_BUILD_JOBS:-4}"
fi

printf '%s\n' "${DEMO_PATH}"
