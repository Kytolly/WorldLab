#!/usr/bin/env bash
# Serve the WorldLab pi05 checkpoint with OpenPI (the third-party/openpi
# submodule is not modified).
#
# Run inside the openpi environment. The serving adapter is bundled in the
# optional worldlab-integrations-openpi package.
# Env vars:
#   OPENPI_ROOT   openpi repository (default: third-party/openpi)
#   OPENPI_CKPT   openpi checkpoint dir (WorldLab model_zoo default)
#   PORT          websocket port (default: 8000)
#   ASSET_ID      norm-stats asset id (default: gesim)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third-party/openpi}"
WORLDLAB_OPENPI_SRC="${REPO_ROOT}/packages/worldlab-integrations-openpi/src"
OPENPI_CKPT="${OPENPI_CKPT:-${REPO_ROOT}/model_zoo/agibot-world/Genie-Envisioner-Sim-v2.0/checkpoints/pi05_gesim_g01op_test}"
PORT="${PORT:-8000}"
ASSET_ID="${ASSET_ID:-gesim}"

# The serving module is owned by the optional WorldLab OpenPI package.
export PYTHONPATH="${WORLDLAB_OPENPI_SRC}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"

exec python -m worldlab_openpi.openpi_serving.serve_pi05 \
    --checkpoint "${OPENPI_CKPT}" \
    --asset-id "${ASSET_ID}" \
    --port "${PORT}" \
    "$@"
