#!/usr/bin/env bash
# Run the forge quant pipeline inside the fni8 container, with the archive disk
# mounted and the HF token passed through. Usage mirrors forge.py:
#   tools/forge.sh one  Qwen/Qwen3.5-0.8B
#   tools/forge.sh one  black-forest-labs/FLUX.1-dev --kind dit
#   tools/forge.sh batch /serve/tools/models.txt
#   tools/forge.sh status
set -euo pipefail

FNI8_DIR="${FNI8_DIR:-/srv/nvme-data/containers/projects/fused_ni8}"
SERVE_DIR="${SERVE_DIR:-/srv/nvme-data/containers/projects/fni8-serve}"
ARCHIVE="${ARCHIVE:-/mnt/24tb}"

docker compose -f "$FNI8_DIR/docker-compose.yml" run --rm \
  -v "$ARCHIVE":"$ARCHIVE" \
  -v "$SERVE_DIR":/serve \
  -e HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN \
  -e HF_HOME=/mnt/24tb/fni8-forge/hf \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -e FORGE_BASE=/mnt/24tb/fni8-forge \
  -e FORGE_MIN_FREE_GB="${FORGE_MIN_FREE_GB:-300}" \
  test bash -lc "pip install -q huggingface_hub hf_transfer safetensors 2>/dev/null || pip install -q huggingface_hub safetensors 2>/dev/null; python3 /serve/tools/forge.py $*"
