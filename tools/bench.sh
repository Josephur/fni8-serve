#!/usr/bin/env bash
# Run tools/bench_llm.py inside the fni8 container's `bench` service, which (per
# docs/CLAUDE-RUNNER-SETUP.md) is already pinned to the profiling-capable Volta
# cards (host idx 4/7/9/11/14) — never the counter-locked CMP cards. Usage:
#   tools/bench.sh /mnt/24tb/fni8-forge/weights/Qwen__Qwen3-8B.b8.fni8
#   tools/bench.sh /mnt/24tb/fni8-forge/weights/Qwen__Qwen3-30B-A3B.b8.fni8 --num-prompts 8
#
# Before running: check `nvidia-smi` on the host for a free card among 4/7/9/11/14
# (the fleet also runs quant/forge jobs) and pass `--gpu-load-caveat "..."` through
# to bench_llm.py so the caveat travels with the recorded numbers.
set -euo pipefail

FNI8_DIR="${FNI8_DIR:-/srv/nvme-data/containers/projects/fused_ni8}"
SERVE_DIR="${SERVE_DIR:-/srv/nvme-data/containers/projects/fni8-serve}"
ARCHIVE="${ARCHIVE:-/mnt/24tb}"
# Compose service name for the GPU-pinned profiling container. Not verified against
# the actual docker-compose.yml from this session (sandboxed away from it) — override
# if it's actually called e.g. "profile" instead of "bench".
BENCH_SERVICE="${BENCH_SERVICE:-bench}"

docker compose -f "$FNI8_DIR/docker-compose.yml" run --rm \
  -v "$ARCHIVE":"$ARCHIVE" \
  -v "$SERVE_DIR":/serve \
  -e FORGE_BASE=/mnt/24tb/fni8-forge \
  "$BENCH_SERVICE" bash -lc "python3 /serve/tools/bench_llm.py $* --out-dir /serve/bench"
