#!/usr/bin/env bash
# Copyright (c) 2026, fni8 authors
# SPDX-License-Identifier: BSD-3-Clause
#
# Produce a byte-reproducible per-layer fp32 reference dump for ANY GGUF that
# llama.cpp supports. Runs the dumper on CPU (-ngl 0) so every op is fp32 — the
# trusted oracle an fni8 / fni8-serve port TDDs against.
#
#   MODEL     path to a *.gguf                              (required)
#   PROMPT    the fixed reference prompt                    (default below)
#   OUT       output directory for the dump                 (default ./ref_data)
#   NCTX      context length                                (default 512)
#   PATTERNS  optional comma-separated tensor-name prefixes to restrict the
#             dump (e.g. "Qcur-,Kcur-,Vcur-,ffn_out-"). Unset => dump ALL named
#             activations (recommended for a first bringup; narrow later).
#   LLAMA_CPP_DIR  built llama.cpp (for the commit stamp; default matches build)
#
# The prompt is a PARAMETER now (unlike the flint8 Qwen3.5-only original). Pick
# one short deterministic prompt and DO NOT change it without re-versioning the
# dump — the consumer pins to these exact activations/tokenization.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
OUT="${OUT:-$HERE/ref_data}"
BIN="${BIN:-$HERE/dump_reference}"
NCTX="${NCTX:-512}"
PROMPT="${PROMPT:-The capital of France is Paris.}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/josh/archives/llama-cpp-turboquant}"

if [ ! -x "$BIN" ]; then echo "build first: bash build_dump.sh" >&2; exit 1; fi
mkdir -p "$OUT"

COMMIT="$(git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# RUNPATH covers only direct deps; ggml's own transitive .so.0 needs the path too.
export LD_LIBRARY_PATH="$LLAMA_CPP_DIR/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FNI8_DUMP_DIR="$OUT"
export FNI8_LLAMA_COMMIT="$COMMIT"
if [ -n "${PATTERNS:-}" ]; then export FNI8_DUMP_PATTERNS="$PATTERNS"; fi

"$BIN" \
  -m "$MODEL" \
  -ngl 0 \
  -c "$NCTX" \
  --temp 0 \
  -p "$PROMPT"

echo "wrote reference dump to $OUT"
echo "manifest: $OUT/manifest.tsv   meta: $OUT/meta.txt"
