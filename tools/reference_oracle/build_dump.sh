#!/usr/bin/env bash
# Copyright (c) 2026, fni8 authors
# SPDX-License-Identifier: BSD-3-Clause
#
# Build the arch-agnostic reference dumper (dump_reference.cpp) against a
# prebuilt on-box llama.cpp. No llama.cpp source changes: we link the shared
# libs the `llama-eval-callback` example already builds, and provide our own
# dumping eval callback.
#
#   LLAMA_CPP_DIR   path to a built llama.cpp checkout (has build/bin/*.so and
#                   the common/, include/, ggml/include/ headers)
#
# The pinned llama.cpp commit is auto-detected from LLAMA_CPP_DIR's git HEAD and
# baked into the dump's meta.txt (FNI8_LLAMA_COMMIT) so every dump records
# exactly which graph produced it. Validated against commit 69d8e4be4 (archs:
# qwen35, qwen35moe, qwen3next, gemma3, gemma3n, gemma4, glm4moe, deepseek2,
# lfm2, lfm2moe, hunyuan-moe, hunyuan-dense, …).
#
# Usage: LLAMA_CPP_DIR=/path/to/llama.cpp bash build_dump.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/josh/archives/llama-cpp-turboquant}"
BIN="${BIN:-$HERE/dump_reference}"

if [ ! -d "$LLAMA_CPP_DIR/build/bin" ]; then
  echo "LLAMA_CPP_DIR=$LLAMA_CPP_DIR has no build/bin — point it at a built llama.cpp" >&2
  exit 1
fi

INC=(-I"$LLAMA_CPP_DIR/common" -I"$LLAMA_CPP_DIR/include"
     -I"$LLAMA_CPP_DIR/ggml/include")
LIBDIR="$LLAMA_CPP_DIR/build/bin"

g++ -std=c++17 -O2 "${INC[@]}" \
  "$HERE/dump_reference.cpp" \
  -L"$LIBDIR" -Wl,-rpath,"$LIBDIR" \
  -lllama-common -lllama -lggml -lggml-base \
  -o "$BIN"

COMMIT="$(git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "built $BIN   (llama.cpp @ $COMMIT)"
echo "run: FNI8_DUMP_DIR=<out> FNI8_LLAMA_COMMIT=$COMMIT $BIN -m <model.gguf> -ngl 0 -c 512 --temp 0 -p \"<prompt>\""
echo "  (or use dump.sh, which sets FNI8_LLAMA_COMMIT for you)"
