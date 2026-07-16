#!/usr/bin/env bash
set -euo pipefail

model="${QWEN35B_MODEL:-/models/Qwen3.6-35B-A3B-MTP-TQ3_4S.gguf}"

if [[ ! -f "${model}" ]]; then
  echo "missing model: ${model}" >&2
  exit 1
fi

args=(
  /app/llama-server
  --host "${LLAMA_ARG_HOST:-0.0.0.0}"
  --port "${LLAMA_ARG_PORT:-8080}"
  --model "${model}"
  --batch-size "${QWEN35B_BATCH_SIZE:-32}"
  --ubatch-size "${QWEN35B_UBATCH_SIZE:-32}"
  --cache-type-k "${QWEN35B_CACHE_TYPE_K:-q8_0}"
  --cache-type-v "${QWEN35B_CACHE_TYPE_V:-tq3_0}"
  --flash-attn "${QWEN35B_FLASH_ATTN:-on}"
  --parallel "${QWEN35B_PARALLEL:-1}"
)

if [[ -n "${QWEN35B_CTX_SIZE:-}" ]]; then
  args+=(--ctx-size "${QWEN35B_CTX_SIZE}")
fi

if [[ -n "${QWEN35B_GPU_LAYERS:-}" ]]; then
  args+=(--gpu-layers "${QWEN35B_GPU_LAYERS}")
fi

chat_template="${QWEN35B_CHAT_TEMPLATE:-/models/chat_template.jinja}"
if [[ -f "${chat_template}" ]]; then
  args+=(--chat-template-file "${chat_template}")
fi

if [[ "${QWEN35B_ENABLE_MTP:-0}" == "1" ]]; then
  args+=(
    --model-draft "${model}"
    --spec-type mtp
    --spec-draft-n-max "${QWEN35B_SPEC_DRAFT_N_MAX:-1}"
  )
fi

if [[ -n "${QWEN35B_REASONING_FORMAT:-}" ]]; then
  args+=(--reasoning-format "${QWEN35B_REASONING_FORMAT}")
fi

if [[ -n "${QWEN35B_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${QWEN35B_EXTRA_ARGS}"
  args+=("${extra_args[@]}")
fi

exec "${args[@]}"
