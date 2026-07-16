# Qwen3.6-35B-A3B TQ3 sidecar

This is the measured fast one-card path for the custom TurboQuant `TQ3_4S` GGUF.
Upstream `gguf` does not recognize its type 46 tensors, so this format cannot be
loaded by `fni8serve.gguf_native` without a dedicated reader and CUDA kernel port.
The sidecar pins the exact turbo-tan runtime commit that implements those kernels
for sm_70 and exposes the same OpenAI-compatible API.

Download the text model (the vision projector is not needed):

```bash
hf download YTan2000/Qwen3.6-35B-A3B-MTP-TQ3_4S \
  Qwen3.6-35B-A3B-MTP-TQ3_4S.gguf chat_template.jinja \
  --local-dir /srv/nvme-data/model-cache/fni8/qwen35b-tq3
```

Build and run:

```bash
docker compose build
docker compose up -d
curl -fsS http://127.0.0.1:8017/health
```

The first build compiles the monolithic Volta flash-attention translation unit and
can take roughly 15 minutes. Later builds use Docker cache. `LLAMA_BUILD_UI=OFF` is
intentional: the historical UI asset bucket is no longer reliable and the service
only needs the API.

Measured on the CMP 100-210 fleet, GPU 8, 2026-07-16: 160-token decode at 40.63
tok/s and 108.9 prompt tok/s for a 28-token prompt. The prior archived runs measured
40-41 tok/s. MTP is disabled because it regressed decode to 36.66 tok/s at draft
width 1 despite 100% acceptance. These numbers are fleet-specific and do not transfer
to a real V100.
