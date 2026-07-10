# SPDX-License-Identifier: MIT
"""Runnable entrypoint: serve a `.fni8` checkpoint behind the OpenAI-compatible API.

    python -m fni8serve.api.server --model qwen3-8b.fni8 --tokenizer Qwen/Qwen3-8B

The `.fni8` file carries weights + architecture config, not tokenizer files, so
`--tokenizer` points at an HF repo id (or local dir) to load the tokenizer /
chat template from -- the base model repo, or the fni8-quant repo if it mirrors one.
`--chat-template` optionally overrides the tokenizer's own embedded Jinja template
with one loaded from a file (matches vLLM's `--chat-template` flag); the override is
rendered through the same sandboxed Jinja environment `apply_chat_template` always
uses, so this grants a custom template no more than the model's own gets.
Needs the `serve` extra: `pip install -e ".[serve]"`.
"""
from __future__ import annotations

import argparse
import itertools
import time

from ..engine import LLMEngine
from ..loader import checkpoint_info, load_fni8_state_dict
from ..models import ModelConfig
from .app import create_app


def load_engine(model_path: str, *, device: str = "cuda", max_num_seqs: int = 16,
                max_len: int = 2048, eos_id: int | None = None) -> LLMEngine:
    """Architecture is auto-detected from the checkpoint; see the README Quickstart."""
    info = checkpoint_info(model_path)
    meta_cfg = info["meta"]["config"]
    # Honor the arch stored at conversion time. The meta config is a ModelConfig dump
    # (carries `arch`, e.g. "qwen3_5_text") but often lacks HF `model_type`/`architectures`,
    # so from_hf's derivation alone would fall through to "unknown". Prefer the stored arch.
    cfg = ModelConfig.from_hf(meta_cfg, arch=meta_cfg.get("arch") or None)
    weights = load_fni8_state_dict(model_path, device=device)
    # Trust the weights over the recorded config: some converted Qwen3-family checkpoints
    # recorded qk_norm=False yet DO carry per-head q_norm/k_norm weights. Qwen3's QK-norm
    # is load-bearing (skipping it feeds un-normalized Q/K into RoPE -> garbage output).
    if not getattr(cfg, "qk_norm", False) and any(".q_norm.weight" in n for n in weights):
        try:
            cfg.qk_norm = True
        except Exception:  # frozen dataclass
            import dataclasses
            cfg = dataclasses.replace(cfg, qk_norm=True)
    return LLMEngine(cfg, weights, device=device, max_num_seqs=max_num_seqs,
                     max_len=max_len, eos_id=eos_id)


def _human_count(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


def build_banner(engine, tokenizer, *, served_model_name, chat_template, load_time_s, max_len):
    """Assemble the one-time startup banner facts from a live engine: model dims,
    GPU, a weights/KV/free VRAM memory breakdown, and the serving config (incl.
    CUDA-graph state + captured batch buckets). Read once at startup -- the one
    place a couple of `.numel()`/`mem_get_info` calls are fine (not the hot loop)."""
    from .. import __version__
    from ..metrics import gpu_stats

    cfg = engine.cfg
    model = engine.model
    tensors = list(itertools.chain(model.parameters(), model.buffers()))
    params = sum(t.numel() for t in tensors)
    weights_bytes = sum(t.numel() * t.element_size() for t in tensors)

    cache = getattr(engine, "cache", None)
    kv_blocks = getattr(cache, "num_blocks", 0)
    kv_bytes = 0
    for attr in ("k_cache", "v_cache", "k_scale", "v_scale"):
        t = getattr(cache, attr, None)
        if t is not None:
            kv_bytes += t.numel() * t.element_size()

    free_gib = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            free_gib = free / 1024**3
    except Exception:
        pass

    graphed = getattr(engine.runner, "graphed", None)
    cuda_graph = graphed is not None
    captured = list(getattr(graphed, "batch_buckets", ()) or ()) if cuda_graph else []

    return {
        "version": __version__,
        "model_name": served_model_name,
        "arch": cfg.arch,
        "quant": f"int8 W8A8 dp4a (weight_bits={cfg.weight_bits})",
        "compute": "sm_70",
        "gpu": gpu_stats(),
        "load_time_s": load_time_s,
        "dims": {
            "params": params,
            "params_str": _human_count(params),
            "layers": cfg.num_hidden_layers,
            "hidden": cfg.hidden_size,
            "num_heads": cfg.num_attention_heads,
            "num_kv_heads": cfg.num_key_value_heads,
            "head_dim": cfg.resolved_head_dim(),
            "vocab": cfg.vocab_size,
            "max_len": max_len,
        },
        "memory": {
            "weights_gib": weights_bytes / 1024**3,
            "kv_gib": kv_bytes / 1024**3,
            "kv_blocks": kv_blocks,
            "free_gib": free_gib,
        },
        "config": {
            "max_num_seqs": engine.scheduler.max_num_seqs,
            "max_len": max_len,
            "cuda_graph": cuda_graph,
            "captured_batch_sizes": captured,
            "tokenizer_id": getattr(tokenizer, "name_or_path", None),
            "chat_template": bool(chat_template) or bool(getattr(tokenizer, "chat_template", None)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Serve a .fni8 checkpoint over an OpenAI-compatible API")
    ap.add_argument("--model", required=True, help="path to a .fni8 checkpoint")
    ap.add_argument("--tokenizer", default=None,
                    help="HF repo id or local dir with the tokenizer (default: --model)")
    ap.add_argument("--chat-template", default=None,
                    help="path to a Jinja file overriding the tokenizer's own chat template")
    ap.add_argument("--tool-parser", default="hermes",
                    help="fni8serve.tool_calls parser for `tool_choice=auto` tool-call "
                         "extraction (default: hermes, Qwen's native tool-calling format)")
    ap.add_argument("--served-model-name", default=None, help="default: --model")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    chat_template = None
    if args.chat_template is not None:
        with open(args.chat_template, encoding="utf-8") as f:
            chat_template = f.read()

    _t0 = time.perf_counter()
    engine = load_engine(args.model, device=args.device, max_num_seqs=args.max_num_seqs,
                         max_len=args.max_len, eos_id=tokenizer.eos_token_id)
    load_time_s = time.perf_counter() - _t0

    from ..metrics import StatsCollector, render_banner, start_heartbeat
    from rich.console import Console

    served = args.served_model_name or args.model
    banner = build_banner(engine, tokenizer, served_model_name=served,
                          chat_template=chat_template, load_time_s=load_time_s,
                          max_len=args.max_len)
    stats = StatsCollector()
    stats.set_banner(banner)

    console = Console()
    console.print(render_banner(banner))
    start_heartbeat(stats, console=console, interval=5.0)

    app = create_app(engine, tokenizer, served_model_name=served,
                     chat_template=chat_template, tool_parser=args.tool_parser, stats=stats)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
