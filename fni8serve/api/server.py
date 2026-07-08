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

from ..engine import LLMEngine
from ..loader import checkpoint_info, load_fni8_state_dict
from ..models import ModelConfig
from .app import create_app


def load_engine(model_path: str, *, device: str = "cuda", max_num_seqs: int = 16,
                max_len: int = 2048, eos_id: int | None = None) -> LLMEngine:
    """Architecture is auto-detected from the checkpoint; see the README Quickstart."""
    info = checkpoint_info(model_path)
    cfg = ModelConfig.from_hf(info["meta"]["config"])
    weights = load_fni8_state_dict(model_path, device=device)
    return LLMEngine(cfg, weights, device=device, max_num_seqs=max_num_seqs,
                     max_len=max_len, eos_id=eos_id)


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

    engine = load_engine(args.model, device=args.device, max_num_seqs=args.max_num_seqs,
                         max_len=args.max_len, eos_id=tokenizer.eos_token_id)
    app = create_app(engine, tokenizer, served_model_name=args.served_model_name or args.model,
                     chat_template=chat_template, tool_parser=args.tool_parser)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
