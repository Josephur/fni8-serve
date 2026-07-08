# SPDX-License-Identifier: MIT
"""Model registry — the seam that makes support fully modular.

A model family registers a builder under one or more architecture keys (HF
`model_type` or `architectures[0]`). `build_model(config, weights)` looks the key
up and hands back a `CausalLM`. Adding a family = one `@register_model(...)` + deco;
no engine, runner, or kernel changes. `list_models()` powers the coverage matrix.
"""
from __future__ import annotations

from typing import Callable

from .base import CausalLM, Weights
from .config import ModelConfig

_REGISTRY: dict[str, Callable[[ModelConfig, Weights], CausalLM]] = {}
_ALIASES: dict[str, str] = {}


def register_model(*keys: str) -> Callable:
    """Register a builder `fn(config, weights) -> CausalLM` under one or more keys
    (lowercased). The first key is canonical; the rest are aliases."""
    def deco(fn: Callable[[ModelConfig, Weights], CausalLM]):
        canon = keys[0].lower()
        _REGISTRY[canon] = fn
        for k in keys:
            _ALIASES[k.lower()] = canon
        return fn
    return deco


def build_model(config: ModelConfig, weights: Weights) -> CausalLM:
    key = _resolve(config.arch)
    if key is None:
        raise KeyError(
            f"no model registered for arch {config.arch!r}. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](config, weights)


def is_supported(arch: str) -> bool:
    return _resolve(arch) is not None


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def _resolve(arch: str) -> str | None:
    a = arch.lower()
    if a in _REGISTRY:
        return a
    if a in _ALIASES:
        return _ALIASES[a]
    # HF arch class names like "Qwen3ForCausalLM" -> "qwen3"
    for key in _REGISTRY:
        if a.startswith(key) or a.replace("forcausallm", "").rstrip("_") == key:
            return key
    return None
