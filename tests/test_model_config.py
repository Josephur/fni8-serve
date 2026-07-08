# SPDX-License-Identifier: MIT
"""ModelConfig.from_hf: plain text configs and multimodal wrappers that nest the
text-model axes under `text_config` (Gemma3-it and friends)."""
import pytest

pytest.importorskip("fni8")   # importing fni8serve.models runs its __init__, which needs fni8

from fni8serve.models import ModelConfig


def _text_cfg(**overrides):
    cfg = dict(
        model_type="gemma3_text", vocab_size=262144, hidden_size=2560,
        num_hidden_layers=34, num_attention_heads=8, num_key_value_heads=4,
        intermediate_size=10240, head_dim=256, query_pre_attn_scalar=256,
        sliding_window=1024, sliding_window_pattern=6,
    )
    cfg.update(overrides)
    return cfg


def test_from_hf_plain_text_config():
    cfg = ModelConfig.from_hf(_text_cfg())
    assert cfg.arch == "gemma3_text"
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 4
    assert cfg.hidden_size == 2560


def test_from_hf_dives_into_text_config_when_top_level_is_bare():
    """The historical case: a multimodal wrapper whose top level has no text-model
    fields at all, only `text_config` + a vision/model-type wrapper."""
    hf = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": _text_cfg(),
        "vision_config": {"hidden_size": 1152, "model_type": "siglip_vision_model"},
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 4
    assert cfg.hidden_size == 2560
    assert cfg.arch == "gemma3"                  # top-level model_type wins for arch


def test_from_hf_merges_text_config_even_when_top_level_partially_duplicates_fields():
    """Regression: some multimodal layouts duplicate a FEW fields at the top level
    (e.g. `hidden_size`, for AutoConfig probing) without duplicating the rest. The old
    guard (`"text_config" in c and "hidden_size" not in c`) skipped the dive whenever
    `hidden_size` leaked to the top level, then KeyError'd on `num_attention_heads`."""
    hf = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "hidden_size": 999,                       # stale/partial top-level duplicate
        "text_config": _text_cfg(),
        "vision_config": {"hidden_size": 1152, "model_type": "siglip_vision_model"},
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.num_attention_heads == 8           # would KeyError before the fix
    assert cfg.hidden_size == 2560                # text_config wins over the stale 999
    assert cfg.arch == "gemma3"


def test_from_hf_unknown_keys_land_in_extra():
    hf = {"text_config": _text_cfg(foo_bar=123)}
    cfg = ModelConfig.from_hf(hf)
    assert cfg.extra.get("foo_bar") == 123
