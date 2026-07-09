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


# ── Multimodal VLM configs ──────────────────────────────────────────────


def _qwen2_5_vl_cfg(**overrides):
    """Simulated Qwen2.5-VL-7B config.json (HF hub shape)."""
    cfg = {
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "vocab_size": 152064,
        "hidden_size": 3584,                           # text hidden_size
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "intermediate_size": 18944,
        "max_position_embeddings": 32768,
        "vision_config": {
            "depth": 32,
            "hidden_size": 3584,
            "patch_size": 14,
            "spatial_merge_size": 2,
        },
        "image_token_id": 151655,
    }
    cfg.update(overrides)
    return cfg


def test_from_hf_qwen2_5_vl():
    """Qwen2.5-VL: is_multimodal=True, vision_config parsed, text axes intact."""
    hf = _qwen2_5_vl_cfg()
    cfg = ModelConfig.from_hf(hf)
    # Multimodal fields
    assert cfg.is_multimodal is True
    assert cfg.vision_config is not None
    assert cfg.vision_config.hidden_size == 3584
    assert cfg.vision_config.patch_size == 14
    assert cfg.vision_config.num_layers == 32
    assert cfg.vision_config.image_token_id == 151655
    assert cfg.vision_config.spatial_merge_size == 2
    assert cfg.image_token_id == 151655
    # Text-model axes (from text_config merge or top-level fallback)
    assert cfg.arch == "qwen2_5_vl"
    assert cfg.vocab_size == 152064
    assert cfg.hidden_size == 3584
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 28
    assert cfg.num_key_value_heads == 4
    # VLM keys are consumed, not leaked into extra
    assert "vision_config" not in cfg.extra
    assert "image_token_id" not in cfg.extra


def test_from_hf_qwen2_5_vl_uses_text_config_subdict():
    """Qwen2.5-VL config that nests text axes under text_config (HF hub style)."""
    text = {
        "vocab_size": 152064, "hidden_size": 3584, "num_hidden_layers": 28,
        "num_attention_heads": 28, "num_key_value_heads": 4,
        "intermediate_size": 18944, "max_position_embeddings": 32768,
    }
    hf = {
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "text_config": text,
        "vision_config": {
            "depth": 32, "hidden_size": 3584, "patch_size": 14,
            "spatial_merge_size": 2,
        },
        "image_token_id": 151655,
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.is_multimodal is True
    assert cfg.vision_config.hidden_size == 3584
    assert cfg.vision_config.num_layers == 32
    assert cfg.hidden_size == 3584
    assert cfg.num_hidden_layers == 28
    assert cfg.image_token_id == 151655


def test_from_hf_llava():
    """LLaVA: uses num_hidden_layers (not depth), image_token_index mapped."""
    text = {
        "vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 11008,
    }
    hf = {
        "model_type": "llava",
        "architectures": ["LlavaForConditionalGeneration"],
        "text_config": text,
        "vision_config": {
            "hidden_size": 1024, "patch_size": 14, "num_hidden_layers": 24,
        },
        "image_token_index": 32001,
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.is_multimodal is True
    assert cfg.vision_config is not None
    assert cfg.vision_config.hidden_size == 1024
    assert cfg.vision_config.patch_size == 14
    assert cfg.vision_config.num_layers == 24
    assert cfg.vision_config.image_token_id == 32001
    assert cfg.vision_config.spatial_merge_size == 1   # default for LLaVA
    assert cfg.image_token_id == 32001
    # Text axes intact
    assert cfg.arch == "llava"
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 32


def test_from_hf_llava_next():
    """LLaVA-Next: same pattern as LLaVA."""
    text = {
        "vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 11008,
    }
    hf = {
        "model_type": "llava_next",
        "architectures": ["LlavaNextForConditionalGeneration"],
        "text_config": text,
        "vision_config": {
            "hidden_size": 1024, "patch_size": 14, "num_hidden_layers": 24,
        },
        "image_token_index": 32001,
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.is_multimodal is True
    assert cfg.vision_config.num_layers == 24
    assert cfg.vision_config.spatial_merge_size == 1
    assert cfg.image_token_id == 32001


def test_from_hf_text_only_configs_unaffected():
    """Existing text-only configs still parse with is_multimodal=False."""
    cfg = ModelConfig.from_hf(_text_cfg())
    assert cfg.is_multimodal is False
    assert cfg.vision_config is None
    assert cfg.image_token_id is None

    # Gemma3 wrapper (text_model has vision_config but arch isn't VLM)
    hf = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": _text_cfg(),
        "vision_config": {"hidden_size": 1152, "model_type": "siglip_vision_model"},
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.is_multimodal is False     # gemma3 is not in _MULTIMODAL_ARCHS
    assert cfg.vision_config is None
    assert cfg.num_attention_heads == 8   # text axes still work
