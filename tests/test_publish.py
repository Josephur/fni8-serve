# SPDX-License-Identifier: MIT
"""Unit tests for the HF publish helpers (pure functions — no network)."""
import pytest

from fni8serve.publish import is_restricted, model_card, parse_fni8_name, repo_name


@pytest.mark.parametrize("fname,parent,kind,bits", [
    ("Qwen__Qwen3-0.6B.b8.fni8", "Qwen/Qwen3-0.6B", "llm", 8),
    ("Qwen__Qwen3-0.6B.b4.fni8", "Qwen/Qwen3-0.6B", "llm", 4),
    ("Qwen__Qwen-Image.dit.b8.fni8", "Qwen/Qwen-Image", "dit", 8),
    ("black-forest-labs__FLUX.1-dev.dit.b8.fni8", "black-forest-labs/FLUX.1-dev", "dit", 8),
    ("Qwen__Qwen-Image-Edit-2509.dit.b4.fni8", "Qwen/Qwen-Image-Edit-2509", "dit", 4),
])
def test_parse_fni8_name(fname, parent, kind, bits):
    r = parse_fni8_name(fname)
    assert r == {"parent_repo": parent, "kind": kind, "bits": bits}


@pytest.mark.parametrize("bad", ["notafni8.txt", "README.md", "x.fni8", "no-bits.fni8"])
def test_parse_fni8_name_rejects_nonmatching(bad):
    assert parse_fni8_name(bad) is None


def test_repo_name_strips_org_and_adds_suffix():
    assert repo_name("black-forest-labs/FLUX.1-dev", "jajmangold") == "jajmangold/FLUX.1-dev-fni8"
    assert repo_name("Qwen/Qwen3-8B", "jajmangold") == "jajmangold/Qwen3-8B-fni8"


def _q(bits, gb=1.0):
    return {"bits": bits, "filename": f"m.b{bits}.fni8", "gb": gb}


def test_model_card_declares_quantized_linkage():
    """The whole point: base_model + base_model_relation=quantized so it files under
    the parent's Quantizations on the Hub."""
    card = model_card("Qwen/Qwen3-8B", "llm", [_q(8, 8.8)], license_tag="apache-2.0",
                      pipeline_tag="text-generation")
    assert card.startswith("---\n")
    assert "base_model: Qwen/Qwen3-8B" in card
    assert "base_model_relation: quantized" in card
    assert "license: apache-2.0" in card
    assert "pipeline_tag: text-generation" in card
    assert card.split("\n")[0] == "---"


def test_model_card_lists_both_quants_distinctly():
    """A repo with int8 AND int4 must make each file's precision unmistakable."""
    card = model_card("Qwen/Qwen3-8B", "llm", [_q(8, 8.8), _q(4, 4.6)],
                      license_tag="apache-2.0")
    assert "m.b8.fni8" in card and "m.b4.fni8" in card
    assert "int8" in card and "int4" in card
    assert "8-bit" in card and "4-bit" in card          # per-bits tags
    assert "| file | precision |" in card               # Files table header


def test_model_card_links_back_to_github():
    card = model_card("x/y", "llm", [_q(8)], license_tag="mit")
    assert "github.com/jajmangold/fni8" in card
    assert "github.com/jajmangold/fni8-serve" in card
    assert "github.com/jajmangold/ComfyUI-fni8" in card


def test_model_card_inherits_parent_license_even_when_restricted():
    card = model_card("black-forest-labs/FLUX.1-dev", "dit", [_q(8)],
                      license_tag="flux-1-dev-non-commercial-license",
                      pipeline_tag="text-to-image", native_dtype="bfloat16")
    assert "base_model: black-forest-labs/FLUX.1-dev" in card
    assert "license: flux-1-dev-non-commercial-license" in card
    assert "bfloat16" in card
    assert "comfyui" in card                       # dit tag set


def test_model_card_omits_missing_optional_fields():
    card = model_card("x/y", "llm", [_q(8)], license_tag=None, pipeline_tag=None)
    assert "license:" not in card
    assert "pipeline_tag:" not in card
    assert "base_model_relation: quantized" in card


def test_model_card_scheme_labels():
    assert "W4A8" in model_card("x/y", "llm", [_q(4)], license_tag="mit")
    assert "W8A8" in model_card("x/y", "llm", [_q(8)], license_tag="mit")


@pytest.mark.parametrize("lic,gated,expected", [
    ("apache-2.0", False, False),
    ("mit", False, False),
    ("llama3.1", False, False),
    ("cc-by-nc-4.0", False, True),
    ("flux-1-dev-non-commercial-license", False, True),
    ("creativeml-openrail-m", False, True),
    ("apache-2.0", "manual", True),          # gated overrides
    (None, False, True),                     # unknown terms -> restricted
    ("other", False, True),
])
def test_is_restricted(lic, gated, expected):
    assert is_restricted(lic, gated) is expected
