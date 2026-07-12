# SPDX-License-Identifier: MIT
"""Multi-Token Prediction (MTP) — the DeepSeek-V3 / Qwen3-Next speculative-decode
head, and the draft+verify loop it drives.

MTP module (per depth k, verified against DeepSeek-V3 §2.2 and vLLM's
Qwen3NextMTP): given the previous depth's hidden `h` for a position and the
embedding of the token to predict, produce the next hidden:

    h'_k = M_k · [ RMSNorm_hidden(h) ; RMSNorm_emb(Emb(t_next)) ]      # fc: 2H -> H
    h_k  = TRM_k(h'_k)                                                 # one decoder block
    logits = SharedHead(SharedNorm(h_k))

The embedding table, final norm, and LM head are PHYSICALLY SHARED with the main
model; each depth has its own `pre_fc_norm_*`, `fc`, and transformer block. The
released models ship depth 1 (predict the 2nd token) at ~80-90% acceptance.

Speculative decoding (chain verify — fni8 already ships the kernel):
  1. draft: from the main model's last hidden, roll the MTP head(s) forward to
     propose k candidate tokens (cheap; no full-model forward per token);
  2. verify: append the k drafts and run the MAIN model over prefix+drafts in ONE
     causal forward — fni8.attn_int8_verify does exactly this (each draft attends
     prefix + preceding drafts), returning the true distribution at each slot;
  3. accept the longest matching prefix (greedy: draft==argmax; sampling: the
     rejection test), commit 1+(#accepted) tokens, repeat.

This module is the head + a greedy single-depth draft. The accept/verify loop is
wired at the engine level (PR3) where the main model's KV cache lives.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.norm import RMSNorm
from .config import ModelConfig


def _fc_to_plain(w):
    """The MTP fc is used as a PLAIN matmul (`fused @ fc.t()`), so it must be a
    dense tensor. A quantized checkpoint ships it as a QTensor (per_row_i8 or
    per_group_i4) which has no `.detach()` — dequantize it to fp16 here. The fc is
    small and the head runs at most once per draft step, so a dense fc is fine (and
    for hybrids where spec-decode is guarded off, the head never runs at all)."""
    if not hasattr(w, "scheme"):
        return w  # already a dense tensor
    if w.scheme == "per_row_i8":
        return (w.data.float() * w.scale.float().unsqueeze(-1)).to(torch.float16)
    if w.scheme == "per_group_i4":
        from fni8.quant.lowbit import dequantize_lowbit

        d = w.data  # [N, K/2] uint8, 2 signed nibbles/byte (even col = low nibble)
        lo = (d & 0xF).to(torch.int16); lo = torch.where(lo >= 8, lo - 16, lo)
        hi = ((d >> 4) & 0xF).to(torch.int16); hi = torch.where(hi >= 8, hi - 16, hi)
        codes = torch.stack([lo, hi], dim=-1).reshape(d.shape[0], -1).to(torch.int8)
        return dequantize_lowbit(codes, w.scale, group_size=w.group_size).to(torch.float16)
    raise ValueError(f"MTP fc: unsupported quant scheme {w.scheme!r}")


class MTPLayer(nn.Module):
    """One MTP depth: two input RMSNorms + fc(2H->H) + a transformer block.

    `block` is a fully-built decoder layer (same class the backbone uses) so MTP
    inherits the family's attention/MLP exactly. `fc_weight` is the M_k projection
    [hidden, 2*hidden]."""

    def __init__(self, cfg: ModelConfig, *, fc_weight, hidden_norm, emb_norm, block,
                 norm_add_unit_offset: bool | None = None):
        super().__init__()
        # Qwen3.5's RMSNorm is zero-centered (gain = 1 + weight) but its config does
        # NOT set `cfg.norm_add_unit_offset` (the builder hardcodes it), so allow the
        # caller to force the offset for the pre-fc norms explicitly.
        off = cfg.norm_add_unit_offset if norm_add_unit_offset is None else norm_add_unit_offset
        self.pre_fc_norm_hidden = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, hidden_norm,
                                          add_unit_offset=off)
        self.pre_fc_norm_embedding = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, emb_norm,
                                             add_unit_offset=off)
        self.fc = nn.Parameter(_fc_to_plain(fc_weight))  # [hidden, 2*hidden], dense
        self.block = block

    def forward(self, prev_hidden, next_token_emb, positions, ctx, layer_idx):
        h = self.pre_fc_norm_hidden(prev_hidden)
        e = self.pre_fc_norm_embedding(next_token_emb)
        fused = torch.cat([h, e], dim=-1) @ self.fc.t().to(h.dtype)   # 2H -> H
        out, _ = self.block(fused, positions, ctx, None)
        return out


class MultiTokenPredictor(nn.Module):
    """N stacked MTP depths over a shared embedding + final norm + LM head (all
    borrowed from the main model). `draft_greedy` rolls the head forward to propose
    `k <= num_depths` tokens from the main model's last hidden state."""

    def __init__(self, cfg: ModelConfig, *, layers, embed, final_norm, lm_head):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.embed = embed              # shared VocabEmbedding
        self.norm = final_norm          # shared final RMSNorm
        self.lm_head = lm_head          # shared LMHead

    def num_depths(self) -> int:
        return len(self.layers)

    @torch.inference_mode()
    def draft_greedy(self, last_hidden, last_token, positions, ctx) -> list[torch.Tensor]:
        """last_hidden [B,1,H], last_token [B,1] -> list of k proposed token ids."""
        drafts, hidden, token = [], last_hidden, last_token
        for depth, layer in enumerate(self.layers):
            emb = self.embed(token)
            hidden = layer(hidden, emb, positions, ctx, depth)
            logits = self.lm_head(self.norm(hidden)[:, -1])
            token = logits.argmax(-1, keepdim=True)
            drafts.append(token)
        return drafts


def build_mtp(cfg: ModelConfig, sd: dict, embed, norm, lm_head, rope,
              decoder_fn) -> MultiTokenPredictor | None:
    """Build MTP heads from weights if cfg.num_mtp_layers > 0.

    *decoder_fn* builds a single decoder layer (attention + MLP + pre/post norms):
    ``decoder_fn(cfg, layer_idx, mtp_prefix, sd, rope) -> module`` whose forward
    is ``(x, positions, ctx, residual) -> (output, residual)``.
    """
    if cfg.num_mtp_layers <= 0:
        return None
    K = cfg.num_mtp_layers
    layers = []
    for k in range(1, K + 1):
        p = f"model.mtp.{k}"
        fc_weight = sd[f"{p}.fc.weight"]
        hidden_norm = sd[f"{p}.pre_fc_norm_hidden.weight"]
        emb_norm = sd[f"{p}.pre_fc_norm_embedding.weight"]
        block = decoder_fn(cfg, k, p, sd, rope)
        layers.append(MTPLayer(cfg, fc_weight=fc_weight, hidden_norm=hidden_norm,
                                emb_norm=emb_norm, block=block))
    return MultiTokenPredictor(cfg, layers=layers, embed=embed, final_norm=norm, lm_head=lm_head)
