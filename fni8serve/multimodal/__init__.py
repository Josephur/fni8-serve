# SPDX-License-Identifier: MIT
"""Multimodal pipeline: vision tower (ViT) with int8 dp4a linears + fp16 attention."""

from __future__ import annotations

import base64
import io
import re
import urllib.request

from PIL import Image

from .preprocess import preprocess_qwen2_5_vl  # noqa: F401
from .projector import build_projector, embed_merge  # noqa: F401
from .vit import VisionTransformer  # noqa: F401


def fetch_image(url: str) -> Image.Image:
    """Fetch an image from an HTTP(S) URL or a base64 data URI and return PIL ``Image`` (RGB).

    Args:
        url: HTTP(S) URL or ``data:image/...;base64,...`` data URI.

    Returns:
        PIL ``Image.Image`` in RGB mode.
    """
    if url.startswith("data:"):
        # data:[<mediatype>][;base64],<data>
        match = re.match(r"^data:[^;]*(?:;base64)?,(.+)$", url)
        if not match:
            raise ValueError(f"Unsupported data URI format: {url[:80]}")
        raw = base64.b64decode(match.group(1))
        return Image.open(io.BytesIO(raw)).convert("RGB")
    # HTTP(S) URL
    with urllib.request.urlopen(url) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")
