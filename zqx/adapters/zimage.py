"""Z-Image (Turbo / Base) adapter.

Source of truth: comfy/ldm/lumina/model.py (NextDiT with z_image_modulation).

* DIFFUSION_MODEL wrapper args: (x, timesteps, context, num_tokens,
  attention_mask) + kwargs (transformer_options, control, ...).
* x is 4-D (B, C, H, W), padded to patch size 2.
* Before the main layers the model runs `context_refiner` (caption only) and
  `noise_refiner` (image only) blocks through the *same* attention function;
  those calls happen before transformer_options["block_index"] is set by the
  main loop.  The wrappers in this pack remove "block_index" before calling
  the model, so refiner calls are recognised by its absence.
* Main-layer joint sequence: [caption (padded to a multiple of
  pad_tokens_multiple=32) | image tokens (h_tok * w_tok) | image padding
  (to a multiple of 32)].  The caption length is only known inside the call,
  so the image start is computed as N - (n_img + n_pad).
* Target RoPE ids: (cap_len + 1, h, w) with h, w starting at 0 (pos_ids_x);
  padding tokens get id (0, 0, 0).  rope_options (scale/shift) would change
  that; the adapter refuses when they are set and a position offset needs them.
* Omni / Ming reference paths (ref_latents, ref_frames, siglip_feats) are not
  supported and raise.
"""
from __future__ import annotations

import torch

from .base import AdapterError, ModelAdapter, TokenLayout, ceil_div


class ZImageAdapter(ModelAdapter):
    name = "z_image"

    import re as _re
    block_key_re = _re.compile(r"(?:^|\.)layers\.(\d+)\.")
    group_rules = [(r"(?:^|\.)(noise_refiner|context_refiner|siglip_refiner)\.", "refiner")]

    @classmethod
    def matches(cls, dm) -> bool:
        try:
            from comfy.ldm.lumina.model import NextDiT
        except Exception:  # pragma: no cover
            return False
        if type(dm) is not NextDiT:
            return False
        # Z-Image: learned pad tokens (pad_tokens_multiple) + z_image modulation, whose adaLN
        # Sequential starts directly with a Linear(min(dim, 256), 4*dim) (Lumina 2 starts with SiLU).
        if getattr(dm, "pad_tokens_multiple", None) is None or len(dm.layers) == 0:
            return False
        mod = getattr(dm.layers[0], "adaLN_modulation", None)
        if mod is None or len(mod) == 0:
            return False
        first = mod[0]
        return hasattr(first, "in_features") and first.in_features == min(dm.dim, 256)

    def get_transformer_options(self, args, kwargs) -> dict:
        to = kwargs.get("transformer_options", None)
        if to is None:
            raise AdapterError("ZQX: could not locate transformer_options in Z-Image forward arguments")
        return to

    def check_supported_call(self, args, kwargs):
        for k in ("ref_latents", "ref_frames", "siglip_feats", "ref_contexts"):
            v = kwargs.get(k, None)
            if v is not None and len(v) > 0:
                raise AdapterError(f"ZQX: Z-Image '{k}' (Omni / Ming reference path) is not supported by this node.")

    @property
    def embedder(self):
        return self.dm.rope_embedder

    @property
    def total_blocks(self) -> int:
        return len(self.dm.layers)

    @property
    def patch_size(self) -> int:
        return int(self.dm.patch_size)

    def token_grid(self, x: torch.Tensor):
        if x.ndim != 4:
            raise AdapterError(f"ZQX: Z-Image latents must be 4-D (B, C, H, W), got {tuple(x.shape)}")
        p = self.patch_size
        return ceil_div(x.shape[-2], p), ceil_div(x.shape[-1], p)

    def n_pad(self, n_img: int) -> int:
        m = self.dm.pad_tokens_multiple
        return (-n_img) % m if m is not None else 0

    def layout(self, x: torch.Tensor, args, kwargs) -> TokenLayout:
        self.check_supported_call(args, kwargs)
        h_tok, w_tok = self.token_grid(x)
        n_img = h_tok * w_tok
        return TokenLayout(n_img=n_img, h_tok=h_tok, w_tok=w_tok, img_from_end=n_img + self.n_pad(n_img))

    def image_position_ids(self, x: torch.Tensor, n_txt: int, transformer_options=None) -> torch.Tensor:
        """ids of the target tokens; n_txt = padded caption length (cap_len in the model)."""
        if transformer_options is not None and transformer_options.get("rope_options", None) is not None:
            raise AdapterError("ZQX: rope_options are set on this Z-Image model; position offsets relative to the image extent are not supported with them.")
        h_tok, w_tok = self.token_grid(x)
        ids = torch.zeros((h_tok, w_tok, 3), dtype=torch.float32)
        ids[..., 0] = n_txt + 1
        ids[..., 1] = torch.arange(h_tok, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(w_tok, dtype=torch.float32)[None, :]
        return ids.reshape(-1, 3)
