"""Qwen-Image / Qwen-Image-2512 / Qwen-Image-Edit(-2509/-2511) adapter.

Source of truth: comfy/ldm/qwen_image/model.py (QwenImageTransformer2DModel).

* DIFFUSION_MODEL wrapper args: (x, timestep, context, attention_mask,
  ref_latents, additional_t_cond, transformer_options) + kwargs
  (control, ref_latents_method, ...).
* x is 5-D (B, C, T=1, H, W).  process_img pads H, W to the patch size 2.
* Joint attention sequence (Attention.forward): [txt (context.shape[1]) |
  target image (h_tok * w_tok) | QIE reference tokens (one block per
  ref latent, appended after the target in _forward)].
* Target image RoPE ids: (frame=0, h - h_tok//2, w - w_tok//2) for
  h in [0, h_tok), w in [0, w_tok) (process_img with index=0, offsets 0).
  QIE "index" refs use frame = 1, 2, ...;  "negative_index" uses -1, -2, ...
* 60 double-stream blocks; transformer_options["block_index"] is set before
  every block call.
"""
from __future__ import annotations

import torch

from .base import AdapterError, ModelAdapter, TokenLayout, ceil_div


class QwenImageAdapter(ModelAdapter):
    name = "qwen_image"

    import re as _re
    block_key_re = _re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.")
    group_rules = []

    @classmethod
    def matches(cls, dm) -> bool:
        try:
            from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel
        except Exception:  # pragma: no cover
            return False
        # exact class only: subclasses (e.g. other Qwen variants) may lay tokens out differently
        return type(dm) is QwenImageTransformer2DModel

    def get_transformer_options(self, args, kwargs) -> dict:
        if len(args) >= 7 and isinstance(args[6], dict):
            return args[6]
        if "transformer_options" in kwargs:
            return kwargs["transformer_options"]
        raise AdapterError("ZQX: could not locate transformer_options in Qwen-Image forward arguments")

    def get_ref_latents(self, args, kwargs):
        refs = args[4] if len(args) > 4 else kwargs.get("ref_latents", None)
        return refs

    @property
    def embedder(self):
        return self.dm.pe_embedder

    @property
    def total_blocks(self) -> int:
        return len(self.dm.transformer_blocks)

    @property
    def patch_size(self) -> int:
        return int(self.dm.patch_size)

    def token_grid(self, x: torch.Tensor):
        if x.ndim != 5:
            raise AdapterError(f"ZQX: Qwen-Image latents must be 5-D (B, C, T, H, W), got {tuple(x.shape)}")
        if x.shape[2] != 1:
            raise AdapterError("ZQX: Qwen-Image adapter supports single-frame image latents only (T == 1)")
        p = self.patch_size
        return ceil_div(x.shape[-2], p), ceil_div(x.shape[-1], p)

    def n_tokens(self, lat: torch.Tensor) -> int:
        p = self.patch_size
        t = lat.shape[2] if lat.ndim == 5 else 1
        return t * ceil_div(lat.shape[-2], p) * ceil_div(lat.shape[-1], p)

    def layout(self, x: torch.Tensor, args, kwargs) -> TokenLayout:
        h_tok, w_tok = self.token_grid(x)
        n_img = h_tok * w_tok
        context = self.get_context(args, kwargs)
        n_txt = int(context.shape[1])
        refs = self.get_ref_latents(args, kwargs)
        n_refs = sum(self.n_tokens(r) for r in refs) if refs is not None else 0
        return TokenLayout(n_img=n_img, h_tok=h_tok, w_tok=w_tok, img_start=n_txt,
                           seq_len=n_txt + n_img + n_refs)

    def image_position_ids(self, x: torch.Tensor, n_txt: int = 0) -> torch.Tensor:
        h_tok, w_tok = self.token_grid(x)
        hh = torch.arange(h_tok, dtype=torch.float32) - (h_tok // 2)
        ww = torch.arange(w_tok, dtype=torch.float32) - (w_tok // 2)
        ids = torch.zeros((h_tok, w_tok, 3), dtype=torch.float32)
        ids[..., 1] = hh[:, None]
        ids[..., 2] = ww[None, :]
        return ids.reshape(-1, 3)
