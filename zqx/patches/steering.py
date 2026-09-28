"""Contrastive activation steering (ActAdd-style, arXiv 2308.10248, applied to DiT image hidden states).

For every model call inside the sigma window two extra forwards run on the *same* x with a "towards" and an
"away" conditioning (e.g. "candid, looking away" vs "posing, looking at the camera").  After each selected
block the image hidden states h_towards and h_away are recorded; in the real forward the image hidden states
become  h + alpha * delta,  delta = h_towards - h_away  (mode 'token') or its mean over image tokens (mode
'mean', a global direction without spatial layout).  Uses the models' double_block patch hook (Qwen: after each
double block, Z-Image: after each main layer; refiners are not touched).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from ..adapters import AdapterError, QwenImageAdapter, ZImageAdapter
from ..core.schedule import in_sigma_window, parse_block_list, sigma_from_transformer_options
from . import passes


class SteeringPatch:
    WRAPPER_KEY = "zqx_activation_steering"

    def __init__(self, adapter, towards, away, alpha, sigma_start, sigma_end, blocks, mode, apply_to):
        if mode not in ("mean", "token"):
            raise ValueError("mode must be 'mean' or 'token'")
        if apply_to not in ("all", "cond_only"):
            raise ValueError("apply_to must be all or cond_only")
        self.adapter = adapter
        self.towards, self.away = towards, away
        self.alpha = float(alpha)
        self.sigma_start, self.sigma_end = sigma_start, sigma_end
        spec = (blocks or "").strip().lower()
        self.mid = spec in ("", "mid", "middle")
        self.blocks = None if self.mid else parse_block_list(blocks, adapter.total_blocks)
        self.mode = mode
        self.apply_to = apply_to
        self.state: Optional[str] = None
        self.store: Dict[str, Dict[int, torch.Tensor]] = {}
        self.n_img = 0
        self.rows = None
        self.calls_patched = 0
        self.injected = []

    def _selected(self, bi, total):
        if self.mid:
            return bi == total // 2
        return self.blocks is None or bi in self.blocks

    # double_block patch -----------------------------------------------------------------------------
    def block_patch(self, args):
        img, txt = args["img"], args["txt"]
        st = self.state
        if st is None:
            return {"img": img, "txt": txt}
        bi = args["block_index"]
        to = args.get("transformer_options", {})
        if not self._selected(bi, to.get("total_blocks", self.adapter.total_blocks)):
            return {"img": img, "txt": txt}
        n = self.n_img
        if st in ("towards", "away"):
            if not passes.blocked(self, ("foreign", "variant")):
                self.store[st][bi] = img[:, :n].detach().clone()
            return {"img": img, "txt": txt}
        # inject (plain and variant passes; not into other patches' foreign passes)
        if passes.blocked(self, ("foreign",)):
            return {"img": img, "txt": txt}
        delta = self.store["towards"][bi] - self.store["away"][bi]
        if self.mode == "mean":
            delta = delta.mean(dim=1, keepdim=True)
        rows = self.rows.to(img.device).view(-1, 1, 1).to(img.dtype)
        img = img.clone()
        img[:, :n] = img[:, :n] + self.alpha * rows * delta.to(img.dtype)
        if not passes.blocked(self, ("foreign", "variant")):
            self.injected.append(bi)
        return {"img": img, "txt": txt}

    # wrapper ---------------------------------------------------------------------------------------
    def _cond_args(self, args, kwargs, cond, batch):
        t, meta = cond[0]
        ctx = t.to(self.adapter.get_context(args, kwargs).device, self.adapter.get_context(args, kwargs).dtype)
        ctx = ctx.expand(batch, -1, -1) if ctx.shape[0] == 1 else ctx
        if ctx.shape[0] != batch:
            raise ValueError("steering conditioning batch must be 1 or equal to the model batch")
        a = list(args)
        a[2] = ctx
        if isinstance(self.adapter, QwenImageAdapter):
            am = meta.get("attention_mask", None)
            a[3] = None if am is None else am.to(ctx.device).expand(batch, -1)
        elif isinstance(self.adapter, ZImageAdapter):
            a[3] = ctx.shape[1]   # num_tokens
            a[4] = None           # attention_mask
        return tuple(a), kwargs

    def wrapper(self, executor, *args, **kwargs):
        if passes.in_foreign_pass():
            return passes.call(executor, args, kwargs)
        with passes.entry("enter", self, args, kwargs):
            return self._wrapper(executor, *args, **kwargs)

    def _wrapper(self, executor, *args, **kwargs):
        ad = self.adapter
        to = ad.get_transformer_options(args, kwargs)
        sigma = sigma_from_transformer_options(to)
        if self.alpha == 0.0 or not in_sigma_window(sigma, self.sigma_start, self.sigma_end):
            return passes.call(executor, args, kwargs)
        x = ad.get_x(args, kwargs)
        b = x.shape[0]
        rows = torch.ones(b)
        if self.apply_to == "cond_only":
            cou = to.get("cond_or_uncond", None)
            if cou is None or b % len(cou) != 0:
                raise RuntimeError("ZQX steering: cannot identify cond/uncond rows")
            per = b // len(cou)
            for i, c in enumerate(cou):
                if c == 1:
                    rows[i * per:(i + 1) * per] = 0
            if not bool(rows.any()):
                return passes.call(executor, args, kwargs)
        self.n_img = ad.layout(x, args, kwargs).n_img
        self.rows = rows
        self.store = {"towards": {}, "away": {}}
        self.injected = []
        try:
            for name, cond in (("towards", self.towards), ("away", self.away)):
                self.state = name
                a2, k2 = self._cond_args(args, kwargs, cond, b)
                with passes.entry("foreign", self, a2, k2):
                    passes.call(executor, a2, k2)
            if sorted(self.store["towards"]) != sorted(self.store["away"]) or not self.store["towards"]:
                raise AdapterError("ZQX steering: no block captured (check the block list)")
            self.state = "inject"
            out = passes.call(executor, args, kwargs)
        finally:
            self.state = None
        self.calls_patched += 1
        return out


def install(model_patcher, **kw):
    from ..adapters import get_adapter
    import comfy.patcher_extension as pe
    m = model_patcher.clone()
    patch = SteeringPatch(get_adapter(m), **kw)
    m.set_model_double_block_patch(patch.block_patch)
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, SteeringPatch.WRAPPER_KEY, patch.wrapper)
    return m, patch
