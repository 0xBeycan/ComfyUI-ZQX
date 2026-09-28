"""Model-output guidance patches: LoRA guidance (LoRA-CFG) and perturbed-attention guidance for DiTs.

LoRA guidance (heuristic, the model-difference form of CFG; related to autoguidance, arXiv 2406.02507):
    out = out_base + w(sigma) * (out_lora - out_base)
w = 1 is the plain LoRA (single forward), w = 0 the base model, w > 1 extrapolates the LoRA's effect.  With a
sigma schedule, e.g. w ~ 0.3 in the layout steps and 1.5 in the detail steps, the base model decides
composition while identity is amplified late.

Perturbed-attention guidance (PAG, arXiv 2403.17377), adapted to joint-attention DiTs through the attention
override: in the perturbed forward the selected blocks replace the self-attention map of every *image* query
by the identity (the query attends only to itself, i.e. its output is its own value vector); text queries are
left unchanged.  out = out + s * (out - out_perturbed) on the selected rows.  ComfyUI's core PAG node patches a
UNet-only location and core SkipLayerGuidanceDiT cannot hook Z-Image; this works for both Qwen-Image and Z-Image.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..adapters import AdapterError
from ..core.schedule import in_sigma_window, linear_sigma_ramp, parse_block_list, sigma_from_transformer_options
from . import passes


def _row_mask(to, batch: int, apply_to: str, device):
    rows = torch.ones(batch, dtype=torch.bool, device=device)
    if apply_to == "all":
        return rows
    cou = to.get("cond_or_uncond", None)
    if cou is None or len(cou) == 0 or batch % len(cou) != 0:
        raise RuntimeError("ZQX: cannot identify cond/uncond rows in this batch")
    per = batch // len(cou)
    for i, c in enumerate(cou):
        if c == 1:
            rows[i * per:(i + 1) * per] = False
    return rows


class LoraGuidancePatch:
    WRAPPER_KEY = "zqx_lora_guidance"

    def __init__(self, adapter, runtime_patch, w_early: float, w_late: float, sigma_hi: float, sigma_lo: float):
        self.adapter = adapter
        self.rt = runtime_patch
        self.w_early, self.w_late, self.hi, self.lo = w_early, w_late, sigma_hi, sigma_lo
        self.calls_double = 0

    def wrapper(self, executor, *args, **kwargs):
        with passes.entry("enter", self, args, kwargs):
            return self._wrapper(executor, *args, **kwargs)

    def _wrapper(self, executor, *args, **kwargs):
        to = self.adapter.get_transformer_options(args, kwargs)
        sigma = sigma_from_transformer_options(to)
        w = linear_sigma_ramp(sigma, self.hi, self.lo, self.w_early, self.w_late)
        if w == 1.0:
            return passes.call(executor, args, kwargs)
        self.rt.enabled = False
        try:
            with passes.entry("variant", self, args, kwargs):
                out0 = passes.call(executor, args, kwargs)
        finally:
            self.rt.enabled = True
        if w == 0.0:
            return out0
        out1 = passes.call(executor, args, kwargs)
        self.calls_double += 1
        return out0 + w * (out1 - out0)


class PerturbedAttentionPatch:
    WRAPPER_KEY = "zqx_dit_pag"

    def __init__(self, adapter, scale: float, sigma_start: float, sigma_end: float, blocks: str, apply_to: str,
                 prev_override=None):
        if apply_to not in ("cond_only", "all"):
            raise ValueError("apply_to must be cond_only or all")
        self.adapter = adapter
        self.scale = float(scale)
        self.sigma_start, self.sigma_end = sigma_start, sigma_end
        spec = (blocks or "").strip().lower()
        self.mid = spec in ("", "mid", "middle")
        self.blocks = None if self.mid else parse_block_list(blocks, adapter.total_blocks)
        self.apply_to = apply_to
        self.prev_override = prev_override
        self.perturb = False
        self.layout = None
        self.calls_patched = 0
        self.perturbed_blocks: List[int] = []

    def _selected(self, bi, total):
        if self.mid:
            return bi == total // 2
        return self.blocks is None or bi in self.blocks

    def _next(self, func, args, kwargs):
        if self.prev_override is not None:
            return self.prev_override(func, *args, **kwargs)
        return func(*args, **kwargs)

    def attention_override(self, func, *args, **kwargs):
        to = kwargs.get("transformer_options", None)
        if not self.perturb or to is None or passes.blocked(self, ("foreign",)):
            return self._next(func, args, kwargs)
        bi = to.get("block_index", None)
        if bi is None or not self._selected(bi, to.get("total_blocks", self.adapter.total_blocks)):
            return self._next(func, args, kwargs)
        q, k, v = args[0], args[1], args[2]
        if not kwargs.get("skip_reshape", False) or v.ndim != 4:
            raise AdapterError("ZQX PAG: expected (B, H, N, D) attention inputs")
        out = self._next(func, args, kwargs)
        # spans from the query length: chained overrides (reference attention) may append extra keys/values,
        # but queries always equal the model's own token sequence and v[:, :, :Nq] are the model's own values
        s0, s1 = self.layout.span(q.shape[2])
        self.perturbed_blocks.append(bi)
        return identity_image_rows(out, v, s0, s1, kwargs.get("skip_output_reshape", False))

    def wrapper(self, executor, *args, **kwargs):
        if passes.in_foreign_pass():
            return passes.call(executor, args, kwargs)
        with passes.entry("enter", self, args, kwargs):
            return self._wrapper(executor, *args, **kwargs)

    def _wrapper(self, executor, *args, **kwargs):
        ad = self.adapter
        to = ad.get_transformer_options(args, kwargs)
        to.pop("block_index", None)
        sigma = sigma_from_transformer_options(to)
        if self.scale == 0.0 or not in_sigma_window(sigma, self.sigma_start, self.sigma_end):
            return passes.call(executor, args, kwargs)
        x = ad.get_x(args, kwargs)
        rows = _row_mask(to, x.shape[0], self.apply_to, x.device)
        if not bool(rows.any()):
            return passes.call(executor, args, kwargs)
        out = passes.call(executor, args, kwargs)
        self.layout = ad.layout(x, args, kwargs)
        self.perturb = True
        self.perturbed_blocks = []
        try:
            to.pop("block_index", None)
            with passes.entry("variant", self, args, kwargs):
                out_p = passes.call(executor, args, kwargs)
        finally:
            self.perturb = False
        if not self.perturbed_blocks:
            raise AdapterError("ZQX PAG: no block was perturbed (check the block list)")
        self.calls_patched += 1
        r = rows.view(-1, *([1] * (out.ndim - 1)))
        return torch.where(r, out + self.scale * (out - out_p), out)


def identity_image_rows(out: torch.Tensor, v: torch.Tensor, s0: int, s1: int, skip_output_reshape: bool) -> torch.Tensor:
    """Replace the attention output of image queries [s0, s1) by their own value vectors (identity attention map)."""
    b, h, n, d = v.shape
    out = out.clone()
    if skip_output_reshape:              # (B, H, N, D)
        out[:, :, s0:s1] = v[:, :, s0:s1].to(out.dtype)
    else:                                # (B, N, H*D)
        out[:, s0:s1] = v[:, :, s0:s1].transpose(1, 2).reshape(b, s1 - s0, h * d).to(out.dtype)
    return out


def install_pag(model_patcher, **kw):
    from ..adapters import get_adapter
    import comfy.patcher_extension as pe
    m = model_patcher.clone()
    ad = get_adapter(m)
    to = m.model_options.setdefault("transformer_options", {})
    patch = PerturbedAttentionPatch(ad, prev_override=to.get("optimized_attention_override", None), **kw)
    to["optimized_attention_override"] = patch.attention_override
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, PerturbedAttentionPatch.WRAPPER_KEY, patch.wrapper)
    return m, patch


def install_lora_guidance(model_patcher, entries_by_key, key, w_early, w_late, sigma_hi, sigma_lo):
    from .runtime_lora import install_runtime_lora
    import comfy.patcher_extension as pe
    if sigma_hi < sigma_lo:
        raise ValueError("sigma_hi must be >= sigma_lo")
    m, rt = install_runtime_lora(model_patcher, entries_by_key, key)
    patch = LoraGuidancePatch(rt.adapter, rt, w_early, w_late, sigma_hi, sigma_lo)
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, key + "_guidance", patch.wrapper)
    return m, patch, rt
