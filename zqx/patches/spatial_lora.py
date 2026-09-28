"""Spatial (token-masked) LoRA: h = W x + m_token * s(sigma) * scale * B (A x).

Idea (LoRAShop, arXiv 2505.23758, applies a subject LoRA only inside the subject's region): instead of merging
the LoRA into the weight (which applies it to every token), the LoRA is run as an additive side branch whose
output is multiplied per token.  With the realism LoRA masked to the background/body and the character LoRA
masked to the face, both can run at full strength in the same step without fighting over the face.

Implementation: for every model call a DIFFUSION_MODEL wrapper computes the token layout, registers PyTorch
forward hooks on the LoRA's target Linear modules for the duration of the call and removes them afterwards.
Each hook adds  weights[token] * s * scale * (x A^T) B^T  to the module output (into the right slice for fused
weights such as Z-Image's attention.qkv).  Token weights by stream:

  image      target image tokens -> spatial mask; trailing tokens (QIE reference tokens, Z-Image padding) -> other_weight
  text       every token -> text_weight
  joint      Z-Image main layers / final layer: [caption -> text_weight | image -> mask | padding -> other_weight]
  nonspatial modulation / timestep layers (input has no token axis) -> nonspatial_weight

With all weights 1 the result equals LoraLoaderModelOnly at the same strength (tested); with all weights 0 no
hook is installed (bitwise identity).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..core.schedule import linear_sigma_ramp, sigma_from_transformer_options
from . import passes
from .reference_attention import resize_mask_to_tokens


class _Branch:
    __slots__ = ("up", "down", "scale", "offset")

    def __init__(self, up, down, scale, offset):
        self.up, self.down, self.scale, self.offset = up, down, scale, offset


def branches_from_patches(patches_by_key) -> Dict[str, List[_Branch]]:
    """Plain LoRA adapters only (comfy LoRAAdapter without DoRA / mid / reshape)."""
    out: Dict[str, List[_Branch]] = {}
    for mk, lst in patches_by_key.items():
        for (v, offset, function) in lst:
            if type(v).__name__ != "LoRAAdapter":
                raise ValueError(f"ZQX Spatial LoRA: only plain LoRA is supported ({mk} is {type(v).__name__})")
            up, down, alpha, mid, dora, reshape = v.weights
            if mid is not None or dora is not None or reshape is not None:
                raise ValueError(f"ZQX Spatial LoRA: LoCon/DoRA/reshape components are not supported ({mk})")
            if function is not None:
                raise ValueError(f"ZQX Spatial LoRA: weight conversion functions are not supported ({mk})")
            if up.ndim != 2 or down.ndim != 2:
                raise ValueError(f"ZQX Spatial LoRA: only linear LoRAs are supported ({mk})")
            if offset is not None and offset[0] != 0:
                raise ValueError(f"ZQX Spatial LoRA: input-dimension slices are not supported ({mk})")
            rank = down.shape[0]
            scale = (float(alpha) / rank) if alpha is not None else 1.0
            out.setdefault(mk, []).append(_Branch(up.float(), down.float(), scale, offset))
    return out


class SpatialLoraPatch:
    def __init__(self, adapter, branches: Dict[str, List[_Branch]], strength_early: float, strength_late: float,
                 sigma_hi: float, sigma_lo: float, mask: Optional[torch.Tensor], invert: bool,
                 text_weight: float, nonspatial_weight: float, other_weight: float, key: str):
        if sigma_hi < sigma_lo:
            raise ValueError("sigma_hi must be >= sigma_lo")
        for name, w in (("text_weight", text_weight), ("nonspatial_weight", nonspatial_weight), ("other_weight", other_weight)):
            if not (0.0 <= w <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        self.adapter = adapter
        self.branches = branches
        self.early, self.late, self.hi, self.lo = strength_early, strength_late, sigma_hi, sigma_lo
        self.mask = mask
        self.invert = invert
        self.text_w, self.nonsp_w, self.other_w = text_weight, nonspatial_weight, other_weight
        self.key = key
        self.streams = {mk: adapter.module_stream(mk) for mk in branches}
        self.modules = {}
        for mk in branches:
            path = mk[: -len(".weight")]
            self.modules[mk] = adapter.model_patcher.get_model_object(path)
        self.calls_patched = 0

    # -------------------------------------------------------------------------------------------------
    def _image_weights(self, n_img, h_tok, w_tok, device):
        if self.mask is None:
            m = torch.ones(n_img, dtype=torch.float32, device=device)
        else:
            m = resize_mask_to_tokens(self.mask, h_tok, w_tok).to(device)
        if self.invert:
            m = 1.0 - m
        return m

    def token_weights(self, stream: str, x: torch.Tensor, layout, img_w: torch.Tensor):
        """Per-token weights (N,) for this module call, or a python float for non-spatial inputs."""
        if stream == "nonspatial" or x.ndim == 2:
            return float(self.nonsp_w)
        n = x.shape[-2]
        dev = x.device
        if stream == "text":
            return torch.full((n,), float(self.text_w), device=dev)
        if stream == "image":
            if n < layout.n_img:
                raise RuntimeError(f"ZQX Spatial LoRA: image-stream input has {n} tokens < {layout.n_img} image tokens")
            rest = torch.full((n - layout.n_img,), float(self.other_w), device=dev)
            return torch.cat([img_w, rest])
        if stream == "joint":
            img_block = layout.img_from_end
            if img_block is None or n < img_block:
                raise RuntimeError("ZQX Spatial LoRA: unexpected joint-sequence layout")
            n_cap = n - img_block
            pad = img_block - layout.n_img
            return torch.cat([torch.full((n_cap,), float(self.text_w), device=dev), img_w,
                              torch.full((pad,), float(self.other_w), device=dev)])
        raise RuntimeError(f"ZQX Spatial LoRA: unknown stream {stream}")

    def _current_layout(self, cache):
        """Token layout of the forward now running (the innermost ZQX pass entry): another patch may be running
        an extra pass on a different image (e.g. the reference-attention capture pass)."""
        args, kwargs, depth = passes.current_call()
        if args is None:
            raise RuntimeError("ZQX Spatial LoRA: hook fired outside a ZQX model call")
        key = (depth, id(args))
        if key not in cache:
            x = self.adapter.get_x(args, kwargs)
            layout = self.adapter.layout(x, args, kwargs)
            cache[key] = (layout, self._image_weights(layout.n_img, layout.h_tok, layout.w_tok, x.device))
        return cache[key]

    def _make_hook(self, mk, strength, cache):
        branches = self.branches[mk]
        stream = self.streams[mk]

        def hook(module, inputs, output):
            x = inputs[0]
            layout, img_w = self._current_layout(cache)
            tw = self.token_weights(stream, x, layout, img_w)
            if isinstance(tw, float):
                if tw == 0.0:
                    return output
                wvec = None
            else:
                if not bool(torch.any(tw != 0)):
                    return output
                wvec = tw.to(output.dtype).view(*([1] * (x.ndim - 2)), -1, 1)
            out = output.clone()
            xf = x.to(torch.float32)
            for b in branches:
                d = (xf @ b.down.to(x.device).T) @ b.up.to(x.device).T
                d = d * (strength * b.scale)
                d = d.to(output.dtype)
                if wvec is not None:
                    d = d * wvec
                elif tw != 1.0:
                    d = d * tw
                if b.offset is None:
                    out = out + d
                else:
                    _, start, size = b.offset
                    out[..., start:start + size] = out[..., start:start + size] + d
            return out

        return hook

    def wrapper(self, executor, *args, **kwargs):
        with passes.entry("enter", self, args, kwargs):
            return self._wrapper(executor, *args, **kwargs)

    def _wrapper(self, executor, *args, **kwargs):
        ad = self.adapter
        to = ad.get_transformer_options(args, kwargs)
        sigma = sigma_from_transformer_options(to)
        strength = linear_sigma_ramp(sigma, self.hi, self.lo, self.early, self.late)
        if strength == 0.0:
            return passes.call(executor, args, kwargs)
        x = ad.get_x(args, kwargs)
        layout = ad.layout(x, args, kwargs)
        img_w = self._image_weights(layout.n_img, layout.h_tok, layout.w_tok, x.device)
        if not bool(torch.any(img_w != 0)) and self.text_w == 0 and self.nonsp_w == 0 and self.other_w == 0:
            return passes.call(executor, args, kwargs)
        handles = []
        cache = {}
        try:
            for mk, mod in self.modules.items():
                handles.append(mod.register_forward_hook(self._make_hook(mk, strength, cache)))
            out = passes.call(executor, args, kwargs)
        finally:
            for h in handles:
                h.remove()
        self.calls_patched += 1
        return out


def install(model_patcher, lora_sd, allow_unmatched: bool, **kw):
    from ..adapters import get_adapter
    from .runtime_lora import load_lora_patches
    import comfy.patcher_extension as pe
    m = model_patcher.clone()
    ad = get_adapter(m)
    pk, unmatched = load_lora_patches(m, lora_sd, allow_unmatched=allow_unmatched)
    key = kw.pop("key", "zqx_spatial_lora")
    patch = SpatialLoraPatch(ad, branches_from_patches(pk), key=key, **kw)
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, key, patch.wrapper)
    return m, patch, unmatched
