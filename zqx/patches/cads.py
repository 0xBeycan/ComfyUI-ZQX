"""CADS as a DIFFUSION_MODEL wrapper that corrupts the text conditioning (context) per call."""
from __future__ import annotations

import torch

from ..core.cads import cads_apply, cads_gamma
from ..core.schedule import sigma_from_transformer_options
from . import passes

APPLY_TO = ["cond_and_uncond", "cond_only"]


class CADSPatch:
    WRAPPER_KEY = "zqx_cads"

    def __init__(self, adapter, tau1: float, tau2: float, noise_scale: float, psi: float, seed: int,
                 apply_to: str = "cond_and_uncond", relative_noise: bool = True):
        if apply_to not in APPLY_TO:
            raise ValueError(f"apply_to must be one of {APPLY_TO}")
        if noise_scale < 0:
            raise ValueError("noise_scale must be >= 0")
        cads_gamma(0.5, tau1, tau2)  # validates taus
        self.adapter = adapter
        self.tau1, self.tau2 = tau1, tau2
        self.s, self.psi, self.seed = noise_scale, psi, seed
        self.apply_to = apply_to
        self.relative = relative_noise
        self.calls_patched = 0

    def _noise(self, shape, sigma: float, device):
        # fresh noise each step (as in the paper), deterministic per (seed, sigma)
        g = torch.Generator(device="cpu").manual_seed(int(self.seed) * 1000003 + int(round(sigma * 1e6)))
        return torch.randn(shape, generator=g, dtype=torch.float32).to(device)

    def wrapper(self, executor, *args, **kwargs):
        ad = self.adapter
        to = ad.get_transformer_options(args, kwargs)
        sigma = sigma_from_transformer_options(to)
        gamma = cads_gamma(sigma, self.tau1, self.tau2)
        if self.s == 0.0 or gamma == 1.0:
            return passes.call(executor, args, kwargs)
        ctx = ad.get_context(args, kwargs)
        if ctx is None:
            return passes.call(executor, args, kwargs)
        b = ctx.shape[0]
        rows = torch.ones(b, dtype=torch.bool)
        if self.apply_to == "cond_only":
            cou = to.get("cond_or_uncond", None)
            if cou is None or b % len(cou) != 0:
                raise RuntimeError("ZQX CADS: cannot identify cond/uncond rows in this batch")
            per = b // len(cou)
            for i, c in enumerate(cou):
                if c == 1:
                    rows[i * per:(i + 1) * per] = False
            if not bool(rows.any()):
                return passes.call(executor, args, kwargs)
        noise = self._noise(ctx.shape, sigma, ctx.device)
        new = cads_apply(ctx, gamma, self.s, self.psi, noise, relative_noise=self.relative)
        rows = rows.to(ctx.device).view(b, *([1] * (ctx.ndim - 1)))
        new = torch.where(rows, new, ctx)
        args, kwargs = ad.replace_args(args, kwargs, context=new)
        self.calls_patched += 1
        return passes.call(executor, args, kwargs)


def install(model_patcher, **kw):
    from ..adapters import get_adapter
    import comfy.patcher_extension as pe
    m = model_patcher.clone()
    patch = CADSPatch(get_adapter(m), **kw)
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, CADSPatch.WRAPPER_KEY, patch.wrapper)
    return m, patch
