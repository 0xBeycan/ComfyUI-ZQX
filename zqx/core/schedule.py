"""Sigma-space windows, ramps and block-spec parsing (pure Python / torch, no ComfyUI).

Convention used by the whole pack
---------------------------------
All timestep windows are defined in **sigma space** of the model's own noise
schedule.  For the rectified-flow models this pack targets (Z-Image, Qwen-Image,
Qwen-Image-Edit) sigma == flow time t in [0, 1]:  x_t = (1 - t) * x0 + t * eps,
t = 1 is pure noise, t = 0 is the clean image.

Why sigma space (and not "fraction of the sampler's steps"): in the two-pass
workflow the img2img pass starts at e.g. sigma = 0.55.  A window defined as
"sigma in [0.6, 1.0]" then correctly never fires in the second pass (the layout
that window is meant to influence is already fixed), whereas a step-fraction
window would silently re-map to a completely different noise level.  A window
is active iff  sigma_end <= sigma <= sigma_start  (both inclusive).
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Tuple


def in_sigma_window(sigma: float, sigma_start: float, sigma_end: float) -> bool:
    """True iff sigma_end <= sigma <= sigma_start.

    sigma_start is the *high-noise* edge (first steps), sigma_end the low-noise edge.
    """
    if sigma_start < sigma_end:
        raise ValueError(
            f"sigma_start ({sigma_start}) must be >= sigma_end ({sigma_end}); "
            "sigma_start is the high-noise (early) edge of the window."
        )
    return (sigma >= sigma_end) and (sigma <= sigma_start)


def linear_sigma_ramp(sigma: float, sigma_hi: float, sigma_lo: float,
                      value_hi: float, value_lo: float) -> float:
    """Piecewise-linear schedule in sigma.

    value_hi for sigma >= sigma_hi, value_lo for sigma <= sigma_lo, linear in
    between.  sigma_hi == sigma_lo gives a hard switch (value_hi at sigma >= sigma_hi).
    """
    if sigma_hi < sigma_lo:
        raise ValueError(f"sigma_hi ({sigma_hi}) must be >= sigma_lo ({sigma_lo})")
    if sigma >= sigma_hi:
        return float(value_hi)
    if sigma <= sigma_lo:
        return float(value_lo)
    # here sigma_lo < sigma < sigma_hi, so sigma_hi > sigma_lo strictly
    a = (sigma - sigma_lo) / (sigma_hi - sigma_lo)
    return float(value_lo + a * (value_hi - value_lo))


_RANGE_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


def parse_block_list(spec: str, total_blocks: Optional[int] = None) -> Optional[List[int]]:
    """Parse "all" | "" | "0-9, 20, 30-35" into a sorted list of block indices.

    Returns None for "all"/"" (meaning every block).  Raises ValueError on bad
    syntax, and on indices >= total_blocks when total_blocks is given.
    """
    s = (spec or "").strip().lower()
    if s in ("", "all", "*"):
        return None
    out = set()
    for part in s.split(","):
        if part.strip() == "":
            continue
        m = _RANGE_RE.match(part)
        if m is None:
            raise ValueError(f"Bad block spec element {part!r} in {spec!r}; use e.g. '0-9, 20, 30-35' or 'all'.")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) is not None else a
        if b < a:
            raise ValueError(f"Bad block range {part!r}: end < start.")
        out.update(range(a, b + 1))
    if total_blocks is not None:
        bad = [i for i in out if i >= total_blocks]
        if bad:
            raise ValueError(f"Block indices {sorted(bad)[:8]} out of range; model has {total_blocks} blocks (0..{total_blocks - 1}).")
    return sorted(out)


def parse_block_weights(spec: str) -> Tuple[Dict[int, float], Dict[str, float]]:
    """Parse a block-weight spec like "0-9:1.0, 10-29:0.5, 30:0, other:1, refiner:0.8".

    Returns (per_block_index_weights, named_group_weights).  Named groups are
    free-form lowercase words (the adapter decides what keys belong to which
    group, e.g. "refiner" for Z-Image noise/context refiners, "other" for
    non-block layers such as img_in / proj_out).  Anything unspecified gets 1.0.
    Later entries override earlier ones.
    """
    per_block: Dict[int, float] = {}
    groups: Dict[str, float] = {}
    s = (spec or "").strip()
    if s == "":
        return per_block, groups
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        if ":" not in part:
            raise ValueError(f"Bad block-weight element {part!r}; expected 'range:weight', e.g. '0-9:0.5'.")
        lhs, rhs = part.rsplit(":", 1)
        try:
            w = float(rhs)
        except ValueError:
            raise ValueError(f"Bad weight {rhs!r} in {part!r}.")
        if not math.isfinite(w):
            raise ValueError(f"Non-finite weight in {part!r}.")
        lhs = lhs.strip().lower()
        m = _RANGE_RE.match(lhs)
        if m is not None:
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) is not None else a
            if b < a:
                raise ValueError(f"Bad block range {lhs!r}: end < start.")
            for i in range(a, b + 1):
                per_block[i] = w
        elif re.match(r"^[a-z_]+$", lhs):
            groups[lhs] = w
        else:
            raise ValueError(f"Bad block-weight selector {lhs!r}.")
    return per_block, groups


def block_weight_for(block_index: Optional[int], group: str,
                     per_block: Dict[int, float], groups: Dict[str, float]) -> float:
    """Weight for a parameter that lives in block `block_index` (or None) of `group`."""
    if block_index is not None and block_index in per_block:
        return per_block[block_index]
    if group in groups:
        return groups[group]
    return 1.0


def sigma_from_transformer_options(transformer_options: dict) -> float:
    """Current sigma of this model call, as set by comfy.samplers (_calc_cond_batch).

    Raises instead of guessing when it is missing (e.g. a custom caller that
    does not go through ComfyUI's sampling path).
    """
    sig = transformer_options.get("sigmas", None)
    if sig is None:
        raise RuntimeError(
            "ZQX: transformer_options['sigmas'] is missing; this node only works when the model is "
            "called through ComfyUI's sampler (KSampler / SamplerCustom[Advanced])."
        )
    try:
        import torch
        if isinstance(sig, torch.Tensor):
            flat = sig.detach().flatten().float()
            if flat.numel() == 0:
                raise RuntimeError("ZQX: empty transformer_options['sigmas'].")
            if flat.numel() > 1 and not bool(torch.all(flat == flat[0])):
                raise RuntimeError(
                    "ZQX: the batch contains different sigmas in one model call; per-sample sigmas are not supported by this node."
                )
            return float(flat[0].item())
    except ImportError:  # pragma: no cover
        pass
    return float(sig)
