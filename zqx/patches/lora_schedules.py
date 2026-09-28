"""Strength schedules for the runtime LoRA patch: sigma-ramped + block-weighted, and K-LoRA selection."""
from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import torch

from ..core.lora_math import klora_gamma, klora_time_scale, klora_use_content
from ..core.schedule import block_weight_for, linear_sigma_ramp, parse_block_weights

KLORA_ATTN_RE = {
    # K-LoRA (2502.18461) patches the attention q/k/v projections.  Qwen: image-stream to_q/k/v (as the
    # official FLUX code, which patches to_q/to_k/to_v of the attention processors).  Z-Image: ComfyUI keeps
    # q/k/v fused in attention.qkv, so the selection is made on the fused delta.
    "qwen_image": re.compile(r"\.attn\.(to_q|to_k|to_v)\.weight$"),
    "z_image": re.compile(r"\.attention\.qkv\.weight$"),
}


def scheduled_entries(adapter, patches_by_key, strength_early: float, strength_late: float,
                      sigma_hi: float, sigma_lo: float, block_weights: str):
    per_block, groups = parse_block_weights(block_weights)
    if sigma_hi < sigma_lo:
        raise ValueError("sigma_hi must be >= sigma_lo")
    out = {}
    report = []
    for mk, lst in patches_by_key.items():
        bi, grp = adapter.block_of_key(mk)
        bw = block_weight_for(bi, grp, per_block, groups)

        def fn(sigma, bw=bw):
            return bw * linear_sigma_ramp(sigma, sigma_hi, sigma_lo, strength_early, strength_late)
        out[mk] = [(fn, v, off, func) for (v, off, func) in lst]
        report.append((mk, bi, grp, bw))
    return out, report


def dense_delta(model_key: str, weight_shape, lst) -> torch.Tensor:
    """Exact dense delta ComfyUI would add at strength 1 (alpha/rank, slices and all)."""
    import comfy.lora
    w = torch.zeros(tuple(weight_shape), dtype=torch.float32)
    patches = [(1.0, v, 1.0, off, func) for (v, off, func) in lst]
    return comfy.lora.calculate_weight(patches, w, model_key, intermediate_dtype=torch.float32)


def _rank_of(lst) -> int:
    r = 0
    for (v, off, func) in lst:
        ws = getattr(v, "weights", None)
        if type(v).__name__ != "LoRAAdapter" or ws is None:
            raise ValueError("ZQX K-LoRA: only plain LoRA adapters are supported (got {})".format(type(v).__name__))
        r += int(ws[1].shape[0])  # down: (rank, in)
    return r


def klora_entries(adapter, weight_shapes: Dict[str, Tuple[int, ...]], patches_c, patches_s,
                  strength_c: float, strength_s: float, alpha: float, beta: float, pattern: str,
                  scope: str, other_layers: str):
    """Per-layer hard selection between the content (character) and style (realism) LoRA.

    Rule (official K-LoRA code, klora.py): with K = r_c * r_s, S_c/S_s = sum of the K largest |dW|,
    gamma = mean over layers of L1(dW_c)/L1(dW_s) (dropping ratios >= 3x the mean), and
    S(t) = alpha * t/T + beta ('s') or (alpha * t/T + beta) mod alpha ('s*'):
        use content  iff  (S_c / gamma) / (S_s * S(t)) > 1.
    Progress t/T is measured in sigma space as 1 - sigma (see DESIGN.md: this keeps the schedule
    meaningful in an img2img pass that starts at sigma < 1; the official code counts steps).
    """
    if scope not in ("attention", "all_shared"):
        raise ValueError("scope must be 'attention' or 'all_shared'")
    if other_layers not in ("both", "character", "realism", "none"):
        raise ValueError("other_layers must be both | character | realism | none")
    rx = KLORA_ATTN_RE[adapter.name]
    shared = [k for k in patches_c if k in patches_s and (scope == "all_shared" or rx.search(k))]
    if not shared:
        raise ValueError("ZQX K-LoRA: the two LoRAs share no layer in the selected scope")
    stats = {}
    ratios = []
    for k in shared:
        dc = dense_delta(k, weight_shapes[k], patches_c[k])
        ds = dense_delta(k, weight_shapes[k], patches_s[k])
        kk = _rank_of(patches_c[k]) * _rank_of(patches_s[k])
        kk = min(kk, dc.numel())
        s_c = float(torch.topk(dc.abs().flatten(), kk, sorted=False).values.sum())
        s_s = float(torch.topk(ds.abs().flatten(), kk, sorted=False).values.sum())
        l1s = float(ds.abs().sum())
        if l1s == 0:
            raise ValueError(f"ZQX K-LoRA: realism LoRA delta is zero at {k}")
        ratios.append(float(dc.abs().sum()) / l1s)
        stats[k] = (s_c, s_s)
    gamma = klora_gamma(ratios)

    out = {}
    report = {"gamma": gamma, "layers": {}}
    for k in shared:
        s_c, s_s = stats[k]

        def use_c(sigma, s_c=s_c, s_s=s_s):
            prog = min(max(1.0 - sigma, 0.0), 1.0)
            return klora_use_content(s_c, s_s, gamma, klora_time_scale(prog, alpha, beta, pattern))

        out[k] = [(lambda sg, u=use_c: strength_c if u(sg) else 0.0, v, off, func) for (v, off, func) in patches_c[k]] + \
                 [(lambda sg, u=use_c: 0.0 if u(sg) else strength_s, v, off, func) for (v, off, func) in patches_s[k]]
        report["layers"][k] = {"S_c": s_c, "S_s": s_s}
    for k, lst in patches_c.items():
        if k in out:
            continue
        s = strength_c if other_layers in ("both", "character") else 0.0
        out.setdefault(k, []).extend([(lambda sg, s=s: s, v, off, func) for (v, off, func) in lst])
    for k, lst in patches_s.items():
        if k in shared:
            continue
        s = strength_s if other_layers in ("both", "realism") else 0.0
        out.setdefault(k, []).extend([(lambda sg, s=s: s, v, off, func) for (v, off, func) in lst])
    return out, report, shared


def klora_switch_sigmas(report, alpha, beta, pattern, gamma=None, n=1001) -> Dict[str, List[float]]:
    """Sigma values (on a fine grid) where each layer uses the character LoRA; for the textual report."""
    gamma = report["gamma"] if gamma is None else gamma
    out = {}
    grid = [1.0 - i / (n - 1) for i in range(n)]
    for k, st in report["layers"].items():
        out[k] = [sg for sg in grid if klora_use_content(st["S_c"], st["S_s"], gamma,
                                                         klora_time_scale(1.0 - sg, alpha, beta, pattern))]
    return out
