"""Image metrics and score combination for candidate selection (pure torch, no model downloads).

Images use ComfyUI's IMAGE layout: (B, H, W, 3) float in [0, 1].
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def to_gray(images: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) -> (B, 1, H, W), ITU-R BT.601 luma."""
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("images must be (B, H, W, 3)")
    w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=images.device)
    return (images[..., :3].to(torch.float32) @ w)[:, None]


def laplacian(gray: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=gray.device).view(1, 1, 3, 3)
    return F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), k)


def region_masks(h: int, w: int, border: float, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """(border band mask, centre mask); border = fraction of each side counted as border."""
    if not (0.0 < border < 0.5):
        raise ValueError("border must be in (0, 0.5)")
    bh, bw = max(1, int(round(h * border))), max(1, int(round(w * border)))
    centre = torch.zeros(h, w, dtype=torch.bool, device=device)
    centre[bh:h - bh, bw:w - bw] = True
    return ~centre, centre


def background_sharpness(images: torch.Tensor, border: float = 0.2, eps: float = 1e-8) -> torch.Tensor:
    """log( var(Laplacian) in the border band / var(Laplacian) in the centre ).

    Shallow depth of field (studio / bokeh portraits) gives a blurred border and a sharp centre -> strongly
    negative; an evenly sharp, natural scene -> around 0.  Heuristic: it assumes the subject is not in the
    border band.  Model-free and scale-invariant.
    """
    lap = laplacian(to_gray(images))[:, 0]
    bmask, cmask = region_masks(lap.shape[-2], lap.shape[-1], border, lap.device)
    vb = lap[:, bmask].var(dim=1, unbiased=False)
    vc = lap[:, cmask].var(dim=1, unbiased=False)
    return torch.log((vb + eps) / (vc + eps))


def off_center(box_xyxy: Sequence[float], w: int, h: int) -> float:
    """Distance of a box centre from the image centre, normalised by the half diagonal (0 = centred, 1 = corner)."""
    x0, y0, x1, y1 = box_xyxy
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return math.hypot(cx - w / 2.0, cy - h / 2.0) / math.hypot(w / 2.0, h / 2.0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.to(torch.float32), b.to(torch.float32), dim=-1)


def zscore(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Standardise across candidates; a constant metric contributes 0."""
    x = x.to(torch.float64)
    sd = x.std(unbiased=False)
    if sd < eps:
        return torch.zeros_like(x)
    return (x - x.mean()) / sd


def combine(metrics: Dict[str, torch.Tensor], weights: Dict[str, float], normalize: bool = True) -> torch.Tensor:
    """score = sum_m w_m * z(metric_m) (normalize) or sum_m w_m * metric_m (raw).  Missing weights -> 0."""
    if not metrics:
        raise ValueError("no metrics")
    n = next(iter(metrics.values())).shape[0]
    total = torch.zeros(n, dtype=torch.float64)
    for name, vals in metrics.items():
        w = float(weights.get(name, 0.0))
        if w == 0.0:
            continue
        if vals.shape[0] != n:
            raise ValueError("metric length mismatch")
        v = vals.to(torch.float64).cpu()
        if torch.any(~torch.isfinite(v)):
            raise ValueError(f"metric {name} has non-finite values")
        total += w * (zscore(v) if normalize else v)
    return total


def top_k(scores: torch.Tensor, k: int) -> List[int]:
    """Indices of the k best scores (stable: ties keep candidate order)."""
    order = sorted(range(scores.shape[0]), key=lambda i: (-float(scores[i]), i))
    return order[:k]
