"""CADS: Condition-Annealed Diffusion Sampler (Sadat et al., arXiv:2310.17347, ICLR 2024).

    gamma(t) = 1                         t <= tau1
             = (tau2 - t) / (tau2 - tau1)  tau1 < t < tau2
             = 0                         t >= tau2
    y_hat    = sqrt(gamma) * y + s * sqrt(1 - gamma) * n,     n ~ N(0, I)  (fresh each step)
    y_resc   = (y_hat - mean(y_hat)) / std(y_hat) * std(y) + mean(y)
    y_final  = psi * y_resc + (1 - psi) * y_hat

t runs from 1 (pure noise) to 0 (clean), i.e. t == sigma for rectified-flow models.
Statistics are scalars over the whole embedding of one sample (the reference
implementations compute them over the whole conditioning tensor; with batch
size 1 per conditioning this is identical; we compute them per sample so a
batched cond/uncond call cannot mix statistics across rows).
"""
from __future__ import annotations

import math

import torch


def cads_gamma(t: float, tau1: float, tau2: float) -> float:
    if not (0.0 <= tau1 < tau2):
        raise ValueError(f"need 0 <= tau1 < tau2, got tau1={tau1}, tau2={tau2}")
    if t <= tau1:
        return 1.0
    if t >= tau2:
        return 0.0
    return (tau2 - t) / (tau2 - tau1)


def cads_apply(y: torch.Tensor, gamma: float, noise_scale: float, psi: float, noise: torch.Tensor,
               relative_noise: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """Apply the CADS corruption to y (B, ...) with per-sample statistics.

    relative_noise=True multiplies s by std(y) of each sample (heuristic for LLM text encoders whose hidden
    states are far from unit variance; with the psi rescale the final statistics are restored anyway).
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if not (0.0 <= psi <= 1.0):
        raise ValueError("psi must be in [0, 1]")
    if noise.shape != y.shape:
        raise ValueError("noise shape must equal y shape")
    y32 = y.to(torch.float32)
    dims = tuple(range(1, y.ndim))
    std_y = y32.std(dim=dims, keepdim=True, unbiased=True)
    mean_y = y32.mean(dim=dims, keepdim=True)
    s = noise_scale * (std_y if relative_noise else 1.0)
    y_hat = math.sqrt(gamma) * y32 + s * math.sqrt(1.0 - gamma) * noise.to(torch.float32)
    if psi > 0:
        mean_h = y_hat.mean(dim=dims, keepdim=True)
        std_h = y_hat.std(dim=dims, keepdim=True, unbiased=True)
        y_resc = (y_hat - mean_h) / (std_h + eps) * std_y + mean_y
        y_hat = psi * y_resc + (1.0 - psi) * y_hat
    return y_hat.to(y.dtype)
