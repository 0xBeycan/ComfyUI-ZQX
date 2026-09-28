"""Single-LoRA surgery and multi-LoRA common-subspace estimation (pure torch).

All operations work on LoraFactors (delta_W = scale * up @ down) and return exact low-rank factors.

Singular value decomposition of a LoRA without forming the dense matrix:
    up = Q_u R_u,  down^T = Q_d R_d                     (thin QR)
    delta_W = scale * Q_u (R_u R_d^T) Q_d^T = Q_u (U_c S V_c^T) Q_d^T
so the singular values of delta_W are those of the small r x r core scale * R_u R_d^T.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .lora_io import LoraFactors


def lora_svd(f: LoraFactors) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact thin SVD of delta_W: returns (U (out, r), S (r,), V (in, r)) in float64, S descending."""
    up = f.up.to(torch.float64)
    down = f.down.to(torch.float64)
    qu, ru = torch.linalg.qr(up)
    qd, rd = torch.linalg.qr(down.T)
    core = f.scale * (ru @ rd.T)
    uc, s, vch = torch.linalg.svd(core)
    return qu @ uc, s, qd @ vch.T


def from_svd(u: torch.Tensor, s: torch.Tensor, v: torch.Tensor, src=None) -> LoraFactors:
    keep = s > 0
    u, s, v = u[:, keep], s[keep], v[:, keep]
    return LoraFactors((u * s).to(torch.float32), v.T.contiguous().to(torch.float32), 1.0, list(src or []))


def truncate_rank(f: LoraFactors, rank: int) -> Tuple[LoraFactors, float]:
    """Best rank-`rank` approximation (Eckart-Young); returns (factors, relative Frobenius error)."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    u, s, v = lora_svd(f)
    tot = float((s ** 2).sum())
    err = math.sqrt(float((s[rank:] ** 2).sum()) / tot) if tot > 0 else 0.0
    return from_svd(u[:, :rank], s[:rank], v[:, :rank], f.source_keys), err


def spectrum_power(f: LoraFactors, power: float, preserve: str = "top") -> LoraFactors:
    """sigma_i -> sigma_1 * (sigma_i / sigma_1)^power.

    power < 1 flattens the spectrum (weak directions gain weight), power > 1 concentrates the update in its
    dominant directions.  preserve = 'top' keeps sigma_1; 'frobenius' rescales so ||delta_W||_F is unchanged.
    """
    if power <= 0 or not math.isfinite(power):
        raise ValueError("power must be > 0")
    u, s, v = lora_svd(f)
    if s.numel() == 0 or s[0] <= 0:
        return f.folded(torch.float32)
    s2 = s[0] * (s / s[0]) ** power
    if preserve == "frobenius":
        s2 = s2 * (s.norm() / s2.norm())
    elif preserve != "top":
        raise ValueError("preserve must be 'top' or 'frobenius'")
    return from_svd(u, s2, v, f.source_keys)


def dare_up(f: LoraFactors, drop_rate: float, generator: torch.Generator) -> LoraFactors:
    """DARE (arXiv 2311.03099) applied to the entries of the up factor: up * (1 - m) / (1 - p), m ~ Bernoulli(p).

    Unbiased (E[result] = delta_W because down is kept) and still exactly low rank, unlike DARE on the dense
    delta, which is full rank.
    """
    if not (0.0 <= drop_rate < 1.0):
        raise ValueError("drop_rate must be in [0, 1)")
    ff = f.folded(torch.float32)
    if drop_rate == 0.0:
        return ff
    keep = (torch.rand(ff.up.shape, generator=generator, dtype=torch.float64) >= drop_rate).to(torch.float32)
    return LoraFactors(ff.up * keep / (1.0 - drop_rate), ff.down, 1.0, list(f.source_keys))


def scale_factors(f: LoraFactors, c: float) -> LoraFactors:
    ff = f.folded(torch.float32)
    return LoraFactors(ff.up * c, ff.down, 1.0, list(f.source_keys))


# ---------------------------------------------------------------------------------------------
# Common subspace of several LoRAs (Iso-CTS-inspired, arXiv 2502.04959)
# ---------------------------------------------------------------------------------------------

def sum_svd(fs: Sequence[LoraFactors]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact thin SVD of sum_t delta_W_t (low rank <= sum of ranks)."""
    ups = torch.cat([f.up.to(torch.float64) * f.scale for f in fs], dim=1)
    downs = torch.cat([f.down.to(torch.float64) for f in fs], dim=0)
    return lora_svd(LoraFactors(ups, downs, 1.0))


def common_subspace(fs: Sequence[LoraFactors], k: int, rtol: float = 1e-9) -> torch.Tensor:
    """Orthonormal basis U_c (out, k') of the top-k left singular vectors of sum_t delta_W_t."""
    if len(fs) < 2:
        raise ValueError("need at least two LoRAs")
    shape = fs[0].shape
    for f in fs:
        if f.shape != shape:
            raise ValueError("all LoRAs must target the same weight shape")
    u, s, _ = sum_svd(fs)
    if s.numel() == 0 or s[0] <= 0:
        return u[:, :0]
    keep = s > rtol * s[0]
    u = u[:, keep]
    return u[:, :k]


def project_left(f: LoraFactors, basis: torch.Tensor, lam: float = 1.0, keep_inside: bool = False) -> LoraFactors:
    """keep_inside=False: (I - lam U U^T) delta_W;  keep_inside=True: U U^T delta_W."""
    u = f.up.to(torch.float64) * f.scale
    b = basis.to(torch.float64)
    inside = b @ (b.T @ u)
    up = inside if keep_inside else u - lam * inside
    return LoraFactors(up.to(torch.float32), f.down.to(torch.float32), 1.0, list(f.source_keys))


def common_component(fs: Sequence[LoraFactors], basis: torch.Tensor) -> LoraFactors:
    """U U^T mean_t(delta_W_t): the shared part of the LoRAs, as an exact low-rank LoRA."""
    n = len(fs)
    ups = torch.cat([f.up.to(torch.float64) * (f.scale / n) for f in fs], dim=1)
    downs = torch.cat([f.down.to(torch.float64) for f in fs], dim=0)
    mean = LoraFactors(ups, downs, 1.0)
    return project_left(mean, basis, keep_inside=True)


def energy_fraction_in(f: LoraFactors, basis: torch.Tensor) -> float:
    u = f.up.to(torch.float64) * f.scale
    d = f.down.to(torch.float64)
    tot = float(torch.trace((u.T @ u) @ (d @ d.T)))
    if tot <= 0:
        return 0.0
    x = basis.to(torch.float64).T @ u
    return float(torch.trace((x.T @ x) @ (d @ d.T))) / tot
