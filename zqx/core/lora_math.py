"""Weight-space LoRA arithmetic (pure torch).

All functions operate on LoraFactors (delta_W = scale * up @ down) of the same
target weight and return LoraFactors whose dense product equals the stated
formula (exactly for the low-rank modes, up to the reported truncation error
for the dense-TIES mode).

Notation: dW1 = character LoRA update, dW2 = second LoRA (realism or, in the
future, an "AI-look" LoRA), lam = strength of the operation.

Modes
-----
add        : dW1 + lam dW2                       exact, rank r1 + r2   (task arithmetic, 2212.04089)
negate     : dW1 - lam dW2                       exact, rank r1 + r2   (task negation)
clean_col  : (I - lam Q2 Q2^T) dW1               exact, rank r1        Q2 = orthonormal basis of col(dW2)
clean_row  : dW1 (I - lam P2 P2^T)               exact, rank r1        P2 = orthonormal basis of row(dW2)
target_sub : dW1 - lam Q1 Q1^T dW2               exact, rank r1 + r2   Q1 = orthonormal basis of col(dW1)
knots_ties : KnOTS (2410.19735) shared-basis TIES (2306.01708), optional DARE (2311.03099); exact, rank <= r1 + r2
ties_dense : TIES on dense dW then truncated SVD to `rank`; reports the relative Frobenius error
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .lora_io import LoraFactors, concat_factors


def orth_basis(m: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis of the column space of m (float64), via SVD with a relative rank tolerance.

    Equivalent to QR for full-column-rank m, but drops numerically-null
    directions of rank-deficient factors so that the projector Q Q^T is exactly
    the projector onto col(m) (a QR basis of a rank-deficient B would also
    project out arbitrary extra directions).
    """
    m64 = m.to(torch.float64)
    if m64.numel() == 0:
        return m64.new_zeros((m64.shape[0], 0))
    u, s, _ = torch.linalg.svd(m64, full_matrices=False)
    if s.numel() == 0 or s[0] <= 0:
        return m64.new_zeros((m64.shape[0], 0))
    keep = s > rtol * s[0]
    return u[:, keep]


def _col_basis(f: LoraFactors) -> torch.Tensor:
    # col(scale * up @ down) == col(up @ down); use the product's column space via up and down
    # (col(up) can be larger than col(up @ down) if down is rank deficient).
    u = f.up.to(torch.float64)
    d = f.down.to(torch.float64)
    # col(U D) = U col(D D^T ...) -> orth basis of U @ (orth basis of row-reduced D)
    qd = orth_basis(d)  # (r, k) basis of col(D) in R^r
    return orth_basis(u @ qd)


def _row_basis(f: LoraFactors) -> torch.Tensor:
    u = f.up.to(torch.float64)
    d = f.down.to(torch.float64)
    qu = orth_basis(u.T)  # basis of col(U^T) = row(U)
    return orth_basis(d.T @ qu)


def op_add(f1: LoraFactors, f2: LoraFactors, lam: float) -> LoraFactors:
    return concat_factors([(1.0, f1), (lam, f2)])


def op_negate(f1: LoraFactors, f2: LoraFactors, lam: float) -> LoraFactors:
    return concat_factors([(1.0, f1), (-lam, f2)])


def op_clean_col(f1: LoraFactors, f2: LoraFactors, lam: float) -> LoraFactors:
    """(I - lam Q2 Q2^T) dW1  ->  up' = s1 (U1 - lam Q2 Q2^T U1), down' = D1."""
    q2 = _col_basis(f2)
    u1 = f1.up.to(torch.float64) * f1.scale
    up = u1 - lam * (q2 @ (q2.T @ u1))
    return LoraFactors(up.to(torch.float32), f1.down.to(torch.float32), 1.0, list(f1.source_keys))


def op_clean_row(f1: LoraFactors, f2: LoraFactors, lam: float) -> LoraFactors:
    """dW1 (I - lam P2 P2^T)  ->  up' = s1 U1, down' = D1 - lam (D1 P2) P2^T."""
    p2 = _row_basis(f2)
    d1 = f1.down.to(torch.float64)
    down = d1 - lam * ((d1 @ p2) @ p2.T)
    return LoraFactors((f1.up.to(torch.float64) * f1.scale).to(torch.float32), down.to(torch.float32), 1.0,
                       list(f1.source_keys))


def op_target_sub(f1: LoraFactors, f2: LoraFactors, lam: float) -> LoraFactors:
    """dW1 - lam Q1 Q1^T dW2  ->  up = [s1 U1, -lam Q1 Q1^T s2 U2], down = [D1; D2]."""
    q1 = _col_basis(f1)
    u2 = f2.up.to(torch.float64) * f2.scale
    proj_u2 = q1 @ (q1.T @ u2)
    up = torch.cat([f1.up.to(torch.float64) * f1.scale, -lam * proj_u2], dim=1)
    down = torch.cat([f1.down.to(torch.float64), f2.down.to(torch.float64)], dim=0)
    return LoraFactors(up.to(torch.float32), down.to(torch.float32), 1.0, list(f1.source_keys) + list(f2.source_keys))


# ---------------------------------------------------------------------------------------------
# TIES / DARE primitives (operate on arbitrary same-shape tensors)
# ---------------------------------------------------------------------------------------------

def trim_topk(t: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the `density` fraction of entries with the largest |t|, zero the rest (TIES step 1)."""
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1], got {density}")
    if density == 1.0:
        return t.clone()
    n = t.numel()
    k = max(1, int(math.ceil(density * n)))
    flat = t.flatten()
    idx = torch.topk(flat.abs(), k, largest=True, sorted=False).indices
    out = torch.zeros_like(flat)
    out[idx] = flat[idx]
    return out.view_as(t)


def dare(t: torch.Tensor, drop_rate: float, generator: torch.Generator) -> torch.Tensor:
    """DARE: delta * (1 - m) / (1 - p), m ~ Bernoulli(p)."""
    if not (0.0 <= drop_rate < 1.0):
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
    if drop_rate == 0.0:
        return t.clone()
    keep = (torch.rand(t.shape, generator=generator, dtype=torch.float64, device="cpu") >= drop_rate)
    return t * keep.to(t.device, t.dtype) / (1.0 - drop_rate)


def ties_merge(tensors: List[torch.Tensor], weights: List[float], density: float,
               sign_method: str = "total") -> torch.Tensor:
    """TIES merge of task tensors (trim -> elect sign -> disjoint mean), peft ordering:
    trim each unweighted tensor, elect the sign on the *weighted* trimmed tensors
    ("total": sgn(sum w_t tau_t);  "frequency": sgn(sum sgn(w_t tau_t))), then
    average the weighted entries that agree with the elected sign.
    """
    if len(tensors) != len(weights) or len(tensors) == 0:
        raise ValueError("tensors/weights length mismatch")
    trimmed = torch.stack([w * trim_topk(t, density) for t, w in zip(tensors, weights)], dim=0)
    if sign_method == "total":
        elect = torch.sign(trimmed.sum(dim=0))
    elif sign_method == "frequency":
        elect = torch.sign(torch.sign(trimmed).sum(dim=0))
    else:
        raise ValueError(f"unknown sign method {sign_method!r}")
    elect = torch.where(elect == 0, torch.ones_like(elect), elect)  # ties -> +1 (peft convention)
    agree = (torch.sign(trimmed) == elect.unsqueeze(0))
    num = (trimmed * agree).sum(dim=0)
    cnt = agree.sum(dim=0).clamp(min=1)
    return num / cnt


def op_knots_ties(f1: LoraFactors, f2: LoraFactors, w1: float, w2: float, density: float,
                  dare_drop: float = 0.0, seed: int = 0, sign_method: str = "total",
                  rtol: float = 1e-6) -> Tuple[LoraFactors, Dict[str, float]]:
    """KnOTS-aligned TIES: U = left singular vectors of [dW1 | dW2]; represent each task in
    that shared basis, C_i = U^T dW_i  (exact since col(dW_i) is inside col(U));
    TIES(-DARE) on the C_i; result dW = U C_merged (exact low-rank, rank <= r1 + r2)."""
    u1 = f1.up.to(torch.float64) * f1.scale
    u2 = f2.up.to(torch.float64) * f2.scale
    d1 = f1.down.to(torch.float64)
    d2 = f2.down.to(torch.float64)
    # left singular vectors of [U1 D1 | U2 D2] = [U1 | U2] blockdiag(D1, D2)
    m = torch.cat([u1, u2], dim=1)                   # (out, r1 + r2)
    n_t = torch.block_diag(d1, d2).T                 # (in1 + in2, r1 + r2)
    q_n, r_n = torch.linalg.qr(n_t)                  # n_t = q_n r_n
    small = m @ r_n.T                                # (out, r1 + r2);  [dW1|dW2] = small @ q_n^T
    uu, ss, _ = torch.linalg.svd(small, full_matrices=False)
    keep = ss > rtol * ss[0] if ss.numel() and ss[0] > 0 else torch.zeros_like(ss, dtype=torch.bool)
    basis = uu[:, keep]                              # (out, k)
    c1 = (basis.T @ u1) @ d1                         # (k, in)
    c2 = (basis.T @ u2) @ d2
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if dare_drop > 0:
        c1 = dare(c1, dare_drop, g)
        c2 = dare(c2, dare_drop, g)
    cm = ties_merge([c1, c2], [w1, w2], density, sign_method)
    info = {"shared_rank": int(basis.shape[1])}
    return LoraFactors(basis.to(torch.float32), cm.to(torch.float32), 1.0,
                       list(f1.source_keys) + list(f2.source_keys)), info


def truncated_svd_factors(dw: torch.Tensor, rank: int) -> Tuple[LoraFactors, float]:
    """Best rank-`rank` approximation (Eckart-Young) and its relative Frobenius error."""
    dw64 = dw.to(torch.float64)
    u, s, vh = torch.linalg.svd(dw64, full_matrices=False)
    r = min(rank, s.numel())
    up = u[:, :r] * s[:r]
    down = vh[:r]
    tot = float(torch.sum(s ** 2))
    err = math.sqrt(float(torch.sum(s[r:] ** 2)) / tot) if tot > 0 else 0.0
    return LoraFactors(up.to(torch.float32), down.to(torch.float32), 1.0), err


def op_ties_dense(f1: LoraFactors, f2: LoraFactors, w1: float, w2: float, density: float, rank: int,
                  dare_drop: float = 0.0, seed: int = 0, sign_method: str = "total") -> Tuple[LoraFactors, Dict[str, float]]:
    t1 = f1.dense()
    t2 = f2.dense()
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if dare_drop > 0:
        t1 = dare(t1, dare_drop, g)
        t2 = dare(t2, dare_drop, g)
    merged = ties_merge([t1, t2], [w1, w2], density, sign_method)
    f, err = truncated_svd_factors(merged, rank)
    return f, {"svd_rel_error": err}


# ---------------------------------------------------------------------------------------------
# Conflict statistics
# ---------------------------------------------------------------------------------------------

@dataclass
class PairStats:
    rank1: int
    rank2: int
    norm1: float
    norm2: float
    cosine: float                 # <dW1, dW2>_F / (|dW1| |dW2|)
    col_overlap: float            # ||Q1^T Q2||_F^2 / min(k1, k2)   (LoRA paper Sec. 7 subspace similarity)
    row_overlap: float
    energy1_in_col2: float        # ||Q2 Q2^T dW1||^2 / ||dW1||^2
    energy2_in_col1: float
    sign_conflict_all: float      # fraction of entries with opposite signs (both non-zero)
    sign_conflict_top: float      # same, restricted to entries in the top-`density` magnitude of both


def pair_stats(f1: LoraFactors, f2: LoraFactors, top_density: float = 0.2, dense: bool = True) -> PairStats:
    u1 = f1.up.to(torch.float64) * f1.scale
    u2 = f2.up.to(torch.float64) * f2.scale
    d1 = f1.down.to(torch.float64)
    d2 = f2.down.to(torch.float64)
    # Frobenius products without forming dense matrices: <U1 D1, U2 D2> = tr((U1^T U2)(D2 D1^T))
    g11 = float(torch.trace((u1.T @ u1) @ (d1 @ d1.T)))
    g22 = float(torch.trace((u2.T @ u2) @ (d2 @ d2.T)))
    g12 = float(torch.trace((u1.T @ u2) @ (d2 @ d1.T)))
    n1, n2 = math.sqrt(max(g11, 0.0)), math.sqrt(max(g22, 0.0))
    cos = g12 / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
    q1, q2 = _col_basis(f1), _col_basis(f2)
    p1, p2 = _row_basis(f1), _row_basis(f2)
    k_c = min(q1.shape[1], q2.shape[1])
    k_r = min(p1.shape[1], p2.shape[1])
    col_ov = float((q1.T @ q2).pow(2).sum()) / k_c if k_c > 0 else 0.0
    row_ov = float((p1.T @ p2).pow(2).sum()) / k_r if k_r > 0 else 0.0

    def energy_in(q, u, d, tot):
        if tot <= 0:
            return 0.0
        x = q.T @ u  # (k, r)
        return float(torch.trace((x.T @ x) @ (d @ d.T))) / tot

    e1 = energy_in(q2, u1, d1, g11)
    e2 = energy_in(q1, u2, d2, g22)
    sc_all = float("nan")
    sc_top = float("nan")
    if dense:
        t1 = u1 @ d1
        t2 = u2 @ d2
        nz = (t1 != 0) & (t2 != 0)
        opp = (torch.sign(t1) != torch.sign(t2)) & nz
        sc_all = float(opp.sum()) / max(int(nz.sum()), 1)
        m1 = trim_topk(t1, top_density) != 0
        m2 = trim_topk(t2, top_density) != 0
        both = m1 & m2 & nz
        sc_top = float((opp & both).sum()) / max(int(both.sum()), 1)
    return PairStats(f1.rank, f2.rank, n1, n2, cos, col_ov, row_ov, e1, e2, sc_all, sc_top)


# ---------------------------------------------------------------------------------------------
# K-LoRA (2502.18461) selection statistics
# ---------------------------------------------------------------------------------------------

def klora_topk_sum(f: LoraFactors, k: int) -> float:
    """Sum of the k largest |dW| entries (K-LoRA S_c / S_s)."""
    dw = f.dense(torch.float32).abs().flatten()
    k = min(k, dw.numel())
    return float(torch.topk(dw, k, sorted=False).values.sum())


def klora_gamma(ratios: List[float]) -> float:
    """K-LoRA average ratio: mean of per-layer L1 ratios after dropping ratios >= 3 * mean (official utils.py)."""
    if not ratios:
        raise ValueError("no layers")
    mean = sum(ratios) / len(ratios)
    kept = [r for r in ratios if r < 3 * mean]
    return sum(kept) / len(kept) if kept else float("inf")


def klora_time_scale(progress: float, alpha: float, beta: float, pattern: str) -> float:
    """S(t) = alpha * t/T + beta   ('s');   (alpha * t/T + beta) mod alpha   ('s*').  progress = t/T in [0, 1]."""
    s = alpha * progress + beta
    if pattern == "s*":
        s = math.fmod(s, alpha)
    elif pattern != "s":
        raise ValueError(f"unknown K-LoRA pattern {pattern!r}")
    return s


def klora_use_content(s_c: float, s_s: float, gamma: float, time_scale: float) -> bool:
    """Official rule: (S_c / gamma) / (S_s * S(t)) > 1  -> content (character) LoRA, else style (realism)."""
    denom = s_s * time_scale
    if denom <= 0:
        return True
    return (s_c / gamma) / denom > 1.0
