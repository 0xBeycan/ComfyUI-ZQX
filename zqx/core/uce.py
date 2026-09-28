"""Closed-form linear-layer concept editing (UCE, Gandikota et al., arXiv 2308.14761).

Given input vectors c_i to edit with desired outputs v*_i, preservation inputs c_j (desired output W c_j) and a
regulariser lambda:

    W' = argmin  sum_i ||W' c_i - v*_i||^2 + sum_j ||W' c_j - W c_j||^2 + lambda ||W' - W||_F^2
       = (lambda W + sum_i v*_i c_i^T + sum_j W c_j c_j^T) (lambda I + sum_i c_i c_i^T + sum_j c_j c_j^T)^{-1}
"""
from __future__ import annotations

from typing import Optional

import torch


def uce_solve(w: torch.Tensor, c_edit: torch.Tensor, v_edit: torch.Tensor, c_keep: Optional[torch.Tensor],
              lam: float) -> torch.Tensor:
    """w: (out, in); c_edit: (n, in); v_edit: (n, out); c_keep: (m, in) or None.  Returns W' (float64)."""
    if lam <= 0:
        raise ValueError("lambda must be > 0")
    w64 = w.to(torch.float64)
    c = c_edit.to(torch.float64)
    v = v_edit.to(torch.float64)
    if c.shape[0] != v.shape[0] or c.shape[1] != w.shape[1] or v.shape[1] != w.shape[0]:
        raise ValueError("shape mismatch between W, edit inputs and edit targets")
    lhs = lam * w64 + v.T @ c
    gram = lam * torch.eye(w.shape[1], dtype=torch.float64) + c.T @ c
    if c_keep is not None and c_keep.shape[0] > 0:
        k = c_keep.to(torch.float64)
        lhs = lhs + (w64 @ k.T) @ k
        gram = gram + k.T @ k
    # W' gram = lhs  ->  gram^T W'^T = lhs^T  (gram is symmetric)
    return torch.linalg.solve(gram, lhs.T).T


def pair_tokens(src: torch.Tensor, tgt: torch.Tensor, mode: str):
    """Build (c_edit, target inputs) from two token sequences (n_s, d) and (n_t, d).

    mean       : one pair (mean of source tokens -> mean of target tokens)
    positional : token i -> token i (requires equal lengths)
    end        : the last min(n_s, n_t) tokens aligned from the end (concept words at the end of the prompt)
    """
    if mode == "mean":
        return src.mean(0, keepdim=True), tgt.mean(0, keepdim=True)
    if mode == "positional":
        if src.shape[0] != tgt.shape[0]:
            raise ValueError(f"positional pairing needs equal token counts ({src.shape[0]} vs {tgt.shape[0]}); "
                             "use 'mean' or 'end'")
        return src, tgt
    if mode == "end":
        n = min(src.shape[0], tgt.shape[0])
        return src[-n:], tgt[-n:]
    raise ValueError(f"unknown pairing mode {mode!r}")
