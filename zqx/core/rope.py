"""RoPE helpers operating on ComfyUI's rotation-matrix representation.

ComfyUI (comfy/ldm/flux/math.py::rope + EmbedND) represents RoPE as explicit
2x2 rotation matrices: freqs[..., n, i, :, :] = [[cos a, -sin a], [sin a, cos a]]
with a = pos_n * omega_i, and applies them with

    out[..., i, r] = F[..., i, r, 0] * x[..., i, 0] + F[..., i, r, 1] * x[..., i, 1]

(see comfy/ldm/flux/math.py::_apply_rope1).  Because 2-D rotations compose
additively in the angle, R(delta) R(p) = R(p + delta).  We use this to move
already-rotated reference keys to a new position *without* re-running the
model:  k_at(p + delta) = R(delta) . k_at(p).
"""
from __future__ import annotations

import torch


def apply_rotation(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply ComfyUI-format rotation matrices to x (last dim = head_dim).

    Mirrors comfy.ldm.flux.math._apply_rope1 exactly (math done in freqs.dtype,
    result cast back to x.dtype).  `freqs` must broadcast against
    x.reshape(*x.shape[:-1], D//2, 1, 2)[..., 0] i.e. have shape
    (..., N or 1, D//2, 2, 2).
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"head dim must be even, got {x.shape[-1]}")
    if freqs.shape[-3:] != (x.shape[-1] // 2, 2, 2):
        raise ValueError(f"freqs trailing shape {tuple(freqs.shape[-3:])} does not match head dim {x.shape[-1]}")
    x_ = x.to(dtype=freqs.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)


def invert_rotation(freqs: torch.Tensor) -> torch.Tensor:
    """R^{-1} = R^T for rotation matrices."""
    return freqs.transpose(-1, -2)


def compose_rotation(freqs_a: torch.Tensor, freqs_b: torch.Tensor) -> torch.Tensor:
    """Matrix product R_a @ R_b (apply R_b first, then R_a)."""
    return freqs_a @ freqs_b
