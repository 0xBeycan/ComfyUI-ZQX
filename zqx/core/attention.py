"""Pure attention math for reference-attention injection.

Shared/extended self-attention with a log-bias on the extra (reference) keys:

    logits = [ Q K^T / sqrt(d) + M ,  Q K_r^T / sqrt(d) + log(w * mq_i * mk_j) ]
    out    = softmax(logits) [V ; V_r]

* M is the model's own additive mask for its own keys (0 when there is none).
* w >= 0 is the global reference weight.  w = 1 is plain concatenation
  (reference-only / ConsiStory / StoryDiffusion style extended attention);
  log(w) is the "attention temperature on the reference keys" used e.g. by
  RefDrop-style and "reference strength" implementations.  w -> 0 removes the
  reference exactly (-inf bias -> zero attention weight).
* mq_i in [0,1] is a per-query mask (e.g. face region of the generated image,
  and 0 for text / padding / QIE-reference queries), mk_j in [0,1] a per-key
  mask over reference tokens (e.g. only the reference face).  Soft masks act
  multiplicatively on the unnormalised attention weight: exp(s + log m) = m e^s.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


def reference_log_bias(query_mask: torch.Tensor, key_mask: torch.Tensor, weight: float,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """log(weight * mq_i * mk_j) as a (B, 1, Nq, Nr) tensor, -inf where the product is 0.

    query_mask: (B, Nq) in [0, 1];  key_mask: (B, Nr) or (1, Nr) in [0, 1].
    """
    if weight < 0 or not math.isfinite(weight):
        raise ValueError(f"reference weight must be finite and >= 0, got {weight}")
    if query_mask.ndim != 2 or key_mask.ndim != 2:
        raise ValueError("query_mask and key_mask must be 2-D (batch, tokens)")
    qm = query_mask.to(torch.float32)
    km = key_mask.to(torch.float32)
    if torch.any(qm < 0) or torch.any(qm > 1) or torch.any(km < 0) or torch.any(km > 1):
        raise ValueError("masks must lie in [0, 1]")
    prod = weight * qm[:, :, None] * km[:, None, :]  # (B, Nq, Nr)
    with torch.no_grad():
        bias = torch.log(prod)  # log(0) = -inf, exactly what we want
    return bias[:, None].to(dtype)


def normalize_existing_mask(mask: Optional[torch.Tensor], batch: int, nq: int, nk: int,
                            dtype: torch.dtype, device) -> torch.Tensor:
    """Turn whatever additive/bool mask the model passed into a (B, 1|H, Nq, Nk) additive float mask.

    Follows comfy.ldm.modules.attention.attention_pytorch broadcasting rules:
    2-D -> (1, 1, Nq?, Nk), 3-D -> add a heads dim at position 1.
    """
    if mask is None:
        return torch.zeros((batch, 1, nq, nk), dtype=dtype, device=device)
    m = mask
    if m.dtype == torch.bool:
        m = torch.zeros(m.shape, dtype=dtype, device=device).masked_fill(~m.to(device), float("-inf"))
    else:
        m = m.to(dtype=dtype, device=device)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim == 3:
        m = m.unsqueeze(1)
    if m.ndim != 4:
        raise ValueError(f"unsupported attention mask rank {mask.ndim}")
    if m.shape[-1] != nk:
        raise ValueError(f"existing attention mask has {m.shape[-1]} keys but attention has {nk}")
    return m.expand(batch, m.shape[1], nq, nk)


def build_extended_mask(existing: Optional[torch.Tensor], ref_bias: torch.Tensor,
                        batch: int, nq: int, nk: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Concatenate the model's own mask (for its Nk keys) with the reference bias (Nr keys)."""
    base = normalize_existing_mask(existing, batch, nq, nk, dtype, device)
    rb = ref_bias.to(dtype=dtype, device=device)
    heads = max(base.shape[1], rb.shape[1])
    base = base.expand(batch, heads, nq, nk)
    rb = rb.expand(batch, heads, nq, rb.shape[-1])
    return torch.cat([base, rb], dim=-1)


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    bias: Optional[torch.Tensor] = None, scale: Optional[float] = None) -> torch.Tensor:
    """softmax(Q K^T * scale + bias) V in float64, (B, H, N, D) layout. Test oracle."""
    q64, k64, v64 = q.double(), k.double(), v.double()
    if scale is None:
        scale = q.shape[-1] ** -0.5
    s = torch.einsum("bhid,bhjd->bhij", q64, k64) * scale
    if bias is not None:
        s = s + bias.double()
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", p, v64)


def adain_tokens(x: torch.Tensor, ref: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """StyleAligned AdaIN over the token axis: (x - mu_x)/sigma_x * sigma_r + mu_r.

    x: (B, H, N, D), ref: (B, H, Nr, D); statistics per (batch, head, channel).
    """
    mu_x = x.mean(dim=-2, keepdim=True)
    sd_x = x.std(dim=-2, keepdim=True, unbiased=False)
    mu_r = ref.mean(dim=-2, keepdim=True)
    sd_r = ref.std(dim=-2, keepdim=True, unbiased=False)
    return (x - mu_x) / (sd_x + eps) * sd_r + mu_r
