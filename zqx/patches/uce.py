"""UCE closed-form edit of the text input projection (Qwen-Image txt_in, Z-Image cap_embedder.1).

Why this layer: in the MMDiT / single-stream DiT the text tokens are re-computed by every block (and modulated
by the timestep in Qwen-Image), so the per-block K/V projections do not see a fixed text embedding the way UNet
cross-attention does (where UCE was proposed).  The input projection is the last linear map that sees the
(RMS-normalised) text-encoder output directly, so the closed-form edit is exact there.
"""
from __future__ import annotations

import torch

from ..core.uce import pair_tokens, uce_solve


def normed_tokens(adapter, conditioning) -> list:
    norm_path, _ = adapter.text_input_projection
    norm = adapter.model_patcher.get_model_object(norm_path)
    out = []
    with torch.no_grad():
        for t, _meta in conditioning:
            if t.ndim != 3:
                raise ValueError("conditioning tensor must be (B, tokens, dim)")
            p = next(norm.parameters(), None)
            dev = p.device if p is not None else "cpu"
            dt = p.dtype if p is not None else torch.float32
            for b in range(t.shape[0]):
                y = norm(t[b:b + 1].to(dev, dt))
                out.append(y[0].to("cpu", torch.float32))
    return out


def build_edit(adapter, src_cond, tgt_cond, keep_cond, pairing: str, lam: float, strength: float):
    """Returns (weight_key, diff, report)."""
    _, wkey = adapter.text_input_projection
    w = adapter.model_patcher.model.state_dict()[wkey].to("cpu", torch.float32)
    src = normed_tokens(adapter, src_cond)
    tgt = normed_tokens(adapter, tgt_cond)
    if len(src) != len(tgt):
        raise ValueError(f"source has {len(src)} conditionings, target {len(tgt)}; they are paired one to one")
    cs, ts = [], []
    for s, t in zip(src, tgt):
        c, tt = pair_tokens(s, t, pairing)
        cs.append(c)
        ts.append(tt)
    c_edit = torch.cat(cs)
    v_edit = torch.cat(ts) @ w.T
    c_keep = torch.cat(normed_tokens(adapter, keep_cond)) if keep_cond else None
    # lambda is given relative to the mean squared norm of the input vectors, so it is scale-free
    allc = c_edit if c_keep is None else torch.cat([c_edit, c_keep])
    lam_eff = lam * float(allc.double().pow(2).sum(1).mean())
    if lam_eff <= 0:
        raise ValueError("ZQX UCE: the conditioning vectors are all zero")
    w_new = uce_solve(w, c_edit, v_edit, c_keep, lam_eff)
    diff = (strength * (w_new - w.double())).to(torch.float32)
    before = float(((c_edit.double() @ w.double().T) - v_edit.double()).norm())
    after = float(((c_edit.double() @ w_new.T) - v_edit.double()).norm())
    rep = {"lambda_effective": lam_eff, "edit_pairs": int(c_edit.shape[0]), "keep_tokens": 0 if c_keep is None else int(c_keep.shape[0]),
           "edit_residual_before": before, "edit_residual_after": after,
           "rel_weight_change": float(diff.norm() / w.norm())}
    if c_keep is not None:
        rep["keep_drift"] = float((c_keep.double() @ (w_new - w.double()).T).norm() / (c_keep.double() @ w.double().T).norm())
    return wkey, diff, rep


def install(model_patcher, src_cond, tgt_cond, keep_cond, pairing="mean", lam=0.1, strength=1.0):
    from ..adapters import get_adapter
    m = model_patcher.clone()
    ad = get_adapter(m)
    if strength == 0.0:
        return m, {"skipped": "strength 0"}
    wkey, diff, rep = build_edit(ad, src_cond, tgt_cond, keep_cond, pairing, lam, strength)
    added = m.add_patches({wkey: ("diff", (diff,))}, 1.0, 1.0)
    if wkey not in added:
        raise RuntimeError(f"ZQX UCE: could not patch {wkey}")
    return m, rep
