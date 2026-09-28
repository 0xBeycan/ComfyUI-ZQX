"""LoRA arithmetic in *model-weight-key space* (ComfyUI glue around zqx.core.lora_math).

Both LoRAs are parsed (kohya / peft / ai-toolkit / musubi key styles), mapped
onto the model's weight keys with ComfyUI's own key map
(comfy.lora.model_lora_keys_unet), slices of fused weights (Z-Image qkv) are
embedded exactly, and the arithmetic is done per model weight.  The result is
written with ComfyUI's generic base keys ("diffusion_model.<weight key>"),
kohya suffixes and alpha = rank, so ComfyUI's LoRA loader reproduces the
computed delta exactly.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch

from ..core import lora_math as LM
from ..core.lora_io import (LoraFactors, LoraFormatError, factors_to_state_dict, parse_lora_state_dict,
                            to_model_key_space)

MODES = ["add", "negate", "clean_col", "clean_row", "target_sub", "knots_ties", "ties_dense"]
_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet_", "lycoris_", "base_model.model.", "unet.")


def _normalize(k: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
                changed = True
    return k.replace(".", "_")


def model_key_map(model_patcher):
    import comfy.lora
    km = comfy.lora.model_lora_keys_unet(model_patcher.model, {})
    norm: Dict[str, object] = {}
    ambiguous = set()
    for k, v in km.items():
        n = _normalize(k)
        if n in norm and norm[n] != v:
            ambiguous.add(n)
        norm[n] = v
    for n in ambiguous:
        norm.pop(n, None)
    return km, norm


def lora_to_model_space(model_patcher, lora_sd) -> Tuple[Dict[str, LoraFactors], List[str], List[str]]:
    """Returns (factors by model weight key, lora keys mapped only via normalised names, unmapped lora keys)."""
    modules = parse_lora_state_dict(lora_sd)
    km, norm = model_key_map(model_patcher)
    full = dict(km)
    via_norm = []
    for base in modules:
        if base not in full:
            n = _normalize(base)
            if n in norm:
                full[base] = norm[n]
                via_norm.append(base)
    shapes = {k: tuple(v.shape) for k, v in model_patcher.model.state_dict().items()}
    mapped, unmapped = to_model_key_space(modules, full, shapes)
    return mapped, via_norm, unmapped


def combine(f1: Dict[str, LoraFactors], f2: Dict[str, LoraFactors], mode: str, lam: float,
            w1: float = 1.0, w2: float = 1.0, density: float = 0.2, rank: int = 64,
            dare_drop: float = 0.0, seed: int = 0, sign_method: str = "total") -> Tuple[Dict[str, LoraFactors], Dict[str, dict]]:
    """Apply `mode` per model weight key.  Keys present in only one LoRA:
    LoRA-1-only keys are kept unchanged (w1 for the TIES modes); LoRA-2-only keys enter only in
    add / negate (+-lam) and the TIES modes (w2, untrimmed); the projection modes use LoRA 2 only as a direction."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    out: Dict[str, LoraFactors] = {}
    info: Dict[str, dict] = {}
    keys = sorted(set(f1) | set(f2))
    for k in keys:
        a, b = f1.get(k), f2.get(k)
        if a is not None and b is not None:
            if mode == "add":
                r = LM.op_add(a, b, lam)
            elif mode == "negate":
                r = LM.op_negate(a, b, lam)
            elif mode == "clean_col":
                r = LM.op_clean_col(a, b, lam)
            elif mode == "clean_row":
                r = LM.op_clean_row(a, b, lam)
            elif mode == "target_sub":
                r = LM.op_target_sub(a, b, lam)
            elif mode == "knots_ties":
                r, i = LM.op_knots_ties(a, b, w1, w2, density, dare_drop, seed, sign_method)
                info[k] = i
            else:
                r, i = LM.op_ties_dense(a, b, w1, w2, density, rank, dare_drop, seed, sign_method)
                info[k] = i
            out[k] = r
        elif a is not None:
            if mode in ("knots_ties", "ties_dense"):
                out[k] = LoraFactors(a.up * (a.scale * w1), a.down, 1.0, list(a.source_keys))
            else:
                out[k] = a.folded(torch.float32)
            info.setdefault(k, {})["only"] = "lora1"
        else:
            if mode in ("add", "negate"):
                sgn = 1.0 if mode == "add" else -1.0
                out[k] = LoraFactors(b.up * (b.scale * sgn * lam), b.down, 1.0, list(b.source_keys))
                info.setdefault(k, {})["only"] = "lora2"
            elif mode in ("knots_ties", "ties_dense"):
                out[k] = LoraFactors(b.up * (b.scale * w2), b.down, 1.0, list(b.source_keys))
                info.setdefault(k, {})["only"] = "lora2"
            # projection modes: nothing to add
    return out, info


def expected_dense(a: Optional[LoraFactors], b: Optional[LoraFactors], mode: str, lam: float) -> torch.Tensor:
    """Dense reference formula for the exact modes (tests)."""
    if a is None:
        return (1 if mode == "add" else -1) * lam * b.dense()
    d1 = a.dense()
    if b is None:
        return d1
    d2 = b.dense()
    if mode == "add":
        return d1 + lam * d2
    if mode == "negate":
        return d1 - lam * d2
    if mode == "clean_col":
        q2 = LM.orth_basis(d2)
        return d1 - lam * q2 @ (q2.T @ d1)
    if mode == "clean_row":
        p2 = LM.orth_basis(d2.T)
        return d1 - lam * (d1 @ p2) @ p2.T
    if mode == "target_sub":
        q1 = LM.orth_basis(d1)
        return d1 - lam * q1 @ (q1.T @ d2)
    raise ValueError(mode)


def conflict_report(adapter, f1: Dict[str, LoraFactors], f2: Dict[str, LoraFactors], density: float = 0.2,
                    dense: bool = True) -> Tuple[str, dict]:
    """Per-layer and per-block interference statistics between two LoRAs (text table + JSON dict)."""
    shared = sorted(set(f1) & set(f2))
    rows = {}
    per_block = defaultdict(list)
    for k in shared:
        st = LM.pair_stats(f1[k], f2[k], density, dense)
        rows[k] = asdict(st)
        bi, grp = adapter.block_of_key(k)
        per_block[(grp, bi)].append(st)
    only1 = sorted(set(f1) - set(f2))
    only2 = sorted(set(f2) - set(f1))

    def mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return sum(xs) / len(xs) if xs else float("nan")

    lines = []
    lines.append(f"shared layers: {len(shared)}   only LoRA1: {len(only1)}   only LoRA2: {len(only2)}")
    lines.append("block      n   |dW1|    |dW2|    cos     colOv  rowOv  E1in2  E2in1  sign%all sign%top")
    blocks_json = {}
    for (grp, bi), sts in sorted(per_block.items(), key=lambda t: (t[0][0], -1 if t[0][1] is None else t[0][1])):
        name = f"{grp}{'' if bi is None else bi}"
        agg = {
            "n": len(sts),
            "norm1": math.sqrt(sum(s.norm1 ** 2 for s in sts)),
            "norm2": math.sqrt(sum(s.norm2 ** 2 for s in sts)),
            "cosine": mean([s.cosine for s in sts]),
            "col_overlap": mean([s.col_overlap for s in sts]),
            "row_overlap": mean([s.row_overlap for s in sts]),
            "energy1_in_col2": mean([s.energy1_in_col2 for s in sts]),
            "energy2_in_col1": mean([s.energy2_in_col1 for s in sts]),
            "sign_conflict_all": mean([s.sign_conflict_all for s in sts]),
            "sign_conflict_top": mean([s.sign_conflict_top for s in sts]),
        }
        blocks_json[name] = agg
        lines.append(f"{name:9s} {agg['n']:3d} {agg['norm1']:8.4f} {agg['norm2']:8.4f} {agg['cosine']:+.3f} "
                     f"{agg['col_overlap']:.3f}  {agg['row_overlap']:.3f}  {agg['energy1_in_col2']:.3f}  "
                     f"{agg['energy2_in_col1']:.3f}  {100 * agg['sign_conflict_all']:6.1f}  {100 * agg['sign_conflict_top']:6.1f}")
    top = sorted(rows.items(), key=lambda kv: -abs(kv[1]["cosine"]) * kv[1]["energy1_in_col2"])[:15]
    lines.append("")
    lines.append("most entangled layers (|cos| * E1in2):")
    for k, r in top:
        lines.append(f"  {k}: cos={r['cosine']:+.3f} E1in2={r['energy1_in_col2']:.3f} colOv={r['col_overlap']:.3f}")
    lines.append("")
    lines.append("Legend: colOv/rowOv = ||Q1^T Q2||_F^2 / min(k1,k2) (LoRA paper Sec. 7 subspace similarity); "
                 "E1in2 = fraction of LoRA1 energy inside LoRA2's output (column) space; sign% = opposite-sign "
                 "entries (all / within both top-density magnitudes, the entries TIES would resolve).")
    return "\n".join(lines), {"layers": rows, "blocks": blocks_json, "only_lora1": only1, "only_lora2": only2}
