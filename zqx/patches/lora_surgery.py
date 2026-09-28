"""Single-LoRA surgery and multi-LoRA common subspace in model-weight-key space (ComfyUI glue)."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..core import lora_surgery as LS
from ..core.lora_io import LoraFactors
from ..core.schedule import block_weight_for, parse_block_weights


def parse_kinds(spec: str, allowed: Sequence[str]) -> List[str]:
    kinds = [k.strip().lower() for k in (spec or "").split(",") if k.strip()]
    bad = [k for k in kinds if k not in allowed]
    if bad:
        raise ValueError(f"unknown module kinds {bad}; allowed: {list(allowed)}")
    return kinds


def surgery(adapter, factors: Dict[str, LoraFactors], block_weights: str = "", drop_kinds: str = "",
            strength: float = 1.0, max_rank: int = 0, energy_keep: float = 1.0, spectrum_power: float = 1.0,
            power_preserve: str = "top", dare_drop: float = 0.0, seed: int = 0) -> Tuple[Dict[str, LoraFactors], dict]:
    """Apply, per model weight: block/kind multiplier -> rank truncation (max_rank and/or energy_keep) ->
    spectrum power -> DARE on the up factor.  Weights whose multiplier is 0 are dropped from the file."""
    if not (0.0 < energy_keep <= 1.0):
        raise ValueError("energy_keep must be in (0, 1]")
    per_block, groups = parse_block_weights(block_weights)
    kinds = parse_kinds(drop_kinds, adapter.MODULE_KINDS)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out: Dict[str, LoraFactors] = {}
    report = {"dropped": [], "layers": {}}
    for k in sorted(factors):
        f = factors[k]
        bi, grp = adapter.block_of_key(k)
        kind = adapter.module_kind(k)
        mult = strength * block_weight_for(bi, grp, per_block, groups)
        if kind in kinds:
            mult = 0.0
        if mult == 0.0:
            report["dropped"].append(k)
            continue
        info = {"kind": kind, "block": bi, "mult": mult, "rank_in": f.rank}
        cur = LS.scale_factors(f, mult)
        u, s, v = LS.lora_svd(cur)
        r = int((s > 0).sum())
        if energy_keep < 1.0 and r > 0:
            e = torch.cumsum(s ** 2, 0) / (s ** 2).sum()
            r = min(r, int(torch.searchsorted(e, torch.tensor(energy_keep, dtype=e.dtype)).item()) + 1)
        if max_rank > 0:
            r = min(r, max_rank)
        tot = float((s ** 2).sum())
        info["trunc_rel_error"] = math.sqrt(float((s[r:] ** 2).sum()) / tot) if tot > 0 else 0.0
        s2 = s[:r]
        if spectrum_power != 1.0 and r > 0:
            s_new = s2[0] * (s2 / s2[0]) ** spectrum_power
            if power_preserve == "frobenius":
                s_new = s_new * (s2.norm() / s_new.norm())
            elif power_preserve != "top":
                raise ValueError("power_preserve must be top or frobenius")
            s2 = s_new
        cur = LS.from_svd(u[:, :r], s2, v[:, :r], f.source_keys)
        if dare_drop > 0:
            cur = LS.dare_up(cur, dare_drop, g)
        info["rank_out"] = cur.rank
        out[k] = cur
        report["layers"][k] = info
    return out, report


def common_subspace_merge(adapter, loras: Sequence[Dict[str, LoraFactors]], k: int, mode: str,
                          target: Optional[Dict[str, LoraFactors]] = None, lam: float = 1.0):
    """mode 'common': LoRA = U_c U_c^T mean_t(dW_t) on weights shared by all LoRAs.
    mode 'clean_target': target with (I - lam U_c U_c^T) applied on shared weights, unchanged elsewhere."""
    if mode not in ("common", "clean_target"):
        raise ValueError("mode must be 'common' or 'clean_target'")
    if len(loras) < 2:
        raise ValueError("need at least two LoRAs")
    shared = set(loras[0])
    for l in loras[1:]:
        shared &= set(l)
    if not shared:
        raise ValueError("the LoRAs share no weight")
    out: Dict[str, LoraFactors] = {}
    energy = defaultdict(list)
    per_block = defaultdict(list)
    for key in sorted(shared):
        fs = [l[key] for l in loras]
        basis = LS.common_subspace(fs, k)
        fr = [LS.energy_fraction_in(f, basis) for f in fs]
        for i, e in enumerate(fr):
            energy[i].append(e)
        bi, grp = adapter.block_of_key(key)
        per_block[f"{grp}{'' if bi is None else bi}"].append(sum(fr) / len(fr))
        if mode == "common":
            out[key] = LS.common_component(fs, basis)
        else:
            if key in target:
                out[key] = LS.project_left(target[key], basis, lam)
    if mode == "clean_target":
        for key, f in target.items():
            if key not in out:
                out[key] = f.folded(torch.float32)
    report = {
        "shared_weights": len(shared),
        "mean_energy_in_common": {f"lora_{i + 1}": sum(v) / len(v) for i, v in energy.items()},
        "per_block_mean_energy_in_common": {b: sum(v) / len(v) for b, v in per_block.items()},
    }
    return out, report
