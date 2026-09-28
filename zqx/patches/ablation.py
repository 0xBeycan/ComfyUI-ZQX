"""Identity-safe realism LoRA: measure which parts of a realism LoRA hurt identity, remove them.

Protocol (fixed seeds / prompt, the model already carries the character LoRA):
  A  = model without the realism LoRA      (identity reference level)
  C  = model + full realism LoRA           (baseline: realism look, identity loss)
  C' = model + realism LoRA minus one unit (a block, a module kind, or one singular direction of a block)

  identity(C')   = sum_m w_m * mean_seeds(metric_m)   from the identity scorer (e.g. ArcFace similarity)
  keep(C')       = mean_seeds cos(CLIP(C'), CLIP(C))  how much of the realism look survives
  gain(unit)     = identity(C') - identity(C)
  drop(unit)     = 1 - keep(C')

A unit is removed when gain >= min_identity_gain and drop <= max_realism_drop.  All removals are then applied
together, the result is evaluated once more and written as a new LoRA.  Heuristic procedure (greedy,
one-at-a-time attribution; interactions between units are only checked by the final evaluation).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..core import lora_surgery as LS
from ..core.lora_io import LoraFactors
from ..core.scoring import combine, cosine


def select_units(results: Sequence[dict], min_gain: float, max_drop: float) -> List[str]:
    """results: [{'name', 'gain', 'drop'}] -> names to remove, best gain first."""
    ok = [r for r in results if r["gain"] >= min_gain and r["drop"] <= max_drop and r["gain"] > 0]
    return [r["name"] for r in sorted(ok, key=lambda r: (-r["gain"], r["name"]))]


def factors_to_entries(factors: Dict[str, LoraFactors], strength: float):
    """Runtime-LoRA entries from model-key-space factors (plain comfy LoRA adapters, scale folded)."""
    from comfy.weight_adapter.lora import LoRAAdapter
    entries = {}
    for k, f in factors.items():
        ff = f.folded(torch.float32)
        ad = LoRAAdapter(set(), (ff.up, ff.down, None, None, None, None))
        entries[k] = [(lambda sigma, s=strength: s, ad, None, None)]
    return entries


def remove_component(f: LoraFactors, index: int) -> LoraFactors:
    u, s, v = LS.lora_svd(f)
    keep = [i for i in range(s.shape[0]) if i != index]
    return LS.from_svd(u[:, keep], s[keep], v[:, keep], f.source_keys)


def build_units(adapter, factors: Dict[str, LoraFactors], mode: str) -> "OrderedDict[str, List[str]]":
    units: "OrderedDict[str, List[str]]" = OrderedDict()
    if mode in ("blocks", "blocks+kinds", "blocks+svd"):
        by = {}
        for k in factors:
            bi, grp = adapter.block_of_key(k)
            name = f"block:{grp}{'' if bi is None else bi}"
            by.setdefault((grp, -1 if bi is None else bi, name), []).append(k)
        for (_, _, name), ks in sorted(by.items()):
            units[name] = sorted(ks)
    if mode in ("kinds", "blocks+kinds"):
        by = {}
        for k in factors:
            by.setdefault(f"kind:{adapter.module_kind(k)}", []).append(k)
        for name in sorted(by):
            units[name] = sorted(by[name])
    if not units:
        raise ValueError(f"unknown unit mode {mode!r}")
    return units


class AblationScan:
    def __init__(self, adapter, model_patcher, factors: Dict[str, LoraFactors], strength: float,
                 render: Callable[[object], torch.Tensor], identity_scorer, clip_vision, log=None):
        self.adapter = adapter
        self.mp = model_patcher
        self.factors = factors
        self.strength = strength
        self.render = render            # model -> images (n_seeds, H, W, 3)
        self.scorer = identity_scorer
        self.clip_vision = clip_vision
        self.n_runs = 0
        self.log = log or (lambda s: None)

    def model_with(self, factors: Dict[str, LoraFactors]):
        from .runtime_lora import install_runtime_lora
        if not factors:
            return self.mp
        m, _ = install_runtime_lora(self.mp, factors_to_entries(factors, self.strength), "zqx_ablation")
        return m

    def evaluate(self, factors, base_emb=None):
        from .scorers import clip_embed
        imgs = self.render(self.model_with(factors))
        self.n_runs += 1
        metrics = self.scorer.metrics(imgs)
        ident = float(combine({k: v.mean().view(1) for k, v in metrics.items()}, self.scorer.weights, normalize=False)[0])
        emb = clip_embed(self.clip_vision, imgs)
        keep = float(cosine(emb, base_emb).mean()) if base_emb is not None else 1.0
        return ident, keep, emb, imgs

    def run(self, units_mode: str, min_gain: float, max_drop: float, svd_blocks: int = 0, svd_components: int = 0):
        i_a, _, _, img_a = self.evaluate({})
        i_c, _, emb_c, img_c = self.evaluate(self.factors)
        self.log(f"A (no realism) identity {i_a:.4f}; C (full realism) identity {i_c:.4f}")
        units = build_units(self.adapter, self.factors, units_mode)
        results = []
        for name, keys in units.items():
            sub = {k: f for k, f in self.factors.items() if k not in set(keys)}
            ident, keep, _, _ = self.evaluate(sub, emb_c)
            results.append({"name": name, "keys": keys, "gain": ident - i_c, "drop": 1.0 - keep})
            self.log(f"{name}: gain {ident - i_c:+.4f} drop {1 - keep:.4f}")
        chosen = select_units(results, min_gain, max_drop)
        removed_keys = set()
        for r in results:
            if r["name"] in chosen:
                removed_keys.update(r["keys"])
        svd_results = []
        chosen_svd: List[Tuple[str, int]] = []
        if units_mode == "blocks+svd" and svd_blocks > 0 and svd_components > 0:
            blocks = [r for r in sorted(results, key=lambda r: -r["gain"])
                      if r["name"].startswith("block:") and r["name"] not in chosen][:svd_blocks]
            for r in blocks:
                for c in range(svd_components):
                    sub = {k: (remove_component(f, c) if k in r["keys"] else f) for k, f in self.factors.items()}
                    ident, keep, _, _ = self.evaluate(sub, emb_c)
                    name = f"{r['name']}/sv{c}"
                    svd_results.append({"name": name, "block": r["name"], "keys": r["keys"], "component": c,
                                        "gain": ident - i_c, "drop": 1.0 - keep})
                    self.log(f"{name}: gain {ident - i_c:+.4f} drop {1 - keep:.4f}")
            chosen_svd_names = select_units(svd_results, min_gain, max_drop)
            chosen_svd = [(r["block"], r["component"]) for r in svd_results if r["name"] in chosen_svd_names]
        final = {}
        for k, f in self.factors.items():
            if k in removed_keys:
                continue
            comps = sorted({c for (blk, c) in chosen_svd if k in units.get(blk, [])}, reverse=True)
            g = f
            if comps:
                u, s, v = LS.lora_svd(f)
                keep_idx = [i for i in range(s.shape[0]) if i not in comps]
                g = LS.from_svd(u[:, keep_idx], s[keep_idx], v[:, keep_idx], f.source_keys)
            final[k] = g
        i_f, keep_f, _, img_f = self.evaluate(final, emb_c)
        report = {
            "identity_A": i_a, "identity_C": i_c, "identity_final": i_f, "keep_final": keep_f,
            "removed_units": chosen, "removed_svd": [f"{b}/sv{c}" for b, c in chosen_svd],
            "units": [{k: v for k, v in r.items() if k != "keys"} for r in results],
            "svd_units": [{k: v for k, v in r.items() if k != "keys"} for r in svd_results],
            "generations": self.n_runs,
        }
        return final, report, torch.cat([img_a, img_c, img_f])
