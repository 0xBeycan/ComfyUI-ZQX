import json
import os

import torch

from ..adapters import get_adapter
from ..patches.lora_schedules import klora_entries, klora_switch_sigmas, scheduled_entries
from ..patches.runtime_lora import install_runtime_lora, load_lora_patches
from .common import CAT_LORA, sigma_input


def _lora_list():
    import folder_paths
    return folder_paths.get_filename_list("loras")


SAVE_DTYPES = ["float16", "bfloat16", "float32"]


def save_lora_file(factors, stem, save_dtype, meta):
    """Write factors (model-key space) to models/loras/zqx/<stem>_NNNN.safetensors; returns (path, relative name)."""
    import folder_paths
    from safetensors.torch import save_file
    from ..core.lora_io import factors_to_state_dict
    if not factors:
        raise ValueError("ZQX: the resulting LoRA is empty (every weight was dropped)")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[save_dtype]
    root = folder_paths.get_folder_paths("loras")[0]
    out_dir = os.path.join(root, "zqx")
    os.makedirs(out_dir, exist_ok=True)
    i = 0
    while True:
        fn = os.path.join(out_dir, f"{stem}_{i:04d}.safetensors")
        if not os.path.exists(fn):
            break
        i += 1
    meta = dict(meta)
    meta.setdefault("zqx_format", "ComfyUI generic keys (diffusion_model.<weight>), alpha = rank")
    save_file(factors_to_state_dict(factors, dtype), fn, metadata={k: str(v) for k, v in meta.items()})
    return fn, os.path.relpath(fn, root)


def write_report(lora_path, obj):
    with open(lora_path[:-len(".safetensors")] + "_report.json", "w") as f:
        json.dump(obj, f, indent=1, default=str)


def load_model_space(model, name):
    """Load a LoRA file and map it to model-weight-key space; refuses unmapped modules."""
    from ..patches.lora_arith import lora_to_model_space
    sd, path = _load_lora_file(name)
    f, via, un = lora_to_model_space(model, sd)
    if un:
        raise ValueError(f"ZQX: {name}: unmapped LoRA modules (refusing to drop them): {un[:10]}")
    return f, via, path


def _load_lora_file(name):
    import comfy.utils
    import folder_paths
    path = folder_paths.get_full_path_or_raise("loras", name)
    return comfy.utils.load_torch_file(path, safe_load=True), path


class ZQXScheduledLoRA:
    DESCRIPTION = ("Runtime (un-merged) LoRA whose strength follows a sigma schedule and per-block weights. "
                   "E.g. realism LoRA strong in the early (layout/pose) steps and weak later, character LoRA the "
                   "opposite - inside one sampler, without extra steps or re-patching.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (_lora_list(),),
            "strength_early": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                                         "tooltip": "Strength for sigma >= sigma_hi (first steps)."}),
            "strength_late": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                                        "tooltip": "Strength for sigma <= sigma_lo (last steps)."}),
            "sigma_hi": sigma_input(0.8, "Upper edge of the linear ramp."),
            "sigma_lo": sigma_input(0.6, "Lower edge of the linear ramp (equal to sigma_hi = hard switch)."),
            "block_weights": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Optional per-block multipliers, e.g. '0-19:1, 20-59:0.3, other:1, refiner:1'. Later entries override earlier ones."}),
            "allow_unmatched_keys": ("BOOLEAN", {"default": False,
                                                 "tooltip": "Ignore LoRA keys that do not map onto the model (LoraLoader silently skips them). Off = error."}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = CAT_LORA

    def apply(self, model, lora_name, strength_early, strength_late, sigma_hi, sigma_lo, block_weights, allow_unmatched_keys):
        sd, _ = _load_lora_file(lora_name)
        ad = get_adapter(model)
        pk, unmatched = load_lora_patches(model, sd, allow_unmatched=allow_unmatched_keys)
        entries, rep = scheduled_entries(ad, pk, strength_early, strength_late, sigma_hi, sigma_lo, block_weights)
        m, _ = install_runtime_lora(model, entries, "zqx_sched_lora_" + lora_name)
        lines = [f"{lora_name}: {len(pk)} model weights patched at runtime; strength {strength_early} (sigma>={sigma_hi}) -> "
                 f"{strength_late} (sigma<={sigma_lo})."]
        if unmatched:
            lines.append(f"IGNORED {len(unmatched)} unmatched LoRA keys (allow_unmatched_keys=True).")
        non1 = [(k, b, g, w) for k, b, g, w in rep if w != 1.0]
        if non1:
            lines.append(f"{len(non1)} weights with block multiplier != 1, e.g.: " +
                         ", ".join(f"{k}={w}" for k, b, g, w in non1[:6]))
        return (m, "\n".join(lines))


class ZQXKLoRA:
    DESCRIPTION = ("K-LoRA (Ouyang et al., CVPR 2025, arXiv:2502.18461): training-free fusion of a subject (character) "
                   "and a style (realism) LoRA. In every attention layer and step exactly one LoRA is active, chosen by "
                   "comparing the sums of their top-K |dW| entries with a timestep-dependent scale.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "character_lora": (_lora_list(),),
            "realism_lora": (_lora_list(),),
            "character_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            "realism_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            "alpha": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "S(t) = alpha * t/T + beta."}),
            "beta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01,
                               "tooltip": "Official: pattern 's' beta=0.5; pattern 's*' (FLUX) beta=0.85*alpha=1.275."}),
            "pattern": (["s", "s*"], {"default": "s"}),
            "scope": (["attention", "all_shared"], {"default": "attention",
                                                    "tooltip": "attention = q/k/v projections only (paper); all_shared = every layer both LoRAs touch."}),
            "other_layers": (["both", "character", "realism", "none"], {"default": "both",
                                                                        "tooltip": "What to apply on layers outside the K-LoRA selection."}),
            "allow_unmatched_keys": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = CAT_LORA

    def apply(self, model, character_lora, realism_lora, character_strength, realism_strength, alpha, beta, pattern,
              scope, other_layers, allow_unmatched_keys):
        sdc, _ = _load_lora_file(character_lora)
        sds, _ = _load_lora_file(realism_lora)
        ad = get_adapter(model)
        pc, _ = load_lora_patches(model, sdc, allow_unmatched=allow_unmatched_keys)
        ps, _ = load_lora_patches(model, sds, allow_unmatched=allow_unmatched_keys)
        shapes = {k: tuple(v.shape) for k, v in model.model.state_dict().items()}
        entries, report, shared = klora_entries(ad, shapes, pc, ps, character_strength, realism_strength, alpha, beta,
                                                pattern, scope, other_layers)
        m, _ = install_runtime_lora(model, entries, "zqx_klora")
        sw = klora_switch_sigmas(report, alpha, beta, pattern)
        lines = [f"K-LoRA gamma = {report['gamma']:.4g}; {len(shared)} layers selected per step."]
        for sg in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1):
            n_c = sum(1 for k in shared if any(abs(x - sg) < 5e-4 for x in sw[k]))
            lines.append(f"  sigma={sg:.2f}: character in {n_c}/{len(shared)} layers, realism in {len(shared) - n_c}")
        return (m, "\n".join(lines))


class ZQXLoRAArithmetic:
    DESCRIPTION = ("Weight-space LoRA arithmetic -> new .safetensors LoRA (saved to models/loras/zqx/) + report. "
                   "Modes: add, negate (task arithmetic), clean_col/clean_row (project the character LoRA off the second "
                   "LoRA's output/input subspace), target_sub (subtract only the part of LoRA2 inside LoRA1's subspace), "
                   "knots_ties (KnOTS-aligned TIES/DARE, exact low rank), ties_dense (dense TIES + truncated SVD).")

    @classmethod
    def INPUT_TYPES(cls):
        from ..patches.lora_arith import MODES
        return {"required": {
            "model": ("MODEL", {"tooltip": "Only used for the key map / weight shapes (nothing is patched)."}),
            "lora_1": (_lora_list(), {"tooltip": "Usually the character LoRA."}),
            "lora_2": (_lora_list(), {"tooltip": "Realism LoRA today; an 'AI-look' LoRA in the future."}),
            "mode": (MODES, {"default": "clean_col"}),
            "lam": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "lambda of the operation."}),
            "w1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "TIES weight of lora_1."}),
            "w2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "TIES weight of lora_2."}),
            "density": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "TIES top-k fraction kept (paper: 0.2)."}),
            "dare_drop": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "DARE drop rate before TIES (0 = off)."}),
            "svd_rank": ("INT", {"default": 64, "min": 1, "max": 1024, "tooltip": "Output rank for ties_dense."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "filename_prefix": ("STRING", {"default": "zqx_merge"}),
            "save_dtype": (SAVE_DTYPES, {"default": "float32"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_file", "report")
    FUNCTION = "run"
    CATEGORY = CAT_LORA
    OUTPUT_NODE = True

    def run(self, model, lora_1, lora_2, mode, lam, w1, w2, density, dare_drop, svd_rank, seed, filename_prefix, save_dtype):
        from ..patches.lora_arith import combine, conflict_report, lora_to_model_space

        sd1, p1 = _load_lora_file(lora_1)
        sd2, p2 = _load_lora_file(lora_2)
        f1, via1, un1 = lora_to_model_space(model, sd1)
        f2, via2, un2 = lora_to_model_space(model, sd2)
        if un1 or un2:
            raise ValueError(f"ZQX LoRA Arithmetic: unmapped LoRA modules (refusing to drop them): {(un1 + un2)[:10]}")
        res, info = combine(f1, f2, mode, lam, w1, w2, density, svd_rank, dare_drop, seed)
        txt, js = conflict_report(get_adapter(model), f1, f2, density, dense=True)
        meta = {"zqx_mode": mode, "zqx_lam": str(lam), "zqx_w1": str(w1), "zqx_w2": str(w2), "zqx_density": str(density),
                "zqx_dare_drop": str(dare_drop), "zqx_seed": str(seed), "zqx_lora_1": os.path.basename(p1),
                "zqx_lora_2": os.path.basename(p2), "zqx_format": "ComfyUI generic keys (diffusion_model.<weight>), alpha = rank"}
        fn, rel = save_lora_file(res, f"{filename_prefix}_{mode}", save_dtype, meta)
        errs = [v.get("svd_rel_error") for v in info.values() if isinstance(v, dict) and "svd_rel_error" in v]
        head = [f"saved: {fn}", f"mode={mode} lam={lam} layers={len(res)} (lora_1 {len(f1)}, lora_2 {len(f2)})"]
        if via1 or via2:
            head.append(f"{len(via1) + len(via2)} modules mapped via normalised names (not loadable by ComfyUI's loader in their original naming).")
        if errs:
            head.append(f"ties_dense truncation: mean rel. Frobenius error {sum(errs) / len(errs):.4f}, max {max(errs):.4f}")
        write_report(fn, {"mode": mode, "info": info, "conflicts": js})
        return {"ui": {"text": ["\n".join(head)]}, "result": (rel, "\n".join(head) + "\n\n" + txt)}


class ZQXLoRAConflictReport:
    DESCRIPTION = ("Per-layer / per-block interference statistics between two LoRAs: cosine, subspace overlap, energy "
                   "of one inside the other's subspace, sign conflicts. Shows where the character and realism LoRAs fight.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_1": (_lora_list(),),
            "lora_2": (_lora_list(),),
            "top_density": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
            "sign_stats": ("BOOLEAN", {"default": True, "tooltip": "Dense sign-conflict statistics (slower)."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = CAT_LORA
    OUTPUT_NODE = True

    def run(self, model, lora_1, lora_2, top_density, sign_stats):
        from ..patches.lora_arith import conflict_report, lora_to_model_space
        sd1, _ = _load_lora_file(lora_1)
        sd2, _ = _load_lora_file(lora_2)
        f1, _, un1 = lora_to_model_space(model, sd1)
        f2, _, un2 = lora_to_model_space(model, sd2)
        txt, _ = conflict_report(get_adapter(model), f1, f2, top_density, dense=sign_stats)
        if un1 or un2:
            txt = f"WARNING: unmapped modules: {(un1 + un2)[:10]}\n" + txt
        return {"ui": {"text": [txt]}, "result": (txt,)}


class ZQXLoRASurgery:
    DESCRIPTION = ("Rewrite a single LoRA: per-block multipliers, drop module kinds (text stream, modulation, attention, "
                   "mlp, io), rank truncation (max rank / energy), spectrum power, DARE on the up factor. Writes a new "
                   "LoRA to models/loras/zqx/ + a JSON report.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "Only used for the key map / weight shapes (nothing is patched)."}),
            "lora_name": (_lora_list(),),
            "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01, "tooltip": "Baked-in global multiplier."}),
            "block_weights": ("STRING", {"default": "", "tooltip": "e.g. '0-9:0, 10-29:1, other:1'. 0 removes the weight from the file."}),
            "drop_kinds": ("STRING", {"default": "", "tooltip": "Comma list of module kinds to remove: text, modulation, attention, mlp, io, other."}),
            "max_rank": ("INT", {"default": 0, "min": 0, "max": 1024, "tooltip": "Keep at most this many singular directions per weight (0 = all)."}),
            "energy_keep": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Keep the smallest rank holding this fraction of the squared singular values."}),
            "spectrum_power": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05, "tooltip": "sigma_i -> sigma_1 (sigma_i/sigma_1)^p. <1 flattens, >1 sharpens."}),
            "power_preserve": (["top", "frobenius"], {"default": "top"}),
            "dare_drop": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "DARE on the up factor (unbiased, stays low rank)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "filename_prefix": ("STRING", {"default": "zqx_surgery"}),
            "save_dtype": (SAVE_DTYPES, {"default": "float32"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_file", "report")
    FUNCTION = "run"
    CATEGORY = CAT_LORA
    OUTPUT_NODE = True

    def run(self, model, lora_name, strength, block_weights, drop_kinds, max_rank, energy_keep, spectrum_power,
            power_preserve, dare_drop, seed, filename_prefix, save_dtype):
        from ..patches.lora_surgery import surgery
        ad = get_adapter(model)
        f, via, path = load_model_space(model, lora_name)
        res, rep = surgery(ad, f, block_weights, drop_kinds, strength, max_rank, energy_keep, spectrum_power,
                           power_preserve, dare_drop, seed)
        meta = {"zqx_op": "surgery", "zqx_source": os.path.basename(path), "zqx_block_weights": block_weights,
                "zqx_drop_kinds": drop_kinds, "zqx_strength": strength, "zqx_max_rank": max_rank,
                "zqx_energy_keep": energy_keep, "zqx_spectrum_power": spectrum_power, "zqx_dare_drop": dare_drop,
                "zqx_seed": seed}
        fn, rel = save_lora_file(res, filename_prefix, save_dtype, meta)
        write_report(fn, rep)
        kinds = {}
        for k, info in rep["layers"].items():
            kinds.setdefault(info["kind"], [0, 0.0])
            kinds[info["kind"]][0] += 1
            kinds[info["kind"]][1] = max(kinds[info["kind"]][1], info["trunc_rel_error"])
        lines = [f"saved: {fn}", f"weights kept {len(res)}, dropped {len(rep['dropped'])}"]
        lines += [f"  {k}: {n} weights, max truncation error {e:.4f}" for k, (n, e) in sorted(kinds.items())]
        if via:
            lines.append(f"{len(via)} modules mapped via normalised names.")
        txt = "\n".join(lines)
        return {"ui": {"text": [txt]}, "result": (rel, txt)}


class ZQXLoRACommonSubspace:
    DESCRIPTION = ("Common subspace of 2-4 LoRAs (Iso-CTS-inspired, arXiv 2502.04959): per weight, the top-k left "
                   "singular vectors of the summed deltas. 'common' writes the shared component (e.g. the 'AI look' "
                   "shared by several character LoRAs made with the same pipeline); 'clean_target' removes that "
                   "subspace from a target LoRA.")

    @classmethod
    def INPUT_TYPES(cls):
        none = ["None"] + _lora_list()
        return {"required": {
            "model": ("MODEL",),
            "lora_1": (_lora_list(),),
            "lora_2": (_lora_list(),),
            "lora_3": (none, {"default": "None"}),
            "lora_4": (none, {"default": "None"}),
            "common_rank": ("INT", {"default": 4, "min": 1, "max": 256}),
            "mode": (["common", "clean_target"], {"default": "common"}),
            "target_lora": (none, {"default": "None", "tooltip": "Required for clean_target."}),
            "lam": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "filename_prefix": ("STRING", {"default": "zqx_common"}),
            "save_dtype": (SAVE_DTYPES, {"default": "float32"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_file", "report")
    FUNCTION = "run"
    CATEGORY = CAT_LORA
    OUTPUT_NODE = True

    def run(self, model, lora_1, lora_2, lora_3, lora_4, common_rank, mode, target_lora, lam, filename_prefix, save_dtype):
        from ..patches.lora_surgery import common_subspace_merge
        names = [n for n in (lora_1, lora_2, lora_3, lora_4) if n != "None"]
        loras = [load_model_space(model, n)[0] for n in names]
        target = None
        if mode == "clean_target":
            if target_lora == "None":
                raise ValueError("ZQX LoRA Common Subspace: clean_target needs target_lora")
            target = load_model_space(model, target_lora)[0]
        res, rep = common_subspace_merge(get_adapter(model), loras, common_rank, mode, target, lam)
        fn, rel = save_lora_file(res, f"{filename_prefix}_{mode}", save_dtype,
                                 {"zqx_op": "common_subspace", "zqx_mode": mode, "zqx_sources": ",".join(names),
                                  "zqx_common_rank": common_rank, "zqx_target": target_lora, "zqx_lam": lam})
        write_report(fn, rep)
        lines = [f"saved: {fn}", f"shared weights: {rep['shared_weights']}"]
        lines += [f"  {k}: {v:.3f} of its energy lies in the common subspace" for k, v in rep["mean_energy_in_common"].items()]
        txt = "\n".join(lines)
        return {"ui": {"text": [txt]}, "result": (rel, txt)}


class ZQXSpatialLoRA:
    DESCRIPTION = ("Token-masked LoRA (LoRAShop-style, arXiv 2505.23758): the LoRA runs as a side branch whose output "
                   "is multiplied per token by a mask. E.g. realism LoRA on everything except the face (mask = face, "
                   "invert on) and character LoRA only on the face. All weights 1 = identical to LoraLoader.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (_lora_list(),),
            "strength_early": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "strength_late": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "sigma_hi": sigma_input(1.0, "Upper edge of the strength ramp."),
            "sigma_lo": sigma_input(0.0, "Lower edge of the strength ramp."),
            "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Apply the LoRA where the mask is 0."}),
            "text_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "LoRA weight on text tokens (Qwen text stream, Z-Image caption tokens)."}),
            "nonspatial_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "LoRA weight on modulation / timestep layers (they act on the whole image)."}),
            "other_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "LoRA weight on Qwen-Image-Edit reference tokens and Z-Image padding tokens."}),
            "allow_unmatched_keys": ("BOOLEAN", {"default": False}),
        }, "optional": {"mask": ("MASK", {"tooltip": "Region of the generated image (resized to the token grid). None = everywhere."})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CAT_LORA

    def apply(self, model, lora_name, strength_early, strength_late, sigma_hi, sigma_lo, invert_mask, text_weight,
              nonspatial_weight, other_weight, allow_unmatched_keys, mask=None):
        from ..patches.spatial_lora import install
        sd, _ = _load_lora_file(lora_name)
        m, _, _ = install(model, sd, allow_unmatched_keys, strength_early=strength_early, strength_late=strength_late,
                          sigma_hi=sigma_hi, sigma_lo=sigma_lo, mask=None if mask is None else mask.clone(),
                          invert=invert_mask, text_weight=text_weight, nonspatial_weight=nonspatial_weight,
                          other_weight=other_weight, key="zqx_spatial_lora_" + lora_name)
        return (m,)


class ZQXLoRAGuidance:
    DESCRIPTION = ("LoRA guidance (LoRA-CFG, heuristic): out = out_base + w * (out_lora - out_base) with a sigma "
                   "schedule for w. w=1 is the plain LoRA, w=0 the base model, w>1 amplifies the LoRA. E.g. character "
                   "LoRA with w 0.3 in the layout steps and 1.5 in the detail steps. Two forwards per step when w != 0, 1.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (_lora_list(),),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "LoRA strength in the 'with LoRA' pass."}),
            "w_early": ("FLOAT", {"default": 0.3, "min": -2.0, "max": 5.0, "step": 0.05, "tooltip": "w for sigma >= sigma_hi."}),
            "w_late": ("FLOAT", {"default": 1.5, "min": -2.0, "max": 5.0, "step": 0.05, "tooltip": "w for sigma <= sigma_lo."}),
            "sigma_hi": sigma_input(0.85, "Upper edge of the w ramp."),
            "sigma_lo": sigma_input(0.6, "Lower edge of the w ramp."),
            "allow_unmatched_keys": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CAT_LORA

    def apply(self, model, lora_name, strength, w_early, w_late, sigma_hi, sigma_lo, allow_unmatched_keys):
        from ..patches.guidance import install_lora_guidance
        sd, _ = _load_lora_file(lora_name)
        pk, _ = load_lora_patches(model, sd, allow_unmatched=allow_unmatched_keys)
        entries, _ = scheduled_entries(get_adapter(model), pk, strength, strength, 1.0, 0.0, "")
        m, _, _ = install_lora_guidance(model, entries, "zqx_lora_guidance_" + lora_name, w_early, w_late, sigma_hi, sigma_lo)
        return (m,)
