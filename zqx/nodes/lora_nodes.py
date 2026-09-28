import json
import os

import torch

from ..adapters import get_adapter
from ..patches.lora_schedules import klora_entries, klora_switch_sigmas, scheduled_entries
from ..patches.runtime_lora import install_runtime_lora, load_lora_patches
from .common import CATEGORY, sigma_input


def _lora_list():
    import folder_paths
    return folder_paths.get_filename_list("loras")


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
    CATEGORY = CATEGORY

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
    CATEGORY = CATEGORY

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
            "save_dtype": (["float16", "bfloat16", "float32"], {"default": "float32"}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_file", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, model, lora_1, lora_2, mode, lam, w1, w2, density, dare_drop, svd_rank, seed, filename_prefix, save_dtype):
        import folder_paths
        from safetensors.torch import save_file
        from ..core.lora_io import factors_to_state_dict
        from ..patches.lora_arith import combine, conflict_report, lora_to_model_space

        sd1, p1 = _load_lora_file(lora_1)
        sd2, p2 = _load_lora_file(lora_2)
        f1, via1, un1 = lora_to_model_space(model, sd1)
        f2, via2, un2 = lora_to_model_space(model, sd2)
        if un1 or un2:
            raise ValueError(f"ZQX LoRA Arithmetic: unmapped LoRA modules (refusing to drop them): {(un1 + un2)[:10]}")
        res, info = combine(f1, f2, mode, lam, w1, w2, density, svd_rank, dare_drop, seed)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[save_dtype]
        out_sd = factors_to_state_dict(res, dtype)
        out_dir = os.path.join(folder_paths.get_folder_paths("loras")[0], "zqx")
        os.makedirs(out_dir, exist_ok=True)
        i = 0
        while True:
            fn = os.path.join(out_dir, f"{filename_prefix}_{mode}_{i:04d}.safetensors")
            if not os.path.exists(fn):
                break
            i += 1
        txt, js = conflict_report(get_adapter(model), f1, f2, density, dense=True)
        meta = {"zqx_mode": mode, "zqx_lam": str(lam), "zqx_w1": str(w1), "zqx_w2": str(w2), "zqx_density": str(density),
                "zqx_dare_drop": str(dare_drop), "zqx_seed": str(seed), "zqx_lora_1": os.path.basename(p1),
                "zqx_lora_2": os.path.basename(p2), "zqx_format": "ComfyUI generic keys (diffusion_model.<weight>), alpha = rank"}
        save_file(out_sd, fn, metadata=meta)
        errs = [v.get("svd_rel_error") for v in info.values() if isinstance(v, dict) and "svd_rel_error" in v]
        head = [f"saved: {fn}", f"mode={mode} lam={lam} layers={len(res)} (lora_1 {len(f1)}, lora_2 {len(f2)})"]
        if via1 or via2:
            head.append(f"{len(via1) + len(via2)} modules mapped via normalised names (not loadable by ComfyUI's loader in their original naming).")
        if errs:
            head.append(f"ties_dense truncation: mean rel. Frobenius error {sum(errs) / len(errs):.4f}, max {max(errs):.4f}")
        with open(fn[:-len(".safetensors")] + "_report.json", "w") as f:
            json.dump({"mode": mode, "info": info, "conflicts": js}, f, indent=1, default=str)
        rel = os.path.relpath(fn, folder_paths.get_folder_paths("loras")[0])
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
    CATEGORY = CATEGORY
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
