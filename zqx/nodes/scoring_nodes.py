import json

import torch

from .common import CAT_LORA, CAT_SAMPLING, CAT_SCORING

SCORER = "ZQX_SCORER"


class ZQXScorerClipSimilarity:
    DESCRIPTION = ("Scorer: cosine similarity of CLIP-vision embeddings to reference image(s). Positive weight = "
                   "closer to the reference (e.g. a real candid photo); negative weight = away from it (e.g. away "
                   "from the passport reference composition).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip_vision": ("CLIP_VISION",), "reference": ("IMAGE",),
                             "weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                             "name": ("STRING", {"default": "clip_ref_sim"})}}

    RETURN_TYPES = (SCORER,)
    FUNCTION = "get"
    CATEGORY = CAT_SCORING

    def get(self, clip_vision, reference, weight, name):
        from ..patches.scorers import ClipSimilarityScorer
        return (ClipSimilarityScorer(clip_vision, reference, weight, name),)


class ZQXScorerFace:
    DESCRIPTION = ("Scorer (InsightFace, optional dependency): face_identity (ArcFace cosine to the reference face), "
                   "face_off_center, head_turn (|yaw| from the 3-D landmarks), face_found. Models are read from "
                   "models/insightface (same layout as IPAdapter / PuLID).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_name": (["buffalo_l", "antelopev2"], {"default": "buffalo_l"}),
            "provider": (["CPU", "CUDA"], {"default": "CUDA"}),
            "w_identity": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            "w_off_center": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.05,
                                       "tooltip": "Positive = prefer faces away from the image centre."}),
            "w_head_turn": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.05,
                                      "tooltip": "Positive = prefer heads turned away from the camera."}),
            "w_face_found": ("FLOAT", {"default": 2.0, "min": -10.0, "max": 10.0, "step": 0.05}),
        }, "optional": {"reference": ("IMAGE", {"tooltip": "Identity reference (e.g. the passport photo). Needed for face_identity."})}}

    RETURN_TYPES = (SCORER,)
    FUNCTION = "get"
    CATEGORY = CAT_SCORING

    def get(self, model_name, provider, w_identity, w_off_center, w_head_turn, w_face_found, reference=None):
        from ..patches.scorers import FaceScorer, load_insightface
        app = load_insightface(model_name, provider)
        return (FaceScorer(app, reference, w_identity, w_off_center, w_head_turn, w_face_found),)


class ZQXScorerBackgroundSharpness:
    DESCRIPTION = ("Scorer (model-free): log ratio of Laplacian variance in the border band vs. the centre. "
                   "Bokeh / studio portraits score low; evenly sharp natural scenes around 0.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"weight": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.05}),
                             "border": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.45, "step": 0.01})}}

    RETURN_TYPES = (SCORER,)
    FUNCTION = "get"
    CATEGORY = CAT_SCORING

    def get(self, weight, border):
        from ..patches.scorers import BackgroundSharpnessScorer
        return (BackgroundSharpnessScorer(weight, border),)


class ZQXScorerCombine:
    DESCRIPTION = "Combine up to four scorers (their metrics and weights are merged)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scorer_1": (SCORER,)},
                "optional": {"scorer_2": (SCORER,), "scorer_3": (SCORER,), "scorer_4": (SCORER,)}}

    RETURN_TYPES = (SCORER,)
    FUNCTION = "get"
    CATEGORY = CAT_SCORING

    def get(self, scorer_1, scorer_2=None, scorer_3=None, scorer_4=None):
        from ..patches.scorers import CombinedScorer
        return (CombinedScorer([s for s in (scorer_1, scorer_2, scorer_3, scorer_4) if s is not None]),)


class ZQXSeedSearch:
    DESCRIPTION = ("First-step seed search: runs n_candidates seeds for probe_steps steps, decodes the x0 predictions, "
                   "scores them (z-normalised weighted metrics) and fully samples the best 'keep' seeds (each exactly as "
                   "a normal run with that seed). Distilled models decide the layout in the first steps.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise": ("NOISE",), "guider": ("GUIDER",), "sampler": ("SAMPLER",), "sigmas": ("SIGMAS",),
            "latent_image": ("LATENT",), "vae": ("VAE",), "scorer": (SCORER,),
            "n_candidates": ("INT", {"default": 16, "min": 1, "max": 512}),
            "probe_steps": ("INT", {"default": 1, "min": 1, "max": 100}),
            "keep": ("INT", {"default": 1, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("samples", "previews", "report", "best_seed")
    FUNCTION = "run"
    CATEGORY = CAT_SAMPLING
    OUTPUT_NODE = True

    def run(self, noise, guider, sampler, sigmas, latent_image, vae, scorer, n_candidates, probe_steps, keep):
        from ..patches.seed_search import seed_search
        out, prev, seeds, report, _, _ = seed_search(noise, guider, sampler, sigmas, latent_image, vae, scorer,
                                                     n_candidates, probe_steps, keep)
        lat = {k: v for k, v in latent_image.items() if k not in ("samples", "noise_mask", "batch_index")}
        lat["samples"] = out.to(torch.float32)
        return {"ui": {"text": [report]}, "result": (lat, prev, report, int(seeds[0]))}


class ZQXRealismLoRAAblation:
    DESCRIPTION = ("Identity-safe realism LoRA: with the character LoRA already on the model, measures for every block "
                   "(and optionally module kind / top singular directions) of the realism LoRA how much removing it "
                   "restores identity (identity scorer, e.g. ArcFace) and how much realism look it costs (CLIP "
                   "similarity to the full-realism baseline), removes the parts that hurt identity cheaply and writes a "
                   "new LoRA. Long-running: one generation per seed per unit.")

    @classmethod
    def INPUT_TYPES(cls):
        from .lora_nodes import SAVE_DTYPES, _lora_list
        import comfy.samplers
        return {"required": {
            "model": ("MODEL", {"tooltip": "Model with the character LoRA already applied."}),
            "realism_lora": (_lora_list(),),
            "realism_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",), "vae": ("VAE",),
            "identity_scorer": (SCORER, {"tooltip": "Typically ZQX Scorer Face with only w_identity > 0."}),
            "clip_vision": ("CLIP_VISION", {"tooltip": "Used to measure how much of the realism look is kept."}),
            "seeds": ("STRING", {"default": "1, 2, 3, 4"}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 200}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "units": (["blocks", "kinds", "blocks+kinds", "blocks+svd"], {"default": "blocks"}),
            "svd_blocks": ("INT", {"default": 3, "min": 0, "max": 64, "tooltip": "blocks+svd: refine this many best non-removed blocks."}),
            "svd_components": ("INT", {"default": 4, "min": 0, "max": 64, "tooltip": "blocks+svd: top singular directions tested per block."}),
            "min_identity_gain": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 10.0, "step": 0.001}),
            "max_realism_drop": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001,
                                           "tooltip": "Largest allowed 1 - cos(CLIP(variant), CLIP(full realism))."}),
            "filename_prefix": ("STRING", {"default": "realism_idsafe"}),
            "save_dtype": (SAVE_DTYPES, {"default": "float32"}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("lora_file", "report", "images_A_C_final")
    FUNCTION = "run"
    CATEGORY = CAT_LORA
    OUTPUT_NODE = True

    def run(self, model, realism_lora, realism_strength, positive, negative, latent_image, vae, identity_scorer,
            clip_vision, seeds, steps, cfg, sampler_name, scheduler, units, svd_blocks, svd_components,
            min_identity_gain, max_realism_drop, filename_prefix, save_dtype):
        from ..adapters import get_adapter
        from ..patches.ablation import AblationScan
        from .lora_nodes import load_model_space, save_lora_file, write_report
        seed_list = parse_seeds(seeds)
        factors, _, path = load_model_space(model, realism_lora)
        render = make_renderer(positive, negative, latent_image, vae, seed_list, steps, cfg, sampler_name, scheduler)
        log_lines = []
        scan = AblationScan(get_adapter(model), model, factors, realism_strength, render, identity_scorer, clip_vision,
                            log=log_lines.append)
        final, rep, imgs = scan.run(units, min_identity_gain, max_realism_drop, svd_blocks, svd_components)
        if final:
            fn, rel = save_lora_file(final, filename_prefix, save_dtype,
                                     {"zqx_op": "realism_ablation", "zqx_source": path, "zqx_strength_baked": 1.0,
                                      "zqx_removed": ",".join(rep["removed_units"] + rep["removed_svd"])})
            write_report(fn, rep)
            saved = f"saved: {fn}  (load it at strength {realism_strength}; the strength is not baked in)"
        else:
            rel = ""
            saved = ("NO LoRA written: every unit of the realism LoRA passed the removal criteria, i.e. with these "
                     "thresholds the whole realism LoRA costs identity more than it is worth.")
        head = [saved,
                f"identity A {rep['identity_A']:.4f} | C {rep['identity_C']:.4f} | final {rep['identity_final']:.4f}; "
                f"realism kept (CLIP cos to C) {rep['keep_final']:.4f}; generations {rep['generations']}",
                "removed: " + (", ".join(rep["removed_units"] + rep["removed_svd"]) or "nothing")]
        txt = "\n".join(head + [""] + log_lines)
        return {"ui": {"text": [txt]}, "result": (rel, txt, imgs)}


def parse_seeds(spec: str):
    out = []
    for part in (spec or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("no seeds given")
    return out


def make_renderer(positive, negative, latent, vae, seeds, steps, cfg, sampler_name, scheduler):
    import comfy.sample

    def render(model):
        samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"], latent.get("downscale_ratio_spacial", None),
                                                         latent.get("downscale_ratio_temporal", None))
        imgs = []
        for s in seeds:
            noise = comfy.sample.prepare_noise(samples, s)
            out = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, samples,
                                      denoise=1.0, disable_pbar=True, seed=s)
            img = vae.decode(out)
            if img.ndim == 5:
                img = img.reshape(-1, *img.shape[-3:])
            imgs.append(img)
        return torch.cat(imgs)

    return render
