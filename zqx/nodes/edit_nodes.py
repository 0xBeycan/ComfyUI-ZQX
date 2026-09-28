from .common import CAT_EDIT, CAT_GUIDANCE, sigma_input


class ZQXActivationSteering:
    DESCRIPTION = ("Contrastive activation steering (ActAdd-style, arXiv 2308.10248): two extra forwards per step with "
                   "a 'towards' and an 'away' prompt; after the selected blocks the image hidden states are shifted by "
                   "alpha * (h_towards - h_away). 'mean' mode uses one global direction (no layout transfer).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "towards": ("CONDITIONING", {"tooltip": "e.g. 'candid photo, looking away, mid-motion'"}),
            "away": ("CONDITIONING", {"tooltip": "e.g. 'posing, looking at the camera, smiling'"}),
            "alpha": ("FLOAT", {"default": 0.5, "min": -5.0, "max": 5.0, "step": 0.05}),
            "sigma_start": sigma_input(1.0, "Active while sigma <= sigma_start."),
            "sigma_end": sigma_input(0.6, "Active while sigma >= sigma_end."),
            "blocks": ("STRING", {"default": "mid", "tooltip": "'mid' = middle block, 'all', or e.g. '10-20'."}),
            "mode": (["mean", "token"], {"default": "mean"}),
            "apply_to": (["all", "cond_only"], {"default": "all"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CAT_GUIDANCE

    def apply(self, model, towards, away, alpha, sigma_start, sigma_end, blocks, mode, apply_to):
        from ..patches.steering import install
        m, _ = install(model, towards=towards, away=away, alpha=alpha, sigma_start=sigma_start, sigma_end=sigma_end,
                       blocks=blocks, mode=mode, apply_to=apply_to)
        return (m,)


class ZQXUCETextEdit:
    DESCRIPTION = ("Closed-form concept edit (UCE, arXiv 2308.14761) of the text input projection (Qwen-Image txt_in, "
                   "Z-Image cap_embedder): the 'source' prompt's embedding is mapped to what the 'target' prompt "
                   "produces, while the 'keep' prompts are preserved. Applied as a weight patch.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "source": ("CONDITIONING", {"tooltip": "Concept to redirect, e.g. 'smiling at the camera, posing'"}),
            "target": ("CONDITIONING", {"tooltip": "What it should mean instead, e.g. 'neutral expression, looking away'"}),
            "pairing": (["mean", "end", "positional"], {"default": "mean"}),
            "lam": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 100.0, "step": 0.01,
                              "tooltip": "Regulariser, relative to the mean squared token norm (bigger = smaller edit)."}),
            "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05}),
        }, "optional": {"keep": ("CONDITIONING", {"tooltip": "Prompts whose embedding must not change (e.g. your trigger word, scene prompts)."})}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = CAT_EDIT

    def apply(self, model, source, target, pairing, lam, strength, keep=None):
        import json
        from ..patches.uce import install
        m, rep = install(model, source, target, keep, pairing, lam, strength)
        return (m, json.dumps(rep, indent=1))
