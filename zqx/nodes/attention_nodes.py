from ..patches.reference_attention import CAPTURE_MODES, POSITION_MODES, RefAttnConfig, install
from .common import CATEGORY, sigma_input


class ZQXReferenceAttention:
    DESCRIPTION = ("Extended self-attention to a reference image (passport / identity reference) for Qwen-Image, "
                   "Qwen-Image-Edit and Z-Image. The target's image tokens attend to the reference's K/V "
                   "(captured by an extra forward pass on the reference), with a log-bias weight, RoPE offset, "
                   "sigma window, block selection and optional face masks.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference": ("LATENT", {"tooltip": "VAE-encoded reference image (batch 1). Any resolution."}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                     "tooltip": "w in log(w) bias on reference logits. 1 = plain concatenation, 0 = off."}),
                "sigma_start": sigma_input(1.0, "High-noise edge of the window (active while sigma <= sigma_start)."),
                "sigma_end": sigma_input(0.0, "Low-noise edge of the window (active while sigma >= sigma_end)."),
                "blocks": ("STRING", {"default": "all", "tooltip": "Main blocks to inject into, e.g. 'all' or '0-19, 30-45'."}),
                "position_mode": (POSITION_MODES, {"default": "frame",
                                                   "tooltip": "RoPE placement of the reference: 'frame' = next frame index (like QIE refs), 'right'/'below' = next to the canvas, 'same' = overlapping positions."}),
                "capture_mode": (CAPTURE_MODES, {"default": "noised",
                                                 "tooltip": "'noised': reference noised to the current sigma each step (extra forward every step). 'cached': captured once at cache_sigma."}),
                "cache_sigma": sigma_input(0.0, "Noise level of the reference in 'cached' mode (0 = clean)."),
                "ref_sigma_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "'noised' mode: reference noise = mult * sigma (FreeCus uses a less-noised reference, < 1)."}),
                "key_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                                        "tooltip": "Multiply reference keys (FreeCus: 1.1)."}),
                "token_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01,
                                            "tooltip": "Randomly drop this fraction of reference tokens each step (ConsiStory: 0.5) to avoid copying the reference layout."}),
                "inject_uncond": ("BOOLEAN", {"default": True,
                                              "tooltip": "Also inject into the uncond branch (CFG > 1). Off = the reference effect is amplified by CFG."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "query_mask": ("MASK", {"tooltip": "Where in the generated image the reference may be attended to (e.g. face region). Resized to the token grid."}),
                "key_mask": ("MASK", {"tooltip": "Which part of the reference is visible (e.g. only the reference face)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, reference, weight, sigma_start, sigma_end, blocks, position_mode, capture_mode, cache_sigma,
              ref_sigma_mult, key_scale, token_dropout, inject_uncond, noise_seed, query_mask=None, key_mask=None):
        ref = reference["samples"]
        if ref.shape[0] != 1:
            raise ValueError("ZQX Reference Attention: reference latent must have batch size 1")
        cfg = RefAttnConfig(ref_latent=ref.clone(), weight=weight, sigma_start=sigma_start, sigma_end=sigma_end,
                            blocks=blocks, position_mode=position_mode, capture_mode=capture_mode,
                            cache_sigma=cache_sigma, noise_seed=noise_seed, inject_uncond=inject_uncond,
                            ref_sigma_mult=ref_sigma_mult, key_scale=key_scale, token_dropout=token_dropout,
                            query_mask=None if query_mask is None else query_mask.clone(),
                            key_mask=None if key_mask is None else key_mask.clone())
        m, _ = install(model, cfg)
        return (m,)
