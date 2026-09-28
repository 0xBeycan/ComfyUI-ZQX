from .common import CAT_GUIDANCE, CAT_SAMPLING, CAT_TOOLS, sigma_input


class ZQXCADS:
    DESCRIPTION = ("CADS (Sadat et al., ICLR 2024, arXiv:2310.17347): anneal the text conditioning with Gaussian noise "
                   "in the high-noise steps (gamma(t) piecewise linear between tau1 and tau2, t = sigma), then rescale "
                   "to the original mean/std (psi). Restores sample diversity (pose, framing, background) lost to "
                   "mode collapse. Keep it conservative on distilled Z-Image Turbo.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tau1": sigma_input(0.6, "gamma = 1 (clean condition) for sigma <= tau1. Paper (SD): 0.6."),
            "tau2": sigma_input(0.9, "gamma = 0 (fully noised condition) for sigma >= tau2. Paper (SD): 0.9."),
            "noise_scale": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "s. Paper (SD): 0.25. 0 = off."}),
            "psi": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Rescale mix. Paper: 1.0."}),
            "relative_noise": ("BOOLEAN", {"default": True,
                                           "tooltip": "Scale s by the std of each embedding (heuristic for LLM text encoders; off = paper's absolute s)."}),
            "apply_to": (["cond_and_uncond", "cond_only"], {"default": "cond_and_uncond"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CAT_GUIDANCE

    def apply(self, model, tau1, tau2, noise_scale, psi, relative_noise, apply_to, seed):
        from ..patches.cads import install
        m, _ = install(model, tau1=tau1, tau2=tau2, noise_scale=noise_scale, psi=psi, seed=seed, apply_to=apply_to,
                       relative_noise=relative_noise)
        return (m,)


class ZQXLowFreqNoise:
    DESCRIPTION = ("NOISE for SamplerCustomAdvanced whose low-frequency band comes from a real photo latent "
                   "(composition / lighting prior without copying detail); unit variance preserved.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "reference": ("LATENT", {"tooltip": "VAE-encoded composition reference (any resolution; resized)."}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "alpha: 0 = plain noise."}),
            "cutoff": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.005,
                                 "tooltip": "Normalised cutoff D0 (1 = Nyquist). FreeInit default 0.25; lower = only the coarsest layout."}),
            "filter": (["gaussian", "butterworth", "ideal"], {"default": "gaussian"}),
            "butterworth_order": ("INT", {"default": 4, "min": 1, "max": 16}),
        }, "optional": {"base_noise": ("NOISE", {"tooltip": "Optional NOISE to modify instead of seeded Gaussian noise."})}}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get"
    CATEGORY = CAT_SAMPLING

    def get(self, noise_seed, reference, strength, cutoff, filter, butterworth_order, base_noise=None):
        from ..patches.noise import LowFreqNoise
        ref = reference["samples"]
        if ref.shape[0] != 1:
            raise ValueError("ZQX Low-Frequency Noise: reference latent must have batch size 1")
        return (LowFreqNoise(noise_seed, ref.clone(), strength, cutoff, filter, butterworth_order, base_noise),)


class ZQXSigmaSplitGuider:
    DESCRIPTION = ("GUIDER: cfg_early for sigma >= switch_sigma, cfg_late below (interval guidance, arXiv:2404.07724); "
                   "optionally a different base model for the early steps (Gandikota & Bau, arXiv:2503.10637).")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "switch_sigma": sigma_input(0.8, "Steps with sigma >= this are 'early'."),
            "cfg_early": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 30.0, "step": 0.05}),
            "cfg_late": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.05}),
        }, "optional": {"model_early": ("MODEL", {"tooltip": "A different base model (same latent space and text encoder) for the early steps."})}}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get"
    CATEGORY = CAT_GUIDANCE

    def get(self, model, positive, negative, switch_sigma, cfg_early, cfg_late, model_early=None):
        from ..patches.guider import SigmaSplitGuider
        g = SigmaSplitGuider(model, switch_sigma, cfg_early, cfg_late, model_early=model_early)
        g.set_conds(positive, negative)
        return (g,)


class ZQXSigmasToText:
    DESCRIPTION = "Print the sigmas of a schedule (to choose sigma windows for both passes)."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sigmas": ("SIGMAS",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = CAT_TOOLS
    OUTPUT_NODE = True

    def run(self, sigmas):
        s = [float(x) for x in sigmas.flatten().tolist()]
        txt = "\n".join(f"step {i}: sigma = {v:.4f}" for i, v in enumerate(s[:-1])) + f"\nend: sigma = {s[-1]:.4f}"
        return {"ui": {"text": [txt]}, "result": (txt,)}


class ZQXBlockSpec:
    DESCRIPTION = "Helper for block ablations: builds a block list / block-weight string for one block index."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "index": ("INT", {"default": 0, "min": 0, "max": 511}),
            "width": ("INT", {"default": 1, "min": 1, "max": 512, "tooltip": "Number of consecutive blocks."}),
            "total_blocks": ("INT", {"default": 60, "min": 1, "max": 512, "tooltip": "Qwen-Image: 60, Z-Image: 30."}),
            "mode": (["only", "except"], {"default": "only"}),
            "weight": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05,
                                 "tooltip": "Block-weight value used for the selected ('except') or unselected ('only') blocks."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("block_list", "block_weights")
    FUNCTION = "run"
    CATEGORY = CAT_TOOLS

    def run(self, index, width, total_blocks, mode, weight):
        a = index
        b = min(index + width - 1, total_blocks - 1)
        if a >= total_blocks:
            raise ValueError("index >= total_blocks")
        sel = f"{a}-{b}" if b > a else f"{a}"
        if mode == "only":
            block_list = sel
            block_weights = f"0-{total_blocks - 1}:{weight}, {sel}:1"
        else:
            parts = []
            if a > 0:
                parts.append(f"0-{a - 1}" if a - 1 > 0 else "0")
            if b < total_blocks - 1:
                parts.append(f"{b + 1}-{total_blocks - 1}" if total_blocks - 1 > b + 1 else f"{b + 1}")
            block_list = ", ".join(parts) if parts else ""
            block_weights = f"{sel}:{weight}"
        return (block_list, block_weights)


class ZQXDiTPAG:
    DESCRIPTION = ("Perturbed-attention guidance for Qwen-Image and Z-Image (PAG, arXiv 2403.17377): an extra forward "
                   "in which the selected blocks use an identity attention map for image queries; "
                   "out += scale * (out - out_perturbed). Works at CFG 1 (unlike CFG tweaks). Core PAG is UNet-only.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.05,
                                "tooltip": "Applied to the cond rows before CFG: with CFG > 1 the effective PAG scale is scale * cfg."}),
            "sigma_start": sigma_input(1.0, "Active while sigma <= sigma_start."),
            "sigma_end": sigma_input(0.0, "Active while sigma >= sigma_end."),
            "blocks": ("STRING", {"default": "mid", "tooltip": "'mid' = the middle block, or a list like '10-14'."}),
            "apply_to": (["cond_only", "all"], {"default": "cond_only"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CAT_GUIDANCE

    def apply(self, model, scale, sigma_start, sigma_end, blocks, apply_to):
        from ..patches.guidance import install_pag
        m, _ = install_pag(model, scale=scale, sigma_start=sigma_start, sigma_end=sigma_end, blocks=blocks, apply_to=apply_to)
        return (m,)
