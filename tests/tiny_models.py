"""Tiny *real* ComfyUI models (random weights) wrapped in the real ModelPatcher.

Built through comfy.supported_models (the same config classes ComfyUI uses
for detection), so model_base, latent formats, model_sampling (flow, shift)
and extra_conds are the production ones; only the sizes are tiny.
"""
import torch


def _init(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            if p.ndim == 1 and "norm" in name:
                p.add_(1.0)  # RMSNorm/LayerNorm gains around 1


def make_patcher(model):
    import comfy.model_patcher
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def qwen_image(num_layers=2, seed=0, dtype=torch.float32, heads=2, head_dim=32):
    import comfy.supported_models as sm
    cfg = dict(image_model="qwen_image", patch_size=2, in_channels=64, out_channels=16, num_layers=num_layers,
               attention_head_dim=head_dim, num_attention_heads=heads, joint_attention_dim=24,
               axes_dims_rope=(8, 12, 12) if head_dim == 32 else (head_dim // 4, 3 * head_dim // 8, 3 * head_dim // 8))
    mc = sm.QwenImage(cfg)
    mc.set_inference_dtype(dtype, None)
    m = mc.get_model({}, device="cpu")
    _init(m, seed)
    m.to(dtype)
    return make_patcher(m)


def z_image(num_layers=2, seed=0, dtype=torch.float32):
    import comfy.supported_models as sm
    cfg = dict(image_model="lumina2", patch_size=2, in_channels=16, dim=256, cap_feat_dim=32, n_layers=num_layers,
               n_refiner_layers=1, qk_norm=True, n_heads=4, n_kv_heads=4, axes_dims=[16, 24, 24],
               axes_lens=[1536, 512, 512], rope_theta=256.0, ffn_dim_multiplier=8.0 / 3.0,
               z_image_modulation=True, time_scale=1000.0, pad_tokens_multiple=32)
    mc = sm.ZImage(cfg)
    mc.set_inference_dtype(dtype, None)
    m = mc.get_model({}, device="cpu")
    _init(m, seed)
    m.to(dtype)
    return make_patcher(m)


def qwen_cond(seed=1, n_txt=7, batch=1):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(batch, n_txt, 24, generator=g), {}]]


def zimage_cond(seed=1, n_txt=7, batch=1):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(batch, n_txt, 32, generator=g), {}]]


def qwen_latent(h=16, w=16, seed=3, batch=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 16, 1, h, w, generator=g)


def zimage_latent(h=16, w=16, seed=3, batch=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 16, h, w, generator=g)


def sample(patcher, positive, negative, latent, steps=4, cfg=1.0, denoise=1.0, seed=11, sampler="euler",
           scheduler="simple"):
    """Drive the real comfy.sample.sample path (what KSampler uses)."""
    import comfy.sample
    noise = comfy.sample.prepare_noise(latent, seed)
    return comfy.sample.sample(patcher, noise, steps, cfg, sampler, scheduler, positive, negative, latent,
                               denoise=denoise, disable_pbar=True, seed=seed)


def direct_call(patcher, x, sigma, context, cond_or_uncond=None, **extra):
    """Call BaseModel.apply_model the way comfy.samplers._calc_cond_batch does (transformer_options with
    the patcher's wrappers, 'sigmas', 'cond_or_uncond'), without a sampler.  Returns the denoised output."""
    import comfy.patcher_extension
    to = comfy.patcher_extension.copy_nested_dicts(patcher.model_options.get("transformer_options", {}))
    to["wrappers"] = comfy.patcher_extension.copy_nested_dicts(patcher.wrappers)
    to["callbacks"] = comfy.patcher_extension.copy_nested_dicts(patcher.callbacks)
    b = x.shape[0]
    sig = torch.full((b,), float(sigma), dtype=torch.float32)
    to["sigmas"] = sig[:1] if cond_or_uncond is None else sig[: max(1, b // len(cond_or_uncond))]
    to["cond_or_uncond"] = cond_or_uncond if cond_or_uncond is not None else [0]
    patcher.model.current_patcher = patcher
    import comfy.model_base
    if isinstance(patcher.model, comfy.model_base.Lumina2) and "num_tokens" not in extra:
        extra["num_tokens"] = context.shape[1]
    return patcher.model.apply_model(x, sig, c_crossattn=context, transformer_options=to, **extra)
