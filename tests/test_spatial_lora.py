import pytest
import torch

from conftest import requires_comfy

pytestmark = requires_comfy


def _setup(kind):
    import lora_utils as lu
    import tiny_models as tm
    if kind == "qwen":
        # text-stream, image-stream and a non-spatial (img_mod) module
        sd = lu.make_lora("qwen", seed=3, alpha=2.0)
        g = torch.Generator().manual_seed(5)
        sd["diffusion_model.transformer_blocks.0.img_mod.1.lora_A.weight"] = torch.randn(4, 64, generator=g) * 0.05
        sd["diffusion_model.transformer_blocks.0.img_mod.1.lora_B.weight"] = torch.randn(384, 4, generator=g) * 0.05
        return tm.qwen_image(2), tm.qwen_cond(1), tm.qwen_cond(2), tm.qwen_latent(10, 12), 2.0, sd
    return tm.z_image(2), tm.zimage_cond(1), tm.zimage_cond(2), tm.zimage_latent(10, 12), 1.0, lu.make_lora("zimage", seed=3, alpha=2.0)


def _install(p, sd, **kw):
    from zqx.patches.spatial_lora import install
    args = dict(strength_early=0.8, strength_late=0.8, sigma_hi=1.0, sigma_lo=0.0, mask=None, invert=False,
                text_weight=1.0, nonspatial_weight=1.0, other_weight=1.0)
    args.update(kw)
    return install(p, sd, False, **args)


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
@pytest.mark.parametrize("denoise", [1.0, 0.6])
def test_all_ones_equals_loraloader(kind, denoise):
    import comfy.sd
    import tiny_models as tm
    p, pos, neg, lat, cfg, sd = _setup(kind)
    lat = lat if denoise < 1 else torch.zeros_like(lat)
    merged, _ = comfy.sd.load_lora_for_models(p, None, sd, 0.8, 0)
    ref = tm.sample(merged, pos, neg, lat, steps=3, cfg=cfg, denoise=denoise)
    m, patch, _ = _install(p, sd)
    out = tm.sample(m, pos, neg, lat, steps=3, cfg=cfg, denoise=denoise)
    assert patch.calls_patched > 0
    assert torch.allclose(out, ref, atol=5e-5), (out - ref).abs().max()


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_all_zero_is_bitwise_identity(kind):
    import tiny_models as tm
    p, pos, neg, lat, cfg, sd = _setup(kind)
    base = tm.sample(p, pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg)
    for kw in (dict(mask=torch.zeros(20, 24), text_weight=0.0, nonspatial_weight=0.0, other_weight=0.0),
               dict(strength_early=0.0, strength_late=0.0),
               dict(mask=torch.ones(20, 24), invert=True, text_weight=0.0, nonspatial_weight=0.0, other_weight=0.0)):
        m, patch, _ = _install(p, sd, **kw)
        assert torch.equal(tm.sample(m, pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg), base), kw
        assert patch.calls_patched == 0


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_token_weights_and_locality(kind):
    """Hook-level check on a real module: the added delta is x A^T B^T * s * scale on weighted tokens only."""
    import tiny_models as tm
    from zqx.patches.spatial_lora import install
    p, pos, neg, lat, cfg, sd = _setup(kind)
    mask = torch.zeros(10, 12)
    mask[:, :6] = 1.0                   # left half of the 5x6 token grid
    m, patch, _ = _install(p, sd, mask=mask, text_weight=0.0, other_weight=0.0)
    ad = patch.adapter
    x = lat
    layout = ad.layout(x, (x, None, pos[0][0], None, None, None, {}), {"transformer_options": {}})
    img_w = patch._image_weights(layout.n_img, layout.h_tok, layout.w_tok, "cpu")
    assert img_w.view(5, 6)[:, :3].eq(1).all() and img_w.view(5, 6)[:, 3:].eq(0).all()
    # pick an image-stream (Qwen) / joint (Z-Image) attention projection and call the hook directly
    mk = [k for k in patch.branches if ("attn.to_q" in k if kind == "qwen" else "layers.0.attention.qkv" in k)][0]
    stream = patch.streams[mk]
    n = layout.n_img if stream == "image" else 32 + layout.img_from_end
    xin = torch.randn(1, n, patch.branches[mk][0].down.shape[1])
    hook = patch._make_hook(mk, 0.8, {})
    mod = patch.modules[mk]
    base_out = mod(xin)
    from zqx.patches import passes
    with passes.entry("enter", patch, (x, None, pos[0][0], None, None, None, {}), {"transformer_options": {}}):
        out = hook(mod, (xin,), base_out)
    delta = out - base_out
    tw = patch.token_weights(stream, xin, layout, img_w)
    full = torch.zeros_like(base_out)
    for b in patch.branches[mk]:
        d = (xin @ b.down.T) @ b.up.T * (0.8 * b.scale)
        if b.offset is None:
            full += d
        else:
            full[..., b.offset[1]:b.offset[1] + b.offset[2]] += d
    assert torch.allclose(delta, full * tw.view(1, -1, 1), atol=1e-5)
    assert delta[0, tw == 0].abs().max() == 0


def test_realism_background_character_face_changes_output():
    import tiny_models as tm
    import lora_utils as lu
    p, pos, neg, lat, cfg, sd = _setup("zimage")
    face = torch.zeros(10, 12)
    face[2:6, 4:8] = 1
    m, _, _ = _install(p, sd, mask=face, invert=True, key="zqx_spatial_realism")
    m2, _, _ = _install(m, lu.make_lora("zimage", seed=9), mask=face, key="zqx_spatial_char")
    base = tm.sample(p, pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg)
    out = tm.sample(m2, pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg)
    assert torch.isfinite(out).all() and not torch.allclose(out, base, atol=1e-4)
