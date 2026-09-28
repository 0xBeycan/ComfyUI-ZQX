import math

import pytest
import torch

from conftest import requires_comfy

pytestmark = requires_comfy


def _install(patcher, ref, **kw):
    from zqx.patches.reference_attention import RefAttnConfig, install
    cfg = RefAttnConfig(ref_latent=ref, **kw)
    return install(patcher, cfg)


# ----------------------------------------------------------------------------------------------
# RoPE: rotation composition with the model's own embedder
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_rope_rotation_composition(kind):
    import tiny_models as tm
    from comfy.ldm.flux.math import apply_rope1
    from zqx.adapters import get_adapter
    from zqx.core.rope import apply_rotation
    p = tm.qwen_image() if kind == "qwen" else tm.z_image()
    ad = get_adapter(p)
    g = torch.Generator().manual_seed(0)
    n, heads = 37, 3
    d = sum(ad.embedder.axes_dim)
    ids = torch.randint(-20, 60, (n, 3), generator=g).float()
    delta = (3.0, -5.0, 11.0)
    k = torch.randn(2, heads, n, d, generator=g)
    f_p = ad.rope_freqs(ids)                                   # (1, 1, N, D/2, 2, 2)
    f_pd = ad.rope_freqs(ids + torch.tensor(delta))
    r_d = ad.rotation_for_offset(delta, "cpu")
    k_p = apply_rope1(k, f_p)                                  # the model's own RoPE application
    lhs = apply_rope1(k, f_pd)                                 # R(p + delta) k
    rhs = apply_rotation(k_p, r_d)                             # R(delta) R(p) k
    # float32 angle computation: |pos * omega| up to ~70 rad -> ~1e-5 relative angle error
    assert torch.allclose(lhs, rhs, atol=5e-5, rtol=0), (lhs - rhs).abs().max()
    # offset 0 is the exact identity
    r0 = ad.rotation_for_offset((0.0, 0.0, 0.0), "cpu")
    assert torch.equal(apply_rotation(k_p, r0), k_p)
    # our apply_rotation == comfy's apply_rope1 on the same freqs
    assert torch.allclose(apply_rotation(k, f_p), k_p, atol=1e-6)


# ----------------------------------------------------------------------------------------------
# Patched attention == naive float64 formula
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("with_mask", [False, True])
def test_override_matches_naive_formula(with_mask):
    import tiny_models as tm
    from comfy.ldm.modules.attention import attention_pytorch, attention_basic
    from zqx.adapters import TokenLayout, get_adapter
    from zqx.core.attention import naive_attention
    from zqx.core.rope import apply_rotation
    from zqx.patches.reference_attention import RefAttnConfig, ReferenceAttentionPatch, _CallState

    p = tm.qwen_image()
    ad = get_adapter(p)
    g = torch.Generator().manual_seed(1)
    b, h, d = 2, 2, 32
    n_txt, n_img, n_ref = 5, 12, 9
    n = n_txt + n_img
    q = torch.randn(b, h, n, d, generator=g)
    k = torch.randn(b, h, n, d, generator=g)
    v = torch.randn(b, h, n, d, generator=g)
    kr = torch.randn(b, h, n_ref, d, generator=g)
    vr = torch.randn(b, h, n_ref, d, generator=g)
    w = 0.7
    cfg = RefAttnConfig(ref_latent=torch.zeros(1, 16, 1, 4, 4), weight=w)
    patch = ReferenceAttentionPatch(ad, cfg)
    delta = (1.0, 2.0, -3.0)
    rot = ad.rotation_for_offset(delta, "cpu")
    qm = torch.rand(b, n_img, generator=g)
    qm[0, :3] = 0.0
    km = torch.rand(1, n_ref, generator=g)
    km[0, 0] = 0.0
    st = _CallState(mode="inject", blocks=None, layout=TokenLayout(n_img=n_img, h_tok=3, w_tok=4, img_start=n_txt, seq_len=n),
                    rot=rot, store={0: (kr, vr)}, qmask_img=qm, kmask=km)
    patch.state = st
    mask = None
    if with_mask:  # Qwen-style additive text padding mask (B, 1, N)
        mask = torch.zeros(b, 1, n)
        mask[1, 0, 1] = -1e4
    for func in (attention_pytorch, attention_basic):
        out = patch.attention_override(func, q, k, v, h, mask, skip_reshape=True, skip_output_reshape=True,
                                       transformer_options={"block_index": 0}, _inside_attn_wrapper=True)
        # naive oracle
        kr_rot = apply_rotation(kr, rot)
        full_q = torch.zeros(b, n)
        full_q[:, n_txt:] = qm
        bias_ref = torch.log(w * full_q[:, :, None] * km[:, None, :])[:, None]
        base = torch.zeros(b, 1, n, n) if mask is None else mask[:, :, None, :].expand(b, 1, n, n)
        bias = torch.cat([base, bias_ref], dim=-1)
        ref = naive_attention(q, torch.cat([k, kr_rot], 2), torch.cat([v, vr], 2), bias)
        assert torch.allclose(out.double(), ref, atol=2e-5), (func.__name__, (out.double() - ref).abs().max())
    # rows with zero query mask equal plain attention without the reference
    plain = naive_attention(q, k, v, None if mask is None else mask[:, :, None, :].expand(b, 1, n, n))
    assert torch.allclose(out.double()[:, :, :n_txt], plain[:, :, :n_txt], atol=2e-5)
    assert torch.allclose(out.double()[0, :, n_txt:n_txt + 3], plain[0, :, n_txt:n_txt + 3], atol=2e-5)


# ----------------------------------------------------------------------------------------------
# Identity invariants through the real sampler (txt2img and img2img)
# ----------------------------------------------------------------------------------------------
def _setup(kind, layers=2):
    import tiny_models as tm
    if kind == "qwen":
        return tm.qwen_image(num_layers=layers), tm.qwen_cond(1), tm.qwen_cond(2), tm.qwen_latent(12, 16, seed=3), tm.qwen_latent(8, 10, seed=4), 2.5
    return tm.z_image(num_layers=layers), tm.zimage_cond(1), tm.zimage_cond(2), tm.zimage_latent(12, 16, seed=3), tm.zimage_latent(8, 10, seed=4), 1.0


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
@pytest.mark.parametrize("denoise", [1.0, 0.6])
def test_disabled_settings_are_bitwise_identity(kind, denoise):
    import tiny_models as tm
    p, pos, neg, lat, ref, cfg = _setup(kind)
    lat_in = lat if denoise < 1.0 else torch.zeros_like(lat)
    base = tm.sample(p, pos, neg, lat_in, steps=4, cfg=cfg, denoise=denoise)
    zero_mask = torch.zeros(20, 20)
    variants = [
        dict(weight=0.0),
        dict(weight=1.0, sigma_start=0.05, sigma_end=0.0),   # no step reaches sigma <= 0.05 except none
        dict(weight=1.0, query_mask=zero_mask),
        dict(weight=1.0, key_mask=zero_mask),
    ]
    for kw in variants:
        m, patch = _install(p, ref, **kw)
        out = tm.sample(m, pos, neg, lat_in, steps=4, cfg=cfg, denoise=denoise)
        assert torch.equal(out, base), kw
        assert patch.calls_patched == 0
    # sanity: an active patch changes the result
    m, patch = _install(p, ref, weight=1.0)
    out = tm.sample(m, pos, neg, lat_in, steps=4, cfg=cfg, denoise=denoise)
    assert patch.calls_patched > 0
    assert not torch.allclose(out, base, atol=1e-4)


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_window_is_sigma_space(kind):
    """With a window [0.5, 0.0] only calls whose sigma <= 0.5 are patched, in both passes."""
    import tiny_models as tm
    p, pos, neg, lat, ref, cfg = _setup(kind)
    m, patch = _install(p, ref, weight=1.0, sigma_start=0.5, sigma_end=0.0)
    seen = []
    orig = patch.diffusion_model_wrapper

    def spy(executor, *a, **k):
        to = patch.adapter.get_transformer_options(a, k)
        seen.append(float(to["sigmas"].flatten()[0]))
        return orig(executor, *a, **k)
    import comfy.patcher_extension as pe
    m.remove_wrappers_with_key(pe.WrappersMP.DIFFUSION_MODEL, patch.WRAPPER_KEY)
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, patch.WRAPPER_KEY, spy)
    tm.sample(m, pos, neg, torch.zeros_like(lat), steps=6, cfg=cfg, denoise=1.0)
    n_in = sum(1 for s in seen if s <= 0.5)
    assert n_in > 0 and n_in < len(seen)
    # each in-window call is patched once (cond+uncond are batched into one call)
    assert patch.calls_patched == n_in


# ----------------------------------------------------------------------------------------------
# Token spans: captured reference K/V are exactly the model's own image-token K/V
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,hw,with_qie", [("qwen", (10, 14), False), ("qwen", (10, 14), True),
                                               ("zimage", (10, 14), False), ("zimage", (16, 16), False)])
def test_spans_capture_equals_target_tokens(kind, hw, with_qie):
    """If the reference input equals the target input (same latent, same noise, same sigma), the captured
    reference K/V must equal the target's own image-token K/V slice in every block.  Checks the span logic
    (Qwen: text first; QIE: native ref tokens after the target; Z-Image: caption padding, image padding,
    refiner calls excluded)."""
    import tiny_models as tm
    from zqx.patches import reference_attention as ra
    p = tm.qwen_image(num_layers=3) if kind == "qwen" else tm.z_image(num_layers=3)
    ctx = (tm.qwen_cond if kind == "qwen" else tm.zimage_cond)(1, n_txt=7)[0][0]
    ref = (tm.qwen_latent if kind == "qwen" else tm.zimage_latent)(*hw, seed=5)
    m, patch = _install(p, ref, weight=1.0, position_mode="same")
    sigma = 0.7
    x = patch._ref_input(torch.zeros_like(ref), sigma)   # x == reference input exactly
    extra = {}
    if with_qie:
        extra["ref_latents"] = [tm.qwen_latent(6, 8, seed=9)]
        extra["ref_latents_method"] = "index"

    # 1) patched call: the capture pass runs the reference (== x) through the unpatched attention
    tm.direct_call(m, x, sigma, ctx, **extra)
    captured = dict(patch._last_store)
    # 2) plain model, spy on its own attention calls and slice with the adapter's layout
    from zqx.adapters import get_adapter
    ad = get_adapter(p)
    seen = {}
    lay_holder = {}

    def wrapper(executor, *a, **k):
        to = ad.get_transformer_options(a, k)
        to.pop("block_index", None)
        lay_holder["lay"] = ad.layout(ad.get_x(a, k), a, k)
        return executor(*a, **k)

    def spy(func, *args, **kwargs):
        to = kwargs.get("transformer_options")
        if to.get("block_index") is not None:
            k, v = args[1], args[2]
            s0, s1 = lay_holder["lay"].span(k.shape[2])
            seen[to["block_index"]] = (k[:, :, s0:s1].clone(), v[:, :, s0:s1].clone(), k.shape[2])
        return func(*args, **kwargs)
    import comfy.patcher_extension as pe
    q = p.clone()
    q.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = spy
    q.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "spy", wrapper)
    tm.direct_call(q, x, sigma, ctx, **extra)
    assert sorted(seen.keys()) == [0, 1, 2] == sorted(captured.keys())   # main blocks only (no Z-Image refiners)
    for bi, (k_t, v_t, n) in seen.items():
        k_r, v_r = captured[bi]
        assert torch.equal(k_t, k_r) and torch.equal(v_t, v_r), bi
    n_img = math.ceil(hw[0] / 2) * math.ceil(hw[1] / 2)
    if kind == "qwen":
        n_q = (3 * 4 if with_qie else 0)
        assert n == 7 + n_img + n_q
    else:
        assert n == 32 + n_img + ((-n_img) % 32)              # caption padded to 32, image padded to 32
    assert patch.last_trace == [("capture", 0), ("capture", 1), ("capture", 2), ("inject", 0), ("inject", 1), ("inject", 2)]


# ----------------------------------------------------------------------------------------------
# Batched cond/uncond do not leak into each other
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
@pytest.mark.parametrize("mode", ["noised", "cached"])
def test_cond_uncond_batch_isolation(kind, mode):
    import tiny_models as tm
    p = tm.qwen_image(num_layers=2) if kind == "qwen" else tm.z_image(num_layers=2)
    mk_c = tm.qwen_cond if kind == "qwen" else tm.zimage_cond
    mk_l = tm.qwen_latent if kind == "qwen" else tm.zimage_latent
    c1, c2 = mk_c(1)[0][0], mk_c(2)[0][0]
    ref = mk_l(8, 8, seed=5)
    x = mk_l(12, 12, seed=6, batch=2)
    m, patch = _install(p, ref, weight=1.3, capture_mode=mode, cache_sigma=0.0)
    both = tm.direct_call(m, x, 0.6, torch.cat([c1, c2]), cond_or_uncond=[0, 1])
    patch.clear_cache()
    a = tm.direct_call(m, x[:1], 0.6, c1, cond_or_uncond=[0])
    b = tm.direct_call(m, x[1:], 0.6, c2, cond_or_uncond=[1])
    assert torch.allclose(both, torch.cat([a, b]), atol=1e-5), (both - torch.cat([a, b])).abs().max()


def test_inject_uncond_false_leaves_uncond_rows_untouched():
    import tiny_models as tm
    p = tm.qwen_image(num_layers=2)
    c1, c2 = tm.qwen_cond(1)[0][0], tm.qwen_cond(2)[0][0]
    ref = tm.qwen_latent(8, 8, seed=5)
    x = tm.qwen_latent(12, 12, seed=6, batch=2)
    base = tm.direct_call(p, x, 0.6, torch.cat([c1, c2]), cond_or_uncond=[0, 1])
    m, patch = _install(p, ref, weight=1.0, inject_uncond=False)
    out = tm.direct_call(m, x, 0.6, torch.cat([c1, c2]), cond_or_uncond=[0, 1])
    assert torch.allclose(out[1], base[1], atol=1e-6)
    assert not torch.allclose(out[0], base[0], atol=1e-4)


# ----------------------------------------------------------------------------------------------
# End-to-end equivalence with Qwen-Image-Edit's native reference concatenation
# ----------------------------------------------------------------------------------------------
def test_matches_native_qie_reference_concat_single_block():
    """1-block Qwen model: injecting the captured reference with position 'frame' (+1 on the frame axis)
    and weight 1 must equal the model's own QIE 'index' ref_latents path for the target tokens, because
    in block 0 the reference tokens' K/V depend only on the reference itself and the timestep embedding,
    and QIE places ref #1 at frame index 1 with the same centred h/w grid."""
    import tiny_models as tm
    p = tm.qwen_image(num_layers=1)
    ctx = tm.qwen_cond(1)[0][0]
    sigma = 0.6
    R = tm.qwen_latent(10, 12, seed=7)               # reference in *model* space
    x = tm.qwen_latent(12, 16, seed=8)
    base_model = p.model
    raw = base_model.process_latent_out(R / (1.0 - sigma))
    m, patch = _install(p, raw, weight=1.0, position_mode="frame")
    patch._noise = torch.zeros_like(R)               # x_ref = (1 - s) * R/(1 - s) + s * 0 = R
    ours = tm.direct_call(m, x, sigma, ctx)
    native = tm.direct_call(p, x, sigma, ctx, ref_latents=[R], ref_latents_method="index")
    assert torch.allclose(ours, native, atol=2e-5), (ours - native).abs().max()
    # and it is not trivially equal to the unpatched output
    plain = tm.direct_call(p, x, sigma, ctx)
    assert not torch.allclose(plain, native, atol=1e-4)


# ----------------------------------------------------------------------------------------------
# Determinism, dtype, different reference resolution, cached mode reuse
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_determinism_and_resolution(kind):
    import tiny_models as tm
    p, pos, neg, lat, ref, cfg = _setup(kind)
    m, _ = _install(p, ref, weight=1.0, token_dropout=0.3, noise_seed=4, position_mode="right")
    a = tm.sample(m, pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg)
    b = tm.sample(m, pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_bf16_runs_and_tracks_fp32(kind):
    import tiny_models as tm
    p32 = tm.qwen_image(num_layers=2) if kind == "qwen" else tm.z_image(num_layers=2)
    p16 = tm.qwen_image(num_layers=2, dtype=torch.bfloat16) if kind == "qwen" else tm.z_image(num_layers=2, dtype=torch.bfloat16)
    mk_c = tm.qwen_cond if kind == "qwen" else tm.zimage_cond
    mk_l = tm.qwen_latent if kind == "qwen" else tm.zimage_latent
    ctx, ref, x = mk_c(1)[0][0], mk_l(8, 8, seed=5), mk_l(12, 12, seed=6)
    m32, _ = _install(p32, ref, weight=1.0)
    m16, _ = _install(p16, ref, weight=1.0)
    o32 = tm.direct_call(m32, x, 0.6, ctx)
    o16 = tm.direct_call(m16, x, 0.6, ctx)
    assert torch.isfinite(o16).all()
    rel = (o16.float() - o32).norm() / o32.norm()
    assert rel < 0.05, rel


def test_cached_mode_captures_once_per_run():
    import tiny_models as tm
    p, pos, neg, lat, ref, cfg = _setup("qwen")
    m, patch = _install(p, ref, weight=1.0, capture_mode="cached", cache_sigma=0.0)
    tm.sample(m, pos, neg, torch.zeros_like(lat), steps=4, cfg=cfg)
    caps = [t for t in patch.last_trace if t[0] == "capture"]
    assert caps == []                                  # last step reused the cache
    assert len(patch._cache) == 0                      # cleared by ON_CLEANUP after the run


def test_refuses_unsupported():
    import tiny_models as tm
    from zqx.adapters import AdapterError
    p = tm.qwen_image()
    with pytest.raises(ValueError):
        _install(p, tm.qwen_latent(8, 8), blocks="0-5")          # only 2 blocks
    m, _ = _install(p, tm.qwen_latent(8, 8))
    with pytest.raises(ValueError):
        _install(m, tm.qwen_latent(8, 8))                         # double install


def test_zimage_fused_rope_kernel_matches_rotation_composition():
    """Z-Image applies RoPE with comfy_kitchen's fused rms_rope kernel.  Shifting the model's own image positions
    by delta (rope_options shift_x / shift_y) must rotate the block-0 keys exactly like R(delta) does
    (no refiner layers, so block-0 inputs are identical in both runs)."""
    import comfy.supported_models as sm
    import tiny_models as tm
    from zqx.adapters import get_adapter
    from zqx.core.rope import apply_rotation
    cfg = dict(image_model="lumina2", patch_size=2, in_channels=16, dim=256, cap_feat_dim=32, n_layers=1,
               n_refiner_layers=0, qk_norm=True, n_heads=4, n_kv_heads=4, axes_dims=[16, 24, 24],
               axes_lens=[1536, 512, 512], rope_theta=256.0, ffn_dim_multiplier=8.0 / 3.0,
               z_image_modulation=True, time_scale=1000.0, pad_tokens_multiple=32)
    mc = sm.ZImage(cfg)
    mc.set_inference_dtype(torch.float32, None)
    model = mc.get_model({}, device="cpu")
    tm._init(model, 0)
    p = tm.make_patcher(model)
    ad = get_adapter(p)
    x = tm.zimage_latent(8, 12, seed=2)
    ctx = tm.zimage_cond(1)[0][0]
    keys = {}

    def run(shift_x, shift_y, tag):
        q = p.clone()

        def spy(func, *args, **kwargs):
            to = kwargs["transformer_options"]
            if to.get("block_index") == 0:
                k = args[1]
                n_img = 4 * 6
                start = k.shape[2] - (n_img + ((-n_img) % 32))
                keys[tag] = k[:, :, start:start + n_img].clone()
            return func(*args, **kwargs)
        to = q.model_options.setdefault("transformer_options", {})
        to["optimized_attention_override"] = spy
        to["rope_options"] = {"scale_x": 1.0, "shift_x": shift_x, "scale_y": 1.0, "shift_y": shift_y}
        tm.direct_call(q, x, 0.6, ctx)

    run(0.0, 0.0, "base")
    run(5.0, 3.0, "shift")
    rot = ad.rotation_for_offset((0.0, 3.0, 5.0), "cpu")
    moved = apply_rotation(keys["base"], rot)
    assert torch.allclose(moved, keys["shift"], atol=5e-5), (moved - keys["shift"]).abs().max()
    assert not torch.allclose(keys["base"], keys["shift"], atol=1e-3)
