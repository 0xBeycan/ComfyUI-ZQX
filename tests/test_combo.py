"""Chained patches: every multi-pass patch together, in several orders, on both models."""
import pytest
import torch

from conftest import requires_comfy

pytestmark = requires_comfy


def _stuff(kind):
    import lora_utils as lu
    import tiny_models as tm
    if kind == "qwen":
        return (tm.qwen_image(2), tm.qwen_cond, tm.qwen_latent, lu.make_lora("qwen", seed=3), lu.make_lora("qwen", seed=4), 2.0)
    return (tm.z_image(2), tm.zimage_cond, tm.zimage_latent, lu.make_lora("zimage", seed=3), lu.make_lora("zimage", seed=4), 1.0)


def _ref(p, ref, **kw):
    from zqx.patches.reference_attention import RefAttnConfig, install
    args = dict(weight=1.0, sigma_start=1.0, sigma_end=0.0)
    args.update(kw)
    return install(p, RefAttnConfig(ref_latent=ref, **args))


def _steer(p, a, b):
    from zqx.patches.steering import install
    return install(p, towards=a, away=b, alpha=0.5, sigma_start=1.0, sigma_end=0.0, blocks="all", mode="mean", apply_to="all")


def _pag(p):
    from zqx.patches.guidance import install_pag
    return install_pag(p, scale=1.0, sigma_start=1.0, sigma_end=0.0, blocks="0", apply_to="cond_only")


def _spatial(p, sd):
    from zqx.patches.spatial_lora import install
    mask = torch.zeros(16, 16)
    mask[4:12, 4:12] = 1
    m, patch, _ = install(p, sd, False, strength_early=1.0, strength_late=1.0, sigma_hi=1.0, sigma_lo=0.0, mask=mask,
                          invert=True, text_weight=1.0, nonspatial_weight=1.0, other_weight=1.0, key="zqx_combo_spatial")
    return m, patch


def _lg(p, sd):
    from zqx.adapters import get_adapter
    from zqx.patches.guidance import install_lora_guidance
    from zqx.patches.lora_schedules import scheduled_entries
    from zqx.patches.runtime_lora import load_lora_patches
    pk, _ = load_lora_patches(p, sd)
    e, _ = scheduled_entries(get_adapter(p), pk, 1.0, 1.0, 1.0, 0.0, "")
    m, patch, _ = install_lora_guidance(p, e, "zqx_combo_lg", 0.5, 1.5, 0.9, 0.5)
    return m, patch


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
@pytest.mark.parametrize("order", ["ref_first", "ref_last"])
@pytest.mark.parametrize("position", ["frame", "matched"])
def test_everything_chained_runs(kind, order, position):
    import tiny_models as tm
    from zqx.patches.cads import install as cads
    p, mkc, mkl, sd1, sd2, cfg = _stuff(kind)
    ref = mkl(12, 10, seed=9)                     # reference at a different resolution than the target
    steps = [lambda m: _spatial(m, sd1)[0], lambda m: _lg(m, sd2)[0], lambda m: _pag(m)[0],
             lambda m: _steer(m, mkc(5), mkc(6, n_txt=9))[0],
             lambda m: cads(m, tau1=0.5, tau2=0.9, noise_scale=0.2, psi=1.0, seed=0)[0]]
    ref_step = lambda m: _ref(m, ref, position_mode=position, match_threshold=0.0)[0]
    steps = [ref_step] + steps if order == "ref_first" else steps + [ref_step]
    m = p
    for s in steps:
        m = s(m)
    pos, neg = mkc(1), mkc(2, n_txt=5)
    for denoise, lat in ((1.0, torch.zeros_like(mkl(8, 8))), (0.6, mkl(8, 8))):
        a = tm.sample(m, pos, neg, lat, steps=3, cfg=cfg, denoise=denoise)
        b = tm.sample(m, pos, neg, lat, steps=3, cfg=cfg, denoise=denoise)
        assert torch.isfinite(a).all() and torch.equal(a, b)


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_reference_capture_not_contaminated_by_outer_patches(kind):
    """The reference K/V captured inside steering / PAG / LoRA-guidance / spatial LoRA must be the reference
    features of the model itself: steering and PAG must not act on the reference pass."""
    import tiny_models as tm
    p, mkc, mkl, sd1, sd2, cfg = _stuff(kind)
    ref = mkl(12, 10, seed=9)
    x = mkl(8, 8, seed=2)
    ctx = mkc(1)[0][0]
    m_a, ref_a = _ref(p, ref)
    tm.direct_call(m_a, x, 0.7, ctx)
    store_a = ref_a._last_store
    for outer in (lambda m: _steer(m, mkc(5), mkc(6))[0], lambda m: _pag(m)[0]):
        m_b, ref_b = _ref(outer(p), ref)         # outer patch installed first -> outermost wrapper
        tm.direct_call(m_b, x, 0.7, ctx)
        assert sorted(ref_b._last_store) == sorted(store_a)
        for bi in store_a:
            assert torch.equal(ref_b._last_store[bi][0], store_a[bi][0])
            assert torch.equal(ref_b._last_store[bi][1], store_a[bi][1])


def test_pass_stack_rules():
    from zqx.patches import passes
    a, b = object(), object()
    assert passes.blocked(a)                                  # outside its wrapper
    with passes.entry("enter", a, (1,), {}):
        assert not passes.blocked(a)
        with passes.entry("foreign", a):
            assert not passes.blocked(a)                      # own foreign pass
        with passes.entry("enter", b, (2,), {}):
            with passes.entry("variant", b):
                assert passes.blocked(a) and not passes.blocked(a, ("foreign",))
                assert not passes.blocked(b)
            with passes.entry("foreign", b, (3,), {}):
                assert passes.blocked(a, ("foreign",))
                assert passes.current_call()[0] == (3,)
    assert passes.current_call() == (None, None, 0)


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_reference_capture_identical_for_inner_activation_patches(kind):
    """Same as above with steering / PAG installed *after* the reference patch (inner wrappers): they must stay
    passive in the reference capture pass, so the captured features are unchanged."""
    import tiny_models as tm
    p, mkc, mkl, sd1, sd2, cfg = _stuff(kind)
    ref = mkl(12, 10, seed=9)
    x = mkl(8, 8, seed=2)
    ctx = mkc(1)[0][0]
    m_a, ref_a = _ref(p, ref)
    tm.direct_call(m_a, x, 0.7, ctx)
    for inner in (lambda m: _steer(m, mkc(5), mkc(6))[0], lambda m: _pag(m)[0]):
        m_b, ref_b = _ref(p, ref)
        m_b = inner(m_b)
        tm.direct_call(m_b, x, 0.7, ctx)
        for bi in ref_a._last_store:
            assert torch.equal(ref_b._last_store[bi][0], ref_a._last_store[bi][0])
