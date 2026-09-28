import pytest
import torch

from conftest import requires_comfy

pytestmark = requires_comfy


def _p(kind, layers=2):
    import tiny_models as tm
    if kind == "qwen":
        return tm.qwen_image(layers), tm.qwen_cond(1)[0][0], tm.qwen_cond(2)[0][0], tm.qwen_latent(10, 12)
    return tm.z_image(layers), tm.zimage_cond(1)[0][0], tm.zimage_cond(2)[0][0], tm.zimage_latent(10, 12)


def _lora_guid(p, kind, w_early, w_late, hi=1.0, lo=0.0, strength=0.9):
    import lora_utils as lu
    from zqx.adapters import get_adapter
    from zqx.patches.guidance import install_lora_guidance
    from zqx.patches.lora_schedules import scheduled_entries
    from zqx.patches.runtime_lora import load_lora_patches
    pk, _ = load_lora_patches(p, lu.make_lora(kind, seed=3))
    entries, _ = scheduled_entries(get_adapter(p), pk, strength, strength, 1.0, 0.0, "")
    return install_lora_guidance(p, entries, "zqx_lg_test", w_early, w_late, hi, lo)


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_lora_guidance_formula(kind):
    import tiny_models as tm
    p, c, _, x = _p(kind)
    base = tm.direct_call(p, x, 0.6, c)
    m1, g1, _ = _lora_guid(p, kind, 1.0, 1.0)
    lora = tm.direct_call(m1, x, 0.6, c)
    assert g1.calls_double == 0
    m0, _, _ = _lora_guid(p, kind, 0.0, 0.0)
    assert torch.equal(tm.direct_call(m0, x, 0.6, c), base)
    m2, g2, _ = _lora_guid(p, kind, 2.5, 2.5)
    out = tm.direct_call(m2, x, 0.6, c)
    assert g2.calls_double == 1
    assert torch.allclose(out, base + 2.5 * (lora - base), atol=1e-5)
    # sigma ramp: at sigma 0.9 (>= hi) w_early applies, at 0.1 (<= lo) w_late
    m3, _, _ = _lora_guid(p, kind, 0.0, 1.0, hi=0.8, lo=0.4)
    assert torch.equal(tm.direct_call(m3, x, 0.9, c), tm.direct_call(p, x, 0.9, c))
    assert torch.allclose(tm.direct_call(m3, x, 0.1, c), tm.direct_call(m1, x, 0.1, c), atol=1e-6)


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_lora_guidance_in_sampler(kind):
    import tiny_models as tm
    p, c, n, x = _p(kind)
    pos, neg = [[c, {}]], [[n, {}]]
    m, g, _ = _lora_guid(p, kind, 0.3, 1.5, hi=0.8, lo=0.5)
    out = tm.sample(m, pos, neg, torch.zeros_like(x), steps=4, cfg=2.0 if kind == "qwen" else 1.0)
    assert torch.isfinite(out).all() and g.calls_double > 0


def test_identity_image_rows():
    from zqx.patches.guidance import identity_image_rows
    b, h, n, d = 2, 3, 10, 4
    v = torch.randn(b, h, n, d)
    out = torch.randn(b, n, h * d)
    r = identity_image_rows(out, v, 4, 9, False)
    assert torch.equal(r[:, :4], out[:, :4]) and torch.equal(r[:, 9:], out[:, 9:])
    assert torch.equal(r[:, 4:9], v[:, :, 4:9].transpose(1, 2).reshape(b, 5, h * d))
    out4 = torch.randn(b, h, n, d)
    r4 = identity_image_rows(out4, v, 4, 9, True)
    assert torch.equal(r4[:, :, 4:9], v[:, :, 4:9]) and torch.equal(r4[:, :, :4], out4[:, :, :4])


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_pag_identity_and_formula(kind):
    import tiny_models as tm
    from zqx.patches.guidance import install_pag
    p, c, n, x = _p(kind, layers=3)
    base = tm.direct_call(p, x, 0.6, c)
    for kw in (dict(scale=0.0), dict(scale=2.0, sigma_start=0.3, sigma_end=0.0)):
        args = dict(scale=2.0, sigma_start=1.0, sigma_end=0.0, blocks="", apply_to="cond_only")
        args.update(kw)
        m, patch = install_pag(p, **args)
        assert torch.equal(tm.direct_call(m, x, 0.6, c), base)
        assert patch.calls_patched == 0
    m, patch = install_pag(p, scale=2.0, sigma_start=1.0, sigma_end=0.0, blocks="", apply_to="cond_only")
    out = tm.direct_call(m, x, 0.6, c)
    assert patch.perturbed_blocks == [1]          # middle of 3 blocks
    # perturbed prediction computed independently: model with the identity attention forced on block 1
    m2, patch2 = install_pag(p, scale=1.0, sigma_start=1.0, sigma_end=0.0, blocks="1", apply_to="cond_only")
    patch2.perturb = True
    import tiny_models as tm2
    from zqx.adapters import get_adapter
    ad = get_adapter(p)

    from zqx.patches import passes

    def pert_wrapper(executor, *a, **k):
        to = ad.get_transformer_options(a, k)
        to.pop("block_index", None)
        patch2.layout = ad.layout(ad.get_x(a, k), a, k)
        with passes.entry("enter", patch2, a, k):
            return executor(*a, **k)
    import comfy.patcher_extension as pe
    m2.remove_wrappers_with_key(pe.WrappersMP.DIFFUSION_MODEL, patch2.WRAPPER_KEY)
    m2.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "pert", pert_wrapper)
    pert = tm.direct_call(m2, x, 0.6, c)
    assert torch.allclose(out, base + 2.0 * (base - pert), atol=1e-5)
    # uncond rows untouched with cond_only
    xb = torch.cat([x, x])
    cb = torch.cat([c, n]) if c.shape == n.shape else None
    if cb is not None:
        both = tm.direct_call(m, xb, 0.6, cb, cond_or_uncond=[0, 1])
        plain = tm.direct_call(p, xb, 0.6, cb, cond_or_uncond=[0, 1])
        assert torch.equal(both[1], plain[1])
        assert not torch.allclose(both[0], plain[0], atol=1e-4)
