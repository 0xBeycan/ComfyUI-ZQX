"""CADS and low-frequency noise: pure math + sampler-level identity tests."""
import math

import pytest
import torch

from conftest import requires_comfy
from zqx.core.cads import cads_apply, cads_gamma
from zqx.core.noise_init import lowfreq_mix, lowpass_filter, prepare_reference


# ------------------------------- CADS (pure) -------------------------------
def test_cads_gamma_schedule_endpoints():
    t1, t2 = 0.6, 0.9
    assert cads_gamma(0.0, t1, t2) == 1.0
    assert cads_gamma(t1, t1, t2) == 1.0           # gamma(tau1) = 1
    assert cads_gamma(t2, t1, t2) == 0.0           # gamma(tau2) = 0
    assert cads_gamma(1.0, t1, t2) == 0.0
    assert cads_gamma(0.75, t1, t2) == pytest.approx(0.5)
    for t in torch.linspace(t1, t2, 11).tolist():
        assert cads_gamma(t, t1, t2) == pytest.approx((t2 - t) / (t2 - t1), abs=1e-12)
    with pytest.raises(ValueError):
        cads_gamma(0.5, 0.9, 0.6)


def test_cads_zero_noise_with_rescale_is_identity_and_rescale_preserves_stats():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(2, 7, 32, generator=g) * 3 + 1.5
    n = torch.randn(y.shape, generator=g)
    # s = 0, psi = 1: (sqrt(g) y - sqrt(g) mu) / (sqrt(g) sd) * sd + mu = y
    out = cads_apply(y, 0.3, 0.0, 1.0, n)
    assert torch.allclose(out, y, atol=1e-5)
    # gamma = 1 -> y unchanged for any s, psi
    assert torch.allclose(cads_apply(y, 1.0, 0.7, 0.3, n), y, atol=1e-6)
    # psi = 1 -> per-sample mean/std equal the input's
    out = cads_apply(y, 0.2, 0.25, 1.0, n)
    for i in range(2):
        assert out[i].mean().item() == pytest.approx(y[i].mean().item(), abs=1e-4)
        assert out[i].std().item() == pytest.approx(y[i].std().item(), rel=1e-4)
    # psi = 0 -> plain formula
    out0 = cads_apply(y, 0.2, 0.25, 0.0, n)
    assert torch.allclose(out0, math.sqrt(0.2) * y + 0.25 * math.sqrt(0.8) * n, atol=1e-5)
    # relative noise scales with std(y) per sample
    outr = cads_apply(y, 0.2, 0.25, 0.0, n, relative_noise=True)
    sd = y.std(dim=(1, 2), keepdim=True)
    assert torch.allclose(outr, math.sqrt(0.2) * y + 0.25 * sd * math.sqrt(0.8) * n, atol=1e-5)


# ------------------------------- low-frequency noise (pure) -------------------------------
def _ideal_split(x, cutoff):
    h, w = x.shape[-2:]
    m = lowpass_filter(h, w, cutoff, "ideal")
    fx = torch.fft.fft2(x.double())
    return torch.fft.ifft2(fx * m).real, torch.fft.ifft2(fx * (1 - m)).real


def test_lowfreq_bands_and_statistics():
    g = torch.Generator().manual_seed(0)
    b, c, h, w = 4, 16, 64, 64
    noise = torch.randn(b, c, h, w, generator=g)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 48), torch.linspace(-1, 1, 40), indexing="ij")
    ref = (torch.stack([torch.sin(3 * xx + k) + torch.cos(2 * yy * (k % 3 + 1)) for k in range(c)])[None] * 5 + 2)
    cutoff = 0.25
    out = lowfreq_mix(noise, ref, 1.0, cutoff, "ideal")
    lo_o, hi_o = _ideal_split(out, cutoff)
    lo_n, hi_n = _ideal_split(noise, cutoff)
    r = prepare_reference(ref, h, w).expand(b, -1, -1, -1)
    lo_r, _ = _ideal_split(r, cutoff)
    # high band exactly the noise's
    assert torch.allclose(hi_o, hi_n, atol=1e-5)
    # low band exactly c * LPF(r), c matching the noise band energy per (b, c)
    cfac = lo_n.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() / lo_r.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
    assert torch.allclose(lo_o, cfac * lo_r, atol=1e-5)
    # total energy per (b, c) equals the noise's -> unit variance, zero mean
    assert torch.allclose(out.double().pow(2).sum(dim=(-2, -1)), noise.double().pow(2).sum(dim=(-2, -1)), rtol=1e-6)
    # partial strength with an ideal filter: low band = sqrt(1-a) LPF(eps) + sqrt(a) c LPF(r)
    out5 = lowfreq_mix(noise, ref, 0.5, cutoff, "ideal")
    lo5, hi5 = _ideal_split(out5, cutoff)
    assert torch.allclose(hi5, hi_n, atol=1e-5)
    assert torch.allclose(lo5, (0.5 ** 0.5) * lo_n + (0.5 ** 0.5) * cfac * lo_r, atol=1e-5)
    assert abs(out.mean().item()) < 0.02 and abs(out.std().item() - 1.0) < 0.02
    # correlation of the low band with the reference is 1
    corr = torch.nn.functional.cosine_similarity(lo_o.flatten(2), lo_r.flatten(2), dim=-1).min().item()
    assert corr > 0.9999


@pytest.mark.parametrize("kind", ["gaussian", "butterworth"])
def test_lowfreq_soft_filters_unit_variance_and_identity(kind):
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(8, 16, 64, 64, generator=g)
    ref = torch.randn(1, 16, 32, 32, generator=g).cumsum(-1).cumsum(-2)   # smooth, low-frequency heavy
    assert lowfreq_mix(noise, ref, 0.0, 0.25, kind) is noise              # strength 0 -> the same tensor
    for a in (0.3, 1.0):
        out = lowfreq_mix(noise, ref, a, 0.25, kind)
        assert abs(out.mean().item()) < 0.02
        assert abs(out.var().item() - 1.0) < 0.05, out.var()
        # (1 - H) band equals the noise's
        hf = lowpass_filter(64, 64, 0.25, kind)
        d = torch.fft.fft2(out.double()) - torch.fft.fft2(noise.double())
        # the difference lives only where H > 0: |d| <= H * (|F eps| + c |F r|)
        assert torch.all((d.abs() < 1e-6) | (hf > 1e-12))


def test_lowfreq_5d_and_errors():
    noise = torch.randn(1, 16, 1, 32, 32)
    ref = torch.randn(1, 16, 1, 16, 16)
    out = lowfreq_mix(noise, ref, 0.5, 0.3)
    assert out.shape == noise.shape
    with pytest.raises(ValueError):
        lowfreq_mix(noise, torch.randn(1, 4, 16, 16), 0.5, 0.3)
    with pytest.raises(ValueError):
        lowfreq_mix(noise, ref, 1.5, 0.3)
    with pytest.raises(ValueError):
        lowfreq_mix(torch.randn(1, 16, 8, 8), torch.ones(1, 16, 8, 8), 0.5, 0.3)   # constant reference


# ------------------------------- sampler level -------------------------------
@requires_comfy
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
@pytest.mark.parametrize("denoise", [1.0, 0.6])
def test_cads_node_identity_and_effect(kind, denoise):
    import tiny_models as tm
    from zqx.patches.cads import install
    p = tm.qwen_image(2) if kind == "qwen" else tm.z_image(2)
    pos = (tm.qwen_cond if kind == "qwen" else tm.zimage_cond)(1)
    neg = (tm.qwen_cond if kind == "qwen" else tm.zimage_cond)(2)
    lat = (tm.qwen_latent if kind == "qwen" else tm.zimage_latent)(8, 8)
    lat = lat if denoise < 1 else torch.zeros_like(lat)
    cfg = 2.0 if kind == "qwen" else 1.0
    base = tm.sample(p, pos, neg, lat, steps=5, cfg=cfg, denoise=denoise)
    variants = [dict(noise_scale=0.0)]
    if denoise < 1.0:   # this img2img pass starts at sigma <= 0.9: a window above 0.97 never fires
        variants.append(dict(noise_scale=0.3, tau1=0.97, tau2=1.0))
    for kw in variants:
        args = dict(tau1=0.6, tau2=0.9, noise_scale=0.25, psi=1.0, seed=0)
        args.update(kw)
        m, patch = install(p, **args)
        assert torch.equal(tm.sample(m, pos, neg, lat, steps=5, cfg=cfg, denoise=denoise), base)
        assert patch.calls_patched == 0
    m, patch = install(p, tau1=0.3, tau2=0.9, noise_scale=0.5, psi=1.0, seed=0)
    out = tm.sample(m, pos, neg, lat, steps=5, cfg=cfg, denoise=denoise)
    assert patch.calls_patched > 0 and not torch.allclose(out, base, atol=1e-4)
    out2 = tm.sample(m, pos, neg, lat, steps=5, cfg=cfg, denoise=denoise)
    assert torch.equal(out, out2)


@requires_comfy
def test_cads_cond_only_leaves_uncond_rows():
    import tiny_models as tm
    from zqx.patches.cads import install
    p = tm.qwen_image(2)
    c1, c2 = tm.qwen_cond(1)[0][0], tm.qwen_cond(2)[0][0]
    x = tm.qwen_latent(8, 8, batch=2)
    base = tm.direct_call(p, x, 0.95, torch.cat([c1, c2]), cond_or_uncond=[0, 1])
    m, _ = install(p, tau1=0.6, tau2=0.9, noise_scale=0.5, psi=1.0, seed=0, apply_to="cond_only")
    out = tm.direct_call(m, x, 0.95, torch.cat([c1, c2]), cond_or_uncond=[0, 1])
    assert torch.equal(out[1], base[1])
    assert not torch.allclose(out[0], base[0], atol=1e-4)


@requires_comfy
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_lowfreq_noise_object(kind):
    import comfy.sample
    import tiny_models as tm
    from zqx.patches.noise import LowFreqNoise
    lat = (tm.qwen_latent if kind == "qwen" else tm.zimage_latent)(16, 16)
    ref = (tm.qwen_latent if kind == "qwen" else tm.zimage_latent)(24, 24, seed=9)
    n0 = LowFreqNoise(5, ref, 0.0, 0.25, "gaussian", 4).generate_noise({"samples": lat})
    assert torch.equal(n0, comfy.sample.prepare_noise(lat, 5))
    n1 = LowFreqNoise(5, ref, 0.8, 0.25, "gaussian", 4).generate_noise({"samples": lat})
    assert n1.shape == lat.shape and torch.isfinite(n1).all()
    assert not torch.allclose(n1, n0)
