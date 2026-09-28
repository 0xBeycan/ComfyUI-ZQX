import pytest
import torch

from conftest import requires_comfy

pytestmark = requires_comfy


def _run(guider, lat, steps=5, seed=11, denoise=1.0):
    import comfy.samplers
    import comfy.sample
    sampler = comfy.samplers.sampler_object("euler")
    ms = guider.model_patcher.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(ms, "simple", steps).cpu()
    if denoise < 1.0:
        sigmas = sigmas[-(int(steps * denoise) + 1):]
    noise = comfy.sample.prepare_noise(lat, seed)
    return guider.sample(noise, lat, sampler, sigmas, disable_pbar=True, seed=seed)


def _base(p, pos, neg, cfg, lat, **kw):
    import comfy.samplers
    g = comfy.samplers.CFGGuider(p)
    g.set_conds(pos, neg)
    g.set_cfg(cfg)
    return _run(g, lat, **kw)


@pytest.mark.parametrize("denoise", [1.0, 0.6])
def test_equal_cfgs_equal_plain_guider(denoise):
    import tiny_models as tm
    from zqx.patches.guider import SigmaSplitGuider
    p = tm.z_image(2)
    pos, neg = tm.zimage_cond(1), tm.zimage_cond(2)
    lat = tm.zimage_latent(8, 8) if denoise < 1 else torch.zeros(1, 16, 8, 8)
    for c in (1.0, 3.0):
        g = SigmaSplitGuider(p, 0.7, c, c)
        g.set_conds(pos, neg)
        assert torch.equal(_run(g, lat, denoise=denoise), _base(p, pos, neg, c, lat, denoise=denoise))


def test_interval_cfg_only_early():
    import tiny_models as tm
    from zqx.patches.guider import SigmaSplitGuider
    p = tm.z_image(2)
    pos, neg = tm.zimage_cond(1), tm.zimage_cond(2)
    lat = torch.zeros(1, 16, 8, 8)
    g = SigmaSplitGuider(p, 0.7, 3.0, 1.0)
    g.set_conds(pos, neg)
    out = _run(g, lat, steps=6)
    assert any(s >= 0.7 and c == 3.0 for s, _, c in g.calls)
    assert all((c == 3.0) == (s >= 0.7) for s, _, c in g.calls)
    assert not torch.equal(out, _base(p, pos, neg, 1.0, lat, steps=6))


def test_early_model_switch_extremes():
    import tiny_models as tm
    from zqx.patches.guider import SigmaSplitGuider
    main, early = tm.z_image(2, seed=0), tm.z_image(2, seed=7)
    pos, neg = tm.zimage_cond(1), tm.zimage_cond(2)
    lat = torch.zeros(1, 16, 8, 8)
    g = SigmaSplitGuider(main, 2.0, 1.0, 1.0, model_early=early)     # never early
    g.set_conds(pos, neg)
    assert torch.equal(_run(g, lat), _base(main, pos, neg, 1.0, lat))
    g = SigmaSplitGuider(main, 0.0, 1.0, 1.0, model_early=early)     # always early
    g.set_conds(pos, neg)
    assert torch.allclose(_run(g, lat), _base(early, pos, neg, 1.0, lat), atol=1e-6)
    g = SigmaSplitGuider(main, 0.8, 1.0, 1.0, model_early=early)
    g.set_conds(pos, neg)
    _run(g, lat, steps=6)
    whiches = [w for _, w, _ in g.calls]
    assert "early_model" in whiches and "model" in whiches


def test_same_weights_refused():
    import tiny_models as tm
    from zqx.patches.guider import SigmaSplitGuider
    p = tm.z_image(2)
    with pytest.raises(ValueError, match="shares its weights"):
        SigmaSplitGuider(p, 0.8, 1.0, 1.0, model_early=p.clone())
