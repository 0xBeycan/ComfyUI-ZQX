import pytest
import torch

from conftest import requires_comfy
from zqx.core.uce import pair_tokens, uce_solve


# ----------------------------------------------------------------------------- UCE (pure)
def test_uce_normal_equations_and_limits():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(12, 8, generator=g, dtype=torch.float64)
    c = torch.randn(3, 8, generator=g, dtype=torch.float64)
    v = torch.randn(3, 12, generator=g, dtype=torch.float64)
    k = torch.randn(5, 8, generator=g, dtype=torch.float64)
    lam = 0.3
    wn = uce_solve(w, c, v, k, lam)
    gram = lam * torch.eye(8, dtype=torch.float64) + c.T @ c + k.T @ k
    rhs = lam * w + v.T @ c + (w @ k.T) @ k
    assert torch.allclose(wn @ gram, rhs, atol=1e-9)
    # tiny lambda, no preservation, fewer constraints than dims -> edits satisfied exactly
    wn2 = uce_solve(w, c, v, None, 1e-9)
    assert torch.allclose(c @ wn2.T, v, atol=1e-5)
    # huge lambda -> W unchanged
    assert torch.allclose(uce_solve(w, c, v, k, 1e9), w, atol=1e-6)
    # targets equal to current outputs -> W unchanged for any lambda
    assert torch.allclose(uce_solve(w, c, c @ w.T, k, 0.5), w, atol=1e-9)


def test_pair_tokens():
    a, b = torch.randn(5, 4), torch.randn(3, 4)
    m1, m2 = pair_tokens(a, b, "mean")
    assert m1.shape == (1, 4) and torch.allclose(m2[0], b.mean(0))
    e1, e2 = pair_tokens(a, b, "end")
    assert torch.equal(e1, a[-3:]) and torch.equal(e2, b)
    with pytest.raises(ValueError):
        pair_tokens(a, b, "positional")


@requires_comfy
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_uce_node_patch(kind):
    import tiny_models as tm
    from zqx.adapters import get_adapter
    from zqx.patches.uce import build_edit, install, normed_tokens
    p = tm.qwen_image(2) if kind == "qwen" else tm.z_image(2)
    mk = tm.qwen_cond if kind == "qwen" else tm.zimage_cond
    src, tgt, keep = mk(1, n_txt=5), mk(2, n_txt=7), mk(3, n_txt=9)
    ad = get_adapter(p)
    # without a preservation set the edit is (almost) satisfied exactly
    wkey, diff, rep = build_edit(ad, src, tgt, None, "mean", 0.01, 1.0)
    assert rep["edit_residual_after"] < 0.05 * rep["edit_residual_before"]
    # a preservation set trades edit accuracy for less drift on the kept tokens
    _, _, rep_k = build_edit(ad, src, tgt, keep, "mean", 0.01, 1.0)
    _, diff_nokeep, _ = build_edit(ad, src, tgt, None, "mean", 0.01, 1.0)
    kt = torch.cat(normed_tokens(ad, keep)).double()
    w0 = p.model.state_dict()[wkey].double()
    drift_nokeep = float((kt @ diff_nokeep.double().T).norm() / (kt @ w0.T).norm())
    assert rep_k["keep_drift"] < drift_nokeep
    assert rep_k["edit_residual_after"] < rep_k["edit_residual_before"]
    m, rep2 = install(p, src, tgt, None, "mean", 0.01, 1.0)
    # the patched projection maps the source mean onto the target mean (approximately; keep set + lambda)
    import comfy.lora
    w = p.model.state_dict()[wkey].float()
    w_new = comfy.lora.calculate_weight(m.patches[wkey], w.clone(), wkey)
    cs = normed_tokens(ad, src)[0].mean(0)
    ct = normed_tokens(ad, tgt)[0].mean(0)
    err_before = (w @ cs - w @ ct).norm()
    err_after = (w_new @ cs - w @ ct).norm()
    assert err_after < 0.2 * err_before
    m0, rep0 = install(p, src, tgt, keep, "mean", 0.01, 0.0)
    assert wkey not in m0.patches
    # output changes when the edited prompt is used, sampling runs
    lat = (tm.qwen_latent if kind == "qwen" else tm.zimage_latent)(8, 8)
    out = tm.sample(m, src, src, torch.zeros_like(lat), steps=2, cfg=1.0)
    base = tm.sample(p, src, src, torch.zeros_like(lat), steps=2, cfg=1.0)
    assert torch.isfinite(out).all() and not torch.allclose(out, base, atol=1e-5)


# ----------------------------------------------------------------------------- steering
def _steer(p, towards, away, **kw):
    from zqx.patches.steering import install
    args = dict(alpha=1.0, sigma_start=1.0, sigma_end=0.0, blocks="0", mode="token", apply_to="all")
    args.update(kw)
    return install(p, towards=towards, away=away, **args)


@requires_comfy
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_steering_identities(kind):
    import tiny_models as tm
    p = tm.qwen_image(1) if kind == "qwen" else tm.z_image(1)
    mk = tm.qwen_cond if kind == "qwen" else tm.zimage_cond
    main, a = mk(1)[0][0], mk(2)
    x = (tm.qwen_latent if kind == "qwen" else tm.zimage_latent)(8, 10)
    base = tm.direct_call(p, x, 0.6, main)
    # alpha 0 / outside the window -> bitwise identity
    for kw in (dict(alpha=0.0), dict(sigma_start=0.3, sigma_end=0.0)):
        m, patch = _steer(p, a, a, **kw)
        assert torch.equal(tm.direct_call(m, x, 0.6, main), base)
        assert patch.calls_patched == 0
    # towards == away -> zero delta -> identical output
    m, patch = _steer(p, a, a)
    assert torch.equal(tm.direct_call(m, x, 0.6, main), base)
    assert patch.injected == [0]
    # 1-block model, token mode, alpha 1, towards = A, away = main prompt:
    # h0 + (h0_A - h0_main) = h0_A, and the final layer only reads image tokens -> output == model run with prompt A
    m, patch = _steer(p, a, [[main, {}]])
    steered = tm.direct_call(m, x, 0.6, main)
    with_a = tm.direct_call(p, x, 0.6, a[0][0])
    assert torch.allclose(steered, with_a, atol=1e-5), (steered - with_a).abs().max()
    assert not torch.allclose(steered, base, atol=1e-4)


@requires_comfy
def test_steering_mean_mode_and_cond_only_rows():
    import tiny_models as tm
    p = tm.qwen_image(2)
    c1, c2 = tm.qwen_cond(1)[0][0], tm.qwen_cond(2)[0][0]
    a, b = tm.qwen_cond(3), tm.qwen_cond(4)
    x = tm.qwen_latent(8, 8, batch=2)
    plain = tm.direct_call(p, x, 0.6, torch.cat([c1, c2]), cond_or_uncond=[0, 1])
    m, patch = _steer(p, a, b, mode="mean", blocks="all", apply_to="cond_only", alpha=0.5)
    out = tm.direct_call(m, x, 0.6, torch.cat([c1, c2]), cond_or_uncond=[0, 1])
    assert patch.injected == [0, 1]
    assert torch.equal(out[1], plain[1])
    assert not torch.allclose(out[0], plain[0], atol=1e-4)
    out2 = tm.sample(m, [[c1, {}]], [[c2, {}]], torch.zeros(1, 16, 1, 8, 8), steps=3, cfg=2.0)
    assert torch.isfinite(out2).all()
