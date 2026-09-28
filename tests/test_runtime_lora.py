import pytest
import torch

from conftest import requires_comfy

pytestmark = requires_comfy


def _merged(patcher, sd, strength):
    import comfy.sd
    m, _ = comfy.sd.load_lora_for_models(patcher, None, sd, strength, 0)
    return m


def _runtime(patcher, sd, early, late, hi=1.0, lo=0.0, blocks=""):
    from zqx.adapters import get_adapter
    from zqx.patches.lora_schedules import scheduled_entries
    from zqx.patches.runtime_lora import install_runtime_lora, load_lora_patches
    ad = get_adapter(patcher)
    pk, _ = load_lora_patches(patcher, sd)
    entries, _ = scheduled_entries(ad, pk, early, late, hi, lo, blocks)
    return install_runtime_lora(patcher, entries, "zqx_test_lora")


def _setup(kind):
    import tiny_models as tm
    if kind == "qwen":
        return tm.qwen_image(2), tm.qwen_cond(1), tm.qwen_cond(2), tm.qwen_latent(8, 8), 2.0
    return tm.z_image(2), tm.zimage_cond(1), tm.zimage_cond(2), tm.zimage_latent(8, 8), 1.0


@pytest.mark.parametrize("kind,fmt,alpha", [("qwen", "peft", None), ("qwen", "kohya", 2.0), ("qwen", "transformer", 8.0),
                                            ("zimage", "peft", None), ("zimage", "peft", 2.0)])
@pytest.mark.parametrize("denoise", [1.0, 0.55])
def test_constant_strength_equals_loraloader(kind, fmt, alpha, denoise):
    import lora_utils as lu
    import tiny_models as tm
    p, pos, neg, lat, cfg = _setup(kind)
    sd = lu.make_lora(kind, fmt=fmt, alpha=alpha, rank=4, seed=3)
    s = 0.8
    ref = tm.sample(_merged(p, sd, s), pos, neg, lat if denoise < 1 else torch.zeros_like(lat), steps=3, cfg=cfg, denoise=denoise)
    m, patch = _runtime(p, sd, s, s)
    out = tm.sample(m, pos, neg, lat if denoise < 1 else torch.zeros_like(lat), steps=3, cfg=cfg, denoise=denoise)
    base = tm.sample(p, pos, neg, lat if denoise < 1 else torch.zeros_like(lat), steps=3, cfg=cfg, denoise=denoise)
    assert not torch.allclose(ref, base, atol=1e-4)       # the LoRA does something
    assert torch.allclose(out, ref, atol=2e-5), (out - ref).abs().max()


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_zero_strength_is_identity(kind):
    import lora_utils as lu
    import tiny_models as tm
    p, pos, neg, lat, cfg = _setup(kind)
    sd = lu.make_lora(kind, seed=3)
    base = tm.sample(p, pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg)
    m, _ = _runtime(p, sd, 0.0, 0.0)
    out = tm.sample(m, pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg)
    # the module now runs through comfy's cast path with an unchanged weight copy: bitwise identical on CPU
    assert torch.equal(out, base)


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_schedule_follows_sigma_ramp(kind):
    import lora_utils as lu
    import tiny_models as tm
    from zqx.core.schedule import linear_sigma_ramp
    p, pos, neg, lat, cfg = _setup(kind)
    sd = lu.make_lora(kind, seed=3)
    m, patch = _runtime(p, sd, 1.0, 0.25, hi=0.8, lo=0.4)
    patch.log_enabled = True
    tm.sample(m, pos, neg, torch.zeros_like(lat), steps=6, cfg=cfg)
    sig = sorted({round(s, 6) for s, k, v in patch.eval_log})
    assert len(sig) >= 4
    for s, k, v in patch.eval_log:
        assert v == pytest.approx(linear_sigma_ramp(s, 0.8, 0.4, 1.0, 0.25), abs=1e-12)
    # a hard switch changes the output relative to both constant strengths
    early = tm.sample(_runtime(p, sd, 1.0, 1.0)[0], pos, neg, torch.zeros_like(lat), steps=6, cfg=cfg)
    late = tm.sample(_runtime(p, sd, 0.25, 0.25)[0], pos, neg, torch.zeros_like(lat), steps=6, cfg=cfg)
    sw = tm.sample(m, pos, neg, torch.zeros_like(lat), steps=6, cfg=cfg)
    assert not torch.allclose(sw, early, atol=1e-5) and not torch.allclose(sw, late, atol=1e-5)


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_block_weights_equal_lora_without_those_blocks(kind):
    import lora_utils as lu
    import tiny_models as tm
    p, pos, neg, lat, cfg = _setup(kind)
    sd = lu.make_lora(kind, seed=3)
    blk = "transformer_blocks.0." if kind == "qwen" else "layers.0."
    sd_wo = {k: v for k, v in sd.items() if blk not in k}
    ref = tm.sample(_merged(p, sd_wo, 0.9), pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg)
    m, _ = _runtime(p, sd, 0.9, 0.9, blocks="0:0")
    out = tm.sample(m, pos, neg, torch.zeros_like(lat), steps=3, cfg=cfg)
    assert torch.allclose(out, ref, atol=2e-5)


def test_unmatched_keys_refused():
    import lora_utils as lu
    import tiny_models as tm
    from zqx.patches.runtime_lora import load_lora_patches
    p = tm.qwen_image(2)
    sd = lu.make_lora("qwen", seed=3)
    sd["diffusion_model.nonexistent.lora_A.weight"] = torch.zeros(4, 8)
    sd["diffusion_model.nonexistent.lora_B.weight"] = torch.zeros(8, 4)
    with pytest.raises(ValueError, match="do not match"):
        load_lora_patches(p, sd)
    pk, unmatched = load_lora_patches(p, sd, allow_unmatched=True)
    assert len(unmatched) >= 1


# ----------------------------------------------------------------------------------------------
# K-LoRA
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_klora_selection_matches_independent_computation(kind):
    """Recompute S_c, S_s, gamma directly from the LoRA files (A, B, alpha) with plain torch and check
    every logged per-step strength follows the official rule."""
    import lora_utils as lu
    import tiny_models as tm
    from zqx.adapters import get_adapter
    from zqx.core.lora_math import klora_time_scale
    from zqx.patches.lora_schedules import klora_entries
    from zqx.patches.runtime_lora import install_runtime_lora, load_lora_patches
    p, pos, neg, lat, cfg = _setup(kind)
    sdc = lu.make_lora(kind, seed=3, rank=4, alpha=2.0, scale=0.05)
    sds = lu.make_lora(kind, seed=4, rank=2, alpha=None, scale=0.08)
    ad = get_adapter(p)
    pc, _ = load_lora_patches(p, sdc)
    ps, _ = load_lora_patches(p, sds)
    shapes = {k: tuple(v.shape) for k, v in p.model.state_dict().items()}
    alpha, beta, pattern = 1.5, 0.5, "s"
    entries, report, shared = klora_entries(ad, shapes, pc, ps, 1.0, 0.7, alpha, beta, pattern, "attention", "both")

    # independent dense deltas per model key
    def dense(sd, rank, a):
        out = {}
        for k in sd:
            if k.endswith(".lora_A.weight"):
                base = k[: -len(".lora_A.weight")]
                d = sd[base + ".lora_B.weight"].double() @ sd[k].double()
                sc = (a / rank) if a is not None else 1.0
                stem = base[len("diffusion_model."):]
                if kind == "zimage" and ".attention.to_" in stem and "to_out" not in stem:
                    i = int(stem.split(".")[1]); j = "qkv".index(stem.split("to_")[1][0])
                    mk = f"diffusion_model.layers.{i}.attention.qkv.weight"
                    full = out.get(mk, torch.zeros(768, 256, dtype=torch.float64))
                    full[j * 256:(j + 1) * 256] += sc * d
                    out[mk] = full
                else:
                    out[f"diffusion_model.{stem}.weight"] = sc * d
        return out
    dc, ds = dense(sdc, 4, 2.0), dense(sds, 2, None)
    import re
    rx = re.compile(r"\.attn\.(to_q|to_k|to_v)\.weight$") if kind == "qwen" else re.compile(r"\.attention\.qkv\.weight$")
    keys = sorted(k for k in dc if rx.search(k))
    assert keys == sorted(shared)
    ratios = [float(dc[k].abs().sum() / ds[k].abs().sum()) for k in keys]
    mean = sum(ratios) / len(ratios)
    kept = [r for r in ratios if r < 3 * mean]
    gamma = sum(kept) / len(kept)
    assert report["gamma"] == pytest.approx(gamma, rel=1e-5)
    rank_c = 4 * (3 if kind == "zimage" else 1)
    rank_s = 2 * (3 if kind == "zimage" else 1)
    for k in keys:
        K = rank_c * rank_s
        s_c = float(torch.topk(dc[k].abs().flatten(), K).values.sum())
        s_s = float(torch.topk(ds[k].abs().flatten(), K).values.sum())
        assert report["layers"][k]["S_c"] == pytest.approx(s_c, rel=1e-4)
        assert report["layers"][k]["S_s"] == pytest.approx(s_s, rel=1e-4)
    m, patch = install_runtime_lora(p, entries, "zqx_klora_test")
    patch.log_enabled = True
    tm.sample(m, pos, neg, torch.zeros_like(lat), steps=6, cfg=cfg)
    n_checked = 0
    for sg, k, v in patch.eval_log:
        if k not in keys:
            continue
        use_c = (report["layers"][k]["S_c"] / gamma) / (report["layers"][k]["S_s"] * klora_time_scale(1 - sg, alpha, beta, pattern)) > 1
        assert v in (0.0, 1.0, 0.7)
        n_checked += 1
    assert n_checked > 0
    # per key and sigma: exactly one of the two LoRAs is active
    by = {}
    for sg, k, v in patch.eval_log:
        if k in keys:
            by.setdefault((sg, k), []).append(v)
    for (sg, k), vals in by.items():
        use_c = (report["layers"][k]["S_c"] / gamma) / (report["layers"][k]["S_s"] * klora_time_scale(1 - sg, alpha, beta, pattern)) > 1
        nc = 3 if kind == "zimage" else 1   # patches per key per LoRA (fused qkv slices)
        c_vals, s_vals = vals[:nc], vals[nc:]
        assert all(v == (1.0 if use_c else 0.0) for v in c_vals)
        assert all(v == (0.0 if use_c else 0.7) for v in s_vals)
