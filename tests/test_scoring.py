import math

import pytest
import torch

from conftest import requires_comfy
from zqx.core import scoring as SC


def test_background_sharpness_detects_bokeh():
    g = torch.Generator().manual_seed(0)
    sharp = torch.rand(1, 64, 64, 3, generator=g)
    # blur the border band only (bokeh-like): smooth everything, then paste the sharp centre back
    blurred = torch.nn.functional.avg_pool2d(sharp.permute(0, 3, 1, 2), 9, stride=1, padding=4).permute(0, 2, 3, 1)
    bokeh = blurred.clone()
    bokeh[:, 13:51, 13:51] = sharp[:, 13:51, 13:51]
    s = SC.background_sharpness(torch.cat([sharp, bokeh]))
    assert abs(float(s[0])) < 0.5
    assert float(s[1]) < -2.0


def test_off_center_and_zscore_combine():
    assert SC.off_center([45, 45, 55, 55], 100, 100) == pytest.approx(0.0)
    assert SC.off_center([0, 0, 0, 0], 100, 100) == pytest.approx(1.0)
    m = {"a": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([5.0, 5.0, 5.0])}
    s = SC.combine(m, {"a": 2.0, "b": 10.0})
    z = (m["a"] - 2) / m["a"].std(unbiased=False)
    assert torch.allclose(s, 2 * z.double())
    raw = SC.combine(m, {"a": 2.0, "b": 1.0}, normalize=False)
    assert torch.allclose(raw, torch.tensor([7.0, 9.0, 11.0], dtype=torch.float64))
    assert SC.top_k(torch.tensor([0.1, 0.9, 0.9, 0.5]), 3) == [1, 2, 3]
    with pytest.raises(ValueError):
        SC.combine({"a": torch.tensor([1.0, float("nan")])}, {"a": 1.0})


def test_face_scorer_with_fake_analyzer():
    import fakes
    from zqx.patches.scorers import CombinedScorer, FaceScorer, BackgroundSharpnessScorer
    imgs = torch.zeros(3, 20, 30, 3)
    imgs[0, 10, 15] = 1.0     # centred "face"
    imgs[1, 2, 28] = 1.0      # corner, turned
    imgs[2, 10, 16] = 1.0
    ref = imgs[:1].clone()
    fs = FaceScorer(fakes.FakeFaceAnalyzer(), ref, 1.0, 0.5, 0.5, 1.0)
    m = fs.metrics(imgs)
    assert m["face_found"].tolist() == [1.0, 1.0, 1.0]
    assert m["face_identity"][0] == pytest.approx(1.0, abs=1e-5)
    assert m["face_off_center"][1] > m["face_off_center"][0]
    assert m["head_turn"][1] > m["head_turn"][0]
    no_face = FaceScorer(fakes.FakeFaceAnalyzer(none_if_dark=True), None, 0.0, 0.5, 0.5, 1.0)
    assert no_face.metrics(torch.zeros(2, 8, 8, 3))["face_found"].tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        FaceScorer(fakes.FakeFaceAnalyzer(), None, 1.0, 0, 0, 0)          # identity needs a reference
    with pytest.raises(ValueError):
        FaceScorer(fakes.FakeFaceAnalyzer(with_pose=False), None, 0, 0, 1.0, 0).metrics(imgs)
    comb = CombinedScorer([fs, BackgroundSharpnessScorer(0.3), BackgroundSharpnessScorer(0.2)])
    mm = comb.metrics(torch.rand(3, 20, 30, 3))
    assert set(mm) == set(comb.weights) and "bg_sharpness_2" in mm


# ------------------------------------------------------------------ seed search
def _guider(p, pos, neg, cfg):
    import comfy.samplers
    g = comfy.samplers.CFGGuider(p)
    g.set_conds(pos, neg)
    g.set_cfg(cfg)
    return g


@requires_comfy
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_seed_search(kind):
    import comfy.samplers
    import fakes
    import tiny_models as tm
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
    from zqx.patches.scorers import BackgroundSharpnessScorer, Scorer
    from zqx.patches.seed_search import seed_search
    p = tm.qwen_image(2) if kind == "qwen" else tm.z_image(2)
    mk = tm.qwen_cond if kind == "qwen" else tm.zimage_cond
    lat = {"samples": torch.zeros(1, 16, 1, 8, 8) if kind == "qwen" else torch.zeros(1, 16, 8, 8)}
    g = _guider(p, mk(1), mk(2), 1.0)
    sampler = comfy.samplers.sampler_object("euler")
    sigmas = comfy.samplers.calculate_sigmas(p.get_model_object("model_sampling"), "simple", 4).cpu()
    vae = fakes.FakeVAE()

    class Bright(Scorer):   # prefer bright previews
        def __init__(self):
            super().__init__()
            self.weights = {"bright": 1.0}

        def metrics(self, images):
            return {"bright": images.mean(dim=(1, 2, 3))}

    out, prev, seeds, report, scores, metrics = seed_search(Noise_RandomNoise(100), g, sampler, sigmas, lat, vae,
                                                            Bright(), 5, 1, 2)
    assert prev.shape[0] == 5 and out.shape[0] == 2
    order = sorted(range(5), key=lambda i: -float(metrics["bright"][i]))
    assert seeds == [100 + order[0], 100 + order[1]]
    # each kept sample equals a normal run with that seed
    import comfy.sample
    for j, s in enumerate(seeds):
        nt = comfy.sample.prepare_noise(lat["samples"], s)
        ref = g.sample(nt, lat["samples"], sampler, sigmas, disable_pbar=True, seed=s)
        assert torch.equal(out[j:j + 1], ref)
    # probing all steps: the decoded x0 preview equals the decoded final sample
    out2, prev2, seeds2, _, _, _ = seed_search(Noise_RandomNoise(7), g, sampler, sigmas, lat, vae, Bright(), 1, 4, 1)
    assert torch.allclose(prev2, vae.decode(out2), atol=1e-5)
    with pytest.raises(ValueError):
        seed_search(Noise_RandomNoise(7), g, sampler, sigmas, lat, vae, Bright(), 2, 9, 1)


# ------------------------------------------------------------------ realism ablation scan
def test_select_units():
    from zqx.patches.ablation import select_units
    res = [{"name": "a", "gain": 0.1, "drop": 0.01}, {"name": "b", "gain": 0.2, "drop": 0.5},
           {"name": "c", "gain": 0.3, "drop": 0.001}, {"name": "d", "gain": -0.1, "drop": 0.0}]
    assert select_units(res, 0.05, 0.02) == ["c", "a"]
    assert select_units(res, 0.0, 1.0) == ["c", "b", "a"]


@requires_comfy
@pytest.mark.parametrize("units", ["blocks", "blocks+kinds", "blocks+svd"])
def test_ablation_scan_end_to_end(units, tmp_path):
    import fakes
    import lora_utils as lu
    import tiny_models as tm
    from zqx.adapters import get_adapter
    from zqx.core.scoring import cosine
    from zqx.nodes.scoring_nodes import make_renderer
    from zqx.patches.ablation import AblationScan, factors_to_entries
    from zqx.patches.lora_arith import lora_to_model_space
    from zqx.patches.scorers import Scorer, clip_embed
    p = tm.z_image(2)
    pos, neg = tm.zimage_cond(1), tm.zimage_cond(2)
    lat = {"samples": torch.zeros(1, 16, 8, 8)}
    vae = fakes.FakeVAE()
    clip = fakes.FakeClipVision()
    render = make_renderer(pos, neg, lat, vae, [1, 2], 3, 1.0, "euler", "simple")
    # synthetic "identity": similarity to the no-realism renders (A)
    img_a = render(p)
    emb_a = clip_embed(clip, img_a)

    class Ident(Scorer):
        def __init__(self):
            super().__init__()
            self.weights = {"id": 1.0}

        def metrics(self, images):
            return {"id": cosine(clip_embed(clip, images), emb_a)}

    f, _, _ = lora_to_model_space(p, lu.make_lora("zimage", seed=3, scale=0.3))
    scan = AblationScan(get_adapter(p), p, f, 1.0, render, Ident(), clip)
    final, rep, imgs = scan.run(units, 0.0, 1.0, svd_blocks=1, svd_components=2)
    assert imgs.shape[0] == 6
    assert rep["identity_A"] == pytest.approx(1.0, abs=1e-6)       # A is the identity reference itself
    n_units = len(rep["units"])
    assert rep["generations"] == 2 + n_units + len(rep["svd_units"]) + 1
    # removals are consistent with the measured gains (min_gain 0, max_drop 1: every positive-gain unit)
    assert set(rep["removed_units"]) == {u["name"] for u in rep["units"] if u["gain"] > 0}
    from zqx.patches.ablation import build_units
    unit_keys = build_units(get_adapter(p), f, "blocks+kinds" if units == "blocks+kinds" else "blocks")
    removed = {k for name in rep["removed_units"] for k in unit_keys[name]}
    assert all(k not in final for k in removed)
    # removing everything reproduces A exactly (no runtime LoRA at all)
    empty = AblationScan(get_adapter(p), p, {}, 1.0, render, Ident(), clip)
    assert torch.equal(render(empty.model_with({})), img_a)
    # the runtime path from factors equals ComfyUI's merged loading of the same factors
    import comfy.sd
    from zqx.core.lora_io import factors_to_state_dict
    merged, _ = comfy.sd.load_lora_for_models(p, None, factors_to_state_dict(f), 1.0, 0)
    assert torch.allclose(render(scan.model_with(f)), render(merged), atol=1e-5)
