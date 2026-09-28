import math

import pytest
import torch

from conftest import requires_comfy
from zqx.core import lora_surgery as LS
from zqx.core.lora_io import LoraFactors


def _f(out, inp, r, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return LoraFactors(torch.randn(out, r, generator=g, dtype=torch.float64).float(),
                       torch.randn(r, inp, generator=g, dtype=torch.float64).float(), scale)


F = _f(30, 20, 6, 1, scale=0.5)


def test_lora_svd_is_exact():
    u, s, v = LS.lora_svd(F)
    d = F.dense()
    assert torch.allclose(u @ torch.diag(s) @ v.T, d, atol=1e-5)
    assert torch.allclose(s, torch.linalg.svdvals(d)[:6], atol=1e-6)
    assert torch.allclose(u.T @ u, torch.eye(6, dtype=torch.float64), atol=1e-10)


def test_truncate_rank_eckart_young():
    t, err = LS.truncate_rank(F, 3)
    s = torch.linalg.svdvals(F.dense())
    assert t.rank == 3
    assert (F.dense() - t.dense()).norm() / F.dense().norm() == pytest.approx(err, abs=1e-6)
    assert err == pytest.approx(math.sqrt(float((s[3:] ** 2).sum() / (s ** 2).sum())), abs=1e-6)
    full, e0 = LS.truncate_rank(F, 6)
    assert e0 == pytest.approx(0.0, abs=1e-12) and torch.allclose(full.dense(), F.dense(), atol=1e-5)


def test_spectrum_power():
    same = LS.spectrum_power(F, 1.0)
    assert torch.allclose(same.dense(), F.dense(), atol=1e-5)
    flat = LS.spectrum_power(F, 0.0001)
    s = torch.linalg.svdvals(flat.dense())[:6]
    assert torch.allclose(s, s[0].expand(6), rtol=1e-2)
    fro = LS.spectrum_power(F, 2.0, "frobenius")
    assert fro.dense().norm() == pytest.approx(F.dense().norm().item(), rel=1e-6)
    top = LS.spectrum_power(F, 2.0, "top")
    assert torch.linalg.svdvals(top.dense())[0] == pytest.approx(torch.linalg.svdvals(F.dense())[0].item(), rel=1e-6)
    # singular vectors are unchanged
    u0, _, _ = LS.lora_svd(F)
    u1, _, _ = LS.lora_svd(top)
    assert torch.allclose((u0.T @ u1).abs(), torch.eye(6, dtype=torch.float64), atol=1e-6)


def test_dare_up_unbiased_and_low_rank():
    acc = torch.zeros_like(F.dense())
    n = 400
    g = torch.Generator().manual_seed(0)
    for _ in range(n):
        d = LS.dare_up(F, 0.5, g)
        assert d.rank == F.rank
        acc += d.dense()
    rel = (acc / n - F.dense()).norm() / F.dense().norm()
    assert rel < 0.08
    assert torch.allclose(LS.dare_up(F, 0.0, g).dense(), F.dense(), atol=1e-6)


def test_common_subspace_recovers_shared_direction():
    g = torch.Generator().manual_seed(3)
    out, inp = 40, 30
    shared_u = torch.linalg.qr(torch.randn(out, 2, generator=g, dtype=torch.float64))[0]
    loras = []
    for i in range(3):
        own_u = torch.randn(out, 2, generator=g, dtype=torch.float64) * 0.3
        up = torch.cat([shared_u * 3.0, own_u], 1)
        down = torch.randn(4, inp, generator=g, dtype=torch.float64)
        loras.append(LoraFactors(up.float(), down.float(), 1.0))
    basis = LS.common_subspace(loras, 2)
    # principal angles between the recovered and the planted subspace ~ 0
    sv = torch.linalg.svdvals(basis.T @ shared_u)
    assert sv.min() > 0.97
    for f in loras:
        d = f.dense()
        exp = float((basis @ basis.T @ d).pow(2).sum() / d.pow(2).sum())
        assert LS.energy_fraction_in(f, basis) == pytest.approx(exp, abs=1e-9)
        assert exp > 0.6   # the planted shared part dominates each LoRA
    cleaned = LS.project_left(loras[0], basis)
    assert (basis.T @ cleaned.dense()).abs().max() < 1e-4
    comp = LS.common_component(loras, basis)
    mean = sum(f.dense() for f in loras) / 3
    assert torch.allclose(comp.dense(), basis @ basis.T @ mean, atol=1e-5)


# -------------------------------------------------------------------- ComfyUI round trip
@requires_comfy
@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_surgery_roundtrip_and_kinds(kind, tmp_path):
    import comfy.lora
    import lora_utils as lu
    import tiny_models as tm
    from test_lora_arith import _roundtrip_weights
    from zqx.adapters import get_adapter
    from zqx.core.lora_io import factors_to_state_dict
    from zqx.patches.lora_arith import lora_to_model_space
    from zqx.patches.lora_surgery import surgery
    p = tm.qwen_image(2) if kind == "qwen" else tm.z_image(2)
    ad = get_adapter(p)
    f, _, _ = lora_to_model_space(p, lu.make_lora(kind, seed=4, rank=4, alpha=2.0))
    kinds = {ad.module_kind(k) for k in f}
    assert "attention" in kinds and "mlp" in kinds
    if kind == "qwen":
        assert "text" in kinds
    res, rep = surgery(ad, f, block_weights="0:0.5", drop_kinds="mlp", strength=0.8, max_rank=2)
    assert all(ad.module_kind(k) != "mlp" for k in res)
    assert set(rep["dropped"]) == {k for k in f if ad.module_kind(k) == "mlp"}
    patched = _roundtrip_weights(p, factors_to_state_dict(res), tmp_path)
    msd = p.model.state_dict()
    for k, r in res.items():
        bi, _ = ad.block_of_key(k)
        mult = 0.8 * (0.5 if bi == 0 else 1.0)
        exp, _ = LS.truncate_rank(LS.scale_factors(f[k], mult), 2)
        assert torch.allclose(patched[k].double() - msd[k].double(), exp.dense(), atol=1e-5)


@requires_comfy
def test_surgery_and_common_nodes(tmp_path, monkeypatch):
    import folder_paths
    import lora_utils as lu
    import tiny_models as tm
    from safetensors.torch import save_file
    from zqx.nodes.lora_nodes import ZQXLoRACommonSubspace, ZQXLoRASurgery
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "loras", ([str(tmp_path)], {".safetensors"}))
    folder_paths.filename_list_cache.clear() if hasattr(folder_paths, "filename_list_cache") else None
    for i in range(3):
        save_file(lu.make_lora("zimage", seed=10 + i, rank=3, alpha=3.0), str(tmp_path / f"c{i}.safetensors"))
    p = tm.z_image(2)
    out = ZQXLoRASurgery().run(p, "c0.safetensors", 1.0, "", "io", 2, 1.0, 1.0, "top", 0.0, 0, "t", "float32")
    rel = out["result"][0]
    assert (tmp_path / rel).exists() and (tmp_path / (rel[:-12] + "_report.json")).exists()
    out = ZQXLoRACommonSubspace().run(p, "c0.safetensors", "c1.safetensors", "c2.safetensors", "None", 2, "common",
                                      "None", 1.0, "cs", "float32")
    assert (tmp_path / out["result"][0]).exists()
    out = ZQXLoRACommonSubspace().run(p, "c0.safetensors", "c1.safetensors", "None", "None", 2, "clean_target",
                                      "c2.safetensors", 1.0, "cs", "float32")
    assert "energy" in out["result"][1]
