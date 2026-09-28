import pytest
import torch

from conftest import requires_comfy

pytestmark = requires_comfy

EXACT = ["add", "negate", "clean_col", "clean_row", "target_sub"]


def _models_and_loras(kind):
    import lora_utils as lu
    import tiny_models as tm
    if kind == "qwen":
        p = tm.qwen_image(2)
        l1 = lu.make_lora("qwen", seed=1, rank=4, fmt="kohya", alpha=2.0)
        l2 = lu.make_lora("qwen", seed=2, rank=3, fmt="peft", only=lambda s: "img_mlp" not in s or "blocks.1" in s)
    else:
        p = tm.z_image(2)
        l1 = lu.make_lora("zimage", seed=1, rank=4, fmt="peft", alpha=8.0)
        # musubi-style kohya names for Z-Image (lora_unet_layers_0_attention_to_q) are not in ComfyUI's key map;
        # they are resolved through normalised names
        l2 = lu.make_lora("zimage", seed=2, rank=3, fmt="kohya", alpha=3.0)
    return p, l1, l2


def _roundtrip_weights(p, sd, tmp_path):
    """Save -> load with comfy.utils.load_torch_file -> comfy.lora.load_lora with the model key map ->
    comfy.lora.calculate_weight on the original weights (exactly what LoraLoader does)."""
    import comfy.lora
    import comfy.utils
    from safetensors.torch import save_file
    fn = str(tmp_path / "out.safetensors")
    save_file(sd, fn)
    loaded_sd = comfy.utils.load_torch_file(fn)
    km = comfy.lora.model_lora_keys_unet(p.model, {})
    patches = comfy.lora.load_lora(loaded_sd, km)
    assert len(patches) * 3 == len(loaded_sd)          # every up/down/alpha triple consumed
    msd = p.model.state_dict()
    out = {}
    for k, v in patches.items():
        mk = k if isinstance(k, str) else k[0]
        w = msd[mk].to(torch.float32).clone()
        out[mk] = comfy.lora.calculate_weight([(1.0, v, 1.0, None, None)], w, mk)
    return out


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
@pytest.mark.parametrize("mode", EXACT)
def test_exact_modes_roundtrip_through_comfy_loader(kind, mode, tmp_path):
    from zqx.core.lora_io import factors_to_state_dict
    from zqx.patches.lora_arith import combine, expected_dense, lora_to_model_space
    p, l1, l2 = _models_and_loras(kind)
    f1, via1, un1 = lora_to_model_space(p, l1)
    f2, via2, un2 = lora_to_model_space(p, l2)
    assert un1 == [] and un2 == []
    if kind == "zimage":
        assert len(via2) > 0                    # musubi kohya names needed the normalised mapping
        assert any(k.endswith("attention.qkv.weight") for k in f1)
    lam = 0.6
    res, info = combine(f1, f2, mode, lam)
    sd = factors_to_state_dict(res)
    patched = _roundtrip_weights(p, sd, tmp_path)
    msd = p.model.state_dict()
    for k in set(f1) | set(f2):
        exp = expected_dense(f1.get(k), f2.get(k), mode, lam)
        if mode not in ("add", "negate") and k not in f1:
            assert k not in patched
            continue
        got = patched[k].double() - msd[k].double()
        assert torch.allclose(got, exp, atol=1e-5), (k, (got - exp).abs().max())


@pytest.mark.parametrize("kind", ["qwen", "zimage"])
def test_ties_modes_roundtrip(kind, tmp_path):
    from zqx.core.lora_io import factors_to_state_dict
    from zqx.patches.lora_arith import combine, lora_to_model_space
    p, l1, l2 = _models_and_loras(kind)
    f1, _, _ = lora_to_model_space(p, l1)
    f2, _, _ = lora_to_model_space(p, l2)
    for mode in ("knots_ties", "ties_dense"):
        res, info = combine(f1, f2, mode, 1.0, w1=1.0, w2=0.7, density=0.3, rank=8, dare_drop=0.2, seed=5)
        patched = _roundtrip_weights(p, factors_to_state_dict(res), tmp_path)
        msd = p.model.state_dict()
        for k, f in res.items():
            got = patched[k].double() - msd[k].double()
            assert torch.allclose(got, f.dense(), atol=1e-5)


def test_conflict_report_runs():
    from zqx.adapters import get_adapter
    from zqx.patches.lora_arith import conflict_report, lora_to_model_space
    p, l1, l2 = _models_and_loras("qwen")
    f1, _, _ = lora_to_model_space(p, l1)
    f2, _, _ = lora_to_model_space(p, l2)
    txt, js = conflict_report(get_adapter(p), f1, f2)
    assert "block0" in txt and "block1" in txt
    assert set(js["layers"]) == set(f1) & set(f2)


def test_refuses_mismatched_shapes():
    import lora_utils as lu
    from zqx.core.lora_io import LoraFormatError
    from zqx.patches.lora_arith import lora_to_model_space
    import tiny_models as tm
    p = tm.qwen_image(2)
    sd = lu.make_lora("qwen", seed=1)
    k = [k for k in sd if k.endswith("lora_B.weight")][0]
    sd[k] = torch.zeros(sd[k].shape[0] + 1, sd[k].shape[1])
    with pytest.raises(LoraFormatError):
        lora_to_model_space(p, sd)
