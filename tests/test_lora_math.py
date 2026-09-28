"""Pure-math LoRA tests (no ComfyUI needed)."""
import math

import pytest
import torch

from zqx.core import lora_math as LM
from zqx.core.lora_io import LoraFactors, LoraFormatError, parse_lora_state_dict, concat_factors, embed_slice


def _f(out, inp, r, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return LoraFactors(torch.randn(out, r, generator=g), torch.randn(r, inp, generator=g), scale)


A = _f(24, 20, 4, 1, scale=0.5)
B = _f(24, 20, 3, 2, scale=2.0)


def test_parse_formats_and_alpha_folding():
    g = torch.Generator().manual_seed(0)
    up, down = torch.randn(6, 2, generator=g), torch.randn(2, 5, generator=g)
    for sd, scale in [
        ({"lora_unet_x.lora_up.weight": up, "lora_unet_x.lora_down.weight": down, "lora_unet_x.alpha": torch.tensor(1.0)}, 0.5),
        ({"diffusion_model.x.lora_B.weight": up, "diffusion_model.x.lora_A.weight": down}, 1.0),
        ({"x.lora_B.default.weight": up, "x.lora_A.default.weight": down, "x.alpha": torch.tensor(8.0)}, 4.0),
        ({"x.lora.up.weight": up, "x.lora.down.weight": down}, 1.0),
    ]:
        mods = parse_lora_state_dict(sd)
        (f,) = mods.values()
        assert f.scale == scale
        assert torch.allclose(f.dense(), scale * up.double() @ down.double())
        ff = f.folded()
        assert ff.scale == 1.0 and torch.allclose(ff.dense(), f.dense(), atol=1e-6)


def test_parse_refuses_unsupported():
    up, down = torch.zeros(4, 2), torch.zeros(2, 4)
    with pytest.raises(LoraFormatError, match="Unsupported"):
        parse_lora_state_dict({"x.lora_up.weight": up, "x.lora_down.weight": down, "x.dora_scale": torch.ones(4, 1)})
    with pytest.raises(LoraFormatError, match="Unrecognised"):
        parse_lora_state_dict({"x.lora_up.weight": up, "x.lora_down.weight": down, "y.something": up})
    with pytest.raises(LoraFormatError, match="no matching"):
        parse_lora_state_dict({"x.lora_up.weight": up})
    with pytest.raises(LoraFormatError, match="2-D"):
        parse_lora_state_dict({"x.lora_up.weight": torch.zeros(4, 2, 1, 1), "x.lora_down.weight": torch.zeros(2, 4, 1, 1)})


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 1.7])
def test_exact_modes_match_dense_formulas(lam):
    d1, d2 = A.dense(), B.dense()
    q2 = LM.orth_basis(d2)
    p2 = LM.orth_basis(d2.T)
    q1 = LM.orth_basis(d1)
    cases = {
        "add": (LM.op_add(A, B, lam), d1 + lam * d2),
        "negate": (LM.op_negate(A, B, lam), d1 - lam * d2),
        "clean_col": (LM.op_clean_col(A, B, lam), d1 - lam * q2 @ q2.T @ d1),
        "clean_row": (LM.op_clean_row(A, B, lam), d1 - lam * d1 @ p2 @ p2.T),
        "target_sub": (LM.op_target_sub(A, B, lam), d1 - lam * q1 @ q1.T @ d2),
    }
    for name, (f, ref) in cases.items():
        assert torch.allclose(f.dense(), ref, atol=1e-4 * ref.abs().max().item() + 1e-6), name
    # negate is exactly the rank-concatenated form
    f = LM.op_negate(A, B, lam)
    assert f.rank == A.rank + B.rank


def test_projections_idempotent_and_orthogonal():
    once = LM.op_clean_col(A, B, 1.0)
    twice = LM.op_clean_col(once, B, 1.0)
    assert torch.allclose(once.dense(), twice.dense(), atol=1e-4)
    q2 = LM.orth_basis(B.dense())
    assert (q2.T @ once.dense()).abs().max() < 1e-4
    r1 = LM.op_clean_row(A, B, 1.0)
    p2 = LM.orth_basis(B.dense().T)
    assert (r1.dense() @ p2).abs().max() < 1e-4
    assert torch.allclose(LM.op_clean_row(r1, B, 1.0).dense(), r1.dense(), atol=1e-4)
    # projector matrix idempotent
    P = q2 @ q2.T
    assert torch.allclose(P @ P, P, atol=1e-10)


def test_rank_deficient_direction_uses_true_column_space():
    g = torch.Generator().manual_seed(5)
    up = torch.randn(10, 3, generator=g)
    down = torch.zeros(3, 8)
    down[0] = torch.randn(8, generator=g)        # rank-1 product even though up has rank 3
    Bd = LoraFactors(up, down, 1.0)
    q = LM._col_basis(Bd)
    assert q.shape[1] == 1
    assert torch.allclose(q @ q.T @ Bd.dense(), Bd.dense(), atol=1e-10)


def test_ties_merge_matches_hand_computation():
    t1 = torch.tensor([[3.0, -1.0, 0.5], [2.0, 0.1, -4.0]])
    t2 = torch.tensor([[-2.0, -3.0, 0.4], [1.0, 0.2, 5.0]])
    # density 1: no trimming.  sign = sgn(t1 + t2) = [[+,-,+],[+,+,+]]
    m = LM.ties_merge([t1, t2], [1.0, 1.0], 1.0)
    exp = torch.tensor([[3.0, -2.0, 0.45], [1.5, 0.15, 5.0]])
    assert torch.allclose(m, exp)
    # density 0.5: keep top-3 by |.| of each tensor
    tr1 = LM.trim_topk(t1, 0.5)
    assert torch.equal(tr1 != 0, torch.tensor([[True, False, False], [True, False, True]]))


def test_dare_unbiased_and_seeded():
    t = torch.randn(200, 200, generator=torch.Generator().manual_seed(0))
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    a, b = LM.dare(t, 0.9, g1), LM.dare(t, 0.9, g2)
    assert torch.equal(a, b)
    kept = (a != 0).float().mean().item()
    assert abs(kept - 0.1) < 0.01
    assert torch.allclose(a[a != 0], t[a != 0] / 0.1)


def test_knots_ties_is_exact_in_shared_basis():
    f, info = LM.op_knots_ties(A, B, 1.0, 1.0, density=1.0)
    assert info["shared_rank"] <= A.rank + B.rank
    # with density 1 and aligned signs, TIES of (C1, C2) in the shared basis; reconstruct manually
    d1, d2 = A.dense(), B.dense()
    u, s, _ = torch.linalg.svd(torch.cat([d1, d2], dim=1), full_matrices=False)
    u = u[:, s > 1e-6 * s[0]]
    c1, c2 = u.T @ d1, u.T @ d2
    ref = u @ LM.ties_merge([c1, c2], [1.0, 1.0], 1.0)
    assert torch.allclose(f.dense(), ref, atol=1e-6)
    # basis spans both (col(dW_i) inside col(U)) -> representation exact
    assert torch.allclose(u @ (u.T @ d1), d1, atol=1e-6)


def test_ties_dense_error_report():
    f, info = LM.op_ties_dense(A, B, 1.0, 1.0, density=0.3, rank=5)
    merged = LM.ties_merge([A.dense(), B.dense()], [1.0, 1.0], 0.3)
    err = (f.dense() - merged).norm() / merged.norm()
    assert abs(err.item() - info["svd_rel_error"]) < 1e-6
    full, info2 = LM.op_ties_dense(A, B, 1.0, 1.0, density=0.3, rank=100)
    assert info2["svd_rel_error"] < 1e-12


def test_pair_stats_against_dense():
    st = LM.pair_stats(A, B, 0.2)
    d1, d2 = A.dense(), B.dense()
    assert st.cosine == pytest.approx(float((d1 * d2).sum() / (d1.norm() * d2.norm())), abs=1e-9)
    q1, q2 = LM.orth_basis(d1), LM.orth_basis(d2)
    assert st.col_overlap == pytest.approx(float((q1.T @ q2).pow(2).sum()) / min(q1.shape[1], q2.shape[1]), abs=1e-9)
    assert st.energy1_in_col2 == pytest.approx(float((q2 @ q2.T @ d1).pow(2).sum() / d1.pow(2).sum()), abs=1e-9)
    same = LM.pair_stats(A, A, 0.2)
    assert same.col_overlap == pytest.approx(1.0) and same.cosine == pytest.approx(1.0) and same.sign_conflict_all == 0.0


def test_klora_primitives():
    assert LM.klora_gamma([1.0, 1.0, 1.0, 1.0, 100.0]) == pytest.approx(1.0)   # 100 >= 3 * mean(20.8) -> dropped
    assert LM.klora_gamma([1.0, 1.0, 100.0]) == pytest.approx(34.0)            # 100 < 3 * 34 -> kept
    assert LM.klora_time_scale(0.0, 1.5, 0.5, "s") == 0.5
    assert LM.klora_time_scale(1.0, 1.5, 0.5, "s") == 2.0
    assert LM.klora_time_scale(0.0, 1.5, 1.275, "s*") == pytest.approx(1.275)
    assert LM.klora_time_scale(0.5, 1.5, 1.275, "s*") == pytest.approx(math.fmod(0.75 + 1.275, 1.5))
    assert LM.klora_use_content(10.0, 1.0, 1.0, 2.0) is True
    assert LM.klora_use_content(1.0, 1.0, 1.0, 2.0) is False


def test_embed_slice_and_concat_exact():
    f = _f(4, 6, 2, 9)
    e = embed_slice(f, (12, 6), (0, 4, 4))
    d = e.dense()
    assert torch.allclose(d[4:8], f.dense()) and d[:4].abs().max() == 0 and d[8:].abs().max() == 0
    c = concat_factors([(1.0, A), (-0.3, B)])
    assert torch.allclose(c.dense(), A.dense() - 0.3 * B.dense(), atol=1e-4)
