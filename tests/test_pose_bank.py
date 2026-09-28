import os

import pytest
import torch

from conftest import requires_comfy
from zqx.core.pose_bank import scan_bank, select


def _bank(tmp_path):
    from PIL import Image
    spec = {"walking/a.png": (64, 96), "walking/b.png": (60, 100), "sitting/c.png": (96, 64), "d.png": (64, 64)}
    for rel, (w, h) in spec.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (w, h), (int(w), 10, 200)).save(p)
    return spec


def test_select_filters_and_is_deterministic(tmp_path):
    spec = _bank(tmp_path)
    entries = [(rel, tag, *spec[rel.replace(os.sep, "/")]) for rel, tag in scan_bank(str(tmp_path))]
    assert {t for _, t, _, _ in entries} == {"walking", "sitting", ""}
    rel, n = select(entries, 3, 832, 1216, 0.15)        # portrait 0.684: a (0.667) and b (0.6) within 15%? b: 0.6/0.684 = 0.877 -> 1.14 ok
    assert n == 2 and rel.startswith("walking")
    assert select(entries, 3, 832, 1216, 0.15) == (rel, n)
    picks = {select(entries, s, 832, 1216, 0.15)[0] for s in range(20)}
    assert len(picks) == 2
    assert select(entries, 0, 1216, 832, 0.1, ["sitting"])[0].startswith("sitting")
    with pytest.raises(ValueError):
        select(entries, 0, 1216, 832, 0.1, ["walking"])


@requires_comfy
def test_pose_bank_node(tmp_path):
    from zqx.nodes.tool_nodes import ZQXPoseBank
    _bank(tmp_path)
    img, rel, n = ZQXPoseBank().pick(str(tmp_path), "walking", 64, 96, 0.2, "crop_to_size", 5)
    assert img.shape == (1, 96, 64, 3) and n == 2 and 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    img2, rel2, _ = ZQXPoseBank().pick(str(tmp_path), "", 64, 64, 0.01, "keep", 1)
    assert rel2 == "d.png" and img2.shape == (1, 64, 64, 3)
