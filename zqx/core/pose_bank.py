"""Deterministic selection of control images (pose / depth maps) from a folder bank."""
from __future__ import annotations

import math
import os
import random
from typing import List, Optional, Sequence, Tuple

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def scan_bank(root: str) -> List[Tuple[str, str]]:
    """[(relative path, tag)] for every image under root; tag = first sub-folder ('' for files in root)."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"pose bank folder not found: {root}")
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                parts = rel.split(os.sep)
                out.append((rel, parts[0] if len(parts) > 1 else ""))
    return sorted(out)


def select(entries: Sequence[Tuple[str, str, int, int]], seed: int, width: int, height: int,
           aspect_tolerance: float, tags: Optional[Sequence[str]] = None) -> Tuple[str, int]:
    """entries: (rel path, tag, w, h).  Keeps entries whose tag is in `tags` (None = any) and whose aspect ratio is
    within a factor (1 + aspect_tolerance) of width/height; picks one with random.Random(seed).
    Returns (rel path, number of candidates)."""
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    target = width / height
    lim = math.log1p(max(aspect_tolerance, 0.0))
    cands = [e for e in entries
             if (tags is None or e[1] in tags) and abs(math.log((e[2] / e[3]) / target)) <= lim + 1e-12]
    if not cands:
        raise ValueError(f"no control image matches tags={list(tags) if tags else 'any'} and aspect {width}x{height} "
                         f"(tolerance {aspect_tolerance})")
    cands = sorted(cands)
    return cands[random.Random(int(seed)).randrange(len(cands))][0], len(cands)
