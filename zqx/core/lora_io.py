"""Parse LoRA state dicts (kohya / diffusers-peft / ai-toolkit / musubi) into low-rank factors.

Only plain LoRA (up @ down, optional alpha) on 2-D linear weights is supported.
DoRA (dora_scale), LoHa/LoKr, conv/Tucker (lora_mid), full "diff" patches and
bias patches are refused with a clear error listing the offending keys (no
silent dropping).

Scale convention (identical to comfy/weight_adapter/lora.py):
    delta_W = (alpha / rank) * up @ down     if an alpha tensor is present
    delta_W = up @ down                      if there is no alpha
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch

# (up_suffix, down_suffix) pairs, most specific first
_PAIR_SUFFIXES: List[Tuple[str, str]] = [
    (".lora_up.weight", ".lora_down.weight"),        # kohya / sd-scripts / musubi-tuner
    (".lora_B.default.weight", ".lora_A.default.weight"),  # peft with adapter name
    (".lora_B.weight", ".lora_A.weight"),            # diffusers / peft / ai-toolkit
    (".lora.up.weight", ".lora.down.weight"),        # old diffusers attn-processor format
]
_ALPHA_SUFFIXES = [".alpha", ".lora_alpha"]
_UNSUPPORTED_MARKERS = [
    ".dora_scale", ".lora_mid.", "hada_", "lokr_", ".diff", ".diff_b", ".oft_", ".boft_",
    ".glora_", ".w_norm", ".b_norm", ".reshape_weight",
]


@dataclass
class LoraFactors:
    """delta_W = scale * up @ down;  up: (out, r), down: (r, in)."""
    up: torch.Tensor
    down: torch.Tensor
    scale: float = 1.0
    source_keys: List[str] = field(default_factory=list)

    @property
    def rank(self) -> int:
        return int(self.down.shape[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.up.shape[0]), int(self.down.shape[1])

    def dense(self, dtype=torch.float64) -> torch.Tensor:
        return self.scale * (self.up.to(dtype) @ self.down.to(dtype))

    def folded(self, dtype=None) -> "LoraFactors":
        dtype = dtype or self.up.dtype
        return LoraFactors(self.up.to(dtype) * self.scale, self.down.to(dtype), 1.0, list(self.source_keys))


class LoraFormatError(ValueError):
    pass


def parse_lora_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, LoraFactors]:
    """Group a LoRA state dict into {module_base_key: LoraFactors}.

    Raises LoraFormatError for unsupported components, unpaired up/down,
    non-2D factors, rank mismatches, and keys that are not recognised.
    """
    keys = list(sd.keys())
    unsupported = [k for k in keys if any(m in k for m in _UNSUPPORTED_MARKERS)]
    if unsupported:
        raise LoraFormatError(
            "Unsupported LoRA components (only plain LoRA up/down[/alpha] on linear layers is supported): "
            + ", ".join(unsupported[:12]) + (" ..." if len(unsupported) > 12 else "")
        )
    used = set()
    modules: Dict[str, LoraFactors] = {}
    for k in keys:
        for up_s, down_s in _PAIR_SUFFIXES:
            if k.endswith(up_s):
                base = k[: -len(up_s)]
                dk = base + down_s
                if dk not in sd:
                    raise LoraFormatError(f"LoRA key {k!r} has no matching {down_s!r}")
                up = sd[k]
                down = sd[dk]
                if up.ndim != 2 or down.ndim != 2:
                    raise LoraFormatError(
                        f"{base}: only 2-D (linear) LoRA factors are supported, got up {tuple(up.shape)} down {tuple(down.shape)}"
                    )
                if up.shape[1] != down.shape[0]:
                    raise LoraFormatError(f"{base}: rank mismatch up {tuple(up.shape)} vs down {tuple(down.shape)}")
                alpha = None
                akey = None
                for a_s in _ALPHA_SUFFIXES:
                    if base + a_s in sd:
                        akey = base + a_s
                        alpha = float(sd[akey].float().reshape(-1)[0].item())
                        break
                rank = int(down.shape[0])
                scale = (alpha / rank) if alpha is not None else 1.0
                if base in modules:
                    raise LoraFormatError(f"module {base!r} appears twice with different key formats")
                src = [k, dk] + ([akey] if akey else [])
                modules[base] = LoraFactors(up.float(), down.float(), scale, src)
                used.update(src)
                break
    unknown = [k for k in keys if k not in used]
    # a down key whose up key was matched is already in `used`; anything left is unknown
    if unknown:
        raise LoraFormatError(
            "Unrecognised LoRA keys (refusing to silently drop them): " + ", ".join(unknown[:12])
            + (" ..." if len(unknown) > 12 else "")
        )
    if not modules:
        raise LoraFormatError("No LoRA modules found in state dict")
    return modules


def embed_slice(f: LoraFactors, full_shape: Tuple[int, int], offset: Optional[Tuple[int, int, int]]) -> LoraFactors:
    """Embed a LoRA that targets a slice of a fused weight into the full weight shape.

    offset = (dim, start, size) as used by ComfyUI's key maps (e.g. Z-Image
    to_q/to_k/to_v -> slices of the fused attention.qkv weight).  The result is
    an exact low-rank factorisation of the zero-padded delta.
    """
    out_full, in_full = full_shape
    if offset is None:
        if f.shape != (out_full, in_full):
            raise LoraFormatError(f"LoRA shape {f.shape} does not match weight shape {full_shape}")
        return f
    dim, start, size = offset
    if dim == 0:
        if f.shape != (size, in_full):
            raise LoraFormatError(f"LoRA slice shape {f.shape} does not match slice ({size}, {in_full})")
        up = torch.zeros((out_full, f.rank), dtype=f.up.dtype)
        up[start:start + size] = f.up
        return LoraFactors(up, f.down, f.scale, list(f.source_keys))
    if dim == 1:
        if f.shape != (out_full, size):
            raise LoraFormatError(f"LoRA slice shape {f.shape} does not match slice ({out_full}, {size})")
        down = torch.zeros((f.rank, in_full), dtype=f.down.dtype)
        down[:, start:start + size] = f.down
        return LoraFactors(f.up, down, f.scale, list(f.source_keys))
    raise LoraFormatError(f"unsupported slice dim {dim}")


def concat_factors(parts: Iterable[Tuple[float, LoraFactors]]) -> LoraFactors:
    """Exact low-rank representation of sum_i c_i * delta_W_i:
    up = [c_1 s_1 U_1, ..., c_n s_n U_n], down = [D_1; ...; D_n], scale 1."""
    ups, downs, src = [], [], []
    shape = None
    for c, f in parts:
        if shape is None:
            shape = f.shape
        elif f.shape != shape:
            raise LoraFormatError(f"cannot concatenate LoRA factors of shapes {shape} and {f.shape}")
        ups.append(f.up.to(torch.float32) * (c * f.scale))
        downs.append(f.down.to(torch.float32))
        src.extend(f.source_keys)
    if not ups:
        raise ValueError("nothing to concatenate")
    return LoraFactors(torch.cat(ups, dim=1), torch.cat(downs, dim=0), 1.0, src)


def to_model_key_space(modules: Dict[str, LoraFactors], key_map: Dict[str, object],
                       weight_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[Dict[str, LoraFactors], List[str]]:
    """Map {lora_base_key: factors} to {model_weight_key: factors} using a ComfyUI key map.

    key_map values are either a model key or (model_key, (dim, start, size)).
    Several LoRA modules targeting slices of one fused weight are combined by
    rank concatenation (exact).  Returns (mapped, unmapped_lora_keys).
    """
    grouped: Dict[str, List[LoraFactors]] = {}
    unmapped = []
    for base, f in modules.items():
        target = key_map.get(base, None)
        if target is None:
            unmapped.append(base)
            continue
        if isinstance(target, tuple):
            mkey, offset = target
        else:
            mkey, offset = target, None
        if mkey not in weight_shapes:
            raise LoraFormatError(f"key map points {base!r} at {mkey!r} which is not a model weight")
        shp = tuple(weight_shapes[mkey])
        if len(shp) != 2:
            raise LoraFormatError(f"{mkey}: only 2-D weights supported, got {shp}")
        grouped.setdefault(mkey, []).append(embed_slice(f, (int(shp[0]), int(shp[1])), offset))
    mapped = {}
    for mkey, fs in grouped.items():
        mapped[mkey] = fs[0] if len(fs) == 1 else concat_factors([(1.0, f) for f in fs])
    return mapped, unmapped


def model_key_to_lora_base(model_key: str) -> str:
    """ComfyUI's generic LoRA format: the model weight key without '.weight'
    (comfy/lora.py model_lora_keys_unet maps k[:-len('.weight')] -> k)."""
    if not model_key.endswith(".weight"):
        raise ValueError(f"not a weight key: {model_key}")
    return model_key[: -len(".weight")]


def factors_to_state_dict(mapped: Dict[str, LoraFactors], dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """Serialise {model_key: factors} in kohya-style suffixes with ComfyUI generic base keys.

    The scale is folded into `up` and alpha is written as the rank, so any
    loader that uses alpha/rank (ComfyUI, kohya) reproduces delta_W exactly.
    """
    out = {}
    for mkey, f in mapped.items():
        base = model_key_to_lora_base(mkey)
        ff = f.folded(torch.float32)
        out[base + ".lora_up.weight"] = ff.up.to(dtype).contiguous()
        out[base + ".lora_down.weight"] = ff.down.to(dtype).contiguous()
        out[base + ".alpha"] = torch.tensor(float(ff.rank), dtype=torch.float32)
    return out
