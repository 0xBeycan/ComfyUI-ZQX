from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


class AdapterError(RuntimeError):
    pass


@dataclass
class TokenLayout:
    """Where the target image tokens live inside a main-block attention call.

    img_start / n_img index the joint sequence (B, H, N, D) seen by attention.
    seq_len is the total N expected (checked at every call; a mismatch raises).
    For Z-Image the caption length is only known inside the call, so
    seq_len is None and `img_from_end` gives the image start relative to N.
    """
    n_img: int
    h_tok: int
    w_tok: int
    img_start: Optional[int] = None      # absolute start (Qwen)
    img_from_end: Optional[int] = None   # N - img_from_end = start (Z-Image: image tokens + padding at the end)
    seq_len: Optional[int] = None

    def span(self, n: int) -> Tuple[int, int]:
        if self.img_start is not None:
            if self.seq_len is not None and n != self.seq_len:
                raise AdapterError(
                    f"ZQX: attention sequence length {n} != expected {self.seq_len}; the token layout changed "
                    "(another patch that adds/removes tokens?). Refusing to guess."
                )
            return self.img_start, self.img_start + self.n_img
        start = n - self.img_from_end
        if start < 0:
            raise AdapterError(f"ZQX: attention sequence length {n} is shorter than the image token block {self.img_from_end}")
        return start, start + self.n_img


class ModelAdapter:
    name = "base"

    def __init__(self, model_patcher, diffusion_model):
        self.model_patcher = model_patcher
        self.dm = diffusion_model

    # -- detection ------------------------------------------------------------------------
    @classmethod
    def matches(cls, dm) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- wrapper argument access ----------------------------------------------------------
    def get_x(self, args, kwargs) -> torch.Tensor:
        return args[0]

    def get_timestep(self, args, kwargs) -> torch.Tensor:
        return args[1]

    def get_context(self, args, kwargs) -> torch.Tensor:
        return args[2]

    def replace_args(self, args, kwargs, x=None, timestep=None, context=None):
        args = list(args)
        if x is not None:
            args[0] = x
        if timestep is not None:
            args[1] = timestep
        if context is not None:
            args[2] = context
        return tuple(args), kwargs

    def get_transformer_options(self, args, kwargs) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- token layout ---------------------------------------------------------------------
    def layout(self, x: torch.Tensor, args, kwargs) -> TokenLayout:  # pragma: no cover - abstract
        raise NotImplementedError

    def image_position_ids(self, x: torch.Tensor, n_txt: int) -> torch.Tensor:  # pragma: no cover
        """(n_img, 3) RoPE ids of the target image tokens, as the model builds them."""
        raise NotImplementedError

    # -- RoPE -----------------------------------------------------------------------------
    @property
    def embedder(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def rope_freqs(self, ids: torch.Tensor) -> torch.Tensor:
        """Rotation matrices for ids (N, n_axes) in (1, 1, N, D/2, 2, 2) layout, via the model's EmbedND."""
        if ids.ndim != 2:
            raise ValueError("ids must be (N, n_axes)")
        f = self.embedder(ids.to(torch.float32).unsqueeze(0))  # EmbedND: (1, 1, N, D/2, 2, 2)
        return f

    def rotation_for_offset(self, delta: Tuple[float, float, float], device) -> torch.Tensor:
        """R(delta) as (1, 1, 1, D/2, 2, 2).  R(delta) R(p) = R(p + delta)."""
        ids = torch.tensor([list(delta)], dtype=torch.float32, device=device)
        return self.rope_freqs(ids)

    # -- weight keys ----------------------------------------------------------------------
    block_key_re = None  # regex with one group = block index, matched against model weight keys
    group_rules: List[Tuple[str, str]] = []  # (regex, group)

    def block_of_key(self, model_key: str) -> Tuple[Optional[int], str]:
        m = self.block_key_re.search(model_key) if self.block_key_re is not None else None
        if m is not None:
            return int(m.group(1)), "block"
        for rx, grp in self.group_rules:
            if re.search(rx, model_key):
                return None, grp
        return None, "other"

    @property
    def total_blocks(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    # (regex, kind) checked in order; kinds: text, modulation, attention, mlp, io
    module_kind_rules: List[Tuple[str, str]] = []
    MODULE_KINDS = ("text", "modulation", "attention", "mlp", "io", "other")

    # token stream a module's input lives in: image | text | joint | nonspatial
    stream_rules: List[Tuple[str, str]] = []

    def module_stream(self, model_key: str) -> str:
        for rx, stream in self.stream_rules:
            if re.search(rx, model_key):
                return stream
        raise AdapterError(f"ZQX: no token-stream rule for {model_key}")

    def module_kind(self, model_key: str) -> str:
        for rx, kind in self.module_kind_rules:
            if re.search(rx, model_key):
                return kind
        return "other"

    # -- misc -----------------------------------------------------------------------------
    @property
    def base_model(self):
        return self.model_patcher.model


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)
