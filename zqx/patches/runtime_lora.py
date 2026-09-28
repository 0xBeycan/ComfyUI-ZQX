"""Runtime (un-merged) LoRA whose strength depends on the current sigma and the block.

Mechanism: ModelPatcher.add_weight_wrapper(key, fn).  ComfyUI calls every
registered weight function inside comfy.ops.cast_bias_weight on each forward
(after casting to the compute dtype and after the regular merged/low-vram LoRA
patches), and forces the cast path for those modules.  Our function returns

    W_eff = W + sum_i s_i(sigma, block) * dW_i

where dW_i is computed by ComfyUI's own comfy.lora.calculate_weight with the
LoRA adapters produced by comfy.lora.load_lora (so alpha/rank, DoRA, LoKr,
fused-qkv slices etc. behave exactly like LoraLoader).  At a constant strength
s the result equals LoraLoaderModelOnly(strength_model=s) up to float rounding
(tested).  Changing the strength between steps needs no re-patching.

The current sigma is published by a DIFFUSION_MODEL wrapper right before the
model runs and cleared afterwards; a weight function called outside of a
sampling call raises instead of guessing.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..core.schedule import sigma_from_transformer_options


StrengthFn = Callable[[float], float]


class RuntimeLoraPatch:
    def __init__(self, adapter, key_prefix: str):
        self.adapter = adapter
        self.key = key_prefix
        self.sigma: Optional[float] = None
        # model_key -> list of (strength_fn, lora_adapter, offset, function)
        self.entries: Dict[str, List[Tuple[StrengthFn, object, object, object]]] = {}
        self.eval_log: List[Tuple[float, str, float]] = []   # (sigma, key, strength) of the last run; for tests
        self.log_enabled = False

    def add(self, model_key: str, strength_fn: StrengthFn, lora_adapter, offset=None, function=None):
        self.entries.setdefault(model_key, []).append((strength_fn, lora_adapter, offset, function))

    # DIFFUSION_MODEL wrapper -------------------------------------------------------------
    def wrapper(self, executor, *args, **kwargs):
        to = self.adapter.get_transformer_options(args, kwargs)
        prev = self.sigma
        self.sigma = sigma_from_transformer_options(to)
        try:
            return executor(*args, **kwargs)
        finally:
            self.sigma = prev

    # weight function ---------------------------------------------------------------------
    def make_weight_fn(self, model_key: str):
        import comfy.lora

        entries = self.entries[model_key]

        def weight_fn(weight: torch.Tensor) -> torch.Tensor:
            sigma = self.sigma
            if sigma is None:
                raise RuntimeError(f"ZQX runtime LoRA: weight {model_key} evaluated outside a sampling call (no sigma).")
            patches = []
            for strength_fn, lora_adapter, offset, function in entries:
                s = float(strength_fn(sigma))
                if not math.isfinite(s):
                    raise ValueError(f"ZQX runtime LoRA: non-finite strength for {model_key}")
                if self.log_enabled:
                    self.eval_log.append((sigma, model_key, s))
                if s != 0.0:
                    patches.append((s, lora_adapter, 1.0, offset, function))
            if not patches:
                return weight
            orig_dtype = weight.dtype
            w32 = weight.to(torch.float32, copy=True)
            w32 = comfy.lora.calculate_weight(patches, w32, model_key, intermediate_dtype=torch.float32)
            return w32.to(orig_dtype)

        return weight_fn


class _CaptureNotLoaded:
    """Collect the keys ComfyUI's load_lora reports as 'lora key not loaded' (its own definition)."""

    def __init__(self):
        import logging
        self.keys = []
        outer = self

        class H(logging.Handler):
            def emit(self, record):
                msg = record.getMessage()
                if msg.startswith("lora key not loaded:"):
                    outer.keys.append(msg.split(":", 1)[1].strip())
        self.handler = H(level=logging.WARNING)

    def __enter__(self):
        import logging
        logging.getLogger().addHandler(self.handler)
        return self

    def __exit__(self, *exc):
        import logging
        logging.getLogger().removeHandler(self.handler)
        return False


def load_lora_patches(model_patcher, lora_sd: Dict[str, torch.Tensor], allow_unmatched: bool = False):
    """comfy.lora.load_lora with the model's key map -> {model_key: [(adapter, offset, function)]}.

    Unmatched LoRA keys (which LoraLoader only logs) raise unless allow_unmatched.
    """
    import comfy.lora
    import comfy.lora_convert

    key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, {})
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    with _CaptureNotLoaded() as cap:
        loaded = comfy.lora.load_lora(lora_sd, key_map, log_missing=True)
    if len(loaded) == 0:
        raise ValueError("ZQX: none of the LoRA keys match this model (wrong model family or unsupported key format).")
    if cap.keys and not allow_unmatched:
        raise ValueError(f"ZQX: {len(cap.keys)} LoRA keys do not match this model (LoraLoader would silently skip them): "
                         + ", ".join(cap.keys[:10]) + (" ..." if len(cap.keys) > 10 else "")
                         + ". Set allow_unmatched_keys=True to ignore them explicitly.")
    out: Dict[str, List[Tuple[object, object, object]]] = {}
    model_sd_keys = set(model_patcher.model.state_dict().keys())
    for k, v in loaded.items():
        if isinstance(k, str):
            mk, offset, function = k, None, None
        else:
            mk, offset = k[0], k[1]
            function = k[2] if len(k) > 2 else None
        if mk not in model_sd_keys:
            raise ValueError(f"ZQX: LoRA maps to {mk}, which is not a model weight")
        out.setdefault(mk, []).append((v, offset, function))
    return out, list(cap.keys)


def install_runtime_lora(model_patcher, entries_by_key: Dict[str, List[Tuple[StrengthFn, object, object, object]]],
                         key_prefix: str):
    """Clone the patcher and register one weight wrapper per model key + the sigma wrapper."""
    from ..adapters import get_adapter
    import comfy.patcher_extension as pe

    m = model_patcher.clone()
    adapter = get_adapter(m)
    patch = RuntimeLoraPatch(adapter, key_prefix)
    for mk, lst in entries_by_key.items():
        for (fn, v, off, func) in lst:
            patch.add(mk, fn, v, off, func)
    for mk in patch.entries:
        m.add_weight_wrapper(mk, patch.make_weight_fn(mk))
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, key_prefix, patch.wrapper)
    return m, patch
