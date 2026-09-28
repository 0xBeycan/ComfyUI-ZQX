"""Reference attention injection (capture pass + inject pass) for Qwen-Image / QIE / Z-Image.

Mechanism (extended self-attention, as in reference-only / ConsiStory /
StoryDiffusion / Personalize-Anything's attention stage, adapted to joint-
attention DiTs with RoPE):

1. For every model call inside the sigma window, a DIFFUSION_MODEL wrapper
   first runs the model on the *reference* latent (same text conditioning,
   same batch layout) and the attention override records, for every selected
   main block, the post-RoPE keys and the values of the reference image tokens.
   - "noised": reference noised to the current sigma with a fixed noise
     sample (x_ref = (1 - s) ref + s eps_fixed), same timestep as the target.
     This is what Personalize Anything's RF-inversion call reduces to with
     gamma = eta = 1, and what reference-only does in UNets.
   - "cached": reference at a fixed `cache_sigma` (0 = clean, like Qwen-Image-Edit's
     'index_timestep_zero' / OminiControl's clean condition branch), computed
     once per sampling run and conditioning, then reused every step.
2. Then the real forward runs; in every selected block the image queries of
   the target attend to [own keys ; R(delta) K_ref] with a log-bias
   log(w * mq * mk) on the reference columns (see zqx/core/attention.py).
   R(delta) moves the reference to a non-overlapping RoPE position using the
   model's own EmbedND (rotation composition, zqx/core/rope.py).

Guarantees (tested): outside the window / weight 0 / all-zero query mask the
wrapper calls the model exactly once with unchanged inputs (bitwise identical
output); cond and uncond rows only ever see references computed from their
own conditioning; capture and inject passes see identical block sequences.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..adapters import AdapterError, ModelAdapter, QwenImageAdapter, ZImageAdapter
from ..core.attention import build_extended_mask, reference_log_bias
from ..core.rope import apply_rotation
from ..core.schedule import in_sigma_window, parse_block_list, sigma_from_transformer_options
from . import passes

POSITION_MODES = ["frame", "right", "below", "same", "matched"]
CAPTURE_MODES = ["noised", "cached"]


@dataclass
class RefAttnConfig:
    ref_latent: torch.Tensor                  # raw VAE latent (as from VAEEncode), batch 1
    weight: float = 1.0
    sigma_start: float = 1.0
    sigma_end: float = 0.0
    blocks: str = "all"
    position_mode: str = "frame"
    capture_mode: str = "noised"
    cache_sigma: float = 0.0
    noise_seed: int = 0
    inject_uncond: bool = True
    ref_sigma_mult: float = 1.0                # noised mode: reference noise level = mult * sigma (FreeCus: < 1)
    key_scale: float = 1.0                     # multiply reference keys (FreeCus uses 1.1): sharper/softer ref attention
    token_dropout: float = 0.0                 # ConsiStory-style: drop this fraction of reference tokens each step
    match_threshold: float = 0.5               # 'matched' mode: minimum cosine similarity of value vectors
    match_mutual: bool = True                  # 'matched' mode: keep only mutual nearest neighbours
    query_mask: Optional[torch.Tensor] = None  # (H, W) in [0, 1], any resolution (resized to the token grid)
    key_mask: Optional[torch.Tensor] = None    # (H, W) in [0, 1] over the reference image


@dataclass
class _CallState:
    mode: str                                  # "capture" | "inject"
    blocks: Optional[List[int]]
    ref_layout: object = None
    layout: object = None
    rot: Optional[torch.Tensor] = None
    store: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    qmask_img: Optional[torch.Tensor] = None   # (B, n_img) in [0,1]
    kmask: Optional[torch.Tensor] = None       # (1, n_ref) in [0,1]
    trace: List[Tuple[str, int]] = field(default_factory=list)
    ref_s0: Optional[int] = None               # start of the reference image span in the capture pass
    ref_x: Optional[torch.Tensor] = None       # reference model input (for its RoPE ids in 'matched' mode)
    tgt_x: Optional[torch.Tensor] = None
    to: Optional[dict] = None
    match_frac: List[float] = field(default_factory=list)


def resize_mask_to_tokens(mask: torch.Tensor, h_tok: int, w_tok: int) -> torch.Tensor:
    """Area-average a (H, W) or (1, H, W) mask to the (h_tok, w_tok) token grid, flattened row-major."""
    m = mask
    if m.ndim == 3:
        if m.shape[0] != 1:
            raise ValueError("ZQX: masks with batch > 1 are not supported; pass a single mask")
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"ZQX: mask must be (H, W), got {tuple(mask.shape)}")
    m = m.to(torch.float32)
    if torch.any(m < 0) or torch.any(m > 1):
        raise ValueError("ZQX: mask values must be in [0, 1]")
    r = F.interpolate(m[None, None], size=(h_tok, w_tok), mode="area")[0, 0]
    return r.clamp(0.0, 1.0).reshape(-1)


def _fingerprint(obj) -> str:
    """Content hash of the tensors in a (nested) structure; used as cache key for 'cached' mode."""
    h = hashlib.sha1()

    def rec(o):
        if torch.is_tensor(o):
            t = o.detach().to("cpu", torch.float64).contiguous()
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(o, (list, tuple)):
            h.update(b"[")
            for e in o:
                rec(e)
            h.update(b"]")
        elif isinstance(o, dict):
            for k in sorted(o.keys(), key=str):
                if k in ("transformer_options", "control"):
                    continue
                h.update(str(k).encode())
                rec(o[k])
        else:
            h.update(repr(o).encode())

    rec(obj)
    return h.hexdigest()


class ReferenceAttentionPatch:
    """Holds config + per-call state; provides the DIFFUSION_MODEL wrapper and the attention override."""

    WRAPPER_KEY = "zqx_reference_attention"

    def __init__(self, adapter: ModelAdapter, cfg: RefAttnConfig, prev_override=None):
        if cfg.position_mode not in POSITION_MODES:
            raise ValueError(f"position_mode must be one of {POSITION_MODES}")
        if cfg.capture_mode not in CAPTURE_MODES:
            raise ValueError(f"capture_mode must be one of {CAPTURE_MODES}")
        if cfg.weight < 0 or not math.isfinite(cfg.weight):
            raise ValueError("weight must be finite and >= 0")
        if cfg.sigma_start < cfg.sigma_end:
            raise ValueError("sigma_start must be >= sigma_end")
        if not (0.0 <= cfg.cache_sigma <= 1.0):
            raise ValueError("cache_sigma must be in [0, 1]")
        if not (0.0 <= cfg.ref_sigma_mult <= 1.0):
            raise ValueError("ref_sigma_mult must be in [0, 1]")
        if not (cfg.key_scale > 0 and math.isfinite(cfg.key_scale)):
            raise ValueError("key_scale must be > 0")
        if not (0.0 <= cfg.token_dropout < 1.0):
            raise ValueError("token_dropout must be in [0, 1)")
        self.adapter = adapter
        self.cfg = cfg
        self.block_list = parse_block_list(cfg.blocks, adapter.total_blocks)
        self.prev_override = prev_override
        self.state: Optional[_CallState] = None
        self._noise: Optional[torch.Tensor] = None
        self._cache: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.last_trace: List[Tuple[str, int]] = []   # for tests / debugging
        self._last_store = {}
        self.calls_patched = 0

    # ------------------------------------------------------------------ helpers
    def clear_cache(self, *args, **kwargs):
        self._cache = {}

    def _ref0(self, like: torch.Tensor) -> torch.Tensor:
        """Reference latent in model space (process_latent_in), shaped like the model input."""
        ref = self.cfg.ref_latent
        if ref.shape[0] != 1:
            raise ValueError("ZQX: reference latent must have batch size 1")
        if like.ndim == 5 and ref.ndim == 4:
            ref = ref.unsqueeze(2)
        elif like.ndim == 4 and ref.ndim == 5:
            if ref.shape[2] != 1:
                raise ValueError("ZQX: multi-frame reference latents are not supported")
            ref = ref[:, :, 0]
        if ref.shape[1] != like.shape[1]:
            raise ValueError(f"ZQX: reference latent has {ref.shape[1]} channels, model input has {like.shape[1]}")
        base = self.adapter.base_model
        ref = base.process_latent_in(ref.to(torch.float32))
        return ref.to(device=like.device)

    def _fixed_noise(self, ref0: torch.Tensor) -> torch.Tensor:
        if self._noise is None or self._noise.shape != ref0.shape:
            g = torch.Generator(device="cpu").manual_seed(int(self.cfg.noise_seed))
            self._noise = torch.randn(ref0.shape, generator=g, dtype=torch.float32)
        return self._noise.to(ref0.device)

    def _ref_input(self, x: torch.Tensor, sigma_value: float) -> torch.Tensor:
        """x_ref at noise level sigma_value, batch-expanded to x.shape[0], in model input space."""
        ref0 = self._ref0(x)
        eps = self._fixed_noise(ref0)
        ms = self.adapter.base_model.model_sampling
        sig = torch.full((1,), float(sigma_value), dtype=torch.float32, device=x.device)
        xr = ms.noise_scaling(sig, eps, ref0)
        xr = ms.calculate_input(sig, xr)
        xr = xr.expand(x.shape[0], *xr.shape[1:]).to(x.dtype)
        return xr.contiguous()

    def _position_delta(self, x: torch.Tensor, ref_in: torch.Tensor, args, kwargs, to) -> Tuple[float, float, float]:
        mode = self.cfg.position_mode
        ad = self.adapter
        if mode == "same":
            return (0.0, 0.0, 0.0)
        if mode == "matched":
            return None
        if isinstance(ad, QwenImageAdapter):
            refs = ad.get_ref_latents(args, kwargs)
            n_native = len(refs) if refs is not None else 0
            if mode == "frame":
                return (float(1 + n_native), 0.0, 0.0)
            ht, wt = ad.token_grid(x)
            hr, wr = ad.token_grid(ref_in)
            if mode == "right":
                # target w ids: [-wt//2, wt-1-wt//2]; ref w ids: [-wr//2, ...]; first ref column right after target
                return (0.0, 0.0, float((wt - 1 - wt // 2) - (-(wr // 2)) + 1))
            if mode == "below":
                return (0.0, float((ht - 1 - ht // 2) - (-(hr // 2)) + 1), 0.0)
        if isinstance(ad, ZImageAdapter):
            if mode == "frame":
                return (1.0, 0.0, 0.0)
            if to.get("rope_options", None) is not None:
                raise AdapterError("ZQX: position_mode right/below is not supported with rope_options on Z-Image")
            ht, wt = ad.token_grid(x)
            if mode == "right":
                return (0.0, 0.0, float(wt))
            if mode == "below":
                return (0.0, float(ht), 0.0)
        raise AdapterError(f"ZQX: position mode {mode!r} not supported for {ad.name}")

    def _batch_gate(self, to, batch: int, device) -> torch.Tensor:
        """(B, 1) gate: 0 for uncond rows when inject_uncond is False."""
        gate = torch.ones((batch, 1), dtype=torch.float32, device=device)
        if self.cfg.inject_uncond:
            return gate
        cou = to.get("cond_or_uncond", None)
        if cou is None:
            raise RuntimeError("ZQX: transformer_options['cond_or_uncond'] missing; cannot exclude uncond rows")
        n = len(cou)
        if n == 0 or batch % n != 0:
            raise RuntimeError(f"ZQX: batch {batch} is not a multiple of the number of cond chunks {n}")
        per = batch // n
        for i, c in enumerate(cou):
            if c == 1:  # comfy.samplers: 0 = cond, 1 = uncond
                gate[i * per:(i + 1) * per] = 0.0
        return gate

    # ------------------------------------------------------------------ wrapper
    def diffusion_model_wrapper(self, executor, *args, **kwargs):
        if passes.in_foreign_pass():
            return passes.call(executor, args, kwargs)
        with passes.entry("enter", self, args, kwargs):
            return self._wrapper(executor, *args, **kwargs)

    def _wrapper(self, executor, *args, **kwargs):
        ad = self.adapter
        to = ad.get_transformer_options(args, kwargs)
        to.pop("block_index", None)
        sigma = sigma_from_transformer_options(to)
        cfg = self.cfg
        if cfg.weight == 0.0 or not in_sigma_window(sigma, cfg.sigma_start, cfg.sigma_end):
            return passes.call(executor, args, kwargs)
        x = ad.get_x(args, kwargs)
        batch = x.shape[0]
        layout = ad.layout(x, args, kwargs)
        gate = self._batch_gate(to, batch, x.device)
        qimg = gate.expand(batch, layout.n_img).clone()
        if cfg.query_mask is not None:
            qimg = qimg * resize_mask_to_tokens(cfg.query_mask, layout.h_tok, layout.w_tok).to(x.device)[None]
        if not bool(torch.any(qimg > 0)):
            return passes.call(executor, args, kwargs)

        if cfg.capture_mode == "noised":
            ref_sigma = sigma * cfg.ref_sigma_mult
        else:
            ref_sigma = cfg.cache_sigma
        ref_in = self._ref_input(x, ref_sigma)
        if cfg.capture_mode == "noised" and cfg.ref_sigma_mult == 1.0:
            ref_t = ad.get_timestep(args, kwargs)  # identical timestep tensor as the target
        else:
            ms = ad.base_model.model_sampling
            t0 = ms.timestep(torch.full((batch,), float(ref_sigma), dtype=torch.float32, device=x.device)).float()
            ref_t = ad.base_model.process_timestep(t0, x=ref_in)
        ref_layout = ad.layout(ref_in, args, kwargs)
        kmask = torch.ones((1, ref_layout.n_img), dtype=torch.float32, device=x.device)
        if cfg.key_mask is not None:
            kmask = resize_mask_to_tokens(cfg.key_mask, ref_layout.h_tok, ref_layout.w_tok).to(x.device)[None]
            if not bool(torch.any(kmask > 0)):
                return passes.call(executor, args, kwargs)
        if cfg.token_dropout > 0:
            # deterministic per (seed, sigma): same drop pattern for every block of this step
            g = torch.Generator(device="cpu").manual_seed(int(cfg.noise_seed) * 1000003 + int(round(sigma * 1e6)))
            keep = (torch.rand((1, ref_layout.n_img), generator=g) >= cfg.token_dropout).to(kmask)
            kmask = kmask * keep.to(kmask.device)
            if not bool(torch.any(kmask > 0)):
                return passes.call(executor, args, kwargs)

        delta = self._position_delta(x, ref_in, args, kwargs, to)
        rot = None if delta is None else ad.rotation_for_offset(delta, x.device)

        # ---- capture pass
        store = None
        ref_s0 = None
        cache_key = None
        if cfg.capture_mode == "cached":
            ctx_args = [a for i, a in enumerate(args) if i not in (0, 1) and a is not to]
            cache_key = _fingerprint((batch, tuple(ref_in.shape), cfg.cache_sigma, ctx_args, kwargs))
            cached = self._cache.get(cache_key, None)
            if cached is not None:
                store, ref_s0 = cached
        if store is None:
            st = _CallState(mode="capture", blocks=self.block_list, ref_layout=ref_layout)
            self.state = st
            try:
                cargs, ckwargs = ad.replace_args(args, kwargs, x=ref_in, timestep=ref_t)
                to.pop("block_index", None)
                with passes.entry("foreign", self, cargs, ckwargs):
                    passes.call(executor, cargs, ckwargs)
            finally:
                self.state = None
            store = st.store
            ref_s0 = st.ref_s0
            capture_trace = [b for m, b in st.trace]
            if cache_key is not None:
                self._cache[cache_key] = (store, ref_s0)
        else:
            capture_trace = None

        # ---- inject pass
        st = _CallState(mode="inject", blocks=self.block_list, layout=layout, rot=rot, store=store,
                        qmask_img=qimg, kmask=kmask, ref_s0=ref_s0, ref_x=ref_in, tgt_x=x, to=to)
        self.state = st
        try:
            to.pop("block_index", None)
            out = passes.call(executor, args, kwargs)
        finally:
            self.state = None
        inject_trace = [b for m, b in st.trace]
        if capture_trace is not None and capture_trace != inject_trace:
            raise AdapterError(f"ZQX: capture and inject passes saw different block sequences: {capture_trace} vs {inject_trace}")
        if capture_trace is None and sorted(store.keys()) != sorted(set(inject_trace)):
            raise AdapterError("ZQX: cached reference does not cover the injected blocks")
        self._last_store = store
        self.last_match_frac = list(st.match_frac)
        self.last_trace = [("capture", b) for b in (capture_trace or [])] + [("inject", b) for b in inject_trace]
        self.calls_patched += 1
        return out

    # ------------------------------------------------------------------ matched positions
    def _match_and_rotate(self, kr, vr, v_tgt, s0, st):
        """FreeGraftor / CharaConsist-style placement: each reference token is matched to the target image token
        with the most similar value vector (cosine over all heads; V carries no RoPE).  Matched reference keys are
        rotated from their own position p_r to the matched target position p_t by R(p_t - p_r); unmatched tokens
        (similarity < threshold, or not mutual nearest neighbours) are masked out."""
        ad = self.adapter
        b, h, nr, d = kr.shape
        n_img = v_tgt.shape[2]
        fr = torch.nn.functional.normalize(vr.to(torch.float32).permute(0, 2, 1, 3).reshape(b, nr, h * d), dim=-1)
        ft = torch.nn.functional.normalize(v_tgt.to(torch.float32).permute(0, 2, 1, 3).reshape(b, n_img, h * d), dim=-1)
        sim = fr @ ft.transpose(1, 2)                      # (B, Nr, n_img)
        conf, best_t = sim.max(dim=2)                      # per reference token
        keep = conf >= self.cfg.match_threshold
        if self.cfg.match_mutual:
            best_r = sim.argmax(dim=1)                     # (B, n_img): best reference token per target token
            keep = keep & (torch.gather(best_r, 1, best_t) == torch.arange(nr, device=kr.device)[None])
        ids_t = ad.image_ids_for_span(st.tgt_x, s0, st.to).to(kr.device)       # (n_img, 3)
        ids_r = ad.image_ids_for_span(st.ref_x, st.ref_s0, st.to).to(kr.device)  # (Nr, 3)
        if ids_r.shape[0] != nr or ids_t.shape[0] != n_img:
            raise AdapterError("ZQX: token/position count mismatch in matched mode")
        delta = ids_t[best_t] - ids_r[None]               # (B, Nr, 3)
        freqs = ad.rope_freqs(delta.reshape(-1, 3).float())          # (1, 1, B*Nr, D/2, 2, 2)
        freqs = freqs.reshape(b, nr, *freqs.shape[-3:])
        kr = apply_rotation(kr, freqs.unsqueeze(1).to(kr.device))
        st.match_frac.append(float(keep.float().mean()))
        return kr, keep.to(torch.float32)

    # ------------------------------------------------------------------ attention override
    def _next(self, func, args, kwargs):
        if self.prev_override is not None:
            return self.prev_override(func, *args, **kwargs)
        return func(*args, **kwargs)

    def attention_override(self, func, *args, **kwargs):
        st = self.state
        to = kwargs.get("transformer_options", None)
        if st is None or to is None:
            return self._next(func, args, kwargs)
        bi = to.get("block_index", None)
        if bi is None:  # Z-Image refiners (block_index removed by our wrapper) or foreign calls
            return self._next(func, args, kwargs)
        if st.blocks is not None and bi not in st.blocks:
            return self._next(func, args, kwargs)
        if len(args) < 4:
            raise AdapterError("ZQX: unexpected attention call signature")
        q, k, v, heads = args[0], args[1], args[2], args[3]
        if not kwargs.get("skip_reshape", False) or q.ndim != 4:
            raise AdapterError("ZQX: expected (B, H, N, D) attention inputs (skip_reshape=True)")
        mask_in_args = len(args) > 4
        mask = args[4] if mask_in_args else kwargs.get("mask", None)
        n = k.shape[2]
        plain = not passes.blocked(self, ("foreign", "variant"))

        if st.mode == "capture":
            if not plain:      # another patch's extra pass inside our capture pass: not ours to record
                return self._next(func, args, kwargs)
            st.trace.append((st.mode, bi))
            s0, s1 = st.ref_layout.span(q.shape[2])
            st.ref_s0 = s0
            st.store[bi] = (k[:, :, s0:s1].detach().clone(), v[:, :, s0:s1].detach().clone())
            return self._next(func, args, kwargs)

        # inject (also into other patches' variant passes, never into their foreign passes)
        if passes.blocked(self, ("foreign",)):
            return self._next(func, args, kwargs)
        if plain:
            st.trace.append((st.mode, bi))
        if bi not in st.store:
            raise AdapterError(f"ZQX: no captured reference for block {bi}")
        kr, vr = st.store[bi]
        b = q.shape[0]
        if kr.shape[0] != b or kr.shape[1] != k.shape[1] or kr.shape[-1] != k.shape[-1]:
            raise AdapterError(f"ZQX: captured reference {tuple(kr.shape)} incompatible with keys {tuple(k.shape)}")
        s0, s1 = st.layout.span(q.shape[2])
        kmask = st.kmask
        if st.rot is not None:
            kr = apply_rotation(kr.to(k.dtype), st.rot.to(k.device))
        else:
            kr, keep = self._match_and_rotate(kr.to(k.dtype), vr, v[:, :, s0:s1], s0, st)
            kmask = kmask.to(keep.device) * keep
        if self.cfg.key_scale != 1.0:
            kr = kr * self.cfg.key_scale
        vr = vr.to(v.dtype)
        nq = q.shape[2]
        qmask = torch.zeros((b, nq), dtype=torch.float32, device=q.device)
        qmask[:, s0:s1] = st.qmask_img
        bias = reference_log_bias(qmask, kmask, self.cfg.weight, dtype=torch.float32)
        mdtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64) else torch.float32
        full = build_extended_mask(mask, bias, b, nq, n, mdtype, q.device)
        k2 = torch.cat([k, kr], dim=2)
        v2 = torch.cat([v, vr], dim=2)
        if mask_in_args:
            new_args = (q, k2, v2, heads, full) + tuple(args[5:])
            new_kwargs = kwargs
        else:
            new_args = (q, k2, v2, heads) + tuple(args[4:])
            new_kwargs = dict(kwargs)
            new_kwargs["mask"] = full
        return self._next(func, new_args, new_kwargs)


def install(model_patcher, cfg: RefAttnConfig):
    """Clone the patcher and install the wrapper + attention override. Returns (clone, patch)."""
    from ..adapters import get_adapter
    import comfy.patcher_extension as pe

    m = model_patcher.clone()
    adapter = get_adapter(m)
    if len(m.get_wrappers(pe.WrappersMP.DIFFUSION_MODEL, ReferenceAttentionPatch.WRAPPER_KEY)) > 0:
        raise ValueError("ZQX: this model already has a ZQX Reference Attention patch; chaining two reference "
                         "patches is not supported (their capture passes would interfere). Use one reference image "
                         "(e.g. a side-by-side face+body composite) instead.")
    to = m.model_options.setdefault("transformer_options", {})
    prev = to.get("optimized_attention_override", None)
    patch = ReferenceAttentionPatch(adapter, cfg, prev_override=prev)
    to["optimized_attention_override"] = patch.attention_override
    m.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, ReferenceAttentionPatch.WRAPPER_KEY, patch.diffusion_model_wrapper)
    m.add_callback_with_key(pe.CallbacksMP.ON_PRE_RUN, ReferenceAttentionPatch.WRAPPER_KEY, patch.clear_cache)
    m.add_callback_with_key(pe.CallbacksMP.ON_CLEANUP, ReferenceAttentionPatch.WRAPPER_KEY, patch.clear_cache)
    return m, patch
