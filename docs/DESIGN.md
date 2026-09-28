# DESIGN

ComfyUI source read and tested against: **comfyanonymous/ComfyUI @ `8d534945ebd53cff61e8def81757c6a6c1b9cf2d`**
(2026-09-27).  All statements about ComfyUI internals below were checked in that source; line-level behaviour is pinned
by the tests in `tests/`.

## 1. Hook points (verified in code)

| Hook | Where (file) | What it gives | Verdict |
|---|---|---|---|
| `patches["attn1_patch"]` | `qwen_image/model.py::Attention.forward` | joint q/k/v (B,H,N,D) **after QK-norm, before RoPE**, `pe`, `img_slice=[txt, txt+img]` | Qwen only; the returned `attn_mask` is *not* used by the attention call (the additive mask is built before the patch), so it cannot add keys with a bias. Not used. |
| `transformer_options["optimized_attention_override"]` | `ldm/modules/attention.py::wrap_attn` | `(func, q, k, v, heads, mask, skip_reshape=True, transformer_options=…)` for **every** attention call, post-RoPE, after `AttentionTensorContainer`s are unwrapped | Works for Qwen **and** Z-Image (the only per-attention hook Z-Image has). **Used** for reference capture/inject; chains to a previously installed override. |
| `patches_replace["dit"]` (`("double_block", i)`) | `qwen_image/model.py::_forward` | whole-block replacement | Qwen only; NextDiT (Z-Image) has no `patches_replace` → core `SkipLayerGuidanceDiT` cannot work on Z-Image. Not used. |
| `patches["double_block"]`, `patches["post_input"]`, `patches["noise_refiner"]` | both models | block outputs / embeddings | not needed |
| `set_model_unet_function_wrapper` | `samplers.py::_calc_cond_batch` | wraps `apply_model` | single slot, conflicts with other nodes → not used |
| `WrappersMP.DIFFUSION_MODEL` | `QwenImageTransformer2DModel.forward`, `NextDiT.forward` | `(executor, x, timestep, context, …)`, composable, keyed | **Used** by every model-patch node (reference passes, CADS context corruption, sigma publication for runtime LoRA). |
| `ModelPatcher.add_weight_wrapper(key, fn)` | `model_patcher.py` (→ `m.weight_function`), `ops.py::cast_bias_weight` | function applied to the cast weight at every forward, after regular/low-vram LoRA patches | **Used** for sigma-scheduled / K-LoRA runtime LoRAs (no re-patching between steps). |
| native `ref_latents` / `ref_latents_method` (Qwen) | `qwen_image/model.py::_forward` | QIE reference tokens at frame index 1..n (`index`), `index_timestep_zero` (2511) doubles the timestep batch | used as the *oracle* of the reference-attention test; the reference patch coexists with it (keys exclude QIE tokens, frame offset skips past them) |
| `transformer_options["sigmas"]`, `["cond_or_uncond"]`, `["sample_sigmas"]`, `["block_index"]`, `["total_blocks"]` | `samplers.py`, both models | current sigma (unbatched), chunk layout (0 = cond, 1 = uncond), full run schedule, block counter | `sigmas`, `cond_or_uncond`, `block_index` used. `block_index` is set only in the **main** loops; the pack's wrappers delete it before calling the model so that Z-Image's refiner blocks (which run through the same attention function first) are recognised by its absence. |
| `PREDICT_NOISE` wrapper / `CFGGuider` subclass | `samplers.py` | per-step guidance | Sigma Split Guider subclasses `CFGGuider` (mirrors core `Guider_DualModel` for loading the second model). |
| NOISE object (`generate_noise(input_latent)`) | `comfy_extras/nodes_custom_sampler.py` | initial noise for SamplerCustomAdvanced | Low-Frequency Noise node. |

## 2. Model adapter layer (`zqx/adapters`)

One adapter per architecture; everything architecture-specific lives there.  `get_adapter()` matches the **exact**
diffusion-model class (subclasses such as `QwenImage21` are refused, since their token layout differs).

| | Qwen-Image / 2512 / Edit 2511 | Z-Image (Turbo / Base) |
|---|---|---|
| wrapper args | `(x, timestep, context, attention_mask, ref_latents, additional_t_cond, transformer_options)` | `(x, timesteps, context, num_tokens, attention_mask, **kw)` with `transformer_options` in kwargs |
| latent | 5-D (B,16,1,H,W) | 4-D (B,16,H,W) |
| joint sequence in main blocks | `[txt (context.shape[1]) │ target (h·w) │ QIE refs…]` | `[caption padded to 32 │ target (h·w) │ image padding to 32]`; refiners run first on caption-only / image-only sequences |
| target span | `[n_txt, n_txt + h·w)`, total length checked every call | `[N − (h·w + pad), N − pad)` (caption length only known in the call) |
| target RoPE ids | `(0, i − h//2, j − w//2)` (centred), QIE refs at frame 1..n | `(cap_len + 1, i, j)`, text at `(1..cap_len, 0, 0)`, pads at 0 |
| embedder | `pe_embedder` (EmbedND, axes (16,56,56), θ = 10000) | `rope_embedder` (EmbedND, axes (32,48,48), θ = 256) |
| block keys | `transformer_blocks.N.` (60) | `layers.N.` (30); `noise_refiner`/`context_refiner` → group `refiner` |
| unsupported → error | T > 1 latents, other Qwen variants | Omni/Ming reference paths (`ref_latents`, `ref_frames`, `siglip_feats`), `rope_options` with right/below offsets |

**RoPE offsets by rotation composition.**  ComfyUI stores RoPE as explicit 2×2 rotation matrices
(`flux/math.py::rope`) applied as `out = F[...,0]·x0 + F[...,1]·x1`.  Rotations compose additively in angle, so a
reference key captured post-RoPE at position p is moved to p + Δ by `R(Δ)·k`, with `R(Δ) = embedder(ids = Δ)` — the
model's own embedder, not a re-implementation.  Tested: `R(Δ)R(p)k = R(p+Δ)k` (float32 angle error ≤ 5e-5) and
`R(0) = I` exactly.

Offsets (`position_mode`): `frame` — Qwen: frame `1 + (#QIE refs)` (native QIE convention, never collides with QIE's own
refs); Z-Image: `cap_len + 2` (next free t, the Z-Image-Omni convention). `right`/`below` — first reference column/row
right after the target's last one (Qwen's centred grids are handled: Δw = (w_t − 1 − w_t//2) + w_r//2 + 1).  `same` —
Δ = 0 (FreeCus / Stable Flow convention; imprints the reference layout).

## 3. Timestep windows: sigma space

All windows are in **sigma of the model's own schedule** (for these flow models sigma == flow time t ∈ [0, 1],
t = 1 noise).  A window is active iff `sigma_end ≤ σ ≤ sigma_start`.  Reason: the two-pass workflow's img2img pass
starts at σ ≈ 0.5–0.9 (depends on denoise and shift).  A σ-window keeps its meaning in both passes ("layout steps" are
σ ≳ 0.75 whichever pass you are in); a fraction-of-steps window would silently re-map to other noise levels in the
second pass.  The current σ comes from `transformer_options["sigmas"]` (set by ComfyUI's sampler for every call); if
it is missing the nodes raise.  `ZQX Sigmas To Text` prints a schedule so windows can be chosen.  Reference: ZIT 8 steps
simple/shift 3 → 1.000, 0.955, 0.900, 0.833, 0.750, 0.643, 0.500, 0.300.

K-LoRA's step progress t/T is also expressed in sigma space (progress = 1 − σ) — a documented deviation from the
official step counter so that an img2img pass does not restart the schedule.

## 4. Nodes and their design

### 4.1 ZQX Reference Attention (`zqx/patches/reference_attention.py`)
Per model call inside the window a DIFFUSION_MODEL wrapper runs **two** passes through the same executor chain:

1. **capture** — the model runs on the reference latent (`process_latent_in`, then `noise_scaling(σ_ref, ε_fixed, ref)`
   and `calculate_input`, i.e. exactly how the sampler builds its inputs; batch-expanded to B with the *same* context,
   masks, QIE refs). The attention override stores post-RoPE K and V of the reference image tokens per selected block.
   `noised`: σ_ref = mult·σ, same timestep as the target when mult = 1.  `cached`: σ_ref = cache_sigma, computed once
   per sampling run and conditioning (content hash of the non-x/t arguments; cleared on ON_PRE_RUN / ON_CLEANUP).
2. **inject** — the real forward; in selected blocks the keys/values become `[K; R(Δ)K_ref·key_scale]`, `[V; V_ref]`
   and the additive mask gets `log(w · m_q · m_k)` on the reference columns (−inf where zero).  `m_q` is 0 for text,
   padding and QIE-reference queries, and the resized query mask on target image tokens (and 0 on uncond rows if
   `inject_uncond` is off).  `m_k` is the resized key mask × the per-step token-dropout mask.
   The attention itself is still the user's backend (`func`), called with the extended mask; `-inf` columns give exactly
   zero weight.

Guarantees (tests): the two passes see identical block sequences (else it raises); cond/uncond rows only see
references computed from their own conditioning (the capture pass is batched identically); weight 0 / out of window /
all-zero masks call the model once with untouched inputs (**bitwise** identical output); with position `frame`,
weight 1 and a 1-block Qwen model the result equals Qwen-Image-Edit's native `index` reference concatenation to 2e-5.
Only one reference patch per model (two would capture each other's passes) — use one composite reference.

Cost: `noised` = 2 forwards per step inside the window; memory for K/V of the reference tokens of all selected blocks
(Qwen, 1024² reference, 60 blocks, bf16: ≈ 3 GB per batch row — use a 512² reference or fewer blocks).  The extended
mask is (B, 1, N, N + N_ref) in the compute dtype.

StyleAligned's AdaIN on Q/K is intentionally not implemented (see RESEARCH §2.1).

### 4.2 ZQX Scheduled LoRA and ZQX K-LoRA (`runtime_lora.py`, `lora_schedules.py`)
LoRAs are parsed by ComfyUI's own `lora_convert` + `load_lora` with the model's key map (so alpha, DoRA, LoKr and
fused-qkv slices behave like LoraLoader).  Keys that ComfyUI would silently skip are an error unless
`allow_unmatched_keys`.  For every touched weight an `add_weight_wrapper` function returns
`W + Σ_i s_i(σ, block)·ΔW_i` using `comfy.lora.calculate_weight` in float32, cast back to the weight dtype.  A
DIFFUSION_MODEL wrapper publishes σ for the duration of each model call.  Tests: constant strength equals
`LoraLoaderModelOnly` through the real sampler (≤ 2e-5, both models, kohya/peft/transformer formats, txt2img and
img2img); strength 0 is bitwise identity; block weights equal a LoRA with those blocks removed.
K-LoRA: S_c, S_s = sums of the top-(r_c·r_s) |ΔW| of the exact dense deltas ComfyUI would add; γ = mean L1 ratio after
dropping ratios ≥ 3·mean (official `utils.py`); per layer and step exactly one LoRA is active.  Selection is
scale-invariant (γ absorbs global strength), the strengths set the applied magnitude.  For Z-Image the selection is made
on the fused `attention.qkv` delta (ComfyUI fuses q/k/v).

### 4.3 ZQX LoRA Arithmetic / Conflict Report (`lora_arith.py`, `core/lora_io.py`, `core/lora_math.py`)
Both LoRAs are mapped to **model-weight-key space** (ComfyUI key map; names ComfyUI cannot map, e.g. musubi Z-Image
`lora_unet_layers_N_attention_to_q`, are resolved by normalised names and reported; ambiguous normalised names are
dropped from that fallback).  Slices of fused weights are embedded exactly (zero-padded factors), several slices of one
weight are rank-concatenated.  Scales are folded (`s = alpha/rank`, or 1 without alpha — ComfyUI's convention).  All
exact modes produce exact low-rank factors; the output file uses ComfyUI's generic base keys
(`diffusion_model.<weight key>.lora_up/down.weight`) with `alpha = rank`.  Orthonormal bases use SVD of the *product*
(col(U·D) = U·col(D)), with a relative rank tolerance, so rank-deficient factors do not over-project.  Tests: every
mode's dense ΔW equals the dense formula; projections are idempotent and orthogonal; save → `load_torch_file` →
`comfy.lora.load_lora` → `calculate_weight` reproduces W + ΔW for both models.  Refused: DoRA, LoHa/LoKr, conv, `diff`
patches, unpaired keys, shape mismatches, unrecognised keys.

### 4.4 ZQX CADS (`core/cads.py`, `patches/cads.py`)
DIFFUSION_MODEL wrapper replacing `context` by the annealed embedding; t = σ; fresh noise per step, seeded by
(seed, σ); statistics per sample (identical to the reference code for batch 1, and never mixes cond/uncond rows).
`relative_noise` (default on, heuristic) multiplies s by std(y) because Qwen2.5-VL / Qwen3 hidden states are far from
unit variance; off = paper's absolute s.  `s = 0` or γ = 1 → the model is called unchanged (bitwise).

### 4.5 ZQX Low-Frequency Noise (`core/noise_init.py`)
Per frequency bin `F(out) = a·F(ε) + b·c·F(r)`, `b = √α·H`, `a = √(1 − α·H²)`, `c` = H²-weighted energy match per
(batch, channel) → unit variance also for soft filters, exact noise in the H = 0 band, exact `c·LPF(r)` for the ideal
filter at α = 1; α = 0 returns the noise tensor itself.  Filters = FreeInit's Gaussian / Butterworth / ideal on the
same normalised frequency grid.

### 4.6 ZQX Sigma Split Guider (`patches/guider.py`)
`CFGGuider` subclass: `cfg_early` for σ ≥ switch, `cfg_late` below; with cfg = 1 ComfyUI skips the uncond pass.
Optional `model_early` is prepared/loaded like core `Guider_DualModel` does (own model_options, wrappers, conds processed
by its own `extra_conds`).  Refuses a `model_early` that shares the base weights with `model` (ComfyUI patches weights
in place, so two LoRA variants of one checkpoint cannot be active in the same run → use Scheduled LoRA) and different
latent formats.

## 5. Principles
* Pure math in `zqx/core` (no ComfyUI import), tested directly; glue in `zqx/patches`, UI in `zqx/nodes`.
* No hidden fallback: unsupported model / arguments / layouts raise with a message.  Explicit opt-outs are node inputs
  (`allow_unmatched_keys`).
* Disabled settings short-circuit to the unpatched call, so "off" is bitwise "off".

## 6. Known limitations
* Area / regional conditioning (`ConditioningSetArea`) crops the latent per cond; query masks then refer to the crop.
  Not supported deliberately; results would be undefined.
* Multi-GPU (`multigpu_clones`) calls the model from worker threads; the patches keep per-call state on the patch
  object and are not thread-safe.  Unsupported.
* `torch.compile` of the diffusion model bypasses Python-level wrappers in parts; untested.
* Reference attention requires one extra forward per in-window step (`noised`) and the memory noted above.
