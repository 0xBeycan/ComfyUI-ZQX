# ZQX nodes — detailed reference

All nodes are under the **ZQX** menu (`ZQX/attention`, `ZQX/lora`, `ZQX/guidance`, `ZQX/model-edit`,
`ZQX/sampling`, `ZQX/scoring`, `ZQX/tools`).  They are all training-free, work with Z-Image Turbo (ZIT),
Qwen-Image 2512 and Qwen-Image-Edit 2511, and can be used in both passes of the **two-pass** workflow (base pass +
latent upscale + img2img pass with denoise < 1).

> **Timestep windows are in sigma space.**  For these flow models sigma = t ∈ [0, 1] (1 = pure noise).  A window is
> active while `sigma_end ≤ σ ≤ sigma_start`.  The second (img2img) pass starts around σ ≈ 0.5–0.9, so a
> "composition window" such as `sigma_start = 1.0, sigma_end = 0.8` never fires in the second pass.  That is the
> intended behaviour, because the composition is already fixed by then.  Use **ZQX Sigmas To Text** to see the sigmas of
> your own schedule.  ZIT, 8 steps, simple, shift 3: `1.000, 0.955, 0.900, 0.833, 0.750, 0.643, 0.500, 0.300`.

> **The recommended values are starting points** and have not been validated on real models.  This pack was tested on
> CPU with tiny random models; see `docs/TEST_RESULTS.md`.  A/B-test all of them with fixed seeds.

---

## 1. ZQX Reference Attention (identity)

**What it does:** in the selected blocks, the image tokens of the generated image also attend to the keys/values (K/V)
of the reference image (the passport face/body reference).  In every step the reference is first run through the model
once (the capture pass) and its K/V are recorded.  In the real pass they are then appended to the attention keys.
Identity then comes **from the reference photo**, and the character LoRA, which has the "AI look" baked in, can be run
weaker.

**Sources:**
* Mechanism: reference-only (sd-webui-controlnet), StoryDiffusion (consistent self-attention).
* Token dropout: ConsiStory (arXiv 2402.03286).
* Less-noised reference and K×1.1: FreeCus (2507.15249).
* Log-bias weight: StyleAligned (2312.02133).
* RoPE placement: separate frame index as in Qwen-Image-Edit / Kontext / Z-Image-Omni; width/height offset as in
  OminiControl / UNO.

Verified on Qwen: with a 1-block model the node's output equals QIE's own reference-concatenation path to 2e-5.

| Parameter | Meaning |
|---|---|
| `model` | A ZIT, Qwen-Image or QIE model. Any other model raises an explicit error. |
| `reference` | VAE-encoded reference (batch 1). May have a different resolution from the target. **About 512² is recommended** (memory). |
| `weight` | `log(w)` is added to the reference logits. 1 = plain concatenation, 0 = off (bitwise-identical output), > 1 = more attention to the reference. |
| `sigma_start`, `sigma_end` | Sigma window (see the note above). |
| `blocks` | Main blocks to inject into: `all` or e.g. `0-19, 30-45`. Qwen has 60 blocks, ZIT has 30. Z-Image refiner blocks are never modified. |
| `position_mode` | RoPE position of the reference. `frame`: the next frame/t index (the placement QIE uses for its references; recommended). `right`/`below`: right of / below the canvas (diptych prior). `same`: the same positions as the target — **copies the reference layout (frontal pose, centred framing)**, which works against the goal. `matched`: every reference token is moved to the position of the target token whose value vector is most similar (FreeGraftor / CharaConsist style), so the reference follows the *generated* pose instead of imposing its own. |
| `capture_mode` | `noised`: the reference is noised to the current sigma every step (+1 forward pass per step). `cached`: captured once at `cache_sigma` and reused for all steps (cheap). |
| `cache_sigma` | Noise level of the reference in `cached` mode; 0 = clean (QIE 2511's `index_timestep_zero` convention). |
| `ref_sigma_mult` | In `noised` mode the reference noise level is mult·σ. FreeCus gives the reference slightly less noise than the target (< 1). |
| `key_scale` | Multiplies the reference keys (FreeCus: 1.1). Sharpens attention to the reference. |
| `token_dropout` | Randomly drops this fraction of reference tokens each step (ConsiStory: 0.5). **Reduces copying of the reference layout.** |
| `inject_uncond` | Also inject into the uncond branch when CFG > 1. Off = the reference effect is amplified by CFG (stronger, riskier). Has no effect on ZIT (CFG = 1). |
| `noise_seed` | Seed for the reference noise and the dropout. |
| `match_threshold` (opt.) | `matched` mode: minimum cosine similarity of value vectors for a reference token to be used (default 0.5). Lower = more tokens matched. |
| `match_mutual` (opt.) | `matched` mode: keep only mutual nearest neighbours (default on; FreeGraftor-like cycle check). |
| `query_mask` (opt.) | Region of the generated image allowed to look at the reference (e.g. the face). Area-averaged down to the token grid. All zeros → node behaves as off. |
| `key_mask` (opt.) | Visible part of the reference (e.g. only the reference face, so its clean/bokeh background does not leak). |

**Starting values**

| | ZIT (8 steps, CFG 1) | Qwen-Image 2512 (CFG ~2.5–4) | QIE 2511 |
|---|---|---|---|
| weight | 1.0 | 1.0 | 1.0 |
| sigma_start / end | 0.90 / 0.30 (skip the first step so the layout stays free) | 0.85 / 0.20 | 0.85 / 0.20 |
| blocks | `all`, then ablate | `all`, then try `20-45` | try `30-45` (AttnRouter, unverified) |
| position_mode | `frame` | `frame` | `frame` |
| capture_mode | `noised` (ref_sigma_mult 0.9) | `noised` | `cached`, cache_sigma 0 |
| token_dropout | 0.3–0.5 | 0.3–0.5 | 0.3 |
| key_mask | reference face mask | same | same |
| pass 2 | same node, window 0.6/0.2 | same | — |

**Risks:**
* A separate frame index is out of distribution for ZIT and Qwen T2I; generalisation is unverified.
* Too much `weight`, or the `same` position mode, copies the reference pose/lighting and brings the AI look back.
* `noised` mode costs ~2× compute per step.
* Memory: Qwen, 1024² reference, 60 blocks, bf16 ≈ 3 GB.
* Two reference nodes cannot be attached to the same model (it raises an error). For face + body, use a single
  side-by-side composite reference.

---

## 2. ZQX Scheduled LoRA (sigma / block)

**What it does:** applies a LoRA without merging it (at runtime).  Its strength follows the sigma: `strength_early` in
the early steps, `strength_late` in the late steps, a linear transition in between.  It can also take per-block
multipliers.  This gives "realism first, character last" inside one sampler, without extra steps and without
re-patching weights.  At a constant strength the output equals ComfyUI's `LoraLoaderModelOnly` (tested).

**Source:** heuristic.  It rests on two findings: flow models fix the layout in the first steps (Gandikota & Bau,
2503.10637; guidance interval, 2404.07724), and LoRAs are additive.  ComfyUI core hook keyframes (`CreateHookLora` +
keyframes) do something similar in percent space by recomputing weights.

| Parameter | Meaning |
|---|---|
| `lora_name` | LoRA in `models/loras`. ComfyUI's key mapping is used (kohya/peft/ai-toolkit/musubi Qwen). |
| `strength_early` | Strength while σ ≥ `sigma_hi`. |
| `strength_late` | Strength while σ ≤ `sigma_lo`. |
| `sigma_hi`, `sigma_lo` | Ends of the linear ramp; equal values give a hard switch. |
| `block_weights` | E.g. `0-19:1, 20-59:0.3, other:1, refiner:1`. `other` = layers outside blocks (img_in, proj_out, …), `refiner` = ZIT refiners. Later entries override earlier ones. |
| `allow_unmatched_keys` | Deliberately ignore LoRA keys that do not map onto the model. When off, such keys raise an error (LoraLoader silently skips them). |

**Starting values (chain two nodes):**

| | ZIT | Qwen-Image 2512 |
|---|---|---|
| Character LoRA | early 0.4, late 1.0, sigma_hi 0.90, sigma_lo 0.75 | early 0.5, late 1.0, 0.85 / 0.65 |
| Realism LoRA | early 1.0, late 0.35, sigma_hi 0.90, sigma_lo 0.75 | early 1.0, late 0.4, 0.85 / 0.65 |
| pass 2 (img2img) | same nodes: for σ < 0.75 the character is at full strength, realism low | same |

**Risks:**
* If the character LoRA is too low in the early steps, face shape and body proportions can drift, and later steps
  cannot fix it.
* ZIT is distilled, so a high realism strength in the early steps may damage texture.
* The compute cost is small: one low-rank product per layer per forward pass.

---

## 3. ZQX K-LoRA (character + realism)

**What it does:** in every attention layer (q/k/v) and every step it selects **only one** of the two LoRAs.
* S_c is the sum of the character LoRA's K = r_c·r_s largest |ΔW| entries; S_s is the same for the realism LoRA.
* If `(S_c/γ) / (S_s·S(t)) > 1` the character LoRA is used, otherwise the realism LoRA.
* γ is the mean of the per-layer L1 ratios, with outliers dropped.
* `S(t) = α·t/T + β` grows over time, so content (identity) dominates the early steps and style (realism) the late ones.

**Source:** K-LoRA, Ouyang et al., CVPR 2025, arXiv 2502.18461 (verified against the official `klora.py`/`utils.py`).
Two deviations:
* The t/T progress is `1 − σ` in sigma space; the official code counts steps.
* On ZIT, ComfyUI keeps q/k/v fused (`attention.qkv`), so the selection is made on the fused ΔW.

| Parameter | Meaning |
|---|---|
| `character_lora`, `realism_lora` | Content (identity) and style (realism) LoRAs. |
| `character_strength`, `realism_strength` | Strength applied when selected (the selection itself is scale-invariant). |
| `alpha`, `beta`, `pattern` | `s`: S(t) = α t/T + β (official: α 1.5, β 0.5). `s*`: (α t/T + β) mod α (recommended for FLUX, β = 0.85α = 1.275). |
| `scope` | `attention`: q/k/v only (as in the paper). `all_shared`: all layers both LoRAs touch. |
| `other_layers` | What to apply on layers outside the selection: `both`, `character`, `realism`, `none`. |
| `allow_unmatched_keys` | See Scheduled LoRA. |

**Start:** ZIT and Qwen: α 1.5, β 0.5, `s`, scope `attention`, other_layers `both`, strengths 1.0 / 0.8.  The report
output shows how many layers use the character / realism LoRA at each σ.  If the character layer count is very low at
early σ, lower β.

**Risks:** untested on DiTs other than FLUX; hard per-layer selection can cause abrupt changes at some steps.

---

## 4. ZQX LoRA Arithmetic

**What it does:** builds a new `.safetensors` LoRA from two LoRAs in weight space and saves it under
`models/loras/zqx/`.  ComfyUI's LoRA loader reproduces the intended ΔW exactly (tested).  It also writes a per-layer /
per-block report (`*_report.json`).

**Modes and formulas** (ΔW₁ = character, ΔW₂ = realism or a future "AI-look" LoRA):

| Mode | Formula | Source |
|---|---|---|
| `add` | ΔW₁ + λΔW₂ (exact, rank r₁+r₂) | task arithmetic 2212.04089 |
| `negate` | ΔW₁ − λΔW₂ (exact: B=[B₁, −λB₂], A=[A₁;A₂]) | task negation |
| `clean_col` | (I − λQ₂Q₂ᵀ)ΔW₁, Q₂ = orthonormal basis of ΔW₂'s column space (exact, rank r₁) | linear algebra (subspace projection) |
| `clean_row` | ΔW₁(I − λP₂P₂ᵀ), P₂ = row (input) space | same |
| `target_sub` | ΔW₁ − λQ₁Q₁ᵀΔW₂: subtract only the part of ΔW₂ inside the character's subspace | same |
| `knots_ties` | TIES (2306.01708) in a shared SVD basis (KnOTS 2410.19735), optional DARE (2311.03099); exact low rank | KnOTS+TIES |
| `ties_dense` | TIES on the dense ΔW + truncated SVD (`svd_rank`); the relative error is reported | TIES |

| Parameter | Meaning |
|---|---|
| `model` | Only used for the key map / weight shapes (nothing is patched). |
| `lora_1`, `lora_2` | Usually character and realism. |
| `mode`, `lam` | See the table above; λ ∈ [0, 4]. In projection modes λ = 1 is a full projection. |
| `w1`, `w2` | TIES weights. |
| `density` | Fraction of largest entries kept by TIES (paper: 0.2). |
| `dare_drop` | DARE drop rate before TIES (0 = off). |
| `svd_rank` | Output rank of `ties_dense`. |
| `seed` | DARE seed. |
| `filename_prefix`, `save_dtype` | Output name and dtype. |

**Key formats.**
* Supported: kohya/musubi (`lora_down/lora_up` + `.alpha`), diffusers/peft (`lora_A/lora_B`, `.lora_A.default`), old
  diffusers (`.lora.down/up`), with `diffusion_model.`/`transformer.`/`lora_unet_` prefixes.
* Musubi Z-Image names that ComfyUI does not recognise are mapped through normalised names and reported.
* Refused with an explicit error: DoRA, LoHa/LoKr, convolutions, `diff` patches, unmatched/unrecognised keys, shape
  mismatches.
* The output uses ComfyUI's generic key format (`diffusion_model.<weight>.lora_up/down.weight`, alpha = rank).  It
  loads in ComfyUI; other tools may not recognise these names.

**Start:** first look at the **Conflict Report** to see where the LoRAs overlap.  Then try:
* `clean_col` with λ 0.5 and 1.0;
* `target_sub` with λ 0.5;
* `knots_ties` with density 0.2, w1 1.0, w2 0.7.

Load the generated LoRA with the normal LoraLoader or with Scheduled LoRA.

**Risks:** the realism LoRA is not the inverse of the "AI look".  Removing its subspace from the character LoRA may also
partly erase identity; watch layers with a high E1in2 in the report.  Once a real "AI-look LoRA" exists, removing that
LoRA's subspace with `clean_col` is the principled operation.

## 5. ZQX LoRA Conflict Report

For two LoRAs, per layer and per block:
* cosine similarity;
* subspace overlap (LoRA paper §7: ‖Q₁ᵀQ₂‖²_F / min(k₁,k₂));
* the fraction of one LoRA's energy inside the other's output space (E1in2, E2in1);
* opposite-sign rates, over all entries and over both LoRAs' top-`top_density` entries.

With `sign_stats` off the dense sign statistics are skipped, which is faster.  Use it to answer "in which blocks do the
character and realism LoRAs fight?", and to pick target blocks for Scheduled LoRA `block_weights`, K-LoRA and
projection.

---

## 6. ZQX CADS (condition annealing)

**What it does:** adds Gaussian noise to the text conditioning in the high-noise steps, then rescales it back to the
original mean/std.  This breaks the distilled model's collapse onto a single mode (always the same pose, gaze, framing,
background).  The condition is clean in the detail steps.

**Source:** CADS, Sadat et al., ICLR 2024, arXiv 2310.17347.
* Schedule: γ(t) = 1 for t ≤ τ1, (τ2 − t)/(τ2 − τ1) in between, 0 for t ≥ τ2.
* Corruption: ŷ = √γ·y + s·√(1−γ)·n, with fresh n every step.
* Then rescaling with ψ.
* SD values τ1 0.6, τ2 0.9, s 0.25, ψ 1 (verified against code).

| Parameter | Meaning |
|---|---|
| `tau1`, `tau2` | σ ≤ τ1: clean condition; σ ≥ τ2: fully noised; linear in between. |
| `noise_scale` | s. 0 = off (bitwise-identical output). |
| `psi` | Rescale mix (1 = mean/std fully preserved). |
| `relative_noise` | Multiply s by the std of each embedding (heuristic; Qwen2.5-VL/Qwen3 hidden states are not unit variance). Off = the paper's absolute s. |
| `apply_to` | `cond_and_uncond` (reference code) or `cond_only`. |
| `seed` | Noise seed (deterministic per σ). |

**Start:**
* ZIT: τ1 0.80, τ2 1.00, s 0.10, ψ 1, relative on (conservative).
* Qwen 2512: τ1 0.60, τ2 0.90, s 0.15–0.25, ψ 1.
* Usually unnecessary in the second pass: with a window above σ 0.8 it does not fire there anyway.

**Risks:** on ZIT a high s lowers prompt adherence and can produce nonsensical scenes.  The character LoRA's trigger
word is noised too, so identity weakens in the first steps (it comes back in the late steps).

---

## 7. ZQX Low-Frequency Noise

**What it does:** produces a NOISE for SamplerCustomAdvanced.
* The low-frequency band comes from the latent of a real photo: large bright/dark masses, the horizon, where the person
  sits in the frame.
* The high-frequency band comes from normal noise.
* Variance is preserved (a² + b² = 1 per frequency, plus energy matching).

It gives a composition/lighting prior without copying detail.

**Source:** a heuristic built on FreeInit's (arXiv 2312.07537) frequency split and filters.  FreeInit's plain
complementary mix lowers the variance, so a power-preserving mix is used instead.

| Parameter | Meaning |
|---|---|
| `noise_seed` | Base noise seed. |
| `reference` | Composition reference (any resolution; resized and standardised per channel). |
| `strength` | α: 0 = normal noise (bitwise identical), 1 = low band entirely from the reference. |
| `cutoff` | Normalised cutoff frequency D0 (1 = Nyquist). FreeInit uses 0.25; 0.05–0.15 for composition. |
| `filter`, `butterworth_order` | `gaussian`, `butterworth` (n), `ideal`. |
| `base_noise` (opt.) | Modify another NOISE instead. |

**Start:** both models: strength 0.4, cutoff 0.1, gaussian, **pass 1 only**.  Reference: a real photo of the kind of
scene you want (street, indoor, natural light, person off-centre).

**Risks:** a high strength or cutoff can cause colour casts and copy the reference silhouette.  In a flow model all
signal is noise at the σ = 1 start, so the effect acts purely through low-band statistics.

---

## 8. ZQX Sigma Split Guider

**What it does:** a GUIDER for SamplerCustomAdvanced.
* `cfg_early` is used while σ ≥ `switch_sigma`, `cfg_late` below it.
* In steps with CFG 1 the uncond pass is skipped.
* Optional `model_early`: the early steps are run with a different base model (e.g. Z-Image Base for ZIT).

**Sources:**
* Guidance interval (Kynkäänniemi et al., 2404.07724).
* Distilling Diversity and Control (Gandikota & Bau, 2503.10637): a distilled model fixes the layout in the first step,
  and running the first step with the base model restores diversity.

For the CFG interval alone, the core `CFGOverride` node does the same job (in percent space).

| Parameter | Meaning |
|---|---|
| `model`, `positive`, `negative` | Main model and conditionings. |
| `switch_sigma` | σ ≥ this value is "early". |
| `cfg_early`, `cfg_late` | Early / late CFG. |
| `model_early` (opt.) | A **different** base model for the early steps (same latent space and text encoder). A LoRA'd copy of the same checkpoint is refused, because weights are patched in place; use Scheduled LoRA instead. |

**Start:**
* ZIT: switch 0.85 (first 2 steps), cfg_early 2.0–3.0, cfg_late 1.0.  Negative prompt: "posed, looking at camera,
  smiling, studio portrait, centered composition, bokeh, clean background".
* Qwen 2512: switch 0.8, cfg_early 4.0, cfg_late 2.5 (same negative).

**Risks:** on ZIT, CFG > 1 can cause oversaturation / burn-in, so keep it to the first 1–2 steps.  `model_early` loads
two models into memory at once.

---

## 9. ZQX Sigmas To Text / ZQX Block Spec (ablation)

* **Sigmas To Text:** prints SIGMAS step by step, for choosing windows.
* **Block Spec:** `index`, `width`, `total_blocks`, `mode` (`only`/`except`), `weight` → `block_list` (for the
  Reference Attention `blocks` input) and `block_weights` (for Scheduled LoRA).  Increment `index` with a primitive to
  run a block ablation (Stable Flow / FreeFlux protocol: which block carries identity, which carries layout).

---

## 10. ZQX Spatial LoRA (masked)

**What it does:** runs a LoRA as a side branch whose output is multiplied **per token** by a mask, instead of
merging it into the weights (which applies it to every token).  Typical use: realism LoRA everywhere except the face
(`mask` = face, `invert_mask` on) and character LoRA only on the face — both at full strength in the same step, so
they no longer fight over the face.  With all weights 1 the output equals `LoraLoaderModelOnly` (tested).

**Source:** LoRAShop (arXiv 2505.23758) restricts subject LoRAs to their region; the token-level side branch is our
implementation (forward hooks, active only during each model call).

| Parameter | Meaning |
|---|---|
| `lora_name` | Plain LoRA only (no DoRA / LoCon mid / LoKr). |
| `strength_early`, `strength_late`, `sigma_hi`, `sigma_lo` | Sigma ramp as in Scheduled LoRA. |
| `mask` (opt.) | Region of the **generated** image (resized to the token grid). None = everywhere. |
| `invert_mask` | Apply where the mask is 0. |
| `text_weight` | Weight on text tokens (Qwen text stream, Z-Image caption tokens). |
| `nonspatial_weight` | Weight on modulation / timestep layers — they act on the whole image, so 0 keeps a masked-out region truly untouched. |
| `other_weight` | Weight on QIE reference tokens and Z-Image padding tokens. |

**Start:** realism LoRA: mask = face, invert on, strength 0.8, nonspatial_weight 0; character LoRA: mask = face,
strength 1.0.  The face position must be known: use it in **pass 2** with a mask of the pass-1 face (drawn, or from a
face-detection node pack), or with a pose from the Pose Bank whose face region you know.

**Risks:** hard mask edges can leave seams — blur the mask; LoRA effects propagate through attention, so a masked
LoRA still influences neighbouring tokens indirectly (by design it is not a pixel-perfect separation).

---

## 11. ZQX LoRA Guidance (LoRA-CFG)

**What it does:** `out = out_base + w · (out_lora − out_base)` with a sigma schedule for `w`.  `w = 1` is the plain LoRA
(one forward), `w = 0` the base model, `w > 1` amplifies the LoRA.  E.g. the character LoRA with `w` 0.3 in the layout
steps (the base model decides pose/scene) and 1.3–1.5 in the detail steps (identity amplified).

**Source:** heuristic; the model-difference form of classifier-free guidance, related to autoguidance
(arXiv 2406.02507).  Two forwards per step when `w ∉ {0, 1}`.

| Parameter | Meaning |
|---|---|
| `lora_name`, `strength` | The LoRA and its strength in the "with LoRA" forward. |
| `w_early`, `w_late`, `sigma_hi`, `sigma_lo` | Guidance weight ramp over sigma. |

**Start:** ZIT: w 0.3 → 1.3, σ 0.85 / 0.6.  Qwen 2512: w 0.5 → 1.3.  **Risks:** `w > 1.5` over-sharpens / burns;
doubles compute inside the ramp.

---

## 12. ZQX Perturbed Attention Guidance (DiT)

**What it does:** an extra forward in which the selected blocks use an *identity* attention map for image queries
(each image token attends only to itself); `out += scale · (out − out_perturbed)`.  Adds a structure/realism signal
**at CFG 1**, where CFG tweaks do nothing.  ComfyUI's core PAG patches a UNet-only location and core
`SkipLayerGuidanceDiT` cannot hook Z-Image; this works for both Qwen-Image and Z-Image.

**Source:** PAG (arXiv 2403.17377), adapted to joint-attention DiTs through the attention override.

| Parameter | Meaning |
|---|---|
| `scale` | Guidance scale.  Applied to cond rows before CFG: with CFG > 1 the effective scale is `scale · cfg`. |
| `sigma_start`, `sigma_end` | Window. |
| `blocks` | `mid` (middle block, default), or a list like `10-14`. |
| `apply_to` | `cond_only` (like core PAG) or `all`. |

**Start:** ZIT: scale 1.0, window 1.0 → 0.5, blocks `mid`.  Qwen: scale 0.5 (× cfg).  **Risks:** PAG tends towards
cleaner, higher-contrast images — it can *increase* the polished look; test it last.

---

## 13. ZQX Activation Steering

**What it does:** per step inside the window, two extra forwards with a "towards" and an "away" prompt (e.g.
"candid, looking away" vs "posing, looking at the camera"); after the selected blocks the image hidden states are
shifted by `alpha · (h_towards − h_away)`.  `mean` mode uses one global direction (no spatial layout transfer),
`token` mode a per-token one.

**Source:** activation addition (ActAdd, arXiv 2308.10248) applied to DiT image hidden states (double-block hook).

| Parameter | Meaning |
|---|---|
| `towards`, `away` | Contrast prompts (text only). |
| `alpha` | Strength (negative reverses). |
| `sigma_start`, `sigma_end` | Window (layout steps by default: 1.0 → 0.6). |
| `blocks` | `mid`, `all`, or a list. |
| `mode` | `mean` or `token`. |
| `apply_to` | `all` or `cond_only`. |

**Start:** alpha 0.3–0.8, blocks `mid`, mode `mean`, window 1.0 → 0.7, pass 1 only.  **Risks:** 3× compute inside the
window; large alpha breaks images; which blocks carry pose/gaze is unmeasured (use Block Spec to sweep).

---

## 14. ZQX UCE Text Edit

**What it does:** closed-form edit of the text input projection (Qwen-Image `txt_in`, Z-Image `cap_embedder`): the
`source` prompt's embedding is mapped to what the `target` prompt produces, while `keep` prompts are preserved.
Applied as a normal weight patch.

**Source:** Unified Concept Editing (UCE, arXiv 2308.14761).  We edit the input projection because in MMDiT /
single-stream DiTs the per-block text K/V projections do not see a fixed text embedding (the text stream is updated by
every block), unlike UNet cross-attention where UCE was proposed.

| Parameter | Meaning |
|---|---|
| `source`, `target` | Paired prompts (one-to-one if you pass several conditionings). |
| `pairing` | `mean` (mean token → mean token), `end` (last tokens aligned from the end), `positional` (equal lengths). |
| `lam` | Regulariser, relative to the mean squared token norm (bigger = smaller edit). |
| `strength` | Fraction of the edit applied (0 = no patch). |
| `keep` (opt.) | Prompts whose embedding must not change — put your trigger word / typical scene prompts here. |

**Start:** lam 0.5, strength 0.5, keep = your trigger prompt.  **Risks:** the AI look is mostly a default mode, not a
word — expect a small effect; an aggressive edit changes unrelated prompts (check with `keep`).  The report output shows
residuals and drift.

---

## 15. ZQX LoRA Surgery

**What it does:** rewrites one LoRA and saves a new file (+ JSON report): per-block multipliers, drop module kinds,
rank truncation, spectrum shaping, DARE.

| Parameter | Meaning |
|---|---|
| `strength` | Baked-in multiplier. |
| `block_weights` | As in Scheduled LoRA; 0 removes the weight from the file. |
| `drop_kinds` | Comma list: `text` (Qwen text stream / Z-Image caption layers), `modulation` (adaLN / timestep layers), `attention`, `mlp`, `io`, `other`. |
| `max_rank` | Keep at most this many singular directions per weight (0 = all). |
| `energy_keep` | Keep the smallest rank that holds this fraction of Σσ². |
| `spectrum_power` | σᵢ → σ₁ (σᵢ/σ₁)ᵖ; p < 1 flattens, p > 1 concentrates. `power_preserve` = keep σ₁ or ‖ΔW‖_F. |
| `dare_drop` | DARE on the up factor (unbiased, stays exactly low rank). |

**Sources:** Eckart–Young (truncation), DARE (arXiv 2311.03099), B-LoRA-style block selection (arXiv 2403.14572).
**Start:** try `drop_kinds = modulation` and `energy_keep = 0.9` on the character LoRA, one at a time.
**Risks:** heuristic hypotheses (e.g. "the AI look lives in the weak directions / modulation layers") — measure.

## 16. ZQX LoRA Common Subspace

**What it does:** for 2–4 LoRAs, per weight, the top-k left singular vectors of the summed deltas = their common
output subspace.  `common` writes the shared component `U Uᵀ mean(ΔW)`; `clean_target` removes that subspace from a
target LoRA.  With several character LoRAs made by the same QIE pipeline, the common part is a training-free estimate
of the shared "AI look" (hypothesis) — though it also contains generic "person" directions.

**Source:** inspired by Iso-CTS common/task-specific subspaces (arXiv 2502.04959).  **Start:** common_rank 2–4.
The report shows how much of each LoRA's energy lies in the common subspace.

## 17. ZQX Realism LoRA Ablation (identity-safe)

**What it does:** with the character LoRA already on the model, measures for each **unit** of the realism LoRA (a
block, a module kind, or a top singular direction of a block) how removing it changes
* identity: the identity scorer (e.g. ArcFace similarity to the passport photo), averaged over the seeds;
* realism kept: CLIP-vision cosine similarity to the full-realism baseline images;

then removes the units whose removal gains ≥ `min_identity_gain` identity while costing ≤ `max_realism_drop`, evaluates
the combination once more and writes a new realism LoRA.  If every unit qualifies, no file is written and the report
says so.  Output images: A (no realism), C (full realism), final — one per seed.

| Parameter | Meaning |
|---|---|
| `model` | Model **with the character LoRA applied**. |
| `realism_lora`, `realism_strength` | The LoRA to clean and the strength you use it at (not baked into the file). |
| `positive`, `negative`, `latent_image`, `steps`, `cfg`, `sampler_name`, `scheduler`, `seeds` | The fixed evaluation set (≥ 4 seeds). |
| `vae`, `identity_scorer`, `clip_vision` | Decoder, identity metric (Face scorer, identity weight only), realism-similarity model. |
| `units` | `blocks`, `kinds`, `blocks+kinds`, `blocks+svd` (then the best `svd_blocks` blocks are refined by their top `svd_components` directions). |
| `min_identity_gain`, `max_realism_drop` | Selection thresholds. |

**Cost:** (2 + #units + svd_blocks·svd_components + 1) × #seeds generations — ZIT: 30 blocks × 4 seeds ≈ 130 runs.
**Risks:** one-at-a-time attribution (interactions only checked at the end); CLIP similarity is a coarse realism
proxy; ArcFace measures the face only.

---

## 18. Scorers and ZQX Seed Search

Scorer nodes output `ZQX_SCORER` (metrics + weights); **ZQX Scorer: Combine** merges up to four.

| Scorer | Metrics | Notes |
|---|---|---|
| **Face (InsightFace)** | `face_identity` (ArcFace cosine to the reference face), `face_off_center` (0 = centred), `head_turn` (\|yaw\|/90), `face_found` | Needs `pip install insightface onnxruntime(-gpu)` and a model pack in `models/insightface/models/<buffalo_l or antelopev2>` (IPAdapter / PuLID layout).  Positive weights on off-centre / head-turn push away from the "centred, looking at camera" look. |
| **CLIP Similarity** | cosine of CLIP-vision embeddings to reference images | Positive weight towards a real candid photo, negative away from the passport composition. |
| **Background Sharpness** | log(var Laplacian in border band / centre) | Model-free; bokeh / studio backgrounds score low. |

**ZQX Seed Search:** probes `n_candidates` seeds (starting at the NOISE seed) for `probe_steps` steps, decodes the x0
prediction, scores the previews (each metric z-normalised across the candidates, then weighted) and fully samples the
best `keep` seeds — each exactly what a normal run with that seed gives.  Outputs the samples, all previews, a ranked
report and the best seed.

**Source:** distilled models fix the layout in the first step (Gandikota & Bau, arXiv 2503.10637); candidate pruning
on early predictions as in Group Inference (FLUX-schnell).  **Start (ZIT):** 16–32 candidates, probe_steps 1, keep 1–2;
Face scorer w_identity 1.0, w_head_turn 0.5, w_off_center 0.3, w_face_found 2; + Background Sharpness 0.5.
**Risks:** one-step previews are blurry — face detection may fail at probe_steps 1 (use 2); cost = n × probe + keep × full.

---

## 19. ZQX Pose Bank

**What it does:** picks a control image (DWPose / OpenPose skeleton, depth map…) from
`ComfyUI/input/<bank_folder>/<tag>/…` by seed, among files whose aspect ratio is within `aspect_tolerance` of the
output, optionally restricted to tags (sub-folders such as `walking`, `sitting`), and crops it to the output size.
Feed it to a ControlNet so pose and composition come from real candid photos instead of the model prior (Z-Image
Turbo: `ModelPatchLoader` + `QwenImageDiffsynthControlnet` with the Fun ControlNet Union; that node has no step range —
lower the strength instead).

| Parameter | Meaning |
|---|---|
| `bank_folder` | Folder inside the input directory, or an absolute path. |
| `tags` | Comma list of sub-folders; empty = all. |
| `width`, `height`, `aspect_tolerance` | Target size and allowed aspect-ratio deviation (0.15 = 15 %). |
| `resize` | `crop_to_size` (centre crop + resize) or `keep`. |
| `seed` | Selection seed (deterministic). |

**Start:** pose maps from 100+ real photos, ControlNet strength 0.5–0.7.  **Risks:** depth maps also carry the donor's
body shape; a small bank just creates a new kind of sameness.

---

## A/B test plan

General rules:
1. **Fixed seeds:** at least 4 seeds (e.g. 1, 2, 3, 4) × 2–3 prompts (everyday scene, street, indoor).  One variable at
   a time.
2. **Baseline:** your current workflow (character LoRA + realism LoRA, two passes).  Change only one node or parameter
   per experiment.
3. **Evaluation:**
   * identity: ArcFace/InsightFace similarity, or by eye;
   * AI-look checklist: gaze at camera, smile, pose, bokeh/clean background, centred framing;
   * texture.
4. **Which pass:**
   * Composition tools go in **pass 1**: CADS, Low-Frequency Noise, Sigma Split Guider, the early part of Scheduled
     LoRA.
   * Identity tools go in **both passes**: Reference Attention, the late part of Scheduled LoRA.

Recommended order (see "Test order" in the README):
1. Scheduled LoRA (character low early / high late; realism the opposite) — passes 1 and 2.
2. Seed Search with the Face + Background scorers — pass 1 (cheap way to *select* against the AI look).
3. Realism LoRA Ablation → use the identity-safe realism LoRA in step 1.
4. Reference Attention (frame or matched, window 0.9→0.3, dropout 0.4, face key_mask), then lower the character LoRA
   strength in steps of 0.2.
5. Pose Bank + ControlNet — pass 1.
6. Spatial LoRA (realism outside the face, character on the face) — pass 2 with a face mask.
7. K-LoRA or LoRA Guidance (instead of Scheduled LoRA) — compare with the same seeds.
8. Sigma Split Guider (early negative), Activation Steering, CADS — pass 1, one at a time.
9. Conflict Report → LoRA Arithmetic / Surgery / Common Subspace → repeat step 1 with the generated LoRA.
10. Low-Frequency Noise, DiT PAG, UCE — last.

After each step, fix the winning setting and then add the next lever.
