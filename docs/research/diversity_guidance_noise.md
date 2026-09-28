# Training-free diversity, composition and realism levers for Z-Image Turbo (ZIT) and Qwen-Image 2512

Research date: 2026-09-28. Target models: **ZIT** (single-stream DiT, Lumina/NextDiT family in ComfyUI `comfy/ldm/lumina/model.py`, rectified flow, distilled to about 8 steps, CFG=1) and **Qwen-Image 2512** (MMDiT, `comfy/ldm/qwen_image/model.py`, flow matching, CFG>1).

## How sources were checked

**Network constraint.** This sandbox's egress proxy blocks arxiv.org, alphaxiv, huggingface, openreview, semanticscholar, CVF, ICLR proceedings and project pages. Only GitHub (code and READMEs) and web-search snippets were reachable.

- **VERIFIED (code):** read directly from the official or reference GitHub source code, which was cloned for this report.
- **VERIFIED (snippet):** taken from web-search snippets that quote the paper.
- **UNVERIFIED:** from memory, or from a secondary source only. Check these against the PDF before relying on them.

**Conventions.**
- Flow time `t` (equal to sigma in ComfyUI for flow models): `x_t = (1-t)·x0 + t·ε`. `t=1` is pure noise.
- EDM-equivalent noise level: `σ_EDM = t/(1-t)`.
- ZIT sigmas for 8 steps with shift 3 (`s = 3t/(1+2t)`): `1.000, 0.955, 0.900, 0.833, 0.750, 0.643, 0.500, 0.300, 0`. The equivalent σ_EDM values are `∞, 21, 9.0, 5.0, 3.0, 1.8, 1.0, 0.43`.
- ZIT sigmas for 9 steps: `1, .96, .913, .857, .789, .706, .6, .462, .273, 0`.

---

## 0. Behaviour of the ComfyUI hooks at CFG=1 (VERIFIED from `comfy/samplers.py`)

- **Uncond branch is skipped.** `sampling_function` sets `uncond_ = None` when `cond_scale≈1` and `disable_cfg1_optimization` is False. So at CFG=1 the uncond branch never runs.
- **Post-CFG functions still run.** `cfg_function` still runs every `sampler_post_cfg_function`. This means PAG/SLG/TPG-style guidance of the form `denoised + s·(cond − perturbed_cond)` works at CFG=1. It costs one extra forward pass per guided step. It needs no negative prompt.
- **Core PAG does nothing on DiTs.** `nodes_pag.py` patches `"attn1"`, `"middle"`, block 0, which is a UNet-only location. On ZIT and Qwen it still pays for the extra pass.
- **Core SLG works on Qwen-Image but not on ZIT.**
  - `SkipLayerGuidanceDiT` (`nodes_slg.py`) patches `patches_replace["dit"][("double_block", i)]`. Defaults: layers 7,8,9; scale 3; start 0.01; end 0.15.
  - Qwen-Image honours `blocks_replace`, so SLG works there.
  - Lumina/ZIT has **no `patches_replace`**. It only has a post-block `patches["double_block"]` hook, which receives `img`, `img_input`, `txt`, `pe`, `vec` and `block_index`. So on ZIT, layer-skip, attention-perturbation or token-shuffle guidance needs a custom hook. Options are object patches of `layers[i].attention` or of `forward`, or `transformer_options` attention overrides.
- **pamparamm/sd-perturbed-attention** (custom, not core) provides PAG, SEG, SWG, PLADIS, NAG, TPG, FDG, MG and SMC-CFG. Its PAG, SEG and TPG patch `BasicTransformerBlock` in UNets (SD1.5/SDXL only). MG and SMC-CFG are model-agnostic.

**Core nodes found in `comfy_extras` (current master):**

| Node | Source | What it does |
|---|---|---|
| `PerturbedAttentionGuidance` | `nodes_pag.py` | UNet only (see above) |
| `SelfAttentionGuidance` | `nodes_sag.py` | |
| `APG` | `nodes_apg.py` | |
| `CFGZeroStar`, `CFGNorm` | `nodes_cfg.py` | |
| `PerpNeg`, `PerpNegGuider` | `nodes_perpneg.py` | |
| `TCFG` | `nodes_tcfg.py` | |
| `Mahiro` | `nodes_mahiro.py` | |
| `Epsilon Scaling`, `TemporalScoreRescaling` | `nodes_eps.py` | TSR is a **diversity / temperature knob** |
| `SkipLayerGuidanceDiT(Simple)` | `nodes_slg.py` | |
| `NAGuidance` | `nodes_nag.py` | |
| `FreSca` | `nodes_fresca.py` | |
| `FreeU`, `FreeU_V2` | `nodes_freelunch.py` | |
| `RescaleCFG` | `nodes_model_advanced.py` | |
| `SamplerEulerCFGpp`, `SamplerEulerAncestralCFGPP` | | |
| `CFGOverride` | `nodes_custom_sampler.py` | Sets cfg over a [start, end] percent range. **This is effectively a guidance-interval node.** |

There is **no core CADS node**. There is **no core node for particle, DPP or diversity guidance**.

---

## 1. CADS: Condition-Annealed Diffusion Sampler

- **Paper:** Sadat, Buhmann, Bradley, Hilliges, Weber. *"CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling."* arXiv **2310.17347**, ICLR 2024.
- **Training-free:** yes. It only needs the conditioning input.

### Mechanism

Status: VERIFIED (code) against v0xie/sd-webui-cads and asagi4/ComfyUI-CADS, both of which follow the paper. The formulas also match the snippet summary of the paper.

- **Time convention.** `t ∈ [0,1]`, and sampling runs **backwards from t=1 (noise) to t=0 (clean)**. The reference implementation uses `t = 1 − step/total_steps`. asagi4 alternatively uses `t = timestep/999`, i.e. the sampler's own sigma or timestep.
- **Annealing schedule** (piecewise-linear):
  ```
  γ(t) = 1                      if t ≤ τ1
       = (τ2 − t)/(τ2 − τ1)     if τ1 < t < τ2
       = 0                      if t ≥ τ2
  ```
  So the condition is fully noised (γ=0) at the start (t ≥ τ2), and fully clean (γ=1) for the last part (t ≤ τ1).
- **Noised condition**, with fresh Gaussian noise `n ~ N(0, I)` drawn at every step, shaped like y:
  ```
  ŷ = sqrt(γ(t))·y + s·sqrt(1 − γ(t))·n
  ```
- **Rescaling**, which is needed because large s blows up the norm:
  ```
  ŷ_rescaled = (ŷ − mean(ŷ)) / std(ŷ) · std(y) + mean(y)
  ŷ_final    = ψ·ŷ_rescaled + (1 − ψ)·ŷ
  ```
  In both reference implementations, mean and std are **scalar statistics over the whole embedding tensor** (`torch.mean(y)`, `torch.std(y)`). The paper's mean/std notation matches this. An unofficial SD3 port (Johnny221B/OSCAR) uses last-dimension (per-token) statistics. OSCAR also has a sign bug in the γ denominator (`τ1 − τ2`) and compounds noise across steps, so do not copy it.
  - Paper wording (VERIFIED, snippet): "the mixing factor ψ ∈ [0,1] prevents divergence especially for high noise scales, but slightly reduces diversity."
  - ψ=1 means the fully rescaled condition is used.
- **Which embeddings.** A1111 applies the noise to both `crossattn` (token sequence) and `vector` (pooled / ADM) embeddings. asagi4 applies it by default to `y` (pooled), with an option for `c_crossattn`.
  - For ZIT and Qwen there is no pooled vector. Apply it to the token sequence `c_crossattn` (Qwen3 or Qwen2.5-VL hidden states).
  - Only non-padding tokens should be noised. SeedVarianceEnhancer auto-masks null-padded tokens because noising them hurts.
- **Unconditional / null branch.**
  - The A1111 implementation noises **both** cond and uncond embeddings. asagi4 has `apply_to = both | cond | uncond`, defaulting to both.
  - The paper frames CADS as annealing *the conditioning vector y* of the conditional branch. With CFG, guidance is formed from `D(z, t, ŷ)` against the unconditional `D(z, t, ∅)`. I believe the null embedding is left untouched in the paper, but this is **UNVERIFIED**.
  - For CFG=1 (ZIT) this question is moot, since only the cond branch exists.
  - The paper compares CADS against "dynamic CFG" (`w(t) = γ(t)·w`). The OSCAR port offers that as an option, but it is a baseline, not CADS.

### Recommended hyperparameters

- **Stable Diffusion text-to-image** (SD 2.1), and also quoted for the DeepFashion pose-to-image table: `τ1=0.6, τ2=0.9, s=0.25, ψ=1`. Status: VERIFIED via snippet quoting Table 13 as used by SPARKE, and matches the defaults of every implementation.
- **ImageNet 256** (class-conditional DiT/LDM): `τ1=0.5, τ2=0.9, s=0.15, ψ=1`. **UNVERIFIED** (snippet from a search summary only).
- **ImageNet 512:** `τ1=0.6, τ2=1.0, s=0.1, ψ=1`. **UNVERIFIED**.
- **Guidance from the implementations:**
  - v0xie: "noise scale default 0.25, recommended ≤ 0.3".
  - asagi4: `s` may be negative; values far from 0 give "garbage unless rescale is also used".
  - Reported result: DeepFashion recall 0.02 → 0.48 and Vendi score roughly doubled (VERIFIED, snippet).

### Adapting CADS to ZIT and Qwen (my recommendations)

- **Use the sampler sigma as t.** For rectified flow, `t` equals sigma, so the "sampler timestep" mode is natural.
- **Shift τ upward for ZIT.** With 8 steps and shift 3, sigmas above 0.9 cover only steps 0–1. Gandikota & Bau (§2) show that distilled models commit the layout at the **first step**, so the first step must be conditioned on a heavily noised y.
  - Suggested: `τ2 ≈ 1.0` (so step 0 has γ≈0.05–0.1, not exactly 0) and `τ1 ≈ 0.75–0.85`.
  - Or define γ over step index: step 0 gets γ≈0.3–0.5, step 1 gets ≈0.7–0.8, then 1.
- **Scale s to the embedding.** `s` must be relative to the embedding scale. Qwen-family text-encoder hidden states have large, outlier-dominated magnitudes. SeedVarianceEnhancer needs "strength 15–30" of uniform noise on 50% of values for ZIT, and notes that the optimal strength is within an order of magnitude of the embedding std.
- **Use ψ=1, or per-token rescale.** Either choice keeps the token norm distribution in-range. Consider per-token std rather than a global scalar, because of outlier channels.

### Existing implementations

| Implementation | Details |
|---|---|
| **asagi4/ComfyUI-CADS** | Unet-function-wrapper node. Params: `noise_scale` 0.25, `t1` 0.6, `t2` 0.9, `rescale` 0 (disabled at 0), `start_step` / `total_steps`, `apply_to`, `key` (y / c_crossattn / both), `noise_type` normal/uniform/exponential, `seed`. The author says it "might not be correct at all". It is not core. |
| **v0xie/sd-webui-cads** | A1111 reference implementation. |
| **ChangeTheConstants/SeedVarianceEnhancer** | ComfyUI; built for **Z-Image Turbo**. A CADS-like hard switch: it adds uniform noise `U(−strength, strength)` to a random `randomize_percent` (default 50%) of embedding values for the first `steps_switchover_percent` (default 20%). It uses two conditionings with `start_percent`/`end_percent`, then switches back to the clean embedding. Recommended for ZIT: strength 15–30, or 40 with switchover at 10%. Has prompt masking and null-pad masking. |

---

## 2. Distilling Diversity and Control in Diffusion Models

- **Paper:** Gandikota & Bau. arXiv **2503.10637**, WACV 2026.
- **Code:** github.com/rohitgandikota/distillation (VERIFIED).
- **Training-free:** yes. It needs the **base (non-distilled) model** as well.

### Findings

Status: VERIFIED (snippet and README).

1. **Control distillation.** Distilled models keep the base model's concept representations. Concept Sliders, LoRAs and DreamBooth trained on the base transfer to the distilled model, and vice versa, without retraining.
2. **DT-Visualization.** This decodes the model's predicted x0 at intermediate steps. It shows that distilled models (DMD, Turbo, LCM, Lightning) **commit to the final structure almost immediately after the first denoising step**. Base models spread structural decisions over many steps.
3. **Diversity distillation (hybrid inference).** Run the **base model for only the first timestep (T=0, the highest noise)**, then switch to the distilled model. The paper claims the "first timestep, not later steps, controls sample diversity", and that the result matches or beats SDXL-Base diversity at close to DMD speed. They also report the bottleneck is architecture-agnostic (UNet and DiT).

### Exact recipe

Status: VERIFIED (code), `evalscripts/diversity_distillation_sdxl.py`.

- Base: SDXL, `base_guidance_scale` 5 (CLI default; the function default is 7), `base_num_inference_steps` 4 by default, same scheduler as the distilled model.
- Distilled: DMD, Turbo, LCM or Lightning. `distilled_num_inference_steps` 4, `distilled_guidance_scale` 0.
- `run_distilled_from_timestep = 1`: the distilled model starts at its **2nd** timestep.
- The base model runs from its first timestep up to the base timestep closest to `distilled_scheduler.timesteps[1]`, which is one distilled-step-equivalent. The base latent is then handed over.

### Application to our models

- **ZIT.** Swap the model for step 0. That could mean Z-Image base, if a base checkpoint is available (**UNVERIFIED** whether the non-distilled Z-Image weights are released and compatible), or ZIT without the character LoRA.
  - A cheap variant: remove or down-weight the **character LoRA** (trained on synthetic data, so collapsed) on step 0–1 only. This follows directly from "the first step controls diversity".
  - In ComfyUI this is two KSamplerAdvanced passes (steps 0–1 with model A, then steps 1–8 with model B, `return_with_leftover_noise`), or a `SplitSigmas` pair.
- **Qwen-Image 2512.** It is not distilled, but its high-CFG early steps behave similarly (see §3).

---

## 3. Guidance interval (Kynkäänniemi et al.)

- **Paper:** *"Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models."* arXiv **2404.07724**, NeurIPS 2024.
- **Code:** github.com/kynkaat/guidance-interval (VERIFIED).
- **Training-free:** yes.

### Finding

Status: VERIFIED (README). Guidance is "clearly harmful toward the beginning of the chain (high noise levels), largely unnecessary toward the end (low noise levels), and only beneficial in the middle." Restricting it improves ImageNet-512 FID from 1.81 to 1.40.

### Exact intervals

- **Code (VERIFIED):** EDM2-XXL, 32 steps, EDM ρ=7, σ from 0.002 to 80, 2nd-order Heun.
  - FID-optimal: guidance on step indices **[17, 22]**, i.e. σ ≈ 1.61 → 0.28, with **G=2.0**.
  - FD_DINOv2-optimal: indices **[13, 19]**, i.e. σ ≈ 5.0 → 0.85, with **G=2.9**.
  - Outside the interval, `D = D_cond` (no uncond evaluation).
- **Paper (UNVERIFIED exact values):**
  - EDM2-XXL (FID) is stated as σ ∈ (0.28, 2.90], w=2.0.
  - **SD-XL: σ ∈ (0.28, 5.42]**, about 50% of the steps. A snippet says w=16, which is **UNVERIFIED**; it may be the large-w setting shown in figures.

### Flow-matching conversion

`t = σ/(1+σ)`:

| σ_EDM | t |
|---|---|
| 5.42 | 0.844 |
| 2.90 | 0.744 |
| 1.61 | 0.617 |
| 0.28 | 0.219 |

**Qwen recommendation:** CFG only for sigma ≈ 0.85 → 0.2, and cfg=1 for the first steps with t > 0.85 and for the tail. This is the main diversity lever for a CFG>1 model, because high-noise CFG is what collapses layouts.

### ComfyUI

- **Core `CFGOverride`**: set `cfg=1` over `start_percent=0 … p_hi` and optionally over the tail. `percent_to_sigma` handles shift.
- Alternatively, use `ConditioningSetTimestepRange` on the negative.

---

## 4. Batch or particle diversity methods (multi-sample, inference-time)

### 4a. Particle Guidance (PG)

- **Paper:** Corso, Xu, De Bortoli, Barzilay, Jaakkola. arXiv **2310.13102**, ICLR 2024.
- **Code:** github.com/gcorso/particle-guidance.
- **Training-free:** yes, for fixed potentials.
- **Core idea:** a joint, time-evolving potential over the batch. Add `−∇ log Φ_t(x_1..x_n)` (repulsive) to each sample's score.
- **Exact SD implementation** (VERIFIED, code, `pipeline_stable_diffusion_particle.py`, non-SVGD branch). With `x_i` the flattened latents of the n batch samples:
  ```
  diff_ij = x_i − x_j                      (for j ≠ i)
  d_ij    = ||diff_ij||_2
  h_t     = median_j(d_ij)^2 / log(n−1)    (median heuristic, per i)
  w_ij    = exp(−d_ij^2 / h_t)             (RBF kernel, power 2)
  grad_i  = Σ_j 2·w_ij·diff_ij / h_t · σ_t · coeff
  ε̂_i     = ε_cfg,i − grad_i               (applied only while σ_t ≥ 1; coeff = 0 when σ < 1)
  ```
  - `coeff` is the CLI `--coeff`, default 0 (i.i.d.).
  - A "feature" variant computes the same kernel on **DINO features** of the decoded x̂0 and backpropagates through the VAE decoder and DINO. That is more semantic, and expensive.
- **Flow adaptation.** For velocity prediction, add the repulsion to x̂0 or subtract it from ε̂ = x + (1−t)·v. Restrict it to high noise (t ≳ 0.5), matching "σ ≥ 1", i.e. t ≥ 0.5.
- **Caveat.** Pixel or latent L2 repulsion mostly changes colour and layout, not pose or semantics.
- **ComfyUI:** none known in core.

### 4b. DiverseFlow

- **Paper:** Morshed & Boddeti. arXiv **2504.07894**, CVPR 2025.
- **Training-free:** yes. It is designed for **flow models**.
- **Mechanism** (VERIFIED at the summary level; exact formula UNVERIFIED):
  - At each ODE step, estimate the endpoint `x̂1` (the clean image) for every sample.
  - Build a DPP kernel `L` over features (CLIP or DINO; DINO worked better) of the n estimates, with a **quality term** folded into the kernel so samples stay on-distribution.
  - Add `γ_t·∇_x log det(L)` to each sample's velocity. This gives coupled ODEs with mutual repulsion.
  - It requires gradients through the feature extractor and decoder.
- **Code:** I did not find a public repository (UNVERIFIED).

### 4c. EDDY: Marginal-Preserving Particle Guidance

- **Paper:** Vinograd, Achituve, Fetaya. arXiv **2605.06553**, May 2026.
- **Mechanism:** divergence-free, anti-symmetric pairwise kernel drift fields (a Fokker–Planck symmetry). They change joint trajectories while **exactly preserving each sample's marginal**, so they add diversity without quality loss. Practical approximations are given for T2I with perceptual embeddings.
- **Applicability:** works for diffusion and flow matching. Details are UNVERIFIED.

### 4d. Scaling Group Inference

- **Paper:** Parmar, Patashnik, Ostashev, Wang, Aberman, Narasimhan, Zhu. arXiv **2508.15773**, ICLR 2026.
- **Code:** github.com/GaParmar/group-inference (VERIFIED).
- **Mechanism:** start with M candidate noises. At each step, denoise all of them, compute
  - a unary quality score `u_i` (CLIP text-image by default; aesthetics and ImageReward are options), and
  - pairwise diversity `D_ij` (DINO by default, or CLIP, or colour) on x̂0 predictions.

  Then solve a QIP, `max_{S,|S|=K} Σ u_i + λ Σ_{i,j∈S} D_ij` (Gurobi), and prune.
- **Defaults** (VERIFIED, code):

  | Model | Steps | Guidance | Candidates M | Keep K | Pruning ratio | λ |
  |---|---|---|---|---|---|---|
  | FLUX-schnell | 4 | 0 | 64 | 4 | 0.9 per step | 1.5 |
  | FLUX-dev | 20 | 3.5 | 128 | 4 | 0.5 | 1.5 |
  | Kontext | — | — | — | — | — | 1.0 |

- **Relevance:** this is the most "guaranteed" gallery diversity for a few-step, CFG=1 model, because it only selects and never perturbs. The cost is many candidates on the first step or two.

### 4e. Other 2025–2026 diversity-for-flow and few-step work (summaries; details UNVERIFIED)

- **"Don't Settle at the Mode!" Feature Self-Guidance** (arXiv **2606.27371**, ECCV 2026).
  - Batch-level dispersion of internal MMDiT features within one block, over a timestep window.
  - A manifold-regularization step re-runs the same block on the dispersed features and uses the projection as a regularizer.
  - Block 2 was most effective. It was tested on FLUX.1, FLUX-Depth, Kontext and the **step-distilled FLUX.2-Klein**. Marginal cost.
  - Very relevant to Qwen (MMDiT) and ZIT.
- **STRIDE** (arXiv **2605.11494**).
  - Single-forward-pass diversity for **1-step** models (FLUX-schnell, SD3.5-Turbo).
  - Injects **spatially coherent (pink) noise** into intermediate transformer features, **projected onto the principal components of the model's own activations** (on-manifold).
  - Only **early blocks** give diversity (FLUX L0–7, SD3.5-Turbo L0–11). Unstructured noise gets "corrected" back to the mode. One scalar strength.
  - Highly relevant to ZIT at CFG=1.
- **"It's Never Too Late": noise optimization for collapse recovery** (Harrington, Koepke, Karthik, Darrell, Efros; arXiv **2601.00090**).
  - Optimizes the initial noise for a diversity objective under a budget.
  - Key cheap finding: **pink-noise initialization** (more low-frequency energy) gives more diverse samples **before any optimization**.
- **SPARKE** (arXiv **2506.10173**): prompt-aware diversity guidance with a scalable kernel. It uses CADS as a baseline, with CADS settings τ1=0.6, τ2=0.9, s=0.25, ψ=1.
- **Minority Guidance.**
  - *"Don't Play Favorites"* (Um, Lee, Ye; arXiv **2301.12334**, ICLR 2024): the minority score is `ℓ(x0) = d(x0, x̂0(√ᾱ_t·x0 + √(1−ᾱ_t)·ε))`, the Tweedie reconstruction loss (LPIPS). Guidance toward high minority score needs a **trained classifier** on minority-score bins, so it is not training-free.
  - *"Self-Guided Generation of Minority Samples"* (Um & Ye, ECCV 2024, arXiv 2407.11555 **UNVERIFIED id**): training-free. It uses the gradient of the self-computed minority score (noise → denoise → LPIPS to x̂0) as guidance. It requires backprop through the model, so it is expensive.
  - Code: github.com/soobin-um/minority-guidance.
- **Diverse Score Distillation / Diversity-preserved DMD** (arXiv 2602.03139) and **Uncertainty DMD** (2609.11265): **training-time** fixes. Not applicable.

---

## 5. Temporal Score Rescaling (TSR): a single-sample temperature knob, **core in ComfyUI**

- **Paper:** Xu, Wu, Park, Zhou, Tulsiani. arXiv **2510.01184**, ICML 2026.
- **Code:** github.com/temporalscorerescaling/TSR (VERIFIED).
- **Training-free:** yes. It works with **flow** models and deterministic samplers, adds **no extra network function evaluations (NFE)**, and runs at CFG=1 as a post-CFG function.
- **Mechanism** (VERIFIED, code). Multiply the noise/score prediction by
  ```
  r_t = (η_t·σ² + 1) / (η_t·σ²/k + 1),   η_t = SNR = (1−t)²/t²  (flow)
  ```
  - As t→0, r→k. At high noise, r→1.
  - Flow version: `ε = x + (1−t)·v`, `ε' = r·ε`, `v' = (r·ε − x)/(1−t)`.
  - ComfyUI core form: `x0' = lerp(x/α, x0, r)`.
- **Parameters** (VERIFIED, snippet):
  - **k > 1** is sharper and less diverse. **k < 1** is flatter and more diverse. k=1 turns it off.
  - σ controls how early the rescaling acts; larger means earlier.
  - Paper-best for image quality: **k=0.93, σ=3.0**. Demo notebook: k=0.95, σ=1.0.
- **ComfyUI:** `TemporalScoreRescaling` core node, defaults `tsr_k=0.95`, `tsr_sigma=1.0`. The node tooltip says lower k gives "more detailed" output and higher k gives "smoother".
- **Caveats for ZIT.** It is theoretically derived for true scores; a distilled model does not predict a true score. Try k 0.85–0.97 and σ 1–3.

---

## 6. Structure and realism guidance (quality levers)

| Method | arXiv | Mechanism (implementable) | Recommended values | Needs uncond / neg? | Extra NFE | CFG=1 distilled OK? | ComfyUI |
|---|---|---|---|---|---|---|---|
| **PAG**, Perturbed-Attention Guidance (Ahn et al. 2024) | 2403.17377 | Replace the self-attention map with identity (`softmax(QKᵀ)→I`, so the output is V) in chosen layers to get a "weak" prediction `ε̂`. Then `ε = ε_c + s(ε_c − ε̂_c)`, added on top of CFG. | s≈3 (SD). UNet: mid block. DiT: some mid layers (per-model search; see HeadHunter) | No | +1 | **Yes** (post-CFG) | Core `PerturbedAttentionGuidance` (UNet only). pamparamm (UNet). DiT needs a custom node |
| **SEG**, Smoothed Energy Guidance (Hong 2024) | 2408.00760 | Gaussian-blur the **queries** (2D spatial, σ) of self-attention. σ→∞ means uniform queries (mean query). Same guidance form as PAG. | scale 3; blur σ 10 or ∞ (pamparamm: negative means ∞) | No | +1 | Yes | pamparamm (UNet) |
| **SWG**, Sliding Window Guidance | 2411.10257 | Weak model is the model applied on sliding crops (reduced receptive field) | — | No | + | Yes | pamparamm |
| **TPG**, Token Perturbation Guidance (Rajabi et al., NeurIPS 2025) | 2506.10036 | **Shuffle tokens** (random permutation along the sequence) of hidden states in selected blocks. It is linear, norm-preserving and destroys local structure. Guidance `cfg + s(ε_c − ε_tpg)` | s≈3; UNet blocks d2.2-9,d3 in pamparamm | No | +1 | Yes | pamparamm (UNet) |
| **SLG / STG**, skip-layer / spatiotemporal skip guidance (STG: Hyung et al. 2024) | STG 2411.18664 | Skip whole DiT blocks for the weak pass | SLG core: layers 7,8,9, scale 3, 1–15% of steps | No | +1 | Yes (Qwen via `blocks_replace`; **not ZIT** without a custom hook) | Core `SkipLayerGuidanceDiT` |
| **HeadHunter / SoftPAG** | 2506.10978 | Head-level PAG on DiTs (SD3, FLUX). Select heads iteratively by objective; specific heads govern structure, style and texture. SoftPAG: `A' = (1−α)A + αI` per head for continuous strength | — | No | +1 | Yes | Official repo cvlab-kaist/HeadHunter; none known in ComfyUI |
| **Autoguidance** (Karras et al. 2024) | 2406.02507 | `D = D_bad + w(D_good − D_bad)`, where D_bad is a smaller or less-trained version of the same model with the **same** conditioning. Improves quality **without the diversity loss of CFG** | EDM2: w≈2–3 | Needs a "bad" model | +1 | Possible: e.g. a heavily quantized or undertrained ZIT as D_bad (speculative) | none core |
| **APG**, Adaptive Projected Guidance (Sadat, Hilliges, Weber) | 2410.02416 | `Δ = ε_c − ε_u`. Momentum: `Δ ← Δ + β·Δ_prev_avg`. Norm clip: `Δ ← Δ·min(1, r/‖Δ‖)`. Project onto `ε_c` (normalised): `Δ_∥`, `Δ_⊥`. Result: `ε = ε_c + (w−1)(Δ_⊥ + η·Δ_∥)` | η=0, r=15 (SDXL), β=−0.5 (β UNVERIFIED; the sd-webui-APG README says the paper's defaults are η=0, r=15, momentum 0) | Yes (CFG>1) | 0 | No (CFG=1) | Core `APG` (eta 1.0, norm_threshold 5, momentum 0 defaults) |
| **CFG-Zero\*** (Fan et al. 2025) | 2503.18886 | `s* = ⟨v_c, v_u⟩/‖v_u‖²` per sample. `v = s*·v_u + w(v_c − s*·v_u)`. **Zero-init:** output v=0 for the first K steps (`i ≤ zero_steps`); the official default is `zero_steps=0`, i.e. only the first step, "about 2.5% of steps" for Wan | zero_steps 0–1 | Yes | 0 | No | Core `CFGZeroStar` (optimized scale only, **no zero-init**) |
| **CFG++** (Chung et al. 2024) | 2406.08070 | Denoise with the guided x̂0 (λ-scaled) but **renoise with ε_uncond**. `λ ∈ [0,1]` | λ≈0.6–1.0 (UNVERIFIED) | Yes | 0 | No | Core `SamplerEulerCFGpp`, `SamplerEulerAncestralCFGPP` |
| **Rescale CFG** (Lin et al. 2024, "Common Diffusion Noise Schedules…") | 2305.08891 | `x_r = x_cfg·std(x_c)/std(x_cfg)`, `x = φ·x_r + (1−φ)·x_cfg` | φ=0.7 | Yes | 0 | No | Core `RescaleCFG` |
| **TCFG**, tangential damping | 2503.18137 | SVD of [ε_u; ε_c]; keep only the uncond component along the top singular vector | — | Yes | 0 | No | Core `TCFG` |
| **NAG** (EXCLUDED per brief) | 2505.21179 | Attention-space negative guidance that works at CFG=1 | — | Neg prompt | + | Yes | Core `NAGuidance` |
| **FDG**, frequency-decoupled guidance | 2506.19713 | Separate CFG weights for low and high frequency | — | Yes | 0 | No | pamparamm |
| **Mahiro** | — | Similarity-adaptive positive-biased CFG | — | Yes | 0 | No | Core |
| **FreSca** | — | FFT scaling of the guidance (low and high separately) | — | Yes | 0 | No | Core |

### Key point for ZIT at CFG=1

Only **perturbation-style guidance** adds a signal without a negative prompt, and each one costs one extra forward pass per guided step:

- **self-perturbation:** PAG, SEG, TPG, SLG, SoftPAG;
- **weak-model guidance:** Autoguidance;
- **negative-in-attention:** NAG (excluded).

Everything else in the table is a CFG modifier and does nothing at CFG=1. **TSR** and **CADS / SeedVarianceEnhancer** are the only zero-extra-NFE levers that act at CFG=1.

Caution: attention-perturbation guidance tends to *sharpen toward the mode*. It improves structure and realism, not diversity. Restrict it to mid or late sigmas if composition diversity is the goal.

---

## 7. Noise and latent initialization

### FreeInit

- **Paper:** Wu et al. arXiv **2312.07537**, ECCV 2024. Built for video; the idea transfers to images.
- **Code:** github.com/TianxingWu/FreeInit (VERIFIED, code).
- **Loop:** generate z0, then diffuse it to the final timestep, `z_T = add_noise(z0, ε, t=999)`. Then mix:
  ```
  z_new = IFFT( FFT(z_T)·LPF + FFT(η)·(1 − LPF) )   η ~ N(0, I) fresh
  ```
  and resample. Default 5 iterations, with a coarse-to-fine step count per iteration.
- **Filters:** d_s and d_t are the normalised spatial and temporal cutoffs, and `D² = ((d_s/d_t)(2t/T−1))² + (2h/H−1)² + (2w/W−1)²` (normalised frequency radius, centred).
  - **Gaussian:** `M = exp(−D²/(2·d_s²))`
  - **Butterworth:** `M = 1/(1 + (D²/d_s²)^n)`, with n=4
  - **Ideal:** `M = 1` if `D² ≤ 2·d_s`
  - **Box** variant also provided.
- **Defaults:** `d_s = d_t = 0.25`. The Gaussian filter is used in the examples; Butterworth n=4 is the alternative.
- **Variance.** FreeInit does **not** re-normalise. A complementary mask `LPF + HPF = 1` on independent signals gives per-frequency variance `M² + (1−M)² < 1` in the transition band, so the result has **less than unit variance**.
  - A correct variance-preserving mix is `FFT(z)·M + FFT(η)·sqrt(1 − M²)`, with M ∈ [0,1].
  - Alternatively, renormalise per channel to std 1 afterward.
  - Also make sure the low-frequency part of the reference is itself unit-variance-scaled noise-like.
- **Rectified-flow caveat.** At t=1, `x_1 = ε` exactly, so no information survives. FreeInit's "diffuse to T" gives **zero** signal in RF. Instead do one of these:
  - diffuse to t<1 (e.g. 0.9–0.97) and start sampling there, or
  - mix the low frequencies of the **clean latent (normalised)** into the noise at a chosen strength, e.g. `z = FFT⁻¹(M·α·FFT(norm(x0_ref)) + sqrt(1 − α²M²)·FFT(η))`.

### Related findings

- **Crystal Ball Hypothesis** (Ban et al., arXiv **2406.01970**, ICLR 2025). Specific "trigger patches" in the initial noise, which are statistical outliers of the Gaussian, induce objects at their locations. They are universal across seeds and prompts, and transplanting a patch moves the object. Rejection sampling improves prompt adherence and **positional diversity**. This supports manipulating the initial noise to vary layout.
- **Golden Noise / NPNet** (Zhou et al., arXiv **2411.09502**, ICCV 2025). Needs a trained NPNet that adds a prompt-dependent perturbation to the noise, adding about 3% time. **Not training-free** (it needs NPNet weights, trained for SDXL, DreamShaper and Hunyuan-DiT). It targets alignment and quality, not diversity.
- **InitNO** (Guo et al., CVPR 2024). Gradient optimisation of the initial noise using cross-attention response and self-attention conflict scores, plus a KL/distribution-alignment loss that keeps it Gaussian. Training-free but gradient-based, and UNet-specific (SD1.x).
- **NoiseCollage** (Shirakawa & Uchida, arXiv **2403.03485**, CVPR 2024). Per-object noise estimates are cropped and merged by layout each step. Needs layout boxes.
- **Colorful-Noise** (arXiv **2605.00548**, SIGGRAPH 2026). Replaces the **low-frequency components of the white Gaussian noise with the low frequencies of a reference image's latent**, which controls global structure and colour while the high frequencies stay free. Training-free, zero overhead, and demonstrated on **FLUX (flow)**.
  - This is the "low-frequency init from a reference for composition" method. The exact cutoff and normalisation are UNVERIFIED.
  - Suggested implementation: the variance-preserving mix above, with a Gaussian LPF cutoff d_s about 0.05–0.15 for layout-only control.
- **Pink-noise initialisation** (from 2601.00090). A 1/f^α spectrum boosts diversity of layout and colour for free. Implement by FFT-shaping white noise, then renormalising to unit std per channel.
- **Frequency or "noise mixing" as a composition lever for ZIT.** Because ZIT decides the layout at step 0, an init with a random low-frequency structure (a pink-noise component, or a low-pass of a random real photo latent from a pool of candid photos) is a strong, cheap composition-diversity lever. It must keep the overall per-channel std near 1 to avoid a colour/brightness cast.

### SDEdit-style partial init (brief)

- **SDEdit** (Meng et al., arXiv **2108.01073**): `x_t = (1−t)·x_ref + t·ε` for RF, then sample from t0 (typically 0.6–0.9 for composition transfer). This is what the ZIT community's "denoise 0.7" tip does implicitly: seeds matter more when some structure is fixed.
- **ReNO** (Eyring et al., arXiv **2406.04312**, NeurIPS 2024): reward-based noise optimisation for one-step models (SD-Turbo, PixArt-δ DMD) using gradients of reward models over a few iterations. It optimises quality and alignment, not diversity, and needs backprop.

---

## 8. Recommended starting points (my synthesis, untested on these models)

**ZIT (CFG=1, 8 steps, character LoRA):**
1. CADS-style condition noise on the Qwen3 hidden states. Use sigma-based γ with τ1≈0.8 and τ2≈1.0, ψ=1 (rescaled), and s ≈ 0.5–1.0 × the per-token std. Or use SeedVarianceEnhancer (strength 15–30 uniform, 50% of values, first 10–20% of steps).
2. Diversity-distillation hybrid: no character LoRA, or LoRA at about 0.3, for step 0 (optionally step 1), then the full LoRA.
3. Variance-preserving pink or low-frequency noise init, optionally from a pool of real candid-photo latents via a Colorful-Noise-style mix.
4. TSR with k≈0.9–0.95 and σ≈1–3 for "temperature" (zero NFE, core node).
5. Optionally a STRIDE-like PCA-directed feature perturbation in early blocks, or batch feature dispersion as in "Don't Settle at the Mode".
6. Realism or structure: a custom PAG/SLG/TPG on mid ZIT layers at mid/late sigmas only (+1 NFE).

**Qwen-Image 2512 (CFG>1):**
1. Guidance interval via core `CFGOverride`: cfg=1 for t > ~0.85 (first 10–20% of steps) and for the last ~10–20%.
2. CADS on the cond branch (paper defaults τ1=0.6, τ2=0.9, s=0.25 × embedding std, ψ=1), leaving the uncond branch alone.
3. APG (η=0, r tuned to latent size) or CFG-Zero\* against oversaturation and the "AI glossy" look.
4. `SkipLayerGuidanceDiT` works natively on Qwen (`blocks_replace`).

## Sources (reachable parts)

- CADS code: github.com/v0xie/sd-webui-cads, github.com/asagi4/ComfyUI-CADS. Paper: arxiv.org/abs/2310.17347.
- github.com/ChangeTheConstants/SeedVarianceEnhancer
- github.com/rohitgandikota/distillation (arXiv 2503.10637)
- github.com/kynkaat/guidance-interval (arXiv 2404.07724)
- github.com/gcorso/particle-guidance (arXiv 2310.13102)
- github.com/GaParmar/group-inference (arXiv 2508.15773)
- github.com/temporalscorerescaling/TSR (arXiv 2510.01184)
- github.com/WeichenFan/CFG-Zero-star (arXiv 2503.18886)
- github.com/seti9585/sd-webui-APG (APG defaults)
- github.com/TianxingWu/FreeInit
- github.com/pamparamm/sd-perturbed-attention
- github.com/comfyanonymous/ComfyUI (`comfy_extras`, `comfy/samplers.py`, `comfy/ldm/lumina`, `comfy/ldm/qwen_image`)
- Web-search snippets for: DiverseFlow 2504.07894; EDDY 2605.06553; Don't Settle at the Mode 2606.27371; STRIDE 2605.11494; Never Too Late 2601.00090; Colorful-Noise 2605.00548; Crystal Ball 2406.01970; Golden Noise 2411.09502; InitNO; NoiseCollage 2403.03485; HeadHunter 2506.10978; TPG 2506.10036; SEG 2408.00760; Minority guidance 2301.12334; SPARKE 2506.10173.
