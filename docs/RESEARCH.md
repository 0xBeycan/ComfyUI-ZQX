# RESEARCH — training-free levers against the scene-level "AI look" while keeping identity

Date: 2026-09-28. Target models: **Z-Image Turbo** (single-stream S3-DiT, rectified flow, distilled, ~8 steps, CFG = 1),
**Qwen-Image 2512** (MMDiT, 60 double-stream blocks, CFG > 1) and **Qwen-Image-Edit 2511** (same architecture, reference
latents concatenated as extra image tokens at frame index 1, 2, ...).

This file is the curated synthesis.  The detailed per-paper notes (mechanism, exact formulas, hyper-parameters, code
paths, existing ComfyUI implementations) are in three appendices:

* [`research/identity_dit.md`](research/identity_dit.md) — training-free subject / identity in DiT, RoPE placement of
  reference tokens, block roles (21 sections).
* [`research/diversity_guidance_noise.md`](research/diversity_guidance_noise.md) — CADS, distilled-model mode collapse,
  interval guidance, particle/group diversity, perturbation guidance at CFG = 1, noise initialisation.
* [`research/lora_arithmetic_inversion.md`](research/lora_arithmetic_inversion.md) — task arithmetic, TIES, DARE,
  ZipLoRA, B-LoRA, K-LoRA, TSV/Iso-C/KnOTS, sliders, LoRA key formats of musubi-tuner / ai-toolkit for Qwen-Image and
  Z-Image; RF-Inversion / RF-Solver / FireFlow / UniEdit-Flow / Tight Inversion and what reference branches really use.

## How sources were checked (read this first)

The sandbox egress proxy **blocked arxiv.org, alphaxiv, huggingface.co, openreview, Semantic Scholar and CVF** for
this session.  Formulas were therefore verified against the **official GitHub code** of each method (cloned during the
research) and against the ComfyUI source; paper-only details were taken from search snippets.  The appendices tag every
statement as *code-verified*, *snippet* or **UNVERIFIED**.  Where the paper and the released code disagree, the
appendices follow the code (that is what produced the published results) and say so.

## 1. Problem restatement and what "training-free" can reach

The character LoRA was trained on ~35 Qwen-Image-Edit 2511 renders generated from one neutral, frontal, passport-style
reference.  Every training image therefore shares the *mode* of QIE + the neutral reference: frontal pose, camera gaze,
posed smile, clean/bokeh background, centred framing.  A LoRA cannot tell identity from these constant factors, so both
are entangled in its ΔW.  Realism LoRAs push against exactly that mode, which is why strong realism loses identity.

Training-free levers fall into five groups, each acting at a different place:

| Where | Lever | What it can change |
|---|---|---|
| attention | extended / shared self-attention to a reference (reference-only, ConsiStory, StoryDiffusion, FreeCus, Personalize-Anything, CharaConsist, FreeGraftor) | adds identity from the passport reference *without* the LoRA having to carry it → the character LoRA (and its baked-in mode) can be run weaker |
| weights (per step / per block) | scheduled LoRA strengths, K-LoRA per-layer selection, block-wise strengths | realism / free composition early (layout steps), identity late (detail steps); keep the LoRA out of blocks that carry layout |
| weights (offline) | task arithmetic, subspace projection, TIES / DARE / KnOTS | remove the part of the character ΔW that overlaps the realism ΔW's subspace; resolve sign conflicts |
| conditioning / guidance | CADS, interval CFG with an anti-AI-look negative, base-model first step | break mode collapse of the distilled model in the high-noise steps (pose, framing, background) |
| initial noise | low-frequency composition from a real photo | steer global layout / lighting away from the centred studio portrait |

## 2. Per-method summary (short; details and UNVERIFIED marks in the appendices)

### 2.1 Identity / subject (appendix A)

| Method (arXiv) | Mechanism in one line | Train-free | DiT/MMDiT | Relevance here |
|---|---|---|---|---|
| Personalize Anything (2503.12590) | copy the (inverted ≈ forward-noised) reference's image tokens into the target region while t > τ (code τ = 0.6), then shared attention | yes | Flux | **locks layout** → reproduces the passport pose; not suited to the goal |
| FreeCus (2507.15249) | reference less noised than the target (negated shift μ), reference foreground K scaled ×1.1 and prepended, only in Flux vital layers [0,1,2,17,18,25,28,53,54,56] | yes | Flux | **implemented as options** (ref_sigma_mult, key_scale, blocks) |
| CharaConsist (2507.11533) | reference K re-RoPE'd to the matched target token position (point tracking, cos-sim > 0.5 from a pre-run), output merge α = 0.8 → 0 | yes | Flux | best for pose freedom; its pre-run correspondences are replaced by per-block value-vector matching → **implemented as `matched` placement (0.2.0)** |
| FreeGraftor (2504.15958) | per-block cycle-consistent matching, grafting, dropout 0.2·t | yes | Flux | per-block matching + mutual-NN check **implemented** in `matched` placement |
| Stable Flow (2411.14430) | vital layers via block-skip + DINOv2; K/V replacement in vital layers | yes | Flux/SD3 | layer-finding protocol; our Block Spec node supports the same ablation |
| FreeFlux (2503.16153) | position-dependent layers [1,2,4,26,30,54,55] vs content-dependent layers in Flux | yes | Flux | inject only into content layers → less layout copying; indices must be re-measured for Qwen / Z-Image |
| DiTCtrl, KV-Edit | KV sharing / KV-cache editing in MM-DiT | yes | MM-DiT | editing-oriented |
| ConsiStory (2402.03286) | subject-driven shared attention, **reference-token dropout p = 0.5**, vanilla-query blending early | yes | UNet | **token dropout implemented** |
| StoryDiffusion consistent self-attention | batch-shared random subset of tokens | yes | UNet | same family as our node |
| 1Prompt1Story | single prompt, embedding re-weighting | yes | UNet | not applicable to the LoRA problem |
| StyleAligned (2312.02133) | shared attention + AdaIN of target Q/K to the reference statistics, log-bias ln 2 | yes | UNet | log-bias implemented; AdaIN **not** implemented: it transfers the reference's global statistics (passport lighting), counter to the goal, and would require un-rotating RoPE in Z-Image |
| RB-Modulation (2405.17401) | attention feature aggregation, stochastic optimal control | yes | UNet/SC | style; not implemented |
| reference-only (sd-webui-controlnet) | reference noised to current t, K/V concatenated in self-attention, style_fidelity blend | yes | UNet | the direct ancestor of our node |
| Visual Style Prompting | K/V swap in late self-attention layers | yes | UNet | style |
| InstantStyle | SDXL block roles (up_blocks.0.attentions.1 style, down_blocks.2.attentions.1 layout) | yes (adapter) | UNet | block-role idea only |
| OminiControl / EasyControl / UNO / Kontext / Qwen-Edit / Z-Image-Omni | RoPE placement of condition tokens: width/height offset, diagonal offset, separate frame index | — | Flux/Qwen/Z-Image | **frame / right / below / same placements implemented** |
| GRAG (Qwen-Image-Edit) | K_ref ← b·μ + δ·(K_ref − μ), 0.8–1.7 | yes | Qwen-Edit | possible future strength knob |
| AttnRouter (2605.01480) | Qwen-Image-Edit-2511 editing circuit ≈ layers 30–45, steps 0–7 (**UNVERIFIED**) | yes | Qwen | only Qwen-specific block localisation found |

Inversion (appendix C, B1–B7): RF-Inversion, RF-Solver, FireFlow, UniEdit-Flow and Tight Inversion give better
reconstructions than forward noising, but the reference branches of Personalize Anything (RF-Inversion with γ = η = 1,
which reduces to forward noising with a fixed ε), FreeCus (no inversion), reference-only and StoryDiffusion do not need
it.  For an 8-step distilled model an inversion would cost as much as the generation.  **Decision:** forward noising with
a fixed ε (and an optional clean, cached capture — the Qwen-Image-Edit 2511 `index_timestep_zero` convention).

### 2.2 Diversity / mode collapse / guidance / noise (appendix B)

| Method (arXiv) | Key formula / finding | CFG = 1 distilled OK? | In ComfyUI core? |
|---|---|---|---|
| CADS (2310.17347) | γ(t) = 1 (t ≤ τ1), (τ2 − t)/(τ2 − τ1), 0 (t ≥ τ2); ŷ = √γ y + s √(1−γ) n (fresh n each step); rescale ψ; SD values τ1 = 0.6, τ2 = 0.9, s = 0.25, ψ = 1 (code-verified) | yes (only touches the conditioning) | no → **implemented** |
| Distilling Diversity and Control (2503.10637) | distilled models fix the layout after the first step; running the **base** model for step 1 restores diversity; LoRAs transfer base↔distilled | yes | no → **implemented** (Sigma Split Guider, model_early) |
| Guidance interval (2404.07724) | CFG only in a middle noise interval improves FID/FD; EDM2 σ ∈ (0.28, 5.42] ≈ flow t ∈ (0.22, 0.84) | the "early-only negative" variant keeps late steps at CFG 1 | **core** `CFGOverride` (start/end percent → sigma, PREDICT_NOISE wrapper) already does interval CFG; our Sigma Split Guider exists for the *base-model early steps* and takes sigma directly |
| Particle guidance, DiverseFlow, EDDY, Group Inference, STRIDE, "Don't settle at the mode" | batch repulsion / selection | needs several samples per run | no; not implemented (batch-coupled, heavy) |
| TSR (2510.01184) | temporal score rescaling, k < 1 = more diversity | yes | **core** (`TemporalScoreRescaling`) — documented, not duplicated |
| PAG (2403.17377), SEG, STG, SLG, TPG | attention-perturbation guidance, +1 forward pass | would add signal without a negative prompt, but may push *towards* the "clean" mode | PAG/SEG/SAG core are **UNet-only**; `SkipLayerGuidanceDiT` is core and works on Qwen, **not** on Z-Image (no `patches_replace` in NextDiT) |
| APG (2410.02416), CFG-Zero* (2503.18886), CFG++, RescaleCFG, TCFG, Mahiro | CFG modifications | no effect at CFG = 1 | **core** (`APG`, `CFGZeroStar`, `CFGNorm`, `TCFG`, `Mahiro`, `RescaleCFG`) — use them on Qwen-Image |
| NAG | normalized attention guidance | **excluded** (breaks Z-Image Turbo per user test) | core `NAGuidance` — not used |
| FreeInit (2312.07537) | LPF(noised latent) + HPF(fresh noise), Gaussian/Butterworth D0 = 0.25 | yes | no → **implemented** as a *variance-preserving* low-frequency composition from a real photo (FreeInit's plain complement loses the 2H(1−H) cross energy) |
| Colorful-Noise (2605.00548), pink-noise init (2601.00090), initial-noise layout findings | low-frequency / coloured noise sets composition | yes | no (our low-frequency node covers the idea) |

### 2.3 LoRA arithmetic (appendix C)

| Method | Exact algorithm | Train-free | Implemented |
|---|---|---|---|
| Task arithmetic / negation (2212.04089) | ΔW₁ ± λΔW₂ = [s₁B₁ ∣ ±λs₂B₂]·[A₁; A₂] (exact) | yes | `add`, `negate` |
| Subspace projection (linear algebra; cf. orthogonal adaptation, TSV) | (I − λQ₂Q₂ᵀ)ΔW₁, ΔW₁(I − λP₂P₂ᵀ), ΔW₁ − λQ₁Q₁ᵀΔW₂ with SVD-based orthonormal bases of the *product's* column/row space | yes | `clean_col`, `clean_row`, `target_sub` |
| TIES (2306.01708) | trim top-k% → elect sign sgn(Σ τ̂) → disjoint mean (peft ordering) | yes | `ties_dense` (+ truncated SVD, error reported) |
| DARE (2311.03099) | δ·(1−m)/(1−p), m ~ Bernoulli(p) | yes | `dare_drop` option |
| KnOTS (2410.19735) | SVD-align the LoRAs in a shared basis U of [ΔW₁ ∣ ΔW₂], TIES on UᵀΔWᵢ, ΔW = U·merged — exact low rank | yes | `knots_ties` |
| K-LoRA (2502.18461, CVPR 2025) | per attention layer, per step: content LoRA iff (S_c/γ)/(S_s·S(t)) > 1, S = sum of top-(r_c·r_s) abs(ΔW), γ = trimmed mean L1 ratio, S(t) = α t/T + β (code-verified) | yes | **ZQX K-LoRA** |
| ZipLoRA (2311.13600), LoRA.rar, Mixture-of-LoRA | learned mergers / hypernetworks | **no** | not implemented |
| B-LoRA (2403.14572) | SDXL content/style blocks (training-based) | no | idea → block-weight tool |
| Iso-C / Iso-CTS (2502.04959), TSV-Merge (2412.00081) | spectrum flattening / common+specific subspaces | yes | not implemented; Iso-CTS's *common subspace of several character LoRAs from the same QIE pipeline* is a promising training-free estimate of the "AI-look" direction (hypothesis) |
| Concept sliders (2311.12092) and training-free variants | slider directions from prompt pairs | originals need training | not implemented |

Key formats (code-verified in musubi-tuner / ai-toolkit): musubi Qwen-Image `lora_unet_transformer_blocks_N_attn_to_q.lora_down/up.weight + .alpha` (default network_alpha = 1 → scale 1/rank); ai-toolkit `diffusion_model.transformer_blocks.N.attn.to_q.lora_A/B.weight` without alpha; Z-Image musubi `lora_unet_layers_N_attention_to_q…` (not loadable by ComfyUI's key map as-is; the arithmetic node resolves them through normalised names), ai-toolkit `diffusion_model.layers.N.attention.to_q.lora_A/B`.  ComfyUI keeps Z-Image q/k/v fused in `attention.qkv`; to_q/to_k/to_v LoRAs are slices of it.

## 3. Ranking for this exact problem

Ranked by (expected impact on the scene-level AI look while preserving identity) × (confidence that the mechanism
transfers to Z-Image / Qwen) ÷ (risk of breaking the distilled model).  All are implemented unless stated.

Items 8–11 were added in 0.2.0; by expected value, seed search (8) and the identity-safe realism LoRA (9) belong
right after item 1, which is also the order used in the README test plan.

1. **Sigma-scheduled LoRA strengths (ZQX Scheduled LoRA)** — heuristic built on the robust finding that flow models fix
   layout/pose in the first steps (Distilling Diversity §2; guidance interval) and that LoRAs act additively.  Weak
   character LoRA + strong realism in σ ≳ 0.8, then the reverse.  Zero cost, no re-patching, works in both passes (the
   2nd pass starts below the layout window, so identity dominates there).  Highest expected value, lowest risk.
2. **Reference attention from the passport photo (ZQX Reference Attention)** — reference-only / ConsiStory / FreeCus
   family, verified exact against QIE's native reference concatenation.  Lets identity come from the reference so the
   character LoRA can be run weaker (or cleaned).  Use a separate frame index, skip the first 1–2 steps, token dropout
   0.3–0.5 and a face key-mask to avoid copying the passport layout.  Medium risk: out-of-distribution positions for
   Z-Image Turbo / Qwen-Image T2I (UNVERIFIED generalisation), 2× compute in `noised` mode.
3. **K-LoRA (character + realism)** — the only published training-free method designed for exactly "subject LoRA +
   style LoRA coexist".  Code-verified rule; untested on DiTs other than FLUX.
4. **Interval CFG with an anti-AI-look negative + optional base-model first step** — guidance interval (core
   `CFGOverride`, or ZQX Sigma Split Guider) + Distilling Diversity (ZQX Sigma Split Guider `model_early`).  On
   Z-Image Turbo only the first 1–2 steps pay the uncond cost.
5. **CADS** — strong, code-verified diversity method; conservative settings needed on the distilled model
   (window τ1 ≈ 0.8, τ2 = 1.0 for 8-step ZIT).
6. **LoRA arithmetic: clean_col / target_sub / knots_ties + conflict report** — the report tells *where* the character
   and realism LoRAs overlap; projection removes the overlapping directions exactly.  Expected impact depends on how
   much of the AI look lives in the realism LoRA's subspace (unknown until measured) — becomes much stronger once an
   "AI-look LoRA" exists (then `clean_col` with that LoRA is the principled operation).
7. **Low-frequency noise from a real photo** — cheap composition/lighting prior, FreeInit-style; risk of colour casts at
   high strength / high cutoff.
8. **First-step seed search with face / background scorers (0.2.0)** — selection instead of modification: distilled
   models fix the layout in the first step, so scoring 1–2-step previews (head turn, off-centre, identity, background
   sharpness) and keeping the best seeds directly targets measurable AI-look attributes without touching the model.
   High value, cost = candidates × probe steps.
9. **Identity-safe realism LoRA via ablation scan (0.2.0)** — the realism LoRA, not the character LoRA, is edited:
   units whose removal restores ArcFace identity at little CLIP-measured realism cost are removed.  Lower risk than
   editing the character LoRA; heuristic one-at-a-time attribution.
10. **Spatial LoRA (LoRAShop, 2505.23758)** — realism and character LoRAs on disjoint token regions in the same step;
    needs a face region (pass 2, or from a known pose).
11. **Pose bank + ControlNet** — composition from real photos; strongest composition lever, needs a curated bank.
12. **LoRA guidance, activation steering (ActAdd, 2308.10248), UCE (2308.14761), DiT PAG (2403.17377), LoRA surgery,
    common subspace (Iso-CTS-inspired)** — implemented as experimental levers; weaker priors that they address the
    scene-level look specifically.
13. Still not implemented: per-model vital-layer measurement protocol automation (Stable Flow / FreeFlux — Block Spec
    + scorers make a manual sweep possible), reward-gradient guidance (impractical memory on Qwen 20B), StyleAligned
    AdaIN (deliberately excluded), NAG (excluded).
