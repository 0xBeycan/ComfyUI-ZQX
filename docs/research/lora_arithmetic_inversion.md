# Research: training-free LoRA arithmetic + flow-model inversion (for ComfyUI-ZIT-QIE-Experimental)

Date: 2026-09-28. Network note: arxiv.org, alphaxiv, huggingface.co and openaccess.thecvf.com were all blocked by the egress proxy. So paper details come
from (a) the official GitHub source code, which I cloned and read (marked **[code-verified]**), (b) web-search snippets
(**[snippet]**), or (c) memory of the papers (**UNVERIFIED**). Sources were shallow-cloned locally during research (not included in this repo).

---------------------------------------------------------------------------------------------------
## TOPIC A: LoRA / weight-space arithmetic

Notation: a LoRA on a linear layer W (d_out x d_in) gives ΔW = s·B·A, with B = lora_up/lora_B (d_out x r),
A = lora_down/lora_A (r x d_in), and s = alpha/r (kohya) or s = 1 (peft/diffusers files without alpha). The user multiplier m is applied on top.
Note that the factorization is **not unique**: (B R)(R⁻¹ A) gives the same ΔW for any invertible R. So any per-element
operation on A or B alone (sign election, magnitude trimming, dropout) does not act on ΔW itself.
Compare subspaces or dense ΔW, never raw factors.

### A1. Task arithmetic / negation. Ilharco et al., "Editing Models with Task Arithmetic", arXiv 2212.04089 (ICLR 2023). Training-free.
- τ_t = θ_ft,t − θ_pre. Addition: θ = θ_pre + λ Σ_t τ_t. Negation (forgetting): θ = θ_pre − λ τ.
  λ is chosen on validation data. UNVERIFIED: grid {0, 0.05, …, 1.0}.
- For LoRAs: τ is ΔW, so "negation" means applying a LoRA with a negative multiplier. This works in any ComfyUI LoRA loader.
- peft: `combination_type="linear"` / `"svd"` / `"cat"` (see A12).
- Relevance: to subtract an "AI look" you need a vector that encodes the AI look *alone*, for example (i) a LoRA
  trained on QIE outputs of a *different* identity with the same pipeline, or (ii) the **common subspace** of ≥2
  character LoRAs trained with the same synthetic pipeline (see Iso-CTS, A9). That common subspace is a plausible
  training-free estimate of the shared "AI look" (my hypothesis, not from a paper). Negating a whole character LoRA also removes identity.

### A2. TIES-Merging. Yadav et al., arXiv 2306.01708 (NeurIPS 2023). Training-free.
Per parameter p, over tasks t (formulas match the peft implementation **[code-verified]** `peft/utils/merge_utils.py`):
1. **Trim**: keep the top-k% magnitude entries of each τ_t and zero the rest → τ̂_t. Paper default k = 20% (keeps 20%) UNVERIFIED-exact.
   peft `density` = fraction kept (per tensor, `torch.topk` on |τ|).
2. **Elect sign**: γ^p = sgn(Σ_t τ̂_t^p). The paper calls this "the sign with the larger total magnitude", and sgn of the signed sum is the same thing.
   peft `majority_sign_method="total"` gives sgn(Σ τ̂); `"frequency"` gives sgn(Σ sgn τ̂). Ties (sum = 0) go to +1.
3. **Disjoint mean**: A^p = {t : sgn(τ̂_t^p) = γ^p}. Then τ_m^p = (1/|A^p|) Σ_{t∈A^p} τ̂_t^p, with the count clamped to at least 1.
4. θ = θ_pre + λ τ_m. Paper: λ from validation. UNVERIFIED: default λ = 1 without validation data.
- peft applies the per-task weights w_t *after* the sign election (the sign is elected on unweighted trimmed tensors) and *before* the disjoint mean.
- **peft pitfall [code-verified]**: `add_weighted_adapter(..., combination_type="ties"|"dare_linear"|"dare_ties"|"magnitude_prune")`
  applies the operation to **A and B separately**, with weight √(w·s) on each side and the sign on A only. That is not TIES on ΔW (gauge issue above).
  The correct ΔW-level versions are `"ties_svd"`, `"dare_linear_svd"`, `"dare_ties_svd"` and `"magnitude_prune_svd"`. These form the dense
  ΔW_t = get_delta_weight (scaling already included), merge, then truncate with `torch.linalg.svd` to `svd_rank`, with optional
  `svd_clamp` quantile clamping (borrowed from kohya `svd_merge_lora.py`). The returned (A = Vh[:r], B = U[:, :r]·diag(S[:r])) has scale 1.
  The non-svd types require equal ranks across adapters.

### A3. DARE ("Language Models are Super Mario"). Yu et al., arXiv 2311.03099 (ICML 2024). Training-free.
- For each delta parameter: m ~ Bernoulli(p) with p = drop rate, δ̃ = (1 − m) ⊙ δ / (1 − p), so E[δ̃] = δ.
  peft `density` = 1 − p (`random_pruning`: mask ~ Bernoulli(density), divide by density) **[code-verified]**.
- It is then combined with task arithmetic (`dare_linear`) or with TIES sign election and disjoint mean (`dare_ties`).
- Paper claims p = 0.9 (even 0.99 for very large models) keeps performance for SFT deltas with small magnitudes.
  It fails when delta values are large (UNVERIFIED threshold ~0.005 for continued-pretraining-sized deltas).
- For LoRAs apply DARE to the dense ΔW. The dropped ΔW is full-rank noise around ΔW, so you must re-factorize (A6) or patch dense.
  Dropping entries of A/B instead preserves the expectation (E[m_B B m_A A]/(1−p)² = BA for independent masks) but has a different variance.
  Stochastic: fix the seed.

### A4. ZipLoRA. Shah et al., arXiv 2311.13600. **Not training-free.**
- Learns merger coefficient vectors m_c, m_s, one scalar per column of each LoRA ΔW. Merged ΔW = m_c ⊗ ΔW_c + m_s ⊗ ΔW_s (column-wise).
  Loss = content reconstruction + style reconstruction (each LoRA's own prompts) + λ Σ_i |cos(m_c^(i)ΔW_c^(i), m_s^(i)ΔW_s^(i))|. About 100 steps, SDXL.
  UNVERIFIED exact loss form.
- Training-free take-aways: (1) LoRA ΔW columns are sparse and most have small norm. (2) Column-wise cosine similarity
  between two LoRAs predicts interference. You can compute cos(ΔW_c[:,i], ΔW_s[:,i]) per column as a diagnostic, or down-weight colliding columns.

### A5. B-LoRA. Frenkel et al., arXiv 2403.14572 (ECCV 2024). Training-based (joint LoRA on two blocks), but the block finding is reusable.
- SDXL: content = `unet.up_blocks.0.attentions.0`, style = `unet.up_blocks.0.attentions.1` **[code-verified from B-LoRA `blora_utils.py`]**.
  These are the "4th and 5th transformer blocks" of the paper.
- DiT analogues are *not* given by B-LoRA. For FLUX, Stable Flow's "vital layers" play a similar role (A/B-tested per layer). For
  Qwen-Image / Z-Image there is no published map. You would have to find it empirically with per-block LoRA ablations: apply each LoRA to one block
  only and measure identity similarity (e.g. ArcFace) vs "realism" (e.g. a detector score).

### A6. Dense ΔW → low rank again: correct linear algebra and trade-offs
Let ΔW_i = s_i B_i A_i with rank r_i.

**(a) Exact low-rank difference/sum (concatenation, peft `"cat"`).**
ΔW_1 − λΔW_2 = [s_1 B_1 | −λ s_2 B_2] · [A_1 ; A_2] = B_cat A_cat, with rank ≤ r_1 + r_2. This is **exact** with no approximation.
To save in kohya format: `lora_up = [B1 | B2]` (d_out x (r1+r2)), `lora_down = [s1·A1 ; −λ·s2·A2]`, `alpha = r1+r2` (so the loader's scale = 1).
Alternatively put √ of the scales on each side for better fp16/bf16 conditioning. The ComfyUI Z-Image converter uses the same trick for QKV (block-sparse up, alpha·3).

**(b) Efficient exact SVD of a LoRA product.** QR: B = Q_B R_B, Aᵀ = Q_A R_A. Small SVD: R_B R_Aᵀ = U' Σ V'ᵀ (r x r).
Then ΔW/s = (Q_B U') Σ (Q_A V')ᵀ. Cost is O((d_out + d_in) r² + r³). Use it for rank truncation, principal directions, and subspace similarity.
For a sum of LoRAs, run it on (B_cat, A_cat).

**(c) Projecting ΔW_1 off LoRA 2's column (output) space.** Q_2 = orth(col(B_2)) via reduced QR of B_2 (d_out x r_2).
Better: use the left singular vectors U_2 from (b), because B_2 may be ill-conditioned; drop directions with σ below a threshold.
ΔW_1' = (I − Q_2 Q_2ᵀ) ΔW_1 = s_1 (B_1 − Q_2 (Q_2ᵀ B_1)) A_1. Only B_1 changes, the rank stays r_1, and the cost is O(d_out r_1 r_2).
- Meaning: ΔW_1' x is orthogonal to span(B_2) for every x. LoRA 1 can no longer write into the output directions LoRA 2 writes into.
  So the two stop fighting or double-counting in those directions. You lose whatever part of LoRA 1 lived there
  (energy lost = ‖Q_2ᵀ B_1 A_1‖²_F s_1², which you can report to the user).
- **Row (input) space**: Q̃_2 = orth(A_2ᵀ) (d_in x r_2). ΔW_1'' = s_1 B_1 (A_1 − (A_1 Q̃_2) Q̃_2ᵀ). LoRA 1 then ignores the input directions
  LoRA 2 reads. Two-sided: (I − Q_2Q_2ᵀ) ΔW_1 (I − Q̃_2Q̃_2ᵀ), with rank ≤ r_1.
- **Soft projection** (keeps more of LoRA 1): P = I − U_2 diag(σ_i²/(σ_i² + τ²)) U_2ᵀ, or project only the top-k directions of ΔW_2.
- Which way to project? To protect identity, project the *realism* LoRA off the character LoRA's subspace (realism is then prevented
  from overwriting identity directions). To make realism win on shared directions, project the character LoRA off realism's subspace.
  Both keep a low-rank format and are exact (no SVD truncation).
- Caveat: rank-r subspaces in d = 3072 are nearly orthogonal by chance: E[φ] ≈ r/d, about 0.5% for r = 16. So measured overlap well above
  r/d is meaningful. Raw col-space projection ignores magnitude, so use the energy-weighted overlap below.

**(d) TIES / DARE / magnitude-prune on dense ΔW, then re-factorized.**
- Form ΔW_t dense per layer (Qwen-Image 3072x3072 ≈ 9.4M entries per linear, ~60 blocks x ~10 linears; fine per-layer on GPU in fp32).
  Run TIES and then truncated SVD to rank r': ΔW_r' = U_{:r'} Σ_{:r'} V_{:r'}ᵀ, with error ‖ΔW − ΔW_r'‖²_F = Σ_{i>r'} σ_i² (Eckart–Young).
- Trade-off: trimming and random dropping produce a *full-rank* matrix. The sparsification "noise" spreads energy across all singular values.
  So a rank-(r1+r2) truncation discards part of the TIES effect, and the result is *not* equal to TIES. At the same time a very large r' is costly.
  Report the retained energy fraction Σ_{i≤r'}σ_i² / Σσ_i² per layer. Use a randomized SVD (`torch.svd_lowrank`) for speed.
- **In ComfyUI you do not need to re-factorize at all.** ComfyUI supports dense weight-diff patches: key `"{x}.diff"` in a
  LoRA file, or in memory `model.add_patches({key: ("diff", (tensor,))}, strength)` **[code-verified comfy/lora.py]**.
  A node can compute the merged dense ΔW per layer and add it as a diff patch (VRAM/RAM cost = one extra full matrix per patched layer).
  Re-factorize only if the user wants to save a small LoRA file.
- Sign election is meaningful on dense ΔW. Doing it on A or B alone (peft non-svd path) is not.

### A7. Subspace overlap measures (for diagnostics nodes)
- **LoRA paper §7** (Hu et al. 2106.09685): φ(A_1, A_2, i, j) = ‖U_{A1}^{i ᵀ} U_{A2}^{j}‖²_F / min(i, j) ∈ [0,1].
  U^i = the top-i singular vectors (right singular vectors of the r x d_in "A" matrix, i.e. the row space). The paper compares r=8 vs r=64 runs.
  Apply it to both sides: U from B/ΔW (output space) and V from A/ΔW (input space), computed via A6(b).
- **Principal angles**: σ_k(Q_1ᵀ Q_2) = cos θ_k. Grassmann distance = ‖θ‖_2. Projection similarity = Σ cos²θ_k / min(r1, r2) (same as φ).
- **Energy-weighted overlap** (more useful here): ‖Q_2ᵀ ΔW_1‖²_F / ‖ΔW_1‖²_F, i.e. the fraction of LoRA 1's energy inside LoRA 2's output space.
  Also the plain cosine ⟨ΔW_1, ΔW_2⟩_F / (‖ΔW_1‖‖ΔW_2‖) = tr((A_1 A_2ᵀ)(B_2ᵀ B_1)) s1 s2 / norms. This is computable in O(r² d) without forming ΔW.
- Per-layer heatmaps of these identify which blocks the two LoRAs collide in (candidate layers for projection, K-LoRA-style switching, or block weights).

### A8. K-LoRA. Ouyang et al., arXiv 2502.18461 (CVPR 2025). **Training-free.** SDXL and FLUX **[code-verified: HVision-NKU/K-LoRA `klora.py`, `utils.py`]**
Per targeted linear layer (SDXL: to_q/to_k/to_v of all attention processors; FLUX: to_q/to_k/to_v of the attn processors), with content LoRA c and style LoRA s:
- ΔW_c = B_c A_c and ΔW_s = B_s A_s (diffusers lora_B @ lora_A; **no alpha scaling** in their code).
- K = r_c · r_s (product of ranks).
- S_c = Σ TopK(|ΔW_c|) and S_s = Σ TopK(|ΔW_s|) (sums of the K largest absolute entries, computed on the dense product).
- γ ("average_ratio"), computed once over all layers: ρ_l = Σ|ΔW_c,l| / Σ|ΔW_s,l| (full L1 sums).
  γ = mean(ρ_l) after dropping layers with ρ_l ≥ 3·mean(ρ). It balances magnitude differences between LoRAs from different sources.
- Timestep scale: S(t) = α · t_now/t_all + β, where t_now = the step index counted from the start of denoising (0 = pure noise).
  Pattern "s": α=1.5, β=0.5, so S goes 0.5 → 2.0. Pattern "s*" (recommended for FLUX): α=1.5, β=0.85α=1.275, S = (α t/T + β) mod α.
  That gives 1.275 → 1.5 in the first 15% of steps, then a wrap to 0, rising to 1.275 at the end.
- **Selection (hard switch, per layer, per step):** if (S_c/γ) / (S_s · S(t)) > 1 use ΔW_c, else use ΔW_s. There is no blending.
  Larger S later on favours the style LoRA, matching the paper's claim of object/content early and style late.
- Implementation quirk: their t_now is a *global forward-call counter* modulo `sum_timesteps`. That is 28000 hard-coded for SDXL, and
  diffuse_step x #LoRA-layers for FLUX (28 steps). In a ComfyUI node use the real progress (sigma or step index from transformer_options).
  Watch out for cond/uncond passes counting twice.
- Cost: the top-K sums don't depend on t, so precompute S_c and S_s per layer once. At runtime it is just a per-layer threshold
  on t: the layer switches from c to s once S(t) > (S_c/γ)/S_s. So the schedule reduces to a per-layer switch time. This is cheap and easy to
  implement as two weight patches plus a per-layer step gate, or as a per-layer LoRA strength of 0/1 that changes with the step.
- For character + realism LoRAs on a DiT: treat character = "content" and realism = "style". This is untested for DiTs other than FLUX.
  K-LoRA's hard per-layer selection is attractive for identity because layers where the character LoRA dominates keep it at full strength.
- Related (search-snippet only, UNVERIFIED): "Dynamic Training-Free Fusion of Subject and Style LoRAs", arXiv 2602.15539 (2026).

### A9. Orthogonal / subspace merging methods
- **TSV-Merge**, "Task Singular Vectors: Reducing Task Interference in Model Merging", Gargiulo et al., arXiv 2412.00081 (CVPR 2025). Training-free
  **[code-verified: AntoAndGar/task_singular_vectors `TSVM_utils.compute_and_sum_svd_mem_reduction`]**:
  per layer, SVD each task matrix Δ_t = U_t Σ_t V_tᵀ and keep the top k = ⌊rank/T⌋ components (T = #tasks). Concatenate
  U = [U_1,k … U_T,k], Σ = diag(Σ_1,k … Σ_T,k), V = [V_1,k … V_T,k]. Orthogonalize (whitening = orthogonal Procrustes):
  U⊥ = P_U Q_Uᵀ where U = P_U S_U Q_Uᵀ is its SVD, and likewise V⊥. Merged Δ = U⊥ Σ V⊥ᵀ, then θ = θ_pre + α Δ (paper α ≈ 1.0, UNVERIFIED).
  For two LoRAs of rank r this is cheap: everything lives in ≤ 2r dimensions. With T=2 each LoRA keeps half its components
  (you can instead keep all r of each, since LoRAs are already low rank). This is essentially "de-interfered concatenation".
- **Iso-C / Iso-CTS**, "No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces", Marczak et al., arXiv 2502.04959 (ICML 2025).
  Training-free **[code-verified: danielm1405/iso-merging `src/utils/iso.py`]**:
  - Iso-C: Δ = Σ_t Δ_t, SVD = U Σ Vᵀ, replace Σ by mean(σ)·I, giving Δ_iso = mean(σ) U Vᵀ (flattened spectrum), then scale α.
  - Iso-CTS: common subspace = top-k of the SVD of Σ_t Δ_t (k = common_space_fraction·min(d)). For each task, the residual
    Δ_t − U_c U_cᵀ Δ_t → SVD → top n_t = (min(d) − k)/T directions. Stack common + task-specific U/V, orthogonalize them
    (U ← P Qᵀ from the SVD), set a flat spectrum mean(σ), and reconstruct.
  - Relevance: the **common/task-specific split** is exactly the tool for "shared AI-look vs identity" when several synthetic-pipeline
    character LoRAs exist (see A1). Spectrum flattening boosts weak directions, which may also amplify artifacts for diffusion LoRAs (untested).
- **KnOTS**, "Model merging with SVD to tie the Knots", Stoica et al., arXiv 2410.19735 (ICLR 2025). Training-free **[code-verified: gstoica27/KnOTS `task_merger.apply_svd`]**:
  concatenate the task updates along the input axis, [ΔW_1 | … | ΔW_n] (d_out x n·d_in), and take the SVD U Σ [V_1 … V_n]
  (keep σ > 1e-5). Each task is represented in the **shared basis U** as ΣV_iᵀ (rank x d_in). Apply any merge (TIES with topK=20%,
  sum_of_values sign, mean; or DARE-TIES) to the ΣV_i, then ΔW = U · merged(ΣV). The idea is that LoRA updates are poorly aligned,
  and SVD aligns them so element-wise TIES becomes meaningful. For LoRAs: U = left singular vectors of [B_1 | B_2] (≤ r1+r2), computed cheaply,
  and the ΣV_i are (r1+r2) x d_in. The result is exactly low-rank (≤ r1+r2) with no truncation needed. **Best-fitting method for "TIES on LoRAs" in a node.**
- **LoRI**, Zhang et al., arXiv 2504.07448 (COLM 2025): frozen random A and sparse task-masked B, giving near-orthogonal adapters. **Requires training** (LLM-focused).
- **Orthogonal Adaptation** (Po et al., arXiv 2312.02432, UNVERIFIED id): trains customization LoRAs with orthogonal fixed A bases so they sum
  without interference. Training-based. "OrthoMerge" by that exact name was not found.
- **Decouple and Orthogonalize: A Data-Free Framework for LoRA Merging**, arXiv 2505.15875 (snippet only, UNVERIFIED details). Data-free.
- **Null-space LoRA merging**, "Label-Free Cross-Task LoRA Merging with Null-Space Compression", arXiv 2603.26317 (snippet only, UNVERIFIED).
- "Subspace Boosting" (model merging at scale via subspace boosting, 2025, UNVERIFIED id/details): addresses rank collapse
  when merging many experts by boosting small singular values. Similar spirit to Iso-C.

### A10. Multi-LoRA composition for diffusion (training status)
- **Multi-LoRA Composition for Image Generation**, Zhong et al., arXiv 2402.16843 (UNVERIFIED id). **Training-free.**
  *LoRA Switch*: only one LoRA is active per step, alternating every τ steps. *LoRA Composite*: compute the CFG guidance with each LoRA separately and
  average the guided scores at every step. Composite costs N x NFE. It keeps identity well because weights are never summed.
- **LoRA-Composer**, arXiv 2403.11627 (UNVERIFIED id). Training-free multi-concept with layout boxes: concept injection constraints,
  concept isolation in cross-attention, latent re-initialization. SD-UNet based.
- **CLoRA**, Meral et al., arXiv 2403.19776 (UNVERIFIED id). Training-free: test-time latent optimization with a contrastive loss on attention maps,
  so each LoRA's tokens attend to their own region, then masked latent fusion.
- **LoRA.rar**, arXiv 2412.05148 (UNVERIFIED id). A hypernetwork predicts the merge coefficients. **Not training-free** (a pretrained hypernet for SDXL).
- **LoRA Soups**, Prabhakar et al., arXiv 2410.13025 (UNVERIFIED id). "CAT" is a concatenation with *learned* per-layer weights. **Training** (LLM skill composition).
- **Mixture of LoRA Experts (MoLE)**, arXiv 2404.13628 (UNVERIFIED id). A learned gating function. **Training.**
- **MultiLoRA** (arXiv 2311.11501, LLM, UNVERIFIED): horizontally scaled LoRA modules. **Training.**

### A11. Concept Sliders and training-free variants
- **Concept Sliders**, Gandikota et al., arXiv 2311.12092 (ECCV 2024). **Training.** A LoRA is trained so that
  ε_θ*(x_t, c_t, t) ≈ ε(x_t, c_t, t) + η Σ_p [ε(x_t, c_+, t) − ε(x_t, c_−, t)], with the preserve concepts p held fixed. Image sliders are also trained from
  paired images. Scaled at inference by multiplier. UNVERIFIED exact form.
- **Training-free equivalent at sampling time** (no LoRA): add a guidance term η[ε(c_t + c_+) − ε(c_t + c_−)] or its velocity equivalent.
  This costs 2 extra NFEs. For "AI look" you could use c_+ = "candid, natural expression, looking away, cluttered background", c_− = "posed, smiling at camera, clean backdrop".
- **Prompt Sliders**, arXiv 2409.16535. Learns a text-embedding token per concept (3KB). **Training** (a textual-inversion-like learned embedding, not a LoRA).
- **SliderSpace** (Gandikota et al., 2025, arXiv 2502.01639, UNVERIFIED id). Discovers directions by PCA over CLIP features of diverse samples, and each
  direction is distilled into a LoRA. The "no training" claim applies to supervision, but LoRAs are still trained (UNVERIFIED).
- **FreeSliders**, arXiv 2511.00103. **Training-free**, modality-agnostic (image/audio/video) [snippet]. Details UNVERIFIED.
- **Text Slider**, arXiv 2509.18831. Its training-free variant does text-embedding vector arithmetic [snippet].

### A12. Existing implementations
- **peft** `LoraModel.add_weighted_adapter(adapters, weights, adapter_name, combination_type, svd_rank, svd_clamp, svd_full_matrices, density, majority_sign_method)`
  with combination types: `svd, linear, cat, ties, ties_svd, dare_ties, dare_linear, dare_ties_svd, dare_linear_svd, magnitude_prune, magnitude_prune_svd`
  **[code-verified]**. `cat` = exact concatenation (rank = Σr), with weight·scaling folded into A. `*_svd` = dense ΔW then SVD. Non-svd types act on A/B separately (see pitfall A2).
- **kohya sd-scripts**: `networks/svd_merge_lora.py` (dense sum then SVD re-factorization, with `--new_rank` and clamp quantile), `merge_lora.py` (concat/merge into model),
  `extract_lora_from_models.py`, `resize_lora.py` (dynamic rank by sv_ratio / sv_fro / sv_cumulative). **musubi-tuner** has `merge_lora.py` and `lora_post_hoc_ema.py`.
- **ComfyUI**: core loaders only sum patches (`add_patches` with strength). Dense "diff" patches are supported. Community nodes for TIES/DARE merging exist,
  e.g. "ComfyUI-DARE-LoRA-Merge" / "LoRA Power-Merger" by larsupb (UNVERIFIED names; I did not inspect them). K-LoRA ComfyUI ports may exist (not verified).

### A13. LoRA file key formats in practice [code-verified unless marked]
**Kohya/sd-scripts style** (musubi-tuner default): `lora_unet_<module.path with . → _>.lora_down.weight`, `.lora_up.weight`, `.alpha` (a scalar tensor).
The scale is alpha/rank. Optional `.dora_scale`. LyCORIS types use `.hada_w1_a/…`, `.lokr_w1/…`.
**musubi-tuner saves alpha**: `register_buffer("alpha")`, with `--network_alpha` **default 1**. So a rank-32 musubi LoRA with default settings has
scale 1/32, and **the alpha must be honoured**.

**Qwen-Image / Qwen-Image-Edit, musubi-tuner** (`networks/lora_qwen_image.py`, prefix `lora_unet`, target class `QwenImageTransformerBlock`,
names relative to the DiT root):
```
lora_unet_transformer_blocks_{N}_attn_to_q / _to_k / _to_v / _to_out_0
lora_unet_transformer_blocks_{N}_attn_add_q_proj / _add_k_proj / _add_v_proj / _to_add_out
lora_unet_transformer_blocks_{N}_img_mlp_net_0_proj / _img_mlp_net_2
lora_unet_transformer_blocks_{N}_txt_mlp_net_0_proj / _txt_mlp_net_2
   + .lora_down.weight / .lora_up.weight / .alpha
```
N = 0..59. Default exclude regex `.*(_mod_).*` is fullmatched against the *dotted* name (e.g. `transformer_blocks.0.img_mod.1`). By my reading
it **does not match**, so `..._img_mod_1` and `..._txt_mod_1` (modulation linears) are probably also trained and saved (the regex was tested in python).
Verify on a real file (UNVERIFIED behaviour). ComfyUI maps `lora_unet_*` generically: `diffusion_model.X.weight` gives `lora_unet_X_with_underscores`.
Qwen ComfyUI keys are identical to the diffusers names, so this loads natively.

**Qwen-Image, ai-toolkit** (`extensions_built_in/diffusion_models/qwen_image/qwen_image.py`: `lora_keys_use_comfy_prefix = True`, is_transformer, which forces peft_format):
```
diffusion_model.transformer_blocks.{N}.attn.to_q.lora_A.weight   (lora_A = down, r x d_in)
diffusion_model.transformer_blocks.{N}.attn.to_q.lora_B.weight   (lora_B = up,  d_out x r)
```
**No alpha keys** (peft_format drops `.alpha`; internally alpha is forced to rank, so scale = 1; LoKr keeps alpha). Target = all Linear modules inside
`QwenImageTransformer2DModel`. UNVERIFIED which non-block linears (img_in, txt_in, proj_out, time embed, norm_out) are included or filtered.
ComfyUI QwenImage also accepts `transformer.` prefixed and bare `transformer_blocks.…` and `lycoris_…` keys, and `lora_B.default.weight` (the "qwen default" peft save).

**Z-Image (Turbo/Base), musubi-tuner** (`networks/lora_zimage.py`, target `ZImageTransformerBlock`, excludes `.*(_modulation|_refiner).*`, which
drops adaLN_modulation and all noise_refiner/context_refiner blocks). The musubi model uses split q/k/v:
```
lora_unet_layers_{N}_attention_to_q / _to_k / _to_v / _to_out_0
lora_unet_layers_{N}_feed_forward_w1 / _w2 / _w3
   + .lora_down.weight / .lora_up.weight / .alpha
```
**ComfyUI's Z-Image (Lumina2 class) uses a fused `attention.qkv` and `attention.out`.** ComfyUI's `z_image_to_diffusers` maps the
diffusers-style names (`layers.N.attention.to_q/to_k/to_v` become slices of `qkv`, `attention.to_out.0` becomes `attention.out`) under the prefixes
`diffusion_model.`, `transformer.`, `lycoris_` and bare. It does **not** register `lora_unet_…` for those diffusers names (only the generic
`lora_unet_layers_N_attention_qkv`). So raw musubi Z-Image files need conversion. musubi docs say to use
`convert_lora.py --target other`, which outputs `diffusion_model.layers.N.attention.to_q.lora_A.weight` etc. and *bakes alpha in* by scaling both
A and B by √(alpha/r) with no alpha key. The alternative, `networks/convert_z_image_lora_to_comfy.py`, keeps kohya style, renames
`attention_to_out_0 → attention_out` and `attention_norm_q/k → attention_q_norm/k_norm`, and fuses q/k/v into `lora_unet_layers_N_attention_qkv` with
block-sparse up (3·d x 3r), cat down (3r x d) and alpha·3.
**Z-Image, ai-toolkit**: `diffusion_model.layers.{N}.attention.to_q.lora_A.weight` / `.lora_B.weight` (also to_k, to_v, to_out.0,
feed_forward.w1/w2/w3), with no alpha (scale 1). `lora_keys_use_comfy_prefix = True`, and the assistant-LoRA loader reads
`diffusion_model.layers.0.attention.to_k.lora_A.weight`. The target is the whole `ZImageTransformer2DModel`, so noise_refiner/context_refiner layers
and adaLN linears are likely included too (UNVERIFIED).

**ComfyUI loader semantics** (`comfy/lora.py`, `comfy/weight_adapter/lora.py`): recognised up/down pairs are `.lora_up/.lora_down`, `.lora_B/.lora_A`,
`_lora.up/_lora.down`, `.lora.up/.lora.down`, `.lora_B/.lora_A` (no .weight, mochi), `.lora_linear_layer.up/down` and `.lora_B.default/.lora_A.default`.
Optional `.alpha`, `.dora_scale`, `.lora_mid`, `.reshape_weight`, `.diff`, `.diff_b`, `.w_norm/.b_norm`. The scale is alpha/rank if `.alpha` exists, else **1.0**.
Slice targets (`(key, (dim, offset, length))`) are used for fused qkv.
**A node that manipulates LoRAs should normalise everything to (B, A, scale) with the scale folded in**, then re-emit in one format.

---------------------------------------------------------------------------------------------------
## TOPIC B: Flow-model inversion for reference-attention (K/V capture) passes

Convention (FLUX/Qwen-Image/Z-Image rectified flow, as in ComfyUI CONST sampling): x_σ = (1−σ)·x_0 + σ·ε, σ ∈ [0,1] with σ = 1 as noise.
The model velocity is v ≈ ε − x_0. In ComfyUI, with denoised D = model(x, σ), v = (x − D)/σ.
Plain inverse Euler: x_{σ_{k+1}} = x_{σ_k} + (σ_{k+1} − σ_k) · v(x_{σ_k}, σ_k) for increasing σ. At σ = 0 v is undefined, so start from a small σ_min or use
the model's prediction at the first grid point (RF-Solver/FireFlow start their grid at t≈0 and query the model there).

### B1. RF-Inversion. Rout et al., arXiv 2410.10792 (ICLR 2025). **[code-verified: diffusers community `pipeline_flux_rf_inversion.py`]**
Paper time convention: t=0 is the image for inversion. Y_0 = y_0 (clean latent), and y_1 ~ N(0,I) is a *fixed* noise sample.
- Controlled forward ODE (Eq. 8): dY_t = [u_t(Y_t) + γ (u_t(Y_t | y_1) − u_t(Y_t))] dt, with u_t(Y_t | y_1) = (y_1 − Y_t)/(1 − t).
  u_t(Y_t) = the model's unconditional velocity toward noise (source prompt, usually "" and guidance ~1).
  Discrete: t_i = i/N, Y_{i+1} = Y_i + [u + γ(u_cond − u)]·(σ_i − σ_{i+1}) in the code's σ bookkeeping.
- Controlled reverse ODE (Eq. 15/17): X_0 = Y_1. v_t(X_t | y_0) = (y_0 − X_t)/(1 − t) (t now 0 = noise).
  v̂ = v_t(X_t) + η_t (v_t(X_t|y_0) − v_t(X_t)), with v_t = −(model output), and X ← X + v̂·Δσ.
  η_t = η for start ≤ i < stop, else 0 (optionally decayed by (1 − i/N)^power).
- γ is the forward controller: γ=0 is plain inverse Euler and γ=1 is a pure straight line to y_1. η trades faithfulness vs editability.
  Defaults from the diffusers example: N = 28, γ = 0.5, η = 0.9, start = 0, stop = 0.25. UNVERIFIED paper grid: η ∈ [0.9, 1.0], τ ∈ [0.2, 0.4].
- Cost: N NFEs for inversion (1 per step), plus editing.
- **Key observation**: with γ = 1 the inversion *ignores the model* and becomes Y_{t} ≈ (1−t)y_0 + t·y_1, a straight line to fixed noise.
  With η = 1 the reverse ODE also ignores the model and walks the straight line back to y_0. Up to the t_i = i/N vs shifted-σ mismatch in the
  diffusers implementation, this is **forward noising with a fixed ε**.

### B2. RF-Solver (RF-Edit). Wang et al., arXiv 2411.04746. **[code-verified: wangjiangshan0725/RF-Solver-Edit `sampling.denoise`]**
Second-order Taylor expansion of the RF ODE, with the time derivative estimated by finite difference:
- v = v(x_t, t); x_mid = x_t + (Δ/2) v with Δ = t_next − t; v_mid = v(x_mid, t + Δ/2);
  v' ≈ (v_mid − v)/(Δ/2); x_next = x_t + Δ·v + ½Δ²·v'.
- 2 NFEs per step. The same formula works for inversion (reversed timesteps) and sampling.
- RF-Edit feature sharing: during inversion it stores the self-attention **V** of single-stream blocks with id > 19, for the first
  `inject_step` steps, and **replaces V** with the stored one during editing. Default inject ~2–5 steps for FLUX (UNVERIFIED default).

### B3. FireFlow. Deng et al., arXiv 2412.07517. **[code-verified: HolmesShuan/FireFlow… `sampling.denoise_fireflow`]**
Midpoint method with velocity reuse:
- x_mid = x_t + (Δ/2)·v_prev; v_mid = v(x_mid, t + Δ/2); x_next = x_t + Δ·v_mid; then v_prev ← v_mid for the next step.
  Only the first step queries v(x_t,t). So there is **~1 NFE per step** with second-order-ish accuracy, and the paper claims O(Δ²) local error.
  UNVERIFIED claim wording.
- It uses the same V-injection editing as RF-Edit.

### B4. UniEdit-Flow (Uni-Inv / Uni-Edit). Jiao et al., arXiv 2504.13109. **[code-verified: DSL-Lab/UniEdit-Flow `sampling.denoise_uniinv`]**
Predictor-corrector inversion:
- x̃ = x_t + Δ·v_prev (predict); v_c = v(x̃, t_next) (velocity at the predicted *next* point); x_next = x_t + Δ·v_c (correct); v_prev ← v_c.
  That is ~1 NFE per step (a backward-Euler-like implicit step, well suited to making the forward sampler reproduce the trajectory).
  Reconstruction then uses the plain Euler sampler.
- Uni-Edit (editing): velocity fusion with `alpha` (fraction of steps; step_threshold = round(alpha·N)) and `omega` (edit guidance strength),
  plus an adaptive mask. Model-agnostic: FLUX, SD3, SDXL, Wan.

### B5. Tight Inversion. Kadosh et al., arXiv 2502.20376 (UNVERIFIED authors). UNVERIFIED details from memory:
- The insight is that the more precise the *condition*, the more accurate the inversion. It conditions the model on the **input image itself** (IP-Adapter for SDXL, FLUX Redux
  / image prompt for FLUX) during both inversion and reconstruction/editing, on top of the text prompt.
- It is orthogonal to the solver and composes with RF-Inversion, RF-Solver, etc. It improves reconstruction while keeping editability.
- For QIE (an edit model that natively conditions on the reference image) this is "free". Conditioning the inversion pass on the reference
  image should similarly tighten it (my inference).

### B6. What do reference-branch methods actually use? [code-verified]
- **Personalize Anything** (fenghora/personalize-anything, FLUX): the reference branch is batch element 0, run in parallel with generation.
  The Gradio demo calls `pipe.invert(..., gamma=1.0)` and then the reverse pass with `eta=1.0, start=0.0, stop=0.99`. Per B1, **both steps ignore the model**,
  so the reference latent at each step is ≈ (1−t)·x_0 + t·y_1 with a fixed y_1. **It is effectively forward noising with a fixed ε**, while its K/V are
  still computed by the transformer on that latent. Token replacement of the reference features into the generation branch applies while
  t > τ (τ from the UI, /100). After that it switches to concatenation.
- **KV-Edit** (Xilluill/KV-Edit, FLUX): plain first-order inverse Euler (`denoise_kv(..., inverse=True)`, `inversion_guidance` default 1.0).
  It caches the inversion latents per t (`info['feature'][t_img]`) and the K/V of image tokens per t and per block. During denoising the background
  tokens are re-blended from cached latents and cached background K/V are reused. The `re_init` option replaces the start latent in the edit region
  with z_0(1−t)+noise·t (forward noising).
- **Stable Flow** (snap-research/stable-flow, FLUX): inverse Euler for 50 steps with guidance 1 and **latent nudging** (clean latent x 1.15 before inversion).
  It stores the whole inverted trajectory. In generation the reference-branch latent is **overwritten every step** with the stored inversion latent
  (`latents[0] = inverted_latent_list[-i]`). Attention is injected only in "vital layers".
- **RF-Edit/FireFlow**: inversion-time V caching (B2/B3).

### B7. Is x_t = (1−t)x_0 + t·ε good enough for K/V capture?
- For **reference-attention capture** (you only need the reference's K/V at the matching noise level, not a reconstruction), forward
  noising is the common cheap choice. Personalize Anything effectively uses it (B6) with good identity results. The important details:
  (1) use **one fixed ε for all steps** (a coherent path, not fresh noise each step); (2) match the **same σ schedule/shift** as
  the generation branch; (3) at high σ the noised reference carries little structure, so K/V injected very early mostly transfer layout/colour.
  Most methods gate injection by timestep for this reason.
- **When inversion helps**: when you need the reference branch to be *on the model's own trajectory*. That matters for (i) exact background
  preservation (KV-Edit), (ii) structure-preserving edits at high σ, and (iii) distilled few-step models, where noised latents off-trajectory may produce
  atypical internal features. Cost = N extra NFEs once (first-order), ~N with FireFlow/Uni-Inv, 2N with RF-Solver. The trajectory can be cached and
  reused across seeds and prompts.
- **Middle ground** (suggestions, UNVERIFIED benefit): RF-Inversion with γ ∈ (0,1) blends model inversion with the straight line.
  Use the **same ε as the generation branch's initial noise** so the reference and generation branches start aligned. For QIE, run the inversion
  conditioned on the reference image (Tight-Inversion spirit).
- For Z-Image-Turbo (8-step distilled, CFG-free) inversion error per step is large, so forward noising with a fixed ε is likely the pragmatic default,
  with inversion as an option.

### Sources
- Code cloned: kohya-ss/musubi-tuner, ostris/ai-toolkit, huggingface/peft, comfyanonymous/ComfyUI, HVision-NKU/K-LoRA,
  diffusers `examples/community/pipeline_flux_rf_inversion.py`, wangjiangshan0725/RF-Solver-Edit, HolmesShuan/FireFlow-…, DSL-Lab/UniEdit-Flow,
  Xilluill/KV-Edit, snap-research/stable-flow, fenghora/personalize-anything, AntoAndGar/task_singular_vectors, danielm1405/iso-merging,
  gstoica27/KnOTS, yardenfren1996/B-LoRA (utils).
- Search snippets: arxiv.org/abs/2502.18461 (K-LoRA S = α·t_now/t_all + β, K = r_c·r_s, γ), 2412.00081 (TSV), 2410.19735 (KnOTS),
  2502.04959 (Iso), 2504.07448 (LoRI), 2403.14572 (B-LoRA), 2511.00103 (FreeSliders), 2409.16535 (Prompt Sliders), 2509.18831 (Text Slider),
  2602.15539, 2505.15875, 2603.26317.
