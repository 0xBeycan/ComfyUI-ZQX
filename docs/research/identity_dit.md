# Training-free identity / subject preservation in DiT: research notes for a Z-Image-Turbo / Qwen-Image(-Edit) ComfyUI pack

Compiled 2026-09-28.

**Sources and access.** arxiv.org, alphaxiv, huggingface.co, openreview, CVF and most project pages were blocked by the sandbox egress proxy. Only GitHub, through git clone and raw.githubusercontent, and web-search snippets were reachable. As a result:

* **Most formulas and hyperparameters below come from the official code.** The repos were shallow-cloned locally during research (not included in this repo), and file paths are given so you can re-check them. When code and paper disagree, the code wins, because that is what produced the published results.
* **Paper-only claims are marked UNVERIFIED.** These are statements found only in search snippets or secondary summaries.
* **Stable Flow is the one paper read in full.** Its full PDF ships in the project-page repo (`repos/stable-flow/static/paper/StableFlow.pdf`), and its text was extracted.
* **The target-model notes come from the local ComfyUI checkout** (`comfy/ldm/{lumina,qwen_image,flux}/model.py` at the commit recorded in docs/DESIGN.md).

---

## 0. Target-model facts you need for any K/V injection (read from ComfyUI source)

### Z-Image / Z-Image-Turbo (`comfy/ldm/lumina/model.py`, NextDiT, `dim == 3840` branch in `model_detection.py`)

**Architecture.**
* dim 3840, 30 heads, head_dim 128.
* RoPE axes_dims = (32, 48, 48), axes_lens = (1536, 512, 512).
* **rope_theta = 256.** This is very low compared with Flux and Qwen, which both use 10000. Every integer position step therefore produces a large rotation. Be careful with fractional or large offsets: an offset of +W in the w axis is an extreme shift.
* The block stack is 2 context_refiner layers (text only), then 2 noise_refiner layers (image only), then the n_layers joint single-stream layers.
* **UNVERIFIED:** the main stack is believed to have 30 layers. ComfyUI detects the count from the state dict.

**Position ids (3 axes: t, h, w).**
* Caption tokens get `t = 1 … L_cap` and h = w = 0. Captions are padded to a multiple of 32 when `pad_tokens_multiple` is set.
* Image tokens get `t = L_cap(+pad) + 1` and `h = 0 … H-1`, `w = 0 … W-1`. The grid is **not centred**; it starts at 0.
* Extra frames (`ref_frames`, the "Ming-Image" path) keep the same h/w grid and use `t = L_cap + 1 + F + f`. In other words, **a reference image is a new t index with the identical h/w grid.**
* In Omni/Edit mode (`ref_latents`), each reference image comes with its own caption segment, and the t offsets accumulate. Each reference is laid out as `[caption_i, siglip_i, image_i]`, with t advancing past each segment. SigLIP tokens get `t = cap_len + 2`, `h, w = linspace(0, 8h-1)`.
* **Implication.** The native way to add a reference in Z-Image is to give its tokens a distinct t coordinate, for example `t_img + 1`, with the same h and w. Injecting reference K with the target's own ids (t, h, w) means "same frame, same location". Section 6 compares the strategies.
* **Hook points.** `transformer_options["rope_options"]` supports `scale_x`, `scale_y`, `shift_x` and `shift_y` for the image grid. That is useful for OminiControl/EasyControl-style offsets or rescaling.

### Qwen-Image / Qwen-Image-Edit (`comfy/ldm/qwen_image/model.py`)

**Architecture.**
* 60 MMDiT blocks. The text and image streams have separate modulation, QKV and FFN weights, then share one joint attention.
* axes_dims_rope = (16, 56, 56), theta = 10000, head_dim 128, 24 heads (3072).

**Image ids.**
* `t` = frame index (0 for the generated image).
* `h = linspace(0, h_len-1) - h_len//2` and `w` likewise. The grid is **centred on 0**.

**Text ids.** `txt_start = max(h_len//2, w_len//2)`. Text token j gets id `txt_start + j` **on all three axes** (a diagonal, as in Qwen2-VL M-RoPE).

**Reference latents (Edit models).** There are three methods:
* `index` (the default for Edit 2509/2511): reference k gets frame index `t = k` (1, 2, …) with the same centred h/w grid.
* `index_timestep_zero`: the reference tokens are also modulated with a timestep = 0 embedding. `timestep_zero_index` splits the modulation.
* `offset`: t = 1, with h/w offsets used to pack multiple references.

The reference tokens are appended to the image stream after the noise tokens.

**Hook points.**
* A `post_input` patch can rewrite `img`, `txt`, `img_ids` and `txt_ids`.
* `transformer_options["reference_image_num_tokens"]` gives the size of each reference segment.

### Flux / Kontext (`comfy/ldm/flux/model.py`), for reference

Flux ids are (t, h, w) with h, w starting at 0. The reference methods are:
* `index`: the reference gets t = 1, 2, …
* `offset`: the Kontext default. t = 1, and h/w offsets are used only to pack several references.
* `uxo`: UNO-style. t = 0, and the reference is shifted diagonally by (H, W).

---

## 1. Personalize Anything for Free with Diffusion Transformer

* **arXiv:** 2503.12590 (2025; AAAI 2026).
* **Code:** github.com/fenghora/personalize-anything
* **ComfyUI:** github.com/smthemex/ComfyUI_Personalize_Anything (a wrapper-mode port).
* **Training-free:** yes. It uses FLUX.1-dev with RF-Inversion (Rout et al. 2410.10792).

### Mechanism, from `src/attn_processor.py` and `src/pipeline.py`

**Batch and inversion.**
* The batch is `[reference, generation]`.
* The reference is inverted with RF-Inversion: `pipe.invert(source_prompt="", gamma=1.0, num_inversion_steps=28)`.
* During generation, the reference branch is driven back along its trajectory by the controlled ODE `v̂ = v + η (v_cond − v)`, with `v_cond = (y0 − x_t)/(1 − t_i)`. η = 1.0 from step 0 to `stop_timestep = 0.99`, i.e. all steps. The reference row therefore reconstructs the reference image. The generation row uses an ordinary Euler step.

**Token replacement.** This runs in every attention layer (all 19 double and 38 single blocks) whenever `t > τ`, where t is the normalised flow time `timestep/1000` and runs from 1 down to 0:

```
concept = hidden_states[ref_row, mask]           # attention INPUT (post-adaLN-norm hidden states), subject tokens
hidden_states[gen_row, shift_mask] = concept     # overwrite generation tokens at (optionally shifted) positions
Q, K, V = to_q/k/v(hidden_states)                # all three recomputed from the replaced hidden states
RoPE: the generation row's ORIGINAL position ids (the tokens sit at shift_mask positions,
      so they get the positions of the location they were pasted into)
```

For single-stream blocks, the text part (the first 512 tokens) is skipped.

**Threshold.**
* Official notebooks use `tau = 60`, passed as `tau/100 = 0.6`, with 28 steps. Replacement therefore happens while normalised time is above 0.6. Because Flux's shifted schedule keeps t high for longer, this covers roughly the first ~30–40 % of steps. **UNVERIFIED** step count.
* The paper's reported ablation optimum is **τ = 0.8T** (CLIP-I 0.882, CLIP-T 0.302). **UNVERIFIED** (from a search snippet).
* A lower τ means more steps with replacement, giving stronger identity and more rigidity.

**Late stage.**
* The paper describes switching to "multi-modal attention" after τ. The reference subject tokens are concatenated as extra Q/K/V: `c_q`, `c_kv = concept_process`.
* In the code this path is behind `concept_process=False`, which is the default. When it is enabled, the concatenated concept tokens get the **same RoPE as the positions they occupy in `shift_mask`**. The code's commented-out alternative gives them "text" RoPE, i.e. identity rotation (cos = 1, sin = 0).
* The released notebooks therefore effectively do **only early-step token replacement, then free generation**.

**Position embeddings.**
* The paper states that DiT semantic features are largely separate from positional encodings, so "reference tokens (excluding positional information)" can be injected at new locations.
* In practice, replacing the hidden state before the Q/K projection and then applying the target position's RoPE means the token carries the reference's content but the target's location. The `shift_tensor(mask, x)` utility translates the mask by x latent columns to move the subject; notebook examples use x = ±8.

**Patch perturbation.** This is **not in the released code**. **UNVERIFIED**, from search snippets: random local token shuffling within 3×3 windows, plus morphological dilation/erosion of the mask with a ~5 px kernel, to reduce texture and structure over-fitting.

### Relevance and portability

**Identity versus composition.** This method produces very strong identity, because the actual tokens are copied. But it **copies pose and structure** unless τ is high and you move or perturb the mask. It is layout-locking during the replacement window, which is the opposite of what you want for natural composition, unless the mask is shifted, warped or perturbed, or the window is kept very short (1–2 steps of 8).

**Z-Image (single-stream).** The port is trivial: replace the image-token hidden states (after `attention_norm1` + modulation) in each layer for t > τ. With 8 steps, τ ≈ 0.8–0.9 means about 1–2 steps.

**Qwen-Image.** Only the image stream is touched. The reference branch needs inversion. With a distilled or Turbo model, RF-Inversion quality is doubtful. A cheaper alternative is to forward-noise the reference, as FreeCus and reference-only do.

---

## 2. FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers

* **arXiv:** 2507.15249 (ICCV 2025).
* **Code:** github.com/Monalissaa/FreeCus
* **ComfyUI:** none known.
* **Training-free:** yes. It runs on FLUX.1-dev and uses Qwen2-VL + Qwen2.5 for captioning and BiRefNet for segmentation.

### Mechanism, from `sa_handler_for_flux_clean.py` and `infer.py`

**1. Reference trajectory ("adjusted dynamic shifting").** This is not inversion.
* x0_ref is **forward-noised** with a fixed noise seed at every step.
* The noise levels come from the scheduler set with `mu = −mu` (`--mu_shift_type negative`), i.e. the dynamic shift is inverted, so the reference sits at *lower* noise than the target step.
* A callback overwrites the reference latent at each step with `scale_noise(x0_ref, t_i, noise)`.
* The reference pass is a separate full pipeline call. It stores K and V for every (step, layer).

**2. Pivotal attention sharing.**
* **Save pass:** K (post-norm, **post-RoPE, using the reference's own positions**) and V of the **foreground reference image tokens only** (BiRefNet mask, bilinearly downsampled to the latent grid, `> 0`). K is pre-multiplied by `scale = 1.1`.
* **Use pass, vital layers only:**
  ```
  K_txt  ← 1.1 · K_txt                               (target text keys also scaled)
  K = [ 1.1·K_ref_fg ; K_txt ; K_img ]      V = [ V_ref_fg ; V_txt ; V_img ]
  O = softmax(Q Kᵀ/√d) V
  ```
  Scaling K by s multiplies those logits by s, i.e. it acts as a temperature/boost on reference and text attention relative to the target image tokens.
* **Vital layers:** `[0, 1, 2, 17, 18, 25, 28, 53, 54, 56]`. These are indices into the 57-layer Flux list (0–18 double, 19–56 single) and are taken from Stable Flow.
* Sharing is applied at **all timesteps** (30 steps).

**3. MLLM caption.** Qwen2-VL describes the subject's attributes, and the text is appended to the prompt. This is optional and cheap for us.

**Positions.** The reference is also generated at 512×512, so its K carries the target grid positions of the same pixel locations. Nothing is offset. Pose freedom comes from sharing only in vital layers and only foreground tokens.

**Hyperparameters.** extend_scale 1.1, 30 steps, guidance 3.5, 512², seed 777.

### Relevance and portability

This is **the most directly portable "identity without copying layout" recipe**. It appends reference K/V (fg only) to a few layers, with a slight logit boost, and uses a forward-noised reference (no inversion, so it is Turbo-friendly).

For Z-Image and Qwen, the vital-layer set must be re-derived; see §4 for the Stable Flow procedure. Because the reference keys keep the reference image's own positions, a subject at the same location is favoured. For more pose and layout freedom, consider a separate t index (Z-Image) or frame index (Qwen) for the reference keys (§6).

---

## 3. CharaConsist: Fine-Grained Consistent Character Generation

* **arXiv:** 2507.11533 (ICCV 2025).
* **Code:** github.com/Murray-Wang/CharaConsist
* **ComfyUI:** none known.
* **Training-free:** yes. Built on FLUX.1-dev.

### Mechanism, from `models/attention_processor_characonsist.py` and `models/pipeline_characonsist.py`, defaults at 50 steps

**Three passes.**
1. **Identity image pass.** Stores the reference K/V (post-norm, **pre-RoPE**) for every step in `[attn_start_step=1, attn_end_step=41)`, and the attention output at step 10 (`save_mask_point_step`).
2. **Pre-run of the new frame up to step 10.** This produces:
   * **The foreground mask**: text-to-image attention, averaged over layers, of the "foreground" prompt tokens versus the "background" prompt tokens, `fg = bg_attn ≤ fg_attn`. The prompt is split into bg and fg parts, and the mask is cleaned with erode 3×3 then dilate 5×5.
   * **Point tracking**: cosine similarity between the current frame's attention outputs and the identity image's attention outputs at step 10, averaged over all single blocks. For each current token, `argmax` gives the matched identity token and `max_sim` its similarity. A correspondence is kept only if both tokens are foreground and `max_sim > sim_thr = 0.5`.
3. **Real run.** The steps are described below.

**Point-tracking attention**, in single-stream blocks only, where `block_ind % fg_share_freq == 0` with `fg_share_freq = 2`, i.e. every other single block (19 of 38):
```
K_ref_fg = K_id[:, id_fg_inds]            (pre-RoPE)
K_ref_fg = RoPE(K_ref_fg, pos = positions of the MATCHED CURRENT tokens curr_fg_inds)
K = [K_cur ; K_ref_fg],  V = [V_cur ; V_id[:, id_fg_inds]]
mask: current fg queries may attend to ref fg keys; current bg queries may attend to ref bg keys (when share_bg);
      text queries never see ref keys
```
This is the key RoPE trick. **Each reference token is re-positioned to the location of its semantic match in the new image.** The reference therefore reinforces identity wherever that body part now is, without dragging the layout.

**Adaptive token merge (`ada_tome`)**, applied on the attention output (before `to_out`) at matched foreground tokens:
```
out[curr] = (1 − α_t·s) · out[curr] + α_t·s · out_id[matched]
s = max_sim (per token)
α_t = 0.8 for steps [1, 11), then cosine decay 0.8 → 0 over [11, 31), 0 afterwards
```

**Background sharing** is optional (`share_bg`), with `bg_share_freq = 1`. Background reference tokens keep the **reference's own positions**.

### Relevance and portability

This is the best-documented recipe for "same character, new pose". The position re-mapping solves exactly our RoPE problem. The cost is a pre-run to step ~20 % to get correspondences. With 8-step Turbo, one could run the pre-run to step 2 and match there.

For Z-Image (single-stream, like Flux single blocks), port it directly: apply to every other layer. For Qwen, apply in the joint attention of image-stream tokens. The step 1–40 / 50 window and the α decay translate to roughly steps 0–6 / 8 and a merge on steps 0–4.

---

## 4. Stable Flow: Vital Layers for Training-Free Image Editing

* **arXiv:** 2411.14430 (CVPR 2025).
* **Code:** github.com/snap-research/stable-flow. The paper was read in full.
* **Training-free:** yes.

### Vital-layer detection (paper §3.1 and App. A.1)

For each layer ℓ, bypass the whole block (keep only the residual) and generate k = 64 prompts × seeds. The protocol was FLUX.1-dev, Euler, 15 steps, guidance 3.5. Then compute

```
vitality(ℓ) = 1 − (1/k) Σ_{s,p} d(M_full(s,p), M_−ℓ(s,p))
```

with d = DINOv2 perceptual similarity. The paper then defines `V = {ℓ : vitality(ℓ) ≥ τ_vit}`. In practice the layers with the lowest similarity are vital.

**Results.**
* **FLUX.1-dev vital layers: [0, 1, 2, 17, 18, 25, 28, 53, 54, 56].** Layer 2 was removable empirically. The code uses `MULTIMODAL_VITAL_LAYERS = [0, 1, 17, 18]` and `SINGLE = [25, 28, 53, 54, 56] − 19`, i.e. single-block indices [6, 9, 34, 35, 37].
* **SD3 vital layers: [0, 7, 8, 9].**
* **Similarity per layer** (from the appendix): [0.222, 0.041, 0.076, 0.08, 0.123, 0.101, 0.135, 0.124, 0.112, 0.105, 0.097, 0.12, 0.118, 0.086, 0.116, 0.067, 0.065, 0.116, 0.146, 0.065, …, 0.079 (53), 0.039, 0.037, 0.026]. The first vital layers' values are near 0 and were omitted from the plot, so the numbers are hard to interpret directly.
* **Qualitative effect of bypassing:** G0 gives noise, G18 changes global structure and identity, and G56 changes texture and fine detail.

### Injection mechanism (code, `attention_processor.py`)

Source and edit are generated in parallel. In vital layers only, **the image-token K and V of the edit are replaced with the source's**:
* single blocks: `key[1:, 512:] = key[:1, 512:]`
* double blocks: `key[1:] = key[:1]`

This happens before QK-norm and RoPE, so it uses the same positions. **Text K/V are untouched**, so text can still steer. It runs at **all timesteps**.

**Attention extension** (concatenating reference K/V instead of replacing) is ablated in the paper: in all layers it hurts text similarity, and in non-vital layers it hurts image similarity.

**Latent nudging** for real images: multiply the clean latent by λ = 1.15 before inverse-Euler inversion.

### Relevance

The vital-layer protocol is directly reusable: bypass-each-block + DINO on ~16–64 prompts. It is cheap for Z-Image's ~30 layers and Qwen's 60.

Replacing K/V (as opposed to extending) is too strong for composition changes, because it copies pose; it was designed for editing. For identity, prefer **extension (concatenation) in vital layers**, as FreeCus does.

---

## 5. FreeFlux: Layer-Specific Roles in RoPE-Based MMDiT (highly relevant to RoPE offsets)

* **arXiv:** 2503.16153 (ICCV 2025).
* **Code:** github.com/wtybest/FreeFlux
* **Training-free:** yes.

### Probing (`probing/`)

For each of FLUX's 57 blocks, modify **only the RoPE of K** (Q unchanged): (a) remove RoPE from K, or (b) shift K's positions. Then measure the PSNR of the output against the unmodified output.
* Low PSNR means the layer is **position-dependent**.
* High PSNR means it is **content-similarity-dependent**.

Layer 2 is the most position-dependent and layer 0 the most content-dependent. **UNVERIFIED** (snippet). The pattern does not correlate with depth.

### Layer sets used by the official editing notebooks

* **Position-dependent layers** (used for object addition / position-controlled injection): **[1, 2, 4, 26, 30, 54, 55]**.
* **Content-similarity-dependent layers** (used for non-rigid editing, i.e. pose change with identity kept): **[0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56]**, all 50 steps. The mechanism (`non_rigid_attn_utils.py`) replaces the target's image K/V (post-RoPE) with the source's in those layers. Because those layers match by content rather than by position, the target can re-arrange pose while pulling appearance.
* **Background replacement:** all layers, with masking.

### Relevance

This is **the key design principle for our injection**. Inject reference K/V only in *content-dependent* layers. In those layers, the reference's RoPE positions barely matter, so the positional offset is irrelevant and the layout is not copied. Avoid *position-dependent* layers, where the reference layout would imprint.

The FreeFlux probe is cheap to reproduce on Z-Image and Qwen-Image: for each layer, shift K's RoPE by e.g. +W/2 or zero it, then compute PSNR against baseline.

---

## 6. RoPE position assignment for reference tokens (OminiControl, EasyControl, UNO, Kontext, Qwen-Edit, Z-Image-Omni, IC-LoRA, Diptych, FreeGraftor, CharaConsist)

| Method | Model | How reference / condition tokens are positioned | Source |
|---|---|---|---|
| OminiControl (2411.15098) | Flux (trained LoRA) | Condition ids = normal grid **+ `position_delta`**. Subject (non-aligned) training uses `delta = (0, −cond_w/16)`, i.e. shifted **left** by the condition width (−32 for 512 px). The released 1024 subject model is used with `position_delta=(0, 32)`. Spatially-aligned tasks use delta (0, 0), i.e. shared positions. Optional `position_scale` s: `ids *= s; ids += (s−1)/2`. | `omini/pipeline/flux_omini.py:128-135`, `train_subject.py:92,110`, `examples/combine_with_style_lora.ipynb` |
| EasyControl (2503.07027) | Flux (trained LoRA) | **Subject**: condition at 512 px (32×32 tokens), `ids[:,1] += 64` (fixed **height-axis** offset of 64). **Spatial**: ids **interpolated** to the target grid, `arange(cond)*scale` with `scale = target/cond` ("position-aware interpolation"). | `src/pipeline.py:62-75, 461-466` |
| UNO (2504.02160, "UnoPE") | Flux (trained) | Reference k: `h_offset = H/2 + Σ prev ref heights`, `w_offset = W/2 + Σ prev ref widths` (pe='d', diagonal). The reference sits beyond the target's max (h, w), t = 0. | `uno/flux/sampling.py:137-155` |
| Flux Kontext (2506.15742) | Flux | Reference at t = 1, same h/w grid (ComfyUI "offset"/"index"). ComfyUI "uxo" reproduces UNO. | ComfyUI `ldm/flux/model.py:362-395` |
| Qwen-Image-Edit 2509/2511 | Qwen | Reference k at **frame index t = k**, same **centred** h/w grid. `index_timestep_zero` also uses t_emb = 0 for reference tokens. | ComfyUI `ldm/qwen_image/model.py:403-496` |
| Z-Image Omni / Edit, Ming-Image frames | Z-Image | Reference = its own caption segment followed by an image segment with **t = next free index**, same h/w grid starting at 0. | ComfyUI `ldm/lumina/model.py:676-800` |
| In-Context LoRA (2410.23775), Diptych (2411.15466), LatentUnfold (2504.11478) | Flux | **Spatial concatenation.** Reference and target form one canvas with ordinary contiguous positions (reference on the left, w ∈ [0, W); target on the right, w ∈ [W, 2W)). | IC-LoRA README, `DiptychPrompting/diptych_prompting_inference.py` |
| Personalize Anything | Flux | Reference content pasted into target tokens, so it takes the target position (optionally shifted mask). | §1 |
| FreeCus, Stable Flow, FreeFlux | Flux | Reference K keeps **its own grid position**, which is the same as the target's (same resolution). | §2, 4, 5 |
| CharaConsist, FreeGraftor | Flux | Reference K is **re-RoPE'd to the position of its semantically matched target token** (point tracking / cycle-consistent matching). | §3, §7 |

### Practical guidance for our injection

1. **Same positions** (FreeCus / Stable Flow).
   * The reference keys attend most strongly to queries at the same location. RoPE makes q·k larger for nearby positions in the position-dependent layers.
   * **Effect:** this biases the output toward the reference layout (centred, frontal, same framing), which is the AI-look we want to escape.
   * **When it's acceptable:** only in content-dependent layers.
2. **Separate frame / t index** (Kontext, Qwen-Edit, Z-Image-Omni).
   * The model was *trained* to treat t ≠ t_target as "another image": for Z-Image-Edit / Omni and Qwen-Image-Edit, t is natively trained. **For Z-Image-Turbo (T2I only) and Qwen-Image T2I, t-offset reference tokens are out of distribution.** **UNVERIFIED** how well they generalise, though Kontext-style behaviour often emerges zero-shot in Flux.
   * The h/w components still align spatially. The relative phase along h/w stays 0 for the same pixel location, so some layout pull remains, but less than with identical ids.
   * **Qwen:** the frame axis has only 16 of 128 dims, and the step per frame index is 1 with θ = 10000.
   * **Z-Image:** the t axis has 32 dims with θ = 256. A +1 t offset rotates the high-frequency pairs by ~1 rad, which is a strong separation. Text sits at small t, and the image t is about L_cap + 1.
3. **Spatial offset beyond the canvas** (UNO diagonal, OminiControl ±W, EasyControl +64 h).
   * This removes spatial alignment entirely: reference tokens act as "a different region" of a bigger canvas, like a diptych.
   * For a T2I model, a w-offset of exactly W reproduces the **diptych prior**, which Flux has, and which Z-Image and Qwen very likely have, since they generate collages and diptychs. **UNVERIFIED** for Z-Image and Qwen.
   * **Caution with Z-Image θ = 256 on 48 dims:** large offsets are fine because RoPE is relative, but non-integer offsets land on untrained phases.
   * **Caution with Qwen's centred grid:** offset by `w_len` (not w_len/2) so the ranges don't overlap.
4. **Matched positions** (CharaConsist / FreeGraftor). Most natural for pose freedom, at the cost of needing correspondences from an early pre-run.

---

## 7. FreeGraftor: Training-Free Cross-Image Feature Grafting

* **arXiv:** 2504.15958 (2025).
* **Code:** github.com/Nihukat/FreeGraftor
* **Training-free:** yes, on FLUX.1-dev.

### Pipeline

1. **Collage construction.** Paste the segmented reference subject into a generated template scene (Grounding + SAM, then scale/position/flip via `configs/*.json`).
2. **Inversion** of the collage.
3. **Generation** with the reference branch run in parallel. The reference branch has its own text and trajectory and runs through every block.

### Per block (double blocks, code `src/flux/modules/layers.py:153-245`)

```
match features = modulated block input  (img_norm1 → (1+scale)·x + shift)   for ref and target
src2tar = argmax cosine(ref_tok, tar_tok); keep if sim > sim_threshold (0.2)
          and cycle-consistent (|flow_src→tar + flow_tar→src| < cyc_threshold = 1.5 px in token units)
ref_ids = grid_sample(target img_ids, src2tar)     # each ref token takes the (h,w) id of its matched target token
ref_pe  = RoPE(ref_ids)
ref_mask = subject_mask · match_mask; dropout with p = t · 0.2  (t = current flow time, i.e. more dropout early)
K = [K_txt ; K_img ; K_ref[mask]],  V likewise,  PE = [pe_txt ; pe_img ; ref_pe[mask]]
```

### Defaults (Gradio)

25 steps, guidance 3.0, inject steps 0–25 (all), inject blocks 0–56 (all), sim 0.2, cycle 1.5, match dropout 0.2.

### Relevance

Same idea as CharaConsist: position-constrained (matched) RoPE. The matching is recomputed **per block, per step**, using the current target features. No pre-run is needed, but the cost is O(N²) similarity per block (chunked).

For identity with free pose, it is a strong candidate. The main risk is that at 8 steps the early target features are noisy, so matching is poor. The time-proportional dropout mitigates that.

---

## 8. DiTCtrl: Attention Control in MM-DiT for multi-prompt video

* **arXiv:** 2412.18597 (CVPR 2025).
* **Code:** github.com/TencentARC/DiTCtrl
* **Model:** CogVideoX-2B (30 layers, 3D full attention).
* **Training-free:** yes.

### Mechanism (`sat/dit_video_concat.py:1785-1870`)

**Mask-guided KV sharing** (MasaCtrl ported to MM-DiT):
* **Target queries attend to the source segment's K/V**: `out_t = Attn(Q_t, K_s, V_s)`.
* **Masks:**
  * Foreground masks are computed from **text→video attention maps** of the subject token indices. Maps are aggregated and binarised at `thres = 0.1`.
  * The source mask is used as a self-attention mask.
  * The target mask blends: `out = out_fg·M + out_bg·(1−M)`, where out_bg comes from ordinary self-attention and out_fg from KV sharing. The text rows of M are 1 only at the subject tokens.
* **Schedule:** steps `start_step = 2 … end_step = 50` and layers `start_layer = 5 … end_layer = 30` (defaults in code; **UNVERIFIED** as the paper's final choice).
* **Latent blending** between overlapping segments smooths transitions.

**Paper claim:** 3D full attention in MM-DiT behaves like UNet cross- and self-attention, so the text→image block of the attention map gives usable masks. **UNVERIFIED** wording.

### Relevance

The fg-mask-from-text-attention trick is useful for restricting identity injection to the character: no segmenter needed, and it works in joint attention. KV *replacement* copies motion and layout, so use it sparingly.

---

## 9. KV-Edit: Training-Free Image Editing for Precise Background Preservation

* **arXiv:** 2502.17363 (2025).
* **Code:** github.com/Xilluill/KV-Edit
* **Training-free:** yes, on Flux.

### Mechanism (`flux/modules/layers.py`, `models/kv_edit.py`)

1. **Inversion pass.** Invert the image (28 steps, guidance 1.5) and cache the image K/V (post-norm, pre-RoPE) for every (t, block).
2. **Denoising.** **Only the masked (foreground) tokens are carried as queries**, with `pe_q = RoPE(mask positions)`. Keys are `[K_txt ; K_bg_cached ∪ K_fg_current]`: the cached background K/V is overwritten at the mask indices with the current foreground K/V. RoPE uses full-image positions for keys.
3. **`attn_scale`** (default 1): add `+scale` to the logits of the background keys, to improve blending with large masks.
4. **Options:** `skip_step` (default 4) skips the first denoising steps. `re_init` starts from blended noise instead of the inverted latent. `attn_mask` optionally blocks fg↔bg attention in inversion.

### Relevance

KV-Edit is not identity transfer, but the **KV cache with query subset** pattern is exactly how to implement "reference K/V computed once, reused at every step". With forward-noised references you need one reference forward per step. Alternatively, cache a single low-noise reference K/V and reuse it at every step, which is cheap and is what ComfyUI "reference" nodes typically do. **UNVERIFIED** quality.

---

## 10. ConsiStory: Training-Free Consistent Text-to-Image Generation

* **arXiv:** 2402.03286 (SIGGRAPH 2024, NVIDIA).
* **Code:** github.com/NVlabs/consistory
* **Model:** SDXL.
* **Training-free:** yes.

### Mechanism (code defaults, 50 steps)

**Subject-Driven Self-Attention (SDSA).**
* Applies in **decoder ("up") self-attention layers only**, at steps 1–50.
* Each image's self-attention keys and values are extended with the **subject-masked** tokens of the other images in the batch (the anchor images).
* Subject masks come from the cross-attention of the subject token, averaged over the last 20 stored maps, at 32² and 64².
* **Mask dropout 0.5:** each subject patch of the other images is dropped with p = 0.5 (fixed seed). The image's own tokens are always kept. This is the anti-layout-copy mechanism: it encourages pose diversity.

**Query "vanilla sharing".**
* For steps 0–5 (`t_range = [0, n_steps//10]`), queries are blended with the queries of a vanilla (no-SDSA) run: `q = s·q_vanilla + (1−s)·q`.
* s is linear from 0.9 down to 0.818 (`strength_start 0.9`, `strength_end 0.81836735`).
* **Purpose:** keep the layout diversity of normal generation.

**Feature injection (DIFT correspondences).**
* For steps 5–16 (`n_steps//10 … n_steps//3`), in 'up' and 'down' layers, the self-attention *output* at subject patches is blended toward the matched anchor's output with α = 0.8: `out = 0.8·out_anchor[nn] + 0.2·out`.
* Correspondences come from DIFT features, `swap_strategy 'min'`, `dist_thr 'dynamic'`.

**Anchors.** The first n images are anchors. Others attend only to anchors (`create_anchor_mapping`).

### Relevance

Two tricks transfer directly:
* **Random dropout of shared reference tokens (p ≈ 0.5)**, to avoid copying pose.
* **Blending queries with a vanilla run early**, to preserve the prompt's natural composition. This is an explicit anti-"same pose" device and maps cleanly to DiT. For Turbo, that means step 0–1 of 8.

---

## 11. StoryDiffusion: Consistent Self-Attention

* **arXiv:** 2405.01434 (NeurIPS 2024).
* **Code:** github.com/HVision-NKU/StoryDiffusion
* **ComfyUI:** github.com/smthemex/ComfyUI_StoryDiffusion
* **Model:** SDXL.
* **Training-free:** yes. Consistent Self-Attention (CSA) is training-free; the video motion predictor is trained.

### Mechanism (`utils/gradio_utils.py:241-267`, `gradio_app_sdxl_specific_id_low_vram.py:140-240`)

* **Token sampling.** For each image, randomly sample tokens of the *other* images in the batch at **sampling rate 0.5** (`sa32 = sa64 = 0.5`) and concatenate them to K/V: `K = [K_self ; K_others_sampled]`.
* **Layers:** up-block self-attention only.
* **Schedule:** step 0 uses no CSA. For steps under 20, CSA is applied with probability 0.7; from step 20 on, with probability 0.9 (`random_number > rand_num` with rand_num 0.3 / 0.1).
* **Id bank.** A "write" pass stores the id images' sampled tokens per step. A "read" pass reuses them for new frames.

### Relevance

Like ConsiStory, but with random, not masked, token subsampling. This is the simplest possible "shared attention" baseline, and it **also shares background style**, which is bad for our goal.

---

## 12. 1Prompt1Story (One-Prompt-One-Story)

* **arXiv:** 2501.13554 (ICLR 2025 spotlight).
* **Code:** github.com/byliutao/1Prompt1Story
* **Model:** SDXL.
* **Training-free:** yes.

### Mechanism (`unet/utils.py`, `unet/unet_controller.py`)

**Prompt consolidation.** All frame prompts are concatenated into one long prompt: id prompt + all frame descriptions. For frame i, the other frames' descriptions are suppressed.

**Singular-Value Reweighting (SVR)** on text embeddings:
* Take the token embeddings X (columns = tokens) of the **express** set (frame i's words + EOT) or the **suppress** set (other frames' words + EOT).
* Compute the SVD `X = U diag(σ) Vᵀ`.
* Reweight `σ' = β · σ · exp(−α σ)`, then reconstruct X' = U diag(σ') Vᵀ.
* **Suppress:** α = 0.01, β = 0.05 (ranges 0.01–0.5 and 0.05–1.0).
* **Enhance:** α = −0.01, β = 1.0 (ranges −0.001 to −0.02 and 1.0–2.0).

**Identity-Preserving Cross-Attention (IPCA).**
* In the conditional branch only, K⁺ and V⁺ are the concatenation over the batch of all frames' **text** keys and values: `K⁺ = [K1 ⊕ … ⊕ KN]`.
* Masks keep only the id-prompt tokens from other frames.
* Applied in all UNet positions, from step 0, dropout 0.

### Relevance

Text-side identity consistency. For Qwen/Z-Image, SVR on the LLM text hidden states could be used to *suppress* composition-cliché tokens ("smiling", "looking at camera") that a character LoRA's trigger drags in. This is speculative: **UNVERIFIED** for LLM encoders.

---

## 13. StyleAligned: Style Aligned Image Generation via Shared Attention

* **arXiv:** 2312.02133 (CVPR 2024, Google).
* **Code:** github.com/google/style-aligned
* **ComfyUI:** github.com/brianfitzgerald/style_aligned_comfy
* **Training-free:** yes.

### Exact mechanism (`sa_handler.py`)

**AdaIN on queries and keys**, per head, per channel, over the token dimension (dim −2):
```
μ(x), σ(x) = mean / sqrt(var + 1e-5) over tokens
AdaIN(Q_t) = (Q_t − μ(Q_t)) / σ(Q_t) · σ(Q_ref) + μ(Q_ref)
AdaIN(K_t) likewise;  V untouched (adain_values=False)
```
The reference is image 0 of each CFG half; the reference itself is left unchanged.

**Shared attention:**
```
K = [K̂_t ; s·K_ref],  V = [V_t ; V_ref]
logits = Q̂_t Kᵀ/√d;  logits[:, :, :, N_t:] += shared_score_shift
```
* `shared_score_scale` s (default 1) multiplies the reference keys.
* `shared_score_shift` defaults to 0. The transfer notebook uses **ln 2**, i.e. doubling the unnormalised attention mass on the reference. Higher means more fidelity.

**Layers.** All self-attention layers of SDXL by default (`only_self_level = 0`). `only_self_level` ∈ (0, 1) disables sharing in an evenly spaced fraction of layers.

**Shared norms.** GroupNorm and LayerNorm statistics are computed over [target ; reference] concatenated (`share_group_norm/share_layer_norm`). This is optional; turn it off for real-image transfer.

### Relevance

This is **style**, not identity. AdaIN-on-QK aligns global statistics (palette, lighting). For identity plus natural photos it is mostly harmful: it copies the reference's photographic look.

But the `shared_score_shift = ln λ` logit-bias formulation is the cleanest knob for **reference strength** in K/V concatenation. Use `+ln(w)` on the reference-key logits.

---

## 14. RB-Modulation: Reference-Based Modulation

* **arXiv:** 2405.17401 (ICLR 2025, Google).
* **Code:** github.com/google/RB-Modulation
* **Model:** StableCascade.
* **Training-free:** yes, but it uses a style descriptor (CSD) and optimisation at inference.

### Mechanism

**Attention Feature Aggregation (AFA)** (`utils.py:305-390`). This is cross-attention with KV = [text ; content-image CLIP ; style-image CLIP]. Instead of one attention, it **averages attention outputs computed with different KV subsets**:
```
x = ( Attn(Q, K_txt) + Attn(Q, K_txt+sub) + Attn(Q, K_txt+sub+style_mean) + Attn(Q, K_txt+style_mean) ) / 4
```
In style-only mode the average is `(x_txt + x_txt_style + x_style)/3`. Here the style tokens are replaced by their mean token (`style_mean`) so that reference *content* does not leak.

**Stochastic optimal controller.** At each step, optimise the predicted x0 or latent to minimise a terminal cost ‖CSD(x̂0) − CSD(ref)‖. **UNVERIFIED** step counts and learning rate (paper-only).

### Relevance

**"Average of attention outputs with and without reference KV"** is a good way to keep text dominance:

`O = (1−w)·Attn(Q, [K_txt;K_img]) + w·Attn(Q, [K_txt;K_img;K_ref])`

It is linear in w, it is stable, and it prevents the reference from swamping the prompt's composition. It costs 2× attention in the injected layers.

---

## 15. Reference-only (ControlNet "reference_only", sd-webui-controlnet)

* **Code:** Mikubill/sd-webui-controlnet `scripts/hook.py` (lines 745-980).
* **ComfyUI:** comfyanonymous/ComfyUI_experiments (the `reference_only` node).
* **Training-free:** yes.

### Mechanism

**Reference trajectory.** The reference latent is **forward-noised to the current timestep** (`q_sample(ref, t)`, no inversion). A write pass stores the normalised input `x_norm1` of each BasicTransformerBlock self-attention.

**Read pass (attention concatenation):**
```
attn_uc = attn1(x_norm1, context = [x_norm1 ; bank])        # self-attn keys/values extended with the ref tokens
attn_c  = attn_uc,  except on the uncond rows: attn1(x_norm1, context = x_norm1) (no reference)
out     = style_fidelity · attn_c + (1 − style_fidelity) · attn_uc
```
Since `attn_c` differs from `attn_uc` only on the uncond rows, **style_fidelity controls how much the *uncond* branch also sees the reference**. The effect is CFG-like amplification of the reference:
* style_fidelity = 1: uncond has no reference, so CFG pushes toward the reference.
* style_fidelity = 0: both branches see it.

The default is 0.5. On SDXL it is cubed (0.5³ = 0.125) "because SDXL attention hacking is unstable".

**Layer selection.** Blocks are sorted by channel width, descending, and `attn_weight = i/N`. A block is used only if `control_weight > attn_weight`. A weight of 1 means all layers; a weight of 0.5 means only the wider (higher-resolution) half.

**`reference_adain`.** Replaces GroupNorm mean/var with the reference's, blended by the same style_fidelity rule, in the mid block and selected up/down blocks.

### Relevance

The "forward-noise the reference at the current t and concatenate its tokens" pattern is **the Turbo-friendly default**: no inversion and one extra forward per step, or a batch of 2. Under CFG = 1 (Z-Image-Turbo), the style_fidelity trick is unavailable. The analogue is to blend outputs with and without the reference (AFA, §14).

---

## 16. Visual Style Prompting (VSP)

* **arXiv:** 2402.12974 (2024, NAVER).
* **Code:** github.com/naver-ai/Visual-Style-Prompting
* **Model:** SDXL.
* **Training-free:** yes.

### Mechanism (`pipelines/inverted_ve_pipeline.py:440-500`, `config/default.json`)

**K/V swap.** In activated self-attention layers, **the target's keys and values are replaced by the reference's**, while the target's queries are kept:
`Attn(Q_t, K_ref, V_ref)`. The reference is image 0 of each CFG chunk.

**Default activation:**
* **Layers:** `activate_layer_indices_list = [[[0,0],[128,140]]]`. These index SDXL's attention modules, of which there are 140, so layers 128–140 are the **last decoder (up) block's self-attentions**.
* **Steps:** `activate_step_indices_list = [[[0, 49]]]`, i.e. all 50 steps.

**Paper claim:** swapping K/V in *late* up-block self-attention transfers style without content leakage; early layers leak content and layout. **UNVERIFIED** wording.

### Relevance

The late-layer = texture/style analogue in DiT is roughly the last few blocks: Flux 53–56, which Stable Flow found affect texture and fine detail. For identity we want mid-level appearance, not late texture.

---

## 17. InstantStyle, plus block roles in SDXL and Flux

* **arXiv:** 2404.02733 (InstantStyle); 2407.00788 (InstantStyle-Plus).
* **Code:** github.com/InstantStyle/InstantStyle

**SDXL findings (README).** For IP-Adapter:
* `up_blocks.0.attentions.1` carries **style** (colour, material, atmosphere).
* `down_blocks.2.attentions.1` carries **spatial layout** (structure, composition).
* **Style only:** inject into `["up_blocks.0.attentions.1"]`.
* **Style + layout:** inject into `["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"]`.
* **Feature subtraction** of CLIP content text features from the image features is used to decouple style from content.

### Flux / MMDiT block-role findings (collected)

* **Stable Flow (2411.14430)** lists the vital layers above. Bypassing layer 0 gives noise, layer 18 changes global structure and identity, and layer 56 changes texture.
* **FreeFlux (2503.16153)** lists the position-dependent layers [1, 2, 4, 26, 30, 54, 55] and the content-dependent layers [0, 7, 8, 9, 10, 18, 25, 28, 37, 42, 45, 50, 56].
* **"Demystifying Flux Architecture" (2507.09595)** and style works such as SplitFlux (2511.15258): early single-stream blocks (~B20–29) affect content and structure; later single-stream blocks (~B30–57) affect style and high-level attributes. **UNVERIFIED** (search-snippet level).
* **"Unraveling MMDiT Blocks" (2601.02211):**
  * analysed on SD3.5 / FLUX / Qwen-Image by removing, disabling and enhancing text hidden states per block;
  * semantics are decided in early blocks and fine details in later blocks;
  * removing a block is usually less disruptive than disabling its text condition;
  * enhancing text in selected blocks improves attributes.

  **UNVERIFIED** block indices.
* **AttnRouter (2605.01480), on Qwen-Image-Edit-2511 (60 blocks).** Its "KVInject" α-blends the source-image K/V into the noise-half K/V: `K_noise ← (1−α)K_noise + α K_src`. The editing-effective sub-circuit is **layers L30–45, steps S0–7**, with α 0.3–0.5. **UNVERIFIED** (snippet only). This is the only Qwen-specific layer localisation found.
* **HeadRouter (2411.15034):** MM-DiT attention heads differ in sensitivity to editing semantics. It routes text guidance per head adaptively. **UNVERIFIED** details.
* **B4M, and a layer-role paper for "ArtiDiffuser":** not found. No evidence was found of papers under those names.

---

## 18. Diptych Prompting, In-Context LoRA, LatentUnfold ("Flux Already Knows")

### Diptych Prompting

* **arXiv:** 2411.15466 (CVPR 2025).
* **Code:** github.com/chaehunshin/DiptychPrompting
* **Model:** FLUX.1-dev with `alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta`, `ctrl_scale 0.95`, 30 steps, guidance 3.5, true_cfg 3.5.

**Setup.**
* Canvas = [reference with background removed (Grounded-SAM) | masked right half]. Each panel is 768 px plus 8 px padding.
* Prompt: `"A diptych with two side-by-side images of same {X}. On the left, a photo of {X}. On the right, replicate this {X} exactly but as {target}"`.

**Reference attention enhancement** (`CustomFluxAttnProcessor2_0`), applied in **all 57 layers and all steps**:
```
P = softmax(QKᵀ/√d)        # full joint attention probs
P[img_right_queries, img_left_keys] *= λ      (λ = attn_enforce = 1.3), NO renormalisation
O = P V
```

### In-Context LoRA

* **arXiv:** 2410.23775 (trained LoRA).
* **Finding:** Flux can already produce consistent multi-panel images when prompted for a panel layout. The LoRA only sharpens this. Condition and target are **one concatenated image with contiguous positions**.

### LatentUnfold ("Flux Already Knows")

* **arXiv:** 2504.11478.
* **Code:** github.com/bytedance/LatentUnfold
* **Mosaic completion.** The subject is replicated in a 3×3 grid of 512 px cells, and the target cell is inpainted.
* **Cascade attention:** extra attention computed on pooled (s×s, s ∈ {2, 3}) Q/K, added with `aug_att ∈ {0, 0.02, 0.05}` for the first 14 of 28 steps.
* **Meta prompting.** **UNVERIFIED** details.

### Relevance

The diptych prior is the zero-shot fallback. For Z-Image and Qwen, the equivalent is to place the reference latent at w-offset = W (a new canvas region) and describe it in the prompt. The λ multiplication (post-softmax, un-normalised) is a simple strength knob. Section 13's logit shift, ln λ, is the normalised equivalent.

---

## 19. GRAG: Group Relative Attention Guidance for Image Editing (Qwen-Image-Edit, training-free)

* **arXiv:** 2510.24657 (2025).
* **Code:** github.com/little-misfit/GRAG-Image-Editing
* **ComfyUI:** github.com/smthemex/ComfyUI_GRAG_Image_Editing and github.com/amir84ferdos/ComfyUI-GRAG-ArchAi3D

### Observation

In MM-attention, the Q and K tokens of a group share a large layer-dependent **bias vector** (the group mean). The token-wise **delta** from that mean carries the content-specific signal.

### Formula (`Qwen-Edit-GRAG/hacked_models/models.py:329-341`)

This is applied after QK-norm and RoPE, in all 60 blocks and at all steps:
```
μ_cond = mean over the reference-image key tokens (last img_len = 4096 image-stream tokens)
K_cond ← b · μ_cond + δ · (K_cond − μ_cond)          (cond_b, cond_delta)
text keys: same form with (1.0, 1.0) by default (i.e. untouched)
```
* **Defaults:** b = δ = 1 (no-op).
* **README:** the recommended range is **0.8–1.7**, in 0.01 steps. δ > 1 sharpens the attention contrast among reference tokens, meaning a stronger pull toward the input image. **UNVERIFIED** direction semantics; check empirically.

### Relevance

GRAG is **directly usable in QIE today** as an "identity strength vs. edit freedom" dial on the reference-image tokens. It is also applicable to our own injected reference K in Z-Image: centre, then scale the delta.

---

## 20. Other 2025–2026 training-free subject / identity methods found (brief; mostly UNVERIFIED beyond abstracts)

**LoRAShop** (2505.23758)
* Training-free multi-concept generation with **LoRAs** on Flux.
* Observation: concept-specific transformer features activate spatially coherent regions **early** in denoising.
* Method: a prior forward pass extracts a disentangled latent mask per concept, then each LoRA's contribution is blended only inside its mask.
* **Relevance:** if identity comes from a *character LoRA*, apply the LoRA only in the character region (and/or only in some layers/steps). This is the natural way to stop the LoRA from imposing its training-set backgrounds, bokeh and framing.

**FreeLoRA** (2507.01792): training-free LoRA fusion for multi-subject personalisation, with autoregressive models (per the title).

**FreeFuse** (2510.23515): multi-subject LoRA fusion via adaptive token-level routing at test time.

**AnyMS** (2512.23537): layout-guided, training-free multi-subject customisation using bottom-up attention decoupling.

**Multi-party Collaborative Attention Control** (2505.01428): image customisation via attention control.

**StorySync** (2508.03735)
* Masked cross-image attention sharing plus "Regional Feature Harmonization".
* Training-free.

**Infinite-Story** (2511.13002): training-free consistent T2I (scale-wise autoregressive, per the snippet).

**ASemConsist** (2512.23245): adaptive semantic feature control for identity-consistent generation. **UNVERIFIED** base model.

**SpatialID / "Inject Where It Matters"** (2602.13994)
* Spatially-adaptive modulation on top of a (trained) ID adapter.
* Uses cross-attention-derived face masks.
* Time schedule: Gaussian prior masks, then attention masks, then relaxation.
* Aims to stop identity features contaminating background and lighting.
* Training-free only as a wrapper.

**FlexID** (2602.07554): training-free flexible identity injection via intent-aware modulation. Also wraps ID adapters, **UNVERIFIED**.

**Qwen / Z-Image specific:** no dedicated training-free identity paper was found for Qwen-Image or Z-Image specifically, other than GRAG (Qwen-Edit) and AttnRouter/KVInject (Qwen-Edit-2511). Z-Image's own report (2511.22699) describes an Omni/Edit continued-training path (reference tokens with separate t index), but no training-free personalisation.

---

## 21. Synthesis: a recommended injection design for Z-Image-Turbo and Qwen-Image(-Edit)

The goal is to keep identity while getting natural composition (off-centre, candid, non-frontal, cluttered or real backgrounds).

### 1. Reference branch

Forward-noise the reference, as reference-only and FreeCus do: `x_ref,t = (1−σ')·x0 + σ'·ε`, using a fixed ε. Optionally use a slightly lower σ' than the target, per FreeCus's negative-μ shift. For example, σ'(t) = σ(t) mapped through `shift(σ, 1/s)`.

**Batch** it with the target (B = 2), so K/V is available in the same forward. No inversion is needed, which is critical for 8-step distilled models.

### 2. Tokens

Use only character tokens. Take them from a segmenter mask, or from the text→image attention of the character tokens (the DiTCtrl / CharaConsist / ConsiStory route: threshold ~0.1, or bg ≤ fg attention). **Drop the reference background entirely**, which removes clean-bokeh leakage.

### 3. Layers

Inject **only in content-similarity-dependent / vital layers**. Derive them per model with:
* **(a)** the Stable Flow bypass test (DINOv2 similarity when skipping block ℓ), and
* **(b)** the FreeFlux probe (PSNR when K's RoPE is removed or shifted in block ℓ).

Pick layers that are vital but not position-dependent. Avoid position-dependent layers, which imprint the reference layout.

For Qwen-Image-Edit-2511, the only known localisation is L30–45 (AttnRouter, UNVERIFIED).

### 4. Positions (see §6)

* **Default:** give reference keys a **separate t / frame index** (Z-Image: `t_img + 1`; Qwen: frame 1, which is native for the Edit models).
* **Alternative:** a **w-offset of +W** (diptych prior).
* **Best, at the cost of an early pre-run:** **matched positions** (CharaConsist / FreeGraftor), via argmax cosine of attention outputs or block inputs, sim > 0.2–0.5, cycle-consistency. Reference tokens then follow the new pose.

### 5. Strength

Pick one of:
* a logit bias `+ln w` on the reference keys (StyleAligned `shared_score_shift`), with w ≈ 1.5–2;
* FreeCus key scaling (1.1);
* AFA-style output blending `O = (1−w)·O_noref + w·O_ref`.

Optionally add GRAG delta scaling on the reference keys.

### 6. Anti-layout-copy devices

* ConsiStory's **random dropout of reference tokens, p ≈ 0.5**, or FreeGraftor's `p = 0.2·t`.
* A **vanilla-query blend in the first step(s)**: `q = 0.9·q_vanilla + 0.1·q` for step 0 of 8.
* **Start injection after step 0–1**, since the layout is decided at the first steps in flow models. This is the reverse of Personalize Anything, which injects *early* to lock identity *and* layout.

### 7. Timesteps

For identity without layout, skip the first 1–2 of 8 steps (layout formation), inject in the middle steps, and taper toward the end. That mirrors CharaConsist's merge schedule (constant, then cosine decay to 0 at ~60 % of steps).

### 8. Character LoRA

Following LoRAShop, restrict the LoRA delta to the character mask and/or to the content layers only. This keeps the LoRA's learned "portrait composition" out of the background and framing.
