# ComfyUI-ZIT-QIE-Experimental

An experimental, **training-free** ComfyUI node pack for **Z-Image Turbo**, **Qwen-Image 2512** and
**Qwen-Image-Edit 2511**.  Goal: **keep the identity** that comes from a character LoRA and/or a passport-style
reference while breaking the scene-level **"AI look"** (posed stance, looking at the camera, perfect smile,
clean/bokeh background, centred framing), and let **character and realism LoRAs coexist**.

All nodes live in the **ZQX Experimental** category and are named `ZQX …`.  Every node works in both passes of the
two-pass workflow (base pass → latent upscale → img2img with denoise < 1).  NAG is deliberately not included (reported
to break Z-Image Turbo).

> **Status:** the code is tested mathematically on CPU.  The tests use tiny random-weight models built from ComfyUI's
> *real* model classes, wrapped in the real `ModelPatcher` and driven through the real sampling path (93 tests).  The
> pack was also run end-to-end on a live ComfyUI server.  **The visual effect on real models has not been verified**;
> that is the next step, on your side.  Details: [`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-ZIT-QIE-Experimental
# no extra dependencies (torch and safetensors ship with ComfyUI)
```
Tested against ComfyUI `8d534945ebd53cff61e8def81757c6a6c1b9cf2d` (2026-09-27).

## Nodes (summary)

All parameters, per-model starting values and risks: [`docs/NODES.md`](docs/NODES.md).

| Node | What it does | Source | ZIT start | Qwen 2512 start | Main risk |
|---|---|---|---|---|---|
| **ZQX Reference Attention** | In selected blocks, image tokens also attend to the K/V of the passport reference (log-bias weight, RoPE offset, sigma window, block selection, face masks, token dropout) | reference-only, ConsiStory, StoryDiffusion, FreeCus, StyleAligned; on Qwen it matches QIE's own reference path exactly (tested) | weight 1, σ 0.90→0.30, `frame`, `noised`, ref_sigma_mult 0.9, dropout 0.4, face key_mask | weight 1, σ 0.85→0.20, `frame`, dropout 0.4 | copies the reference pose/lighting; 2× compute; memory |
| **ZQX Scheduled LoRA** | LoRA strength varies with sigma and block (runtime, no re-patching) | heuristic; layout is decided in the first steps (2503.10637, 2404.07724) | character 0.4→1.0, realism 1.0→0.35, σ 0.90/0.75 | character 0.5→1.0, realism 1.0→0.4, σ 0.85/0.65 | face/body drift if the early character strength is too low |
| **ZQX K-LoRA** | In every attention layer and step, *either* the character *or* the realism LoRA (top-K abs(ΔW) rule) | K-LoRA, CVPR 2025 (2502.18461) | α 1.5, β 0.5, `s` | same | untested on DiTs other than FLUX |
| **ZQX LoRA Arithmetic** | New LoRA from two LoRAs: add/negate/clean_col/clean_row/target_sub/knots_ties/ties_dense + report | task arithmetic, TIES, DARE, KnOTS, subspace projection | clean_col λ 0.5–1 | same | the realism subspace may also contain identity |
| **ZQX LoRA Conflict Report** | Measures in which blocks the character and realism LoRAs conflict | LoRA paper §7 subspace similarity, TIES sign conflicts | — | — | — |
| **ZQX CADS** | Noises and rescales the text condition in the high-noise steps → breaks mode collapse | CADS, ICLR 2024 (2310.17347) | τ 0.80/1.00, s 0.10 | τ 0.60/0.90, s 0.20 | lower prompt adherence |
| **ZQX Low-Frequency Noise** | Low frequencies of the initial noise come from a real photo (composition/lighting prior), variance preserved | heuristic built on FreeInit (2312.07537) | strength 0.4, cutoff 0.1 | same | colour cast, silhouette copying |
| **ZQX Sigma Split Guider** | Different CFG in the early steps (anti-AI-look negative) and an optional different base model | guidance interval (2404.07724), Distilling Diversity (2503.10637) | switch 0.85, cfg 2.0→1.0 | switch 0.80, cfg 4.0→2.5 | burn-in with CFG > 1 on ZIT |
| **ZQX Sigmas To Text** | Prints the sigmas of a schedule (for choosing windows) | — | — | — | — |
| **ZQX Block Spec (ablation)** | Builds block-list / block-weight strings for block ablations | Stable Flow / FreeFlux protocol | — | — | — |

**Timestep windows are in sigma space** (σ = t ∈ [0,1], 1 = noise).  The second (img2img) pass starts around
σ ≈ 0.5–0.9, so early (composition) windows do not fire there on their own, while identity windows do.  ZIT, 8 steps
(simple, shift 3): `1.000, 0.955, 0.900, 0.833, 0.750, 0.643, 0.500, 0.300`.

Already in ComfyUI core (not duplicated): `CFGOverride` (interval CFG), `TemporalScoreRescaling`, `APG`,
`CFGZeroStar`, `CFGNorm`, `TCFG`, `Mahiro`, `RescaleCFG`, `SkipLayerGuidanceDiT` (works only on Qwen; Z-Image has no
block-replacement hook) and hook-keyframed LoRAs.  Details: [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Example workflows

`examples/` holds API-format workflows.  Open them in ComfyUI with *Workflow → Open* and change the file names to your
models:
* `zit_two_pass_api.json` — Z-Image Turbo: Scheduled LoRA ×2 → Reference Attention → CADS → pass 1
  SamplerCustomAdvanced (Low-Frequency Noise + Sigma Split Guider) → LatentUpscaleBy 1.5 → pass 2 KSampler denoise 0.45.
* `qwen2512_two_pass_api.json` — the same for Qwen-Image 2512.
* `lora_tools_api.json` — Conflict Report, LoRA Arithmetic, K-LoRA, Block Spec.

`python examples/build_examples.py` regenerates them.

## Test order (on real models, against the scene-level AI look)

Use fixed seeds (≥ 4) × 2–3 everyday-scene prompts, change **one variable** at a time, and keep your current workflow
as the baseline.  Evaluate identity similarity, an AI-look checklist (gaze, smile, pose, background, framing) and
texture.

1. **Scheduled LoRA** (cheapest, lowest risk): character early 0.4 / late 1.0; realism early 1.0 / late 0.35;
   σ 0.90/0.75.  Use the same model chain in both passes.  Then sweep the character's early strength from 0.2 to 0.6.
2. Add **Reference Attention** (frame, σ 0.90→0.30, dropout 0.4, face key_mask) and lower the character LoRA's *late*
   strength in steps of 0.2.  Identity then comes from the reference while the LoRA's baked-in mode weakens.  Use block
   ablation (Block Spec) to find which blocks carry identity, and restrict the node to those.
3. **K-LoRA** — an alternative to step 1; compare with the same seeds.
4. **Sigma Split Guider** — CFG 2–3 with an anti-AI-look negative prompt, in the first 1–2 steps only (ZIT).
5. **CADS** — conservative (ZIT τ 0.8/1.0, s 0.1).  Check whether pose/framing diversity increases while identity
   holds.
6. **Conflict Report → LoRA Arithmetic** (`clean_col` λ 0.5/1.0, `target_sub`, `knots_ties`), then repeat step 1 with
   the generated LoRA.  Once an "AI-look LoRA" exists, remove its subspace with `clean_col`.
7. **Low-Frequency Noise** — with a real photo of the kind of scene you want, pass 1 only.

Fix the winning settings, then add the next lever.  Detailed plan:
[`docs/NODES.md`](docs/NODES.md#ab-test-plan).

## Documentation
* [`docs/RESEARCH.md`](docs/RESEARCH.md) — literature survey (2023–2026), method tables, ranking; detailed notes in
  `docs/research/`.
* [`docs/DESIGN.md`](docs/DESIGN.md) — ComfyUI hook points (verified in code), model adapter, sigma windows, design
  decisions.
* [`docs/NODES.md`](docs/NODES.md) — node reference.
* [`docs/TEST_RESULTS.md`](docs/TEST_RESULTS.md) — test table and what could not be verified on CPU.

## Running the tests
```bash
COMFYUI_PATH=/path/to/ComfyUI python -m pytest tests/ -q
```
End-to-end: `tests/e2e/` (test-only stub nodes — **do not** copy them into a real ComfyUI install).

## License
MIT — see [`LICENSE`](LICENSE).
