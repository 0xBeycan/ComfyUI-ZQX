# Changelog

## 0.2.0 — 2026-09-28
Renamed to **ComfyUI-ZQX**; menu categories are now `ZQX/attention`, `ZQX/lora`, `ZQX/guidance`, `ZQX/model-edit`,
`ZQX/sampling`, `ZQX/scoring`, `ZQX/tools` (node ids unchanged, saved workflows keep working).

New nodes:
- **ZQX Spatial LoRA** — token-masked LoRA side branch (realism outside the face, character on the face).
- **ZQX LoRA Guidance** — LoRA-CFG with a sigma schedule.
- **ZQX Realism LoRA Ablation** — identity-safe realism LoRA by block / module-kind / singular-direction ablation,
  scored with an identity scorer and CLIP similarity.
- **ZQX LoRA Surgery**, **ZQX LoRA Common Subspace** — single-LoRA rewriting; shared subspace of several LoRAs.
- **ZQX Seed Search** + scorers (**Face (InsightFace)**, **CLIP Similarity**, **Background Sharpness**, **Combine**).
- **ZQX Pose Bank** — seeded control-image picker for ControlNet.
- **ZQX Activation Steering**, **ZQX UCE Text Edit**, **ZQX Perturbed Attention Guidance (DiT)**.

Changes:
- Reference Attention: new `matched` position mode (optional inputs `match_threshold`, `match_mutual`).
- Pass-context stack so multi-pass patches can be chained safely; fixes found by the new chaining tests: stale
  `block_index` on Z-Image refiners in multi-pass wrappers, hooks acting inside other patches' extra passes, spans
  computed from extended key lengths.
- 151 tests, E2E graph with all new nodes, 7 example workflows.

## 0.1.1 — 2026-09-28
- README.md and docs/NODES.md translated to English (the whole repository is now English).

## 0.1.0 — 2026-09-28
First release.

- **ZQX Reference Attention**: capture/inject extended self-attention for Qwen-Image, Qwen-Image-Edit and Z-Image
  (log-bias weight, RoPE offset by rotation composition, sigma window, block list, query/key masks, noised/cached
  capture, FreeCus reference noise/key scale, ConsiStory token dropout, uncond control).
- **ZQX Scheduled LoRA**: runtime LoRA with sigma ramp and block weights (weight wrappers, no re-patching).
- **ZQX K-LoRA**: per-layer, per-step character/realism selection (arXiv 2502.18461).
- **ZQX LoRA Arithmetic** and **ZQX LoRA Conflict Report**: exact low-rank add/negate/projection modes,
  KnOTS-aligned TIES/DARE, dense TIES + SVD; model-key-space mapping for kohya/peft/ai-toolkit/musubi formats.
- **ZQX CADS** (arXiv 2310.17347), **ZQX Low-Frequency Noise** (variance-preserving FreeInit-style composition),
  **ZQX Sigma Split Guider** (interval CFG + base-model early steps), **ZQX Sigmas To Text**, **ZQX Block Spec**.
- Docs: RESEARCH, DESIGN, NODES, TEST_RESULTS; example API workflows; 93 tests + end-to-end API test.
