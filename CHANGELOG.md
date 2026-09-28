# Changelog

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

## 0.1.1 — 2026-09-28
- README.md and docs/NODES.md translated to English (the whole repository is now English).
