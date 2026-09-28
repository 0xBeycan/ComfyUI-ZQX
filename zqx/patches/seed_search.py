"""First-step seed search: probe many seeds for a few steps, score the x0 previews, fully sample the best.

Rationale: distilled flow models fix layout/pose in the first step(s) (Gandikota & Bau, arXiv 2503.10637);
Group Inference (FLUX-schnell) prunes candidates on early predictions.  The x0 prediction after the probe
steps is decoded with the VAE and scored (face pose / off-centre / identity, CLIP similarity, background
sharpness...).  The kept candidates are re-sampled from scratch with the same noise, so each final sample is
exactly what a normal run with that seed would give.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import torch

from ..core.scoring import combine, top_k


def _decode(vae, model_patcher, x0):
    lat = model_patcher.model.process_latent_out(x0.to(torch.float32))
    img = vae.decode(lat)
    if img.ndim == 5:          # video-style VAEs return (B, T, H, W, C)
        img = img.reshape(-1, *img.shape[-3:])
    return img


def seed_search(noise, guider, sampler, sigmas, latent: Dict, vae, scorer, n_candidates: int, probe_steps: int,
                keep: int, normalize: bool = True):
    import comfy.sample
    if n_candidates < 1 or keep < 1 or keep > n_candidates:
        raise ValueError("need 1 <= keep <= n_candidates")
    total_steps = sigmas.shape[-1] - 1
    if not (1 <= probe_steps <= total_steps):
        raise ValueError(f"probe_steps must be in [1, {total_steps}]")
    if not hasattr(noise, "seed"):
        raise ValueError("the NOISE object has no seed; seed search needs a seeded noise (RandomNoise / ZQX noise)")
    latent = latent.copy()
    samples = latent["samples"]
    samples = comfy.sample.fix_empty_latent_channels(guider.model_patcher, samples, latent.get("downscale_ratio_spacial", None),
                                                     latent.get("downscale_ratio_temporal", None))
    if samples.shape[0] != 1:
        raise ValueError("seed search expects a latent with batch size 1")
    latent["samples"] = samples
    mask = latent.get("noise_mask", None)

    base_seed = int(noise.seed)
    noises, previews = [], []
    for i in range(n_candidates):
        nz = copy.copy(noise)
        nz.seed = base_seed + i
        nt = nz.generate_noise(latent)
        noises.append((nz.seed, nt))
        cap = {}

        def cb(step, x0, x, total, cap=cap):
            cap["x0"] = x0

        guider.sample(nt, samples, sampler, sigmas[: probe_steps + 1], denoise_mask=mask, callback=cb,
                      disable_pbar=True, seed=nz.seed)
        if "x0" not in cap:
            raise RuntimeError("sampler did not report an x0 prediction through the callback")
        previews.append(_decode(vae, guider.model_patcher, cap["x0"]))
    prev = torch.cat(previews, dim=0)
    metrics = scorer.metrics(prev)
    scores = combine(metrics, scorer.weights, normalize=normalize)
    best = top_k(scores, keep)
    outs = []
    for i in best:
        seed, nt = noises[i]
        outs.append(guider.sample(nt, samples, sampler, sigmas, denoise_mask=mask, disable_pbar=True, seed=seed))
    result = torch.cat(outs, dim=0)
    lines = ["rank  seed        score   " + "  ".join(f"{k:>14s}" for k in metrics)]
    order = top_k(scores, n_candidates)
    for r, i in enumerate(order):
        vals = "  ".join(f"{float(metrics[k][i]):14.4f}" for k in metrics)
        mark = "*" if i in best else " "
        lines.append(f"{r:3d}{mark} {noises[i][0]:<10d} {float(scores[i]):7.3f}   {vals}")
    lines.append("weights: " + ", ".join(f"{k}={v}" for k, v in scorer.weights.items()))
    return result, prev, [noises[i][0] for i in best], "\n".join(lines), scores, metrics
