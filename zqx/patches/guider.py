"""Sigma-split guider: different CFG scale (and optionally a different model) above / below a sigma.

* Interval guidance (Kynkaenniemi et al., arXiv:2404.07724): CFG only inside a noise-level interval.
  Here: cfg_early for sigma >= switch_sigma, cfg_late below.  With cfg = 1.0 the uncond pass is skipped
  (ComfyUI's cfg1 optimisation), so e.g. Z-Image Turbo can use a negative prompt only in its first
  1-2 steps ("posed, looking at camera, studio bokeh") and stays at its distilled CFG = 1 afterwards.
* Base-model first step (Gandikota & Bau, "Distilling Diversity and Control in Diffusion Models",
  arXiv:2503.10637): distilled models fix the layout in the first step; running the *base* model for the
  high-noise steps restores diversity.  Optional `model_early` is used for sigma >= switch_sigma.
  It must be a different base model (e.g. Z-Image base for Z-Image Turbo) with the same latent space and
  text encoder; two patchers of the *same* weights with different LoRAs cannot be swapped mid-sampling
  (ComfyUI patches weights in place) - use ZQX Scheduled LoRA for that.
"""
from __future__ import annotations

import math

import torch

import comfy.model_patcher
import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.samplers


class SigmaSplitGuider(comfy.samplers.CFGGuider):
    def __init__(self, model_patcher, switch_sigma: float, cfg_early: float, cfg_late: float, model_early=None):
        super().__init__(model_patcher)
        if not (0.0 <= switch_sigma <= 1.0e4):
            raise ValueError("switch_sigma out of range")
        self.switch_sigma = float(switch_sigma)
        self.cfg_early = float(cfg_early)
        self.cfg_late = float(cfg_late)
        self.cfg = self.cfg_late
        self.model_early = model_early
        if model_early is not None:
            if model_early.model is model_patcher.model:
                raise ValueError(
                    "ZQX Sigma Split Guider: model_early shares its weights with model (same base checkpoint, "
                    "different patches). ComfyUI patches weights in place, so both cannot be active in one "
                    "sampling run. Use 'ZQX Scheduled LoRA' to change LoRA strengths over sigma instead.")
            a = type(model_early.model.latent_format)
            b = type(model_patcher.model.latent_format)
            if a is not b:
                raise ValueError(f"ZQX Sigma Split Guider: latent formats differ ({a.__name__} vs {b.__name__})")
        self.early_inner = None
        self.calls = []  # (sigma, which, cfg) for tests

    def _early(self, sigma: float) -> bool:
        return sigma >= self.switch_sigma

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False,
                     seed=None, latent_shapes=None):
        self.early_inner = None
        self._early_loaded = []
        if self.model_early is not None:
            conds = {k: list(map(lambda a: a.copy(), v)) for k, v in self.original_conds.items()}
            self._early_opts = comfy.model_patcher.create_model_options_clone(self.model_early.model_options)
            to = self._early_opts.setdefault("transformer_options", {})
            comfy.patcher_extension.merge_nested_dicts(to.setdefault("wrappers", {}), self.model_early.wrappers, copy_dict1=False)
            comfy.patcher_extension.merge_nested_dicts(to.setdefault("callbacks", {}), self.model_early.callbacks, copy_dict1=False)
            self.early_inner, self._early_conds_raw, self._early_loaded = comfy.sampler_helpers.prepare_sampling(
                self.model_early, noise.shape, conds, self._early_opts)
            self.model_early.pre_run()
        try:
            return super().outer_sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed,
                                        latent_shapes=latent_shapes)
        finally:
            if self.early_inner is not None:
                self.model_early.cleanup()
                comfy.sampler_helpers.cleanup_models(self._early_conds_raw, self._early_loaded)
                self.early_inner = None

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed,
                     latent_shapes=None):
        if self.early_inner is not None:
            li = latent_image
            if li is not None and torch.count_nonzero(li) > 0:
                li = self.early_inner.process_latent_in(li)
            self._early_conds = comfy.samplers.process_conds(self.early_inner, noise, self._early_conds_raw, device, li,
                                                             denoise_mask, seed, latent_shapes=latent_shapes)
            self._early_opts.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        return super().inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar,
                                    seed, latent_shapes=latent_shapes)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        sigma = float(timestep.flatten()[0])
        early = self._early(sigma)
        cfg = self.cfg_early if early else self.cfg_late
        if early and self.early_inner is not None:
            self.calls.append((sigma, "early_model", cfg))
            return comfy.samplers.sampling_function(self.early_inner, x, timestep, self._early_conds.get("negative", None),
                                                    self._early_conds.get("positive", None), cfg,
                                                    model_options=self._early_opts, seed=seed)
        self.calls.append((sigma, "model", cfg))
        return comfy.samplers.sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None),
                                                self.conds.get("positive", None), cfg, model_options=model_options, seed=seed)
