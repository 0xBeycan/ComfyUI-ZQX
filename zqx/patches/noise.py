"""NOISE object (for SamplerCustomAdvanced) whose low-frequency band comes from a reference latent."""
from __future__ import annotations

import torch

from ..core.noise_init import lowfreq_mix


class LowFreqNoise:
    def __init__(self, seed: int, ref_latent: torch.Tensor, strength: float, cutoff: float, kind: str, order: int,
                 base_noise=None):
        self.seed = seed
        self.ref = ref_latent
        self.strength = strength
        self.cutoff = cutoff
        self.kind = kind
        self.order = order
        self.base_noise = base_noise

    def generate_noise(self, input_latent):
        import comfy.sample
        if self.base_noise is not None:
            noise = self.base_noise.generate_noise(input_latent)
        else:
            latent = input_latent["samples"]
            batch_inds = input_latent.get("batch_index", None)
            noise = comfy.sample.prepare_noise(latent, self.seed, batch_inds)
        ref = self.ref
        if noise.ndim == 5 and ref.ndim == 4:
            ref = ref.unsqueeze(2)
        return lowfreq_mix(noise, ref, self.strength, self.cutoff, self.kind, self.order)
