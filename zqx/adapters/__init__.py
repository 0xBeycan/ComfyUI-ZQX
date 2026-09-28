"""Model adapter layer: the only place that knows Qwen-Image vs Z-Image internals.

Everything the nodes need from a specific architecture goes through
`get_adapter(model_patcher)`:

* how the DIFFUSION_MODEL wrapper's (args, kwargs) are laid out
  (x, timestep, context, transformer_options, native reference latents);
* where the target image tokens sit inside every *main-block* attention call
  (text first in both models; Qwen-Image-Edit appends its own reference
  tokens after the target; Z-Image pads the caption and the image to a
  multiple of 32 tokens and runs separate refiner blocks before the main
  layers);
* the RoPE position ids of image tokens and a rotation R(delta) built with
  the model's *own* EmbedND embedder;
* which block a weight key belongs to (for block-wise LoRA weights).

Unsupported models / settings raise AdapterError; there is no fallback.
"""
from .base import AdapterError, ModelAdapter, TokenLayout
from .qwen import QwenImageAdapter
from .zimage import ZImageAdapter


def get_adapter(model_patcher) -> ModelAdapter:
    dm = model_patcher.get_model_object("diffusion_model")
    for cls in (QwenImageAdapter, ZImageAdapter):
        if cls.matches(dm):
            return cls(model_patcher, dm)
    raise AdapterError(
        f"ZQX: unsupported diffusion model {type(dm).__module__}.{type(dm).__name__}. "
        "Supported: Qwen-Image / Qwen-Image-Edit (comfy.ldm.qwen_image.model.QwenImageTransformer2DModel) "
        "and Z-Image (comfy.ldm.lumina.model.NextDiT with z_image_modulation)."
    )


__all__ = ["AdapterError", "ModelAdapter", "TokenLayout", "QwenImageAdapter", "ZImageAdapter", "get_adapter"]
