"""TEST-ONLY ComfyUI nodes: tiny random-weight models / conditioning / latents so the ZQX nodes can be run
end-to-end through the real ComfyUI server + executor on CPU without downloading model weights.
Never install this into a real ComfyUI."""
import os
import sys

import torch

_here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))   # tests/
import tiny_models as tm  # noqa: E402


class ZQXTestTinyModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"kind": (["zimage", "qwen"],), "layers": ("INT", {"default": 2, "min": 1, "max": 8})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "ZQX test stubs"

    def run(self, kind, layers):
        return (tm.z_image(layers) if kind == "zimage" else tm.qwen_image(layers),)


class ZQXTestRandomCond:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"kind": (["zimage", "qwen"],), "seed": ("INT", {"default": 1})}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "run"
    CATEGORY = "ZQX test stubs"

    def run(self, kind, seed):
        return ((tm.zimage_cond if kind == "zimage" else tm.qwen_cond)(seed),)


class ZQXTestLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"kind": (["zimage", "qwen"],), "h": ("INT", {"default": 8}), "w": ("INT", {"default": 8}),
                             "seed": ("INT", {"default": 3}), "zero": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "ZQX test stubs"

    def run(self, kind, h, w, seed, zero):
        lat = (tm.zimage_latent if kind == "zimage" else tm.qwen_latent)(h, w, seed=seed)
        return ({"samples": torch.zeros_like(lat) if zero else lat},)


class ZQXTestLatentStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "tag": ("STRING", {"default": ""})}}
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "ZQX test stubs"

    def run(self, latent, tag):
        s = latent["samples"]
        txt = f"{tag}: shape={tuple(s.shape)} finite={bool(torch.isfinite(s).all())} mean={s.mean().item():.4f} std={s.std().item():.4f}"
        return {"ui": {"text": [txt]}}


NODE_CLASS_MAPPINGS = {"ZQXTestTinyModel": ZQXTestTinyModel, "ZQXTestRandomCond": ZQXTestRandomCond,
                       "ZQXTestLatent": ZQXTestLatent, "ZQXTestLatentStats": ZQXTestLatentStats}
