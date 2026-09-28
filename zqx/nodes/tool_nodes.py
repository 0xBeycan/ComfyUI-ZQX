import os

import numpy as np
import torch

from ..core.pose_bank import scan_bank, select
from .common import CAT_TOOLS


class ZQXPoseBank:
    DESCRIPTION = ("Pick a control image (OpenPose / DWPose skeleton, depth map...) from a folder bank by seed, "
                   "matching the output aspect ratio and optional tags (= sub-folder names, e.g. walking, sitting). "
                   "Feed it to a ControlNet so the pose/composition comes from real candid photos, not the model prior.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "bank_folder": ("STRING", {"default": "zqx_pose_bank", "tooltip": "Folder inside ComfyUI's input directory (or an absolute path)."}),
            "tags": ("STRING", {"default": "", "tooltip": "Comma list of sub-folders to draw from; empty = all."}),
            "width": ("INT", {"default": 832, "min": 16, "max": 16384, "step": 8}),
            "height": ("INT", {"default": 1216, "min": 16, "max": 16384, "step": 8}),
            "aspect_tolerance": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01,
                                           "tooltip": "Allowed aspect-ratio factor deviation (0.15 = within 15%)."}),
            "resize": (["crop_to_size", "keep"], {"default": "crop_to_size"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "filename", "candidates")
    FUNCTION = "pick"
    CATEGORY = CAT_TOOLS

    @classmethod
    def IS_CHANGED(cls, bank_folder, **kw):
        # re-run when files in the bank change (the choice itself is deterministic in the seed)
        root = cls._root(bank_folder)
        if not os.path.isdir(root):
            return ""
        return str([(p, os.path.getmtime(os.path.join(root, p))) for p, _ in scan_bank(root)])

    @staticmethod
    def _root(bank_folder):
        if os.path.isabs(bank_folder):
            return bank_folder
        import folder_paths
        return os.path.join(folder_paths.get_input_directory(), bank_folder)

    def pick(self, bank_folder, tags, width, height, aspect_tolerance, resize, seed):
        from PIL import Image
        root = self._root(bank_folder)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] or None
        entries = []
        for rel, tag in scan_bank(root):
            with Image.open(os.path.join(root, rel)) as im:
                w, h = im.size
            entries.append((rel, tag, w, h))
        rel, n = select(entries, seed, width, height, aspect_tolerance, tag_list)
        with Image.open(os.path.join(root, rel)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        img = torch.from_numpy(arr)[None]
        if resize == "crop_to_size":
            import comfy.utils
            img = comfy.utils.common_upscale(img.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        return (img, rel, n)
