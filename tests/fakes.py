"""Deterministic stand-ins for models that cannot be downloaded here (VAE, CLIP vision, InsightFace).
They only exercise the plumbing; the real models are used on the user's machine."""
import types

import numpy as np
import torch


class FakeVAE:
    """latent (B, C, [T,] H, W) -> image (B, H, W, 3) in [0, 1] via sigmoid of the first 3 channels."""

    def decode(self, lat):
        if lat.ndim == 5:
            lat = lat[:, :, 0]
        return torch.sigmoid(lat[:, :3].float()).permute(0, 2, 3, 1).contiguous()


class FakeClipVision:
    def __init__(self, dim=16, seed=0):
        self.proj = torch.randn(3 * 4 * 4, dim, generator=torch.Generator().manual_seed(seed))

    def encode_image(self, images):
        x = torch.nn.functional.adaptive_avg_pool2d(images.permute(0, 3, 1, 2).float(), (4, 4)).flatten(1)
        return {"image_embeds": x @ self.proj}


class FakeFaceAnalyzer:
    """'Face' = brightest 3x3 region; embedding = colour histogram-ish vector; yaw from its horizontal position."""

    def __init__(self, with_pose=True, none_if_dark=False):
        self.with_pose = with_pose
        self.none_if_dark = none_if_dark

    def get(self, bgr):
        g = bgr.astype(np.float32).mean(-1)
        if self.none_if_dark and g.max() < 128:
            return []
        y, x = np.unravel_index(np.argmax(g), g.shape)
        emb = np.concatenate([bgr.reshape(-1, 3).mean(0), bgr.reshape(-1, 3).std(0), [x, y]]).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        f = types.SimpleNamespace(bbox=np.array([x - 1, y - 1, x + 2, y + 2], dtype=np.float32), normed_embedding=emb)
        if self.with_pose:
            f.pose = np.array([0.0, (x / max(g.shape[1] - 1, 1) - 0.5) * 120.0, 0.0], dtype=np.float32)
        return [f]
