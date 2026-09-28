"""ZQX_SCORER objects: turn a batch of images into named metrics with weights.

A scorer has `metrics(images) -> {name: tensor(B)}` and `weights: {name: float}` (positive = higher is better).
Scorers are combined by ZQX Scorer Combine; selection nodes use zqx.core.scoring.combine.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from ..core.scoring import background_sharpness, cosine, off_center


class Scorer:
    name = "scorer"

    def __init__(self):
        self.weights: Dict[str, float] = {}

    def metrics(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:  # pragma: no cover - abstract
        raise NotImplementedError


def clip_embed(clip_vision, images: torch.Tensor) -> torch.Tensor:
    out = clip_vision.encode_image(images)
    return out["image_embeds"].to(torch.float32)


class ClipSimilarityScorer(Scorer):
    """Cosine similarity of CLIP-vision image embeddings to reference image(s) (mean reference embedding).
    A negative weight pushes candidates *away* from the reference (e.g. away from the passport composition)."""

    def __init__(self, clip_vision, reference: torch.Tensor, weight: float, name: str = "clip_ref_sim"):
        super().__init__()
        self.clip_vision = clip_vision
        self.ref = clip_embed(clip_vision, reference).mean(0, keepdim=True)
        self.metric = name
        self.weights = {name: float(weight)}

    def metrics(self, images):
        e = clip_embed(self.clip_vision, images)
        return {self.metric: cosine(e, self.ref.to(e.device).expand_as(e)).cpu()}


class BackgroundSharpnessScorer(Scorer):
    """Model-free: log variance-of-Laplacian ratio border/centre (bokeh -> strongly negative)."""

    def __init__(self, weight: float, border: float = 0.2):
        super().__init__()
        self.border = border
        self.weights = {"bg_sharpness": float(weight)}

    def metrics(self, images):
        return {"bg_sharpness": background_sharpness(images, self.border).cpu()}


def to_bgr_uint8(img: torch.Tensor) -> np.ndarray:
    a = (img[..., :3].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(a[..., ::-1])


class FaceScorer(Scorer):
    """InsightFace-based face metrics (largest detected face):
      face_found     1 if a face was detected, else 0
      face_identity  cosine(ArcFace embedding, reference embedding)    (needs a reference)
      face_off_center distance of the face centre from the image centre / half diagonal
      head_turn      min(|yaw|, 90) / 90 from the 3-D landmark pose     (looking away from the camera)
    Candidates without a face get identity 0, off_center 0, head_turn 0 and face_found 0."""

    def __init__(self, analyzer, reference: Optional[torch.Tensor], w_identity: float, w_off_center: float,
                 w_head_turn: float, w_face_found: float):
        super().__init__()
        self.analyzer = analyzer
        self.weights = {"face_identity": float(w_identity), "face_off_center": float(w_off_center),
                        "head_turn": float(w_head_turn), "face_found": float(w_face_found)}
        self.ref_emb = None
        if reference is not None:
            embs = []
            for i in range(reference.shape[0]):
                f = self._largest(reference[i])
                if f is None:
                    raise ValueError("ZQX Face scorer: no face found in the reference image")
                embs.append(self._emb(f))
            self.ref_emb = torch.stack(embs).mean(0)
        elif w_identity != 0.0:
            raise ValueError("ZQX Face scorer: face_identity needs a reference image")

    @staticmethod
    def _emb(face):
        e = getattr(face, "normed_embedding", None)
        if e is None:
            e = face.embedding
        return torch.as_tensor(np.asarray(e), dtype=torch.float32)

    def _largest(self, img):
        faces = self.analyzer.get(to_bgr_uint8(img))
        if not faces:
            return None
        return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))

    def metrics(self, images):
        b, h, w, _ = images.shape
        found, ident, offc, turn = [], [], [], []
        for i in range(b):
            f = self._largest(images[i])
            if f is None:
                found.append(0.0), ident.append(0.0), offc.append(0.0), turn.append(0.0)
                continue
            found.append(1.0)
            ident.append(float(cosine(self._emb(f), self.ref_emb)) if self.ref_emb is not None else 0.0)
            offc.append(off_center([float(v) for v in f.bbox[:4]], w, h))
            pose = getattr(f, "pose", None)
            if pose is None:
                if self.weights["head_turn"] != 0.0:
                    raise ValueError("ZQX Face scorer: the face model provides no pose (use buffalo_l / antelopev2 "
                                     "with the 3-D landmark model) or set the head_turn weight to 0")
                turn.append(0.0)
            else:
                turn.append(min(abs(float(pose[1])), 90.0) / 90.0)
        t = lambda v: torch.tensor(v, dtype=torch.float32)
        return {"face_found": t(found), "face_identity": t(ident), "face_off_center": t(offc), "head_turn": t(turn)}


def load_insightface(model_name: str, provider: str, det_size: int = 640):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise ImportError("ZQX Face scorer needs the 'insightface' package (pip install insightface onnxruntime[-gpu]) "
                          "and a model pack in models/insightface/models/<name> (as used by IPAdapter / PuLID).") from e
    import folder_paths
    root = os.path.join(folder_paths.models_dir, "insightface")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "CUDA" else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=model_name, root=root, providers=providers)
    app.prepare(ctx_id=0 if provider == "CUDA" else -1, det_size=(det_size, det_size))
    return app


class CombinedScorer(Scorer):
    def __init__(self, scorers: List[Scorer]):
        super().__init__()
        if not scorers:
            raise ValueError("no scorers")
        self.scorers = scorers
        self.names = []
        for i, s in enumerate(scorers):
            ren = {}
            for k, w in s.weights.items():
                nk = k if k not in self.weights else f"{k}_{i}"
                ren[k] = nk
                self.weights[nk] = w
            self.names.append(ren)

    def metrics(self, images):
        out = {}
        for s, ren in zip(self.scorers, self.names):
            for k, v in s.metrics(images).items():
                out[ren.get(k, k)] = v
        return out
