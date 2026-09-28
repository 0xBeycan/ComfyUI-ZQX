from .attention_nodes import ZQXReferenceAttention
from .edit_nodes import ZQXActivationSteering, ZQXUCETextEdit
from .lora_nodes import (ZQXKLoRA, ZQXLoRAArithmetic, ZQXLoRACommonSubspace, ZQXLoRAConflictReport, ZQXLoRAGuidance,
                         ZQXLoRASurgery, ZQXScheduledLoRA, ZQXSpatialLoRA)
from .sampling_nodes import (ZQXBlockSpec, ZQXCADS, ZQXDiTPAG, ZQXLowFreqNoise, ZQXSigmaSplitGuider,
                             ZQXSigmasToText)
from .scoring_nodes import (ZQXRealismLoRAAblation, ZQXScorerBackgroundSharpness, ZQXScorerClipSimilarity,
                            ZQXScorerCombine, ZQXScorerFace, ZQXSeedSearch)
from .tool_nodes import ZQXPoseBank

_NODES = [
    # attention / identity
    (ZQXReferenceAttention, "ZQX Reference Attention (identity)"),
    # LoRA
    (ZQXScheduledLoRA, "ZQX Scheduled LoRA (sigma / block)"),
    (ZQXSpatialLoRA, "ZQX Spatial LoRA (masked)"),
    (ZQXLoRAGuidance, "ZQX LoRA Guidance (LoRA-CFG)"),
    (ZQXKLoRA, "ZQX K-LoRA (character + realism)"),
    (ZQXLoRAArithmetic, "ZQX LoRA Arithmetic"),
    (ZQXLoRASurgery, "ZQX LoRA Surgery"),
    (ZQXLoRACommonSubspace, "ZQX LoRA Common Subspace"),
    (ZQXLoRAConflictReport, "ZQX LoRA Conflict Report"),
    (ZQXRealismLoRAAblation, "ZQX Realism LoRA Ablation (identity-safe)"),
    # guidance / conditioning
    (ZQXCADS, "ZQX CADS (condition annealing)"),
    (ZQXSigmaSplitGuider, "ZQX Sigma Split Guider"),
    (ZQXDiTPAG, "ZQX Perturbed Attention Guidance (DiT)"),
    (ZQXActivationSteering, "ZQX Activation Steering"),
    # model edit
    (ZQXUCETextEdit, "ZQX UCE Text Edit"),
    # sampling
    (ZQXLowFreqNoise, "ZQX Low-Frequency Noise"),
    (ZQXSeedSearch, "ZQX Seed Search (first-step scoring)"),
    # scoring
    (ZQXScorerFace, "ZQX Scorer: Face (InsightFace)"),
    (ZQXScorerClipSimilarity, "ZQX Scorer: CLIP Similarity"),
    (ZQXScorerBackgroundSharpness, "ZQX Scorer: Background Sharpness"),
    (ZQXScorerCombine, "ZQX Scorer: Combine"),
    # tools
    (ZQXPoseBank, "ZQX Pose Bank"),
    (ZQXSigmasToText, "ZQX Sigmas To Text"),
    (ZQXBlockSpec, "ZQX Block Spec (ablation)"),
]

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls, _ in _NODES}
NODE_DISPLAY_NAME_MAPPINGS = {cls.__name__: name for cls, name in _NODES}
