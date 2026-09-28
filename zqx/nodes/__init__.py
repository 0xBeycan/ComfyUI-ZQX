from .attention_nodes import ZQXReferenceAttention
from .lora_nodes import ZQXKLoRA, ZQXLoRAArithmetic, ZQXLoRAConflictReport, ZQXScheduledLoRA
from .sampling_nodes import ZQXBlockSpec, ZQXCADS, ZQXLowFreqNoise, ZQXSigmaSplitGuider, ZQXSigmasToText

NODE_CLASS_MAPPINGS = {
    "ZQXReferenceAttention": ZQXReferenceAttention,
    "ZQXScheduledLoRA": ZQXScheduledLoRA,
    "ZQXKLoRA": ZQXKLoRA,
    "ZQXLoRAArithmetic": ZQXLoRAArithmetic,
    "ZQXLoRAConflictReport": ZQXLoRAConflictReport,
    "ZQXCADS": ZQXCADS,
    "ZQXLowFreqNoise": ZQXLowFreqNoise,
    "ZQXSigmaSplitGuider": ZQXSigmaSplitGuider,
    "ZQXSigmasToText": ZQXSigmasToText,
    "ZQXBlockSpec": ZQXBlockSpec,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZQXReferenceAttention": "ZQX Reference Attention (identity)",
    "ZQXScheduledLoRA": "ZQX Scheduled LoRA (sigma / block)",
    "ZQXKLoRA": "ZQX K-LoRA (character + realism)",
    "ZQXLoRAArithmetic": "ZQX LoRA Arithmetic",
    "ZQXLoRAConflictReport": "ZQX LoRA Conflict Report",
    "ZQXCADS": "ZQX CADS (condition annealing)",
    "ZQXLowFreqNoise": "ZQX Low-Frequency Noise",
    "ZQXSigmaSplitGuider": "ZQX Sigma Split Guider",
    "ZQXSigmasToText": "ZQX Sigmas To Text",
    "ZQXBlockSpec": "ZQX Block Spec (ablation)",
}
