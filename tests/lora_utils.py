"""Synthetic LoRA state dicts for the tiny models, in several real-world key formats."""
import torch


def _targets(kind, n_layers):
    if kind == "qwen":
        # (model-weight-key stem, out, in) with tiny dims: inner = heads*head_dim = 64
        t = []
        for i in range(n_layers):
            for n in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
                t.append((f"transformer_blocks.{i}.attn.{n}", 64, 64))
            t.append((f"transformer_blocks.{i}.attn.to_out.0", 64, 64))
            t.append((f"transformer_blocks.{i}.img_mlp.net.0.proj", 256, 64))
            t.append((f"transformer_blocks.{i}.img_mlp.net.2", 64, 256))
        return t
    if kind == "zimage":
        t = []
        for i in range(n_layers):
            for n in ("to_q", "to_k", "to_v"):
                t.append((f"layers.{i}.attention.{n}", 256, 256))   # diffusers names -> fused qkv slices in ComfyUI
            t.append((f"layers.{i}.attention.to_out.0", 256, 256))
            t.append((f"layers.{i}.feed_forward.w1", 768, 256))
            t.append((f"layers.{i}.feed_forward.w2", 256, 768))
        return t
    raise ValueError(kind)


def make_lora(kind, n_layers=2, rank=4, seed=0, fmt="peft", alpha=None, scale=0.05, only=None):
    """fmt: 'peft' (diffusion_model.X.lora_A/B.weight), 'kohya' (lora_unet_X_with_underscores.lora_down/up + alpha),
    'transformer' (transformer.X.lora_A/B)."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for stem, out_d, in_d in _targets(kind, n_layers):
        if only is not None and not only(stem):
            continue
        a = torch.randn(rank, in_d, generator=g) * scale
        b = torch.randn(out_d, rank, generator=g) * scale
        if fmt == "peft":
            base = f"diffusion_model.{stem}"
            sd[base + ".lora_A.weight"] = a
            sd[base + ".lora_B.weight"] = b
        elif fmt == "transformer":
            base = f"transformer.{stem}"
            sd[base + ".lora_A.weight"] = a
            sd[base + ".lora_B.weight"] = b
        elif fmt == "kohya":
            base = "lora_unet_" + stem.replace(".", "_")
            sd[base + ".lora_down.weight"] = a
            sd[base + ".lora_up.weight"] = b
        else:
            raise ValueError(fmt)
        if alpha is not None:
            sd[base + ".alpha"] = torch.tensor(float(alpha))
    return sd


def zimage_ff_dim():
    # NextDiT FeedForward: hidden = int(ffn_dim_multiplier * dim) rounded up to multiple_of (256): int(8/3*256)=682 -> 768
    return 768
