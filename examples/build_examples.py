"""Generates the API-format example workflows in this folder (load them in ComfyUI via Workflow > Open,
or POST them to /prompt).  File names of models/LoRAs/images are placeholders: change them to yours.

  python examples/build_examples.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NEG = ("posed, posing, looking at camera, eye contact, perfect smile, studio portrait, centered composition, "
       "symmetrical framing, bokeh, blurred background, clean background, plastic skin")


def common(model, clip, clip_type, vae, shift, char_lora, real_lora, prompt, sched_char, sched_real):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": clip_type, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": shift}},
        # sigma-scheduled LoRAs: character weak early (pose/scene free) and full late; realism the opposite
        "5": {"class_type": "ZQXScheduledLoRA", "inputs": {"model": ["4", 0], "lora_name": char_lora, **sched_char,
                                                             "block_weights": "", "allow_unmatched_keys": False}},
        "6": {"class_type": "ZQXScheduledLoRA", "inputs": {"model": ["5", 0], "lora_name": real_lora, **sched_real,
                                                             "block_weights": "", "allow_unmatched_keys": False}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": NEG}},
        # passport / identity reference + face mask of the reference (white = face)
        "9": {"class_type": "LoadImage", "inputs": {"image": "passport_reference.png"}},
        "10": {"class_type": "ImageScale", "inputs": {"image": ["9", 0], "upscale_method": "lanczos", "width": 512,
                                                        "height": 640, "crop": "center"}},
        "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "LoadImageMask", "inputs": {"image": "passport_reference_face_mask.png", "channel": "red"}},
        # real photo with the composition / light you want (not the person)
        "13": {"class_type": "LoadImage", "inputs": {"image": "composition_photo.jpg"}},
        "14": {"class_type": "VAEEncode", "inputs": {"pixels": ["13", 0], "vae": ["3", 0]}},
    }


def zit():
    g = common("z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "lumina2", "ae.safetensors", 3.0,
               "my_character_zit.safetensors", "my_realism_zit.safetensors",
               "photo of <trigger> woman buying bread at a crowded street market, candid, walking, side light",
               dict(strength_early=0.4, strength_late=1.0, sigma_hi=0.90, sigma_lo=0.75),
               dict(strength_early=1.0, strength_late=0.35, sigma_hi=0.90, sigma_lo=0.75))
    g.update({
        "20": {"class_type": "ZQXReferenceAttention", "inputs": {
            "model": ["6", 0], "reference": ["11", 0], "weight": 1.0, "sigma_start": 0.90, "sigma_end": 0.30,
            "blocks": "all", "position_mode": "frame", "capture_mode": "noised", "cache_sigma": 0.0,
            "ref_sigma_mult": 0.9, "key_scale": 1.0, "token_dropout": 0.4, "inject_uncond": True, "noise_seed": 0,
            "key_mask": ["12", 0]}},
        "21": {"class_type": "ZQXCADS", "inputs": {"model": ["20", 0], "tau1": 0.80, "tau2": 1.00, "noise_scale": 0.10,
                                                     "psi": 1.0, "relative_noise": True, "apply_to": "cond_and_uncond", "seed": 0}},
        # ---- pass 1 (txt2img): low-frequency composition noise + early-only negative CFG
        "30": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "31": {"class_type": "ZQXLowFreqNoise", "inputs": {"noise_seed": 1, "reference": ["14", 0], "strength": 0.4,
                                                             "cutoff": 0.1, "filter": "gaussian", "butterworth_order": 4}},
        "32": {"class_type": "ZQXSigmaSplitGuider", "inputs": {"model": ["21", 0], "positive": ["7", 0], "negative": ["8", 0],
                                                                 "switch_sigma": 0.85, "cfg_early": 2.0, "cfg_late": 1.0}},
        "33": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "34": {"class_type": "BasicScheduler", "inputs": {"model": ["21", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "35": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["31", 0], "guider": ["32", 0], "sampler": ["33", 0],
                                                                   "sigmas": ["34", 0], "latent_image": ["30", 0]}},
        "36": {"class_type": "ZQXSigmasToText", "inputs": {"sigmas": ["34", 0]}},
        # ---- latent upscale + pass 2 (img2img, denoise 0.45): same model chain; sigma windows decide what fires
        "40": {"class_type": "LatentUpscaleBy", "inputs": {"samples": ["35", 0], "upscale_method": "bislerp", "scale_by": 1.5}},
        "41": {"class_type": "KSampler", "inputs": {"model": ["21", 0], "seed": 1, "steps": 8, "cfg": 1.0,
                                                      "sampler_name": "res_multistep", "scheduler": "simple",
                                                      "positive": ["7", 0], "negative": ["8", 0], "latent_image": ["40", 0],
                                                      "denoise": 0.45}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["3", 0]}},
        "43": {"class_type": "SaveImage", "inputs": {"images": ["42", 0], "filename_prefix": "zqx_zit"}},
    })
    return g


def qwen():
    g = common("qwen_image_2512_bf16.safetensors", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image",
               "qwen_image_vae.safetensors", 3.1, "my_character_qwen.safetensors", "my_realism_qwen.safetensors",
               "photo of <trigger> woman buying bread at a crowded street market, candid, walking, side light",
               dict(strength_early=0.5, strength_late=1.0, sigma_hi=0.85, sigma_lo=0.65),
               dict(strength_early=1.0, strength_late=0.4, sigma_hi=0.85, sigma_lo=0.65))
    g.update({
        "20": {"class_type": "ZQXReferenceAttention", "inputs": {
            "model": ["6", 0], "reference": ["11", 0], "weight": 1.0, "sigma_start": 0.85, "sigma_end": 0.20,
            "blocks": "all", "position_mode": "frame", "capture_mode": "noised", "cache_sigma": 0.0,
            "ref_sigma_mult": 1.0, "key_scale": 1.0, "token_dropout": 0.4, "inject_uncond": True, "noise_seed": 0,
            "key_mask": ["12", 0]}},
        "21": {"class_type": "ZQXCADS", "inputs": {"model": ["20", 0], "tau1": 0.60, "tau2": 0.90, "noise_scale": 0.20,
                                                     "psi": 1.0, "relative_noise": True, "apply_to": "cond_and_uncond", "seed": 0}},
        "30": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 928, "height": 1664, "batch_size": 1}},
        "31": {"class_type": "ZQXLowFreqNoise", "inputs": {"noise_seed": 1, "reference": ["14", 0], "strength": 0.4,
                                                             "cutoff": 0.1, "filter": "gaussian", "butterworth_order": 4}},
        "32": {"class_type": "ZQXSigmaSplitGuider", "inputs": {"model": ["21", 0], "positive": ["7", 0], "negative": ["8", 0],
                                                                 "switch_sigma": 0.80, "cfg_early": 4.0, "cfg_late": 2.5}},
        "33": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "34": {"class_type": "BasicScheduler", "inputs": {"model": ["21", 0], "scheduler": "simple", "steps": 30, "denoise": 1.0}},
        "35": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["31", 0], "guider": ["32", 0], "sampler": ["33", 0],
                                                                   "sigmas": ["34", 0], "latent_image": ["30", 0]}},
        "36": {"class_type": "ZQXSigmasToText", "inputs": {"sigmas": ["34", 0]}},
        "40": {"class_type": "LatentUpscaleBy", "inputs": {"samples": ["35", 0], "upscale_method": "bislerp", "scale_by": 1.5}},
        "41": {"class_type": "KSampler", "inputs": {"model": ["21", 0], "seed": 1, "steps": 20, "cfg": 2.5,
                                                      "sampler_name": "euler", "scheduler": "simple",
                                                      "positive": ["7", 0], "negative": ["8", 0], "latent_image": ["40", 0],
                                                      "denoise": 0.45}},
        "42": {"class_type": "VAEDecode", "inputs": {"samples": ["41", 0], "vae": ["3", 0]}},
        "43": {"class_type": "SaveImage", "inputs": {"images": ["42", 0], "filename_prefix": "zqx_qwen2512"}},
    })
    return g


def lora_tools():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "ZQXLoRAConflictReport", "inputs": {"model": ["1", 0], "lora_1": "my_character_zit.safetensors",
                                                                  "lora_2": "my_realism_zit.safetensors", "top_density": 0.2,
                                                                  "sign_stats": True}},
        "3": {"class_type": "ZQXLoRAArithmetic", "inputs": {"model": ["1", 0], "lora_1": "my_character_zit.safetensors",
                                                              "lora_2": "my_realism_zit.safetensors", "mode": "clean_col",
                                                              "lam": 0.5, "w1": 1.0, "w2": 0.7, "density": 0.2,
                                                              "dare_drop": 0.0, "svd_rank": 64, "seed": 0,
                                                              "filename_prefix": "char_clean", "save_dtype": "bfloat16"}},
        "4": {"class_type": "ZQXKLoRA", "inputs": {"model": ["1", 0], "character_lora": "my_character_zit.safetensors",
                                                     "realism_lora": "my_realism_zit.safetensors", "character_strength": 1.0,
                                                     "realism_strength": 0.8, "alpha": 1.5, "beta": 0.5, "pattern": "s",
                                                     "scope": "attention", "other_layers": "both", "allow_unmatched_keys": False}},
        "5": {"class_type": "ZQXBlockSpec", "inputs": {"index": 0, "width": 5, "total_blocks": 30, "mode": "only", "weight": 0.0}},
    }


if __name__ == "__main__":
    for name, g in (("zit_two_pass_api.json", zit()), ("qwen2512_two_pass_api.json", qwen()),
                    ("lora_tools_api.json", lora_tools())):
        with open(os.path.join(HERE, name), "w") as f:
            json.dump(g, f, indent=1)
        print("wrote", name)
