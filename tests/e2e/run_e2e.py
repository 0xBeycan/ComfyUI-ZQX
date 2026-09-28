"""End-to-end API run against a live ComfyUI (see docs/TEST_RESULTS.md).

  python tests/e2e/run_e2e.py http://127.0.0.1:8199

Requires tests/e2e/zqx_e2e_stubs symlinked into ComfyUI/custom_nodes and the synthetic LoRAs
zqx_e2e_char_<kind>.safetensors / zqx_e2e_real_<kind>.safetensors in models/loras (written by
tests/e2e/make_e2e_loras.py)."""
import json
import sys
import time
import urllib.request


def graph(kind):
    cfg = 1.0 if kind == "zimage" else 2.5
    g = {
        "1": {"class_type": "ZQXTestTinyModel", "inputs": {"kind": kind, "layers": 2}},
        "2": {"class_type": "ZQXTestRandomCond", "inputs": {"kind": kind, "seed": 1}},
        "3": {"class_type": "ZQXTestRandomCond", "inputs": {"kind": kind, "seed": 2}},
        "4": {"class_type": "ZQXTestLatent", "inputs": {"kind": kind, "h": 8, "w": 8, "seed": 3, "zero": True}},
        "5": {"class_type": "ZQXTestLatent", "inputs": {"kind": kind, "h": 12, "w": 10, "seed": 5, "zero": False}},
        "10": {"class_type": "ZQXScheduledLoRA", "inputs": {"model": ["1", 0], "lora_name": f"zqx_e2e_real_{kind}.safetensors",
                                                              "strength_early": 1.0, "strength_late": 0.3, "sigma_hi": 0.8,
                                                              "sigma_lo": 0.6, "block_weights": "0:1, 1:0.5", "allow_unmatched_keys": False}},
        "11": {"class_type": "ZQXKLoRA", "inputs": {"model": ["10", 0], "character_lora": f"zqx_e2e_char_{kind}.safetensors",
                                                      "realism_lora": f"zqx_e2e_real_{kind}.safetensors", "character_strength": 1.0,
                                                      "realism_strength": 0.5, "alpha": 1.5, "beta": 0.5, "pattern": "s",
                                                      "scope": "attention", "other_layers": "both", "allow_unmatched_keys": False}},
        "12": {"class_type": "ZQXReferenceAttention", "inputs": {"model": ["11", 0], "reference": ["5", 0], "weight": 1.0,
                                                                   "sigma_start": 0.95, "sigma_end": 0.2, "blocks": "all",
                                                                   "position_mode": "frame", "capture_mode": "noised",
                                                                   "cache_sigma": 0.0, "ref_sigma_mult": 0.9, "key_scale": 1.1,
                                                                   "token_dropout": 0.3, "inject_uncond": True, "noise_seed": 0}},
        "13": {"class_type": "ZQXCADS", "inputs": {"model": ["12", 0], "tau1": 0.8, "tau2": 1.0, "noise_scale": 0.1, "psi": 1.0,
                                                     "relative_noise": True, "apply_to": "cond_and_uncond", "seed": 0}},
        # pass 1: SamplerCustomAdvanced with low-frequency noise + sigma-split guider
        "20": {"class_type": "ZQXLowFreqNoise", "inputs": {"noise_seed": 7, "reference": ["5", 0], "strength": 0.5, "cutoff": 0.2,
                                                             "filter": "gaussian", "butterworth_order": 4}},
        "21": {"class_type": "ZQXSigmaSplitGuider", "inputs": {"model": ["13", 0], "positive": ["2", 0], "negative": ["3", 0],
                                                                 "switch_sigma": 0.8, "cfg_early": 2.0, "cfg_late": cfg}},
        "22": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "23": {"class_type": "BasicScheduler", "inputs": {"model": ["13", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0}},
        "24": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["20", 0], "guider": ["21", 0], "sampler": ["22", 0],
                                                                   "sigmas": ["23", 0], "latent_image": ["4", 0]}},
        "25": {"class_type": "ZQXSigmasToText", "inputs": {"sigmas": ["23", 0]}},
        # latent upscale + pass 2 with the regular KSampler (img2img, denoise 0.5)
        "30": {"class_type": "LatentUpscaleBy", "inputs": {"samples": ["24", 0], "upscale_method": "bilinear", "scale_by": 1.5}},
        "31": {"class_type": "KSampler", "inputs": {"model": ["13", 0], "seed": 9, "steps": 4, "cfg": cfg, "sampler_name": "euler",
                                                      "scheduler": "simple", "positive": ["2", 0], "negative": ["3", 0],
                                                      "latent_image": ["30", 0], "denoise": 0.5}},
        "32": {"class_type": "ZQXTestLatentStats", "inputs": {"latent": ["31", 0], "tag": f"{kind} two-pass"}},
        # LoRA tools
        "40": {"class_type": "ZQXLoRAArithmetic", "inputs": {"model": ["1", 0], "lora_1": f"zqx_e2e_char_{kind}.safetensors",
                                                               "lora_2": f"zqx_e2e_real_{kind}.safetensors", "mode": "clean_col",
                                                               "lam": 1.0, "w1": 1.0, "w2": 1.0, "density": 0.2, "dare_drop": 0.0,
                                                               "svd_rank": 8, "seed": 0, "filename_prefix": f"e2e_{kind}",
                                                               "save_dtype": "float32"}},
        "41": {"class_type": "ZQXLoRAConflictReport", "inputs": {"model": ["1", 0], "lora_1": f"zqx_e2e_char_{kind}.safetensors",
                                                                   "lora_2": f"zqx_e2e_real_{kind}.safetensors", "top_density": 0.2,
                                                                   "sign_stats": True}},
        "42": {"class_type": "ZQXBlockSpec", "inputs": {"index": 1, "width": 1, "total_blocks": 2, "mode": "only", "weight": 0.0}},
    }
    return g


def reload_graph(kind, lora_rel):
    """Second prompt: the LoRA written by ZQXLoRAArithmetic is loaded again through ZQXScheduledLoRA
    (with a block-ablation spec from ZQXBlockSpec) and sampled."""
    cfg = 1.0 if kind == "zimage" else 2.5
    g = {k: v for k, v in graph(kind).items() if k in ("1", "2", "3", "4", "42")}
    g["43"] = {"class_type": "ZQXScheduledLoRA", "inputs": {"model": ["1", 0], "lora_name": lora_rel, "strength_early": 1.0,
                                                          "strength_late": 1.0, "sigma_hi": 1.0, "sigma_lo": 0.0,
                                                          "block_weights": ["42", 1], "allow_unmatched_keys": False}}
    g["44"] = {"class_type": "KSampler", "inputs": {"model": ["43", 0], "seed": 9, "steps": 3, "cfg": cfg, "sampler_name": "euler",
                                                  "scheduler": "simple", "positive": ["2", 0], "negative": ["3", 0],
                                                  "latent_image": ["4", 0], "denoise": 1.0}}
    g["45"] = {"class_type": "ZQXTestLatentStats", "inputs": {"latent": ["44", 0], "tag": f"{kind} merged-lora reload"}}
    return g


def run(url, g, expect):
    r = post(url, "/prompt", {"prompt": g})
    pid = r["prompt_id"]
    for _ in range(600):
        h = json.loads(urllib.request.urlopen(f"{url}/history/{pid}").read())
        if pid in h:
            break
        time.sleep(0.5)
    entry = h[pid]
    status = entry["status"]
    ok = status.get("status_str") == "success"
    print("status:", status.get("status_str"), "completed:", status.get("completed"))
    if not ok:
        for m in status.get("messages", []):
            if m[0] == "execution_error":
                print(json.dumps(m[1], indent=1)[:3000])
    missing = [n for n in expect if n not in entry.get("outputs", {})]
    if missing:
        print("MISSING OUTPUTS:", missing)
        ok = False
    for nid, out in entry.get("outputs", {}).items():
        for t in out.get("text", []):
            print(f"  [{nid}]", t.replace("\n", "\n        ")[:1500])
    return ok, entry


def new_nodes_graph(kind):
    """Third prompt: every node added in 0.2.0 in one graph."""
    cfg = 1.0 if kind == "zimage" else 2.5
    g = {k: v for k, v in graph(kind).items() if k in ("1", "2", "3", "4", "5", "22", "23")}
    g["23"] = dict(g["23"], inputs=dict(g["23"]["inputs"], model=["55", 0]))
    char, real = f"zqx_e2e_char_{kind}.safetensors", f"zqx_e2e_real_{kind}.safetensors"
    g.update({
        "50": {"class_type": "ZQXSpatialLoRA", "inputs": {"model": ["1", 0], "lora_name": real, "strength_early": 1.0,
                                                            "strength_late": 0.5, "sigma_hi": 1.0, "sigma_lo": 0.0,
                                                            "invert_mask": True, "text_weight": 1.0, "nonspatial_weight": 0.0,
                                                            "other_weight": 1.0, "allow_unmatched_keys": False,
                                                            "mask": ["60", 0]}},
        "51": {"class_type": "ZQXLoRAGuidance", "inputs": {"model": ["50", 0], "lora_name": char, "strength": 1.0,
                                                             "w_early": 0.3, "w_late": 1.5, "sigma_hi": 0.8, "sigma_lo": 0.5,
                                                             "allow_unmatched_keys": False}},
        "52": {"class_type": "ZQXDiTPAG", "inputs": {"model": ["51", 0], "scale": 1.0, "sigma_start": 1.0, "sigma_end": 0.5,
                                                       "blocks": "mid", "apply_to": "cond_only"}},
        "53": {"class_type": "ZQXActivationSteering", "inputs": {"model": ["52", 0], "towards": ["2", 0], "away": ["3", 0],
                                                                   "alpha": 0.3, "sigma_start": 1.0, "sigma_end": 0.6,
                                                                   "blocks": "mid", "mode": "mean", "apply_to": "all"}},
        "54": {"class_type": "ZQXUCETextEdit", "inputs": {"model": ["53", 0], "source": ["3", 0], "target": ["2", 0],
                                                            "pairing": "mean", "lam": 0.1, "strength": 1.0}},
        "55": {"class_type": "ZQXReferenceAttention", "inputs": {"model": ["54", 0], "reference": ["5", 0], "weight": 1.0,
                                                                   "sigma_start": 0.95, "sigma_end": 0.2, "blocks": "all",
                                                                   "position_mode": "matched", "capture_mode": "noised",
                                                                   "cache_sigma": 0.0, "ref_sigma_mult": 1.0, "key_scale": 1.0,
                                                                   "token_dropout": 0.0, "inject_uncond": True, "noise_seed": 0,
                                                                   "match_threshold": 0.0, "match_mutual": True}},
        "60": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 64, "height": 64}},
        "61": {"class_type": "ZQXTestFakeVAE", "inputs": {}},
        "62": {"class_type": "ZQXScorerBackgroundSharpness", "inputs": {"weight": 1.0, "border": 0.2}},
        "63": {"class_type": "ZQXScorerBackgroundSharpness", "inputs": {"weight": -0.5, "border": 0.3}},
        "64": {"class_type": "ZQXScorerCombine", "inputs": {"scorer_1": ["62", 0], "scorer_2": ["63", 0]}},
        "65": {"class_type": "RandomNoise", "inputs": {"noise_seed": 3}},
        "66": {"class_type": "CFGGuider", "inputs": {"model": ["55", 0], "positive": ["2", 0], "negative": ["3", 0], "cfg": cfg}},
        "67": {"class_type": "ZQXSeedSearch", "inputs": {"noise": ["65", 0], "guider": ["66", 0], "sampler": ["22", 0],
                                                           "sigmas": ["23", 0], "latent_image": ["4", 0], "vae": ["61", 0],
                                                           "scorer": ["64", 0], "n_candidates": 4, "probe_steps": 1, "keep": 2}},
        "68": {"class_type": "ZQXTestLatentStats", "inputs": {"latent": ["67", 0], "tag": f"{kind} seed search"}},
        "70": {"class_type": "ZQXLoRASurgery", "inputs": {"model": ["1", 0], "lora_name": char, "strength": 1.0,
                                                            "block_weights": "0:0.5", "drop_kinds": "io", "max_rank": 2,
                                                            "energy_keep": 1.0, "spectrum_power": 0.8, "power_preserve": "top",
                                                            "dare_drop": 0.0, "seed": 0, "filename_prefix": f"e2e_{kind}_surg",
                                                            "save_dtype": "float32"}},
        "71": {"class_type": "ZQXLoRACommonSubspace", "inputs": {"model": ["1", 0], "lora_1": char, "lora_2": real,
                                                                   "lora_3": "None", "lora_4": "None", "common_rank": 2,
                                                                   "mode": "clean_target", "target_lora": char, "lam": 1.0,
                                                                   "filename_prefix": f"e2e_{kind}_common", "save_dtype": "float32"}},
        "72": {"class_type": "ZQXTestFakeClipVision", "inputs": {}},
        "73": {"class_type": "ZQXRealismLoRAAblation", "inputs": {
            "model": ["1", 0], "realism_lora": real, "realism_strength": 1.0, "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "vae": ["61", 0], "identity_scorer": ["62", 0], "clip_vision": ["72", 0],
            "seeds": "1, 2", "steps": 2, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "units": "blocks+svd",
            "svd_blocks": 1, "svd_components": 1, "min_identity_gain": 0.0, "max_realism_drop": 1.0,
            "filename_prefix": f"e2e_{kind}_ablation", "save_dtype": "float32"}},
        "74": {"class_type": "ZQXPoseBank", "inputs": {"bank_folder": "zqx_e2e_pose_bank", "tags": "", "width": 64,
                                                         "height": 96, "aspect_tolerance": 0.2, "resize": "crop_to_size", "seed": 1}},
        "75": {"class_type": "PreviewImage", "inputs": {"images": ["74", 0]}},
    })
    return g


def post(url, path, data):
    req = urllib.request.Request(url + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def main(url):
    ok = True
    for kind in ("zimage", "qwen"):
        print("==", kind, "two-pass graph")
        o, entry = run(url, graph(kind), ["25", "32", "40", "41"])
        ok &= o
        saved = entry["outputs"]["40"]["text"][0].split("\n")[0].split("saved: ")[1]
        rel = "zqx/" + saved.split("/zqx/")[1]
        print("==", kind, "reload merged LoRA", rel)
        o, _ = run(url, reload_graph(kind, rel), ["45"])
        ok &= o
        print("==", kind, "new nodes graph")
        o, _ = run(url, new_nodes_graph(kind), ["67", "68", "70", "71", "73", "75"])
        ok &= o
    print("E2E", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8199") else 1)
