"""Validate examples/*_api.json against a live ComfyUI /object_info: node types exist, every required input is
given, links point to existing nodes/outputs with matching types, combo values are valid (except file names,
which depend on the user's model folders).  python tests/e2e/validate_examples.py http://127.0.0.1:8199"""
import glob
import json
import os
import sys
import urllib.request

FILE_INPUTS = {"unet_name", "clip_name", "vae_name", "lora_name", "lora_1", "lora_2", "lora_3", "lora_4", "target_lora",
               "character_lora", "realism_lora", "image", "name"}


def main(url):
    info = json.loads(urllib.request.urlopen(url + "/object_info").read())
    ok = True
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples")
    for fn in sorted(glob.glob(os.path.join(root, "*_api.json"))):
        g = json.load(open(fn))
        errs = []
        for nid, node in g.items():
            ct = node["class_type"]
            if ct not in info:
                errs.append(f"{nid}: unknown node {ct}")
                continue
            spec = info[ct]["input"]
            req, opt = spec.get("required", {}), spec.get("optional", {})
            for k in req:
                if k not in node["inputs"]:
                    errs.append(f"{nid} {ct}: missing required input {k}")
            for k, v in node["inputs"].items():
                s = req.get(k) or opt.get(k)
                if s is None:
                    errs.append(f"{nid} {ct}: unknown input {k}")
                    continue
                typ = s[0]
                if isinstance(v, list):
                    src = g.get(v[0])
                    if src is None:
                        errs.append(f"{nid}: link to missing node {v[0]}")
                        continue
                    out_t = info[src["class_type"]]["output"][v[1]]
                    exp = typ if isinstance(typ, str) else "COMBO"
                    if out_t != exp and not (exp == "COMBO" and out_t == "COMBO"):
                        errs.append(f"{nid} {ct}.{k}: link type {out_t} != {exp}")
                elif isinstance(typ, list) and k not in FILE_INPUTS and v not in typ:
                    errs.append(f"{nid} {ct}.{k}: {v!r} not in combo")
                elif typ == "COMBO" and k not in FILE_INPUTS and v not in s[1].get("options", []):
                    errs.append(f"{nid} {ct}.{k}: {v!r} not in combo options")
        print(os.path.basename(fn), "OK" if not errs else "ERRORS")
        for e in errs:
            print("  ", e)
        ok &= not errs
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8199") else 1)
