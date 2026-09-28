"""Write the synthetic LoRAs used by run_e2e.py into <ComfyUI>/models/loras."""
import os
import sys

from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lora_utils as lu  # noqa: E402

out = os.path.join(sys.argv[1], "models", "loras")
os.makedirs(out, exist_ok=True)
for kind in ("zimage", "qwen"):
    save_file(lu.make_lora(kind, seed=1, rank=4, fmt="kohya" if kind == "qwen" else "peft", alpha=2.0),
              os.path.join(out, f"zqx_e2e_char_{kind}.safetensors"))
    save_file(lu.make_lora(kind, seed=2, rank=3, fmt="peft"), os.path.join(out, f"zqx_e2e_real_{kind}.safetensors"))
print("ok", out)
