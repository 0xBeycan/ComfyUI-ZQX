"""Test bootstrap: put a ComfyUI checkout and this repo on sys.path, force CPU.

Set COMFYUI_PATH to a ComfyUI checkout (the commit used for the recorded
results is listed in docs/TEST_RESULTS.md).  Tests that need ComfyUI are
skipped with a clear message when it is missing; pure-math tests always run.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "")
HAVE_COMFY = bool(COMFYUI_PATH) and os.path.isdir(os.path.join(COMFYUI_PATH, "comfy"))

if HAVE_COMFY:
    if COMFYUI_PATH not in sys.path:
        sys.path.insert(1, COMFYUI_PATH)
    # comfy.cli_args parses sys.argv at import time when enabled; force CPU mode.
    _argv = sys.argv
    sys.argv = [_argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args  # noqa: F401  (parses --cpu)
    sys.argv = _argv

import pytest  # noqa: E402

requires_comfy = pytest.mark.skipif(not HAVE_COMFY, reason="COMFYUI_PATH not set to a ComfyUI checkout")
