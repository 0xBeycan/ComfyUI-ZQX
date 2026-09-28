"""ComfyUI-ZIT-QIE-Experimental: training-free levers for identity vs. "AI look" on
Z-Image Turbo, Qwen-Image 2512 and Qwen-Image-Edit 2511.  See README.md."""
if __package__:
    # ComfyUI imports custom-node folders as packages (nodes.py::load_custom_node).
    from .zqx.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    # Imported as a plain top-level module (pytest collects the repo root as a Package); there are no
    # nodes to register in that context.  The test-suite imports `zqx` directly.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
