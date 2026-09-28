CATEGORY_ROOT = "ZQX"
CAT_ATTENTION = "ZQX/attention"
CAT_LORA = "ZQX/lora"
CAT_SAMPLING = "ZQX/sampling"
CAT_GUIDANCE = "ZQX/guidance"
CAT_SCORING = "ZQX/scoring"
CAT_TOOLS = "ZQX/tools"
CAT_EDIT = "ZQX/model-edit"


def sigma_input(default, tooltip):
    return ("FLOAT", {"default": default, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.0001, "tooltip": tooltip})
