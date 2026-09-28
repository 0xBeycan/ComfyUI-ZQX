CATEGORY = "ZQX Experimental"


def sigma_input(default, tooltip):
    return ("FLOAT", {"default": default, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.0001, "tooltip": tooltip})
