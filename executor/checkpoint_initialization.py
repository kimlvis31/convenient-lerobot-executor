"""Apply native WALL-X precision before allocating its parameters on CUDA."""

from contextlib import contextmanager
from functools import wraps
from inspect import signature
from unittest.mock import patch


@contextmanager
def checkpoint_model_placement(factory, config, args, kwargs):
    """Keep native casting/loading while avoiding a temporary full-float GPU copy."""
    del factory, args, kwargs
    if getattr(config, "type", None) != "wall_x" or not str(getattr(config, "device", "")).startswith("cuda"):
        yield
        return
    from lerobot.policies.wall_x.modeling_wall_x import Qwen2_5_VLMoEForAction

    model_class = Qwen2_5_VLMoEForAction
    cast = getattr(model_class, "to_bfloat16_for_selected_params", None)
    if not callable(cast) or list(signature(cast).parameters) != ["self"]:
        yield
        return
    original = model_class.to

    @wraps(original)
    def move(model, *values, **options):
        target = values[0] if len(values) == 1 and not options else (
            options["device"] if not values and set(options) == {"device"} else None)
        kind = target.split(":", 1)[0] if isinstance(target, str) else getattr(target, "type", None)
        if kind == "cuda":
            parameters = list(model.parameters())
            if parameters and all(parameter.device.type == "cpu" for parameter in parameters):
                print("Applying native WALL-X precision on host before CUDA placement.", flush=True)
                cast(model)
        return original(model, *values, **options)

    with patch.object(model_class, "to", move):
        yield
