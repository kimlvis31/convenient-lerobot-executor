"""Upgrade the saved input pipeline of an older native Wall-X evaluation checkpoint."""

import json
from pathlib import Path


LEGACY_STEPS = (
    "rename_observations_processor", "to_batch_processor", "wall_x_task_processor",
    "normalizer_processor", "device_processor",
)
PREPROCESSOR = "policy_preprocessor.json"


def is_legacy_processor(pipeline):
    if not isinstance(pipeline, dict) or not isinstance(pipeline.get("steps"), list):
        return False
    return tuple(step.get("registry_name") if isinstance(step, dict) else None
                 for step in pipeline["steps"]) == LEGACY_STEPS


def migrate_processor(saved, current):
    """Use the runtime order, with the checkpoint's original statistics and settings."""
    if not is_legacy_processor(saved):
        return None
    old = {step["registry_name"]: step for step in saved["steps"]}
    names = [step.get("registry_name") for step in current["steps"]]
    if not {"wall_x_prompt", "wall_x_tokenizer"} <= set(names):
        return None
    if any(names.count(name) != 1 for name in old):
        raise ValueError("The installed Wall-X processor cannot preserve all saved processing steps.")
    return {**saved, "steps": [old.get(step.get("registry_name"), step) for step in current["steps"]]}


def prepare_wallx_evaluation(reference, destination, revision=None):
    from lerobot.policies.wall_x import processor_wall_x
    from lerobot.configs import PreTrainedConfig
    from checkpoint_compat import write_processor_view

    if not hasattr(processor_wall_x, "WallXTokenizerStep"):
        return None
    base = Path(reference).expanduser()
    if base.is_dir():
        fetch = lambda name: base / name
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            pipeline_path = Path(hf_hub_download(reference, PREPROCESSOR, revision=revision))
        except EntryNotFoundError:
            return None
        revision = pipeline_path.parent.name

        def fetch(name):
            return Path(hf_hub_download(reference, name, revision=revision))

    saved = json.loads(fetch(PREPROCESSOR).read_text(encoding="utf-8"))
    if not is_legacy_processor(saved):
        return None
    config = PreTrainedConfig.from_pretrained(fetch("config.json").parent)
    if config.type != "wall_x":
        return None
    pre, _ = processor_wall_x.make_wall_x_pre_post_processors(config)
    migrated = migrate_processor(saved, pre.get_config())
    if migrated is None:
        return None
    return write_processor_view(destination, fetch, {PREPROCESSOR: migrated})
