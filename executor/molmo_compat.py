"""Restore the inference topology recorded by older native MolmoAct2 checkpoints."""

import json
from dataclasses import fields
from pathlib import Path


LEGACY_FLAGS = ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only")
LEGACY_FIELDS = frozenset((*LEGACY_FLAGS, "model_dtype"))


def evaluation_config(config):
    """Migrate only the known full-weight topology, only for an incompatible loader."""
    if config.get("type") != "molmoact2" or not LEGACY_FIELDS <= config.keys():
        return None
    if any(config[key] is not False for key in LEGACY_FLAGS):
        raise ValueError("Legacy MolmoAct2 adapter or frozen-expert settings require their original LeRobot loader.")
    dtype = config["model_dtype"]
    if (dtype not in ("bfloat16", "float32") or config.get("dtype", dtype) != dtype
            or config.get("train_mode_vlm", "fft") != "fft"):
        raise ValueError("Conflicting legacy MolmoAct2 precision or training topology.")
    migrated = {key: value for key, value in config.items() if key not in LEGACY_FIELDS}
    return {**migrated, "dtype": dtype, "train_mode_vlm": "fft"}


def prepare_molmo_evaluation(reference, destination, revision=None):
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    supported = {field.name for field in fields(MolmoAct2Config)}
    if LEGACY_FIELDS & supported or not {"dtype", "train_mode_vlm"} <= supported:
        return None

    base = Path(reference).expanduser()
    local = base.is_dir()
    if local:
        config_path = base / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(reference, "config.json", revision=revision))
        # Hub resolves a branch or tag into an immutable snapshot directory.
        revision = config_path.parent.name
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not LEGACY_FIELDS <= source_config.keys():
        return None
    from draccus.utils import DecodingError
    from lerobot.configs import PreTrainedConfig

    try:
        PreTrainedConfig.from_pretrained(config_path.parent)
    except DecodingError:
        pass
    else:
        # Newer loaders may support old names without exposing dataclass fields.
        return None
    config = evaluation_config(source_config)
    if config is None:
        return None

    def fetch(name):
        if local:
            return base / name
        return Path(hf_hub_download(reference, name, revision=revision))

    from checkpoint_compat import write_processor_view

    return write_processor_view(destination, fetch, {"config.json": config})
