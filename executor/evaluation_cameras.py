"""Restore saved camera names when a native evaluation uses a checkpoint."""

import json
from pathlib import Path
import sys


# Raw camera names declared by the corresponding LeRobot environment configs.
ENVIRONMENT_CAMERAS = {
    "robocasa": ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"),
    "vlabench": ("image", "second_image", "wrist_image"),
}


def checkpoint_directory(reference, revision, *, download):
    path = Path(reference).expanduser()
    if path.is_dir():
        return path
    if path.is_absolute() or len(reference.split("/")) != 2:
        return None
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled
        from httpx import TransportError
    except ImportError:
        return None
    processor = "policy_preprocessor.json"
    try:
        if download:
            saved = hf_hub_download(reference, processor, revision=revision)
            path = Path(saved).parent
            hf_hub_download(reference, "config.json", revision=path.name)
        else:
            saved = try_to_load_from_cache(reference, processor, revision=revision)
            if not isinstance(saved, str):
                return None
            path = Path(saved).parent
        return path if (path / "config.json").is_file() else None
    except (ValueError, OSError, HfHubHTTPError, OfflineModeIsEnabled, TransportError):
        return None


def camera_mapping(arguments, *, download=False):
    """Return a verified map, {} if none applies, or None if the checkpoint is pending."""
    options = dict(arg[2:].split("=", 1) for arg in arguments if arg.startswith("--") and "=" in arg)
    kind = options.get("env.type")
    reference = options.get("policy.path")
    if kind not in ENVIRONMENT_CAMERAS or not reference:
        return {}
    if any(key == field or key.startswith(field + ".") for key in options
           for field in ("rename_map", "env.features", "env.features_map", "policy.input_features")):
        return {}
    names = ENVIRONMENT_CAMERAS[kind]
    if kind == "robocasa" and "env.camera_name" in options:
        names = [name.strip() for name in options["env.camera_name"].split(",") if name.strip()]
    cameras = {f"observation.images.{name}" for name in names}
    directory = checkpoint_directory(reference, options.get("policy.pretrained_revision"), download=download)
    if directory is None:
        return None
    try:
        policy = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        processor = json.loads((directory / "policy_preprocessor.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    inputs = policy.get("input_features") if isinstance(policy, dict) else None
    steps = processor.get("steps") if isinstance(processor, dict) else None
    if not isinstance(inputs, dict) or not isinstance(steps, list):
        return {}
    visual = {name for name, spec in inputs.items() if isinstance(spec, dict) and spec.get("type") == "VISUAL"}
    if not visual or visual <= cameras or cameras <= visual:
        return {}

    renames = [step.get("config") for step in steps if isinstance(step, dict)
               and step.get("registry_name") == "rename_observations_processor"]
    if len(renames) != 1 or not isinstance(renames[0], dict):
        return {}
    mapping = renames[0].get("rename_map")
    if not isinstance(mapping, dict):
        return {}
    mapping = {name: target for name, target in mapping.items()
               if name in cameras and isinstance(target, str) and target in visual and name != target}
    renamed = {mapping.get(name, name) for name in cameras}
    if len(renamed) != len(cameras) or not (visual <= renamed or renamed <= visual):
        return {}
    return mapping


def restore_camera_arguments():
    """Use the now-available checkpoint for uncached Hub and train-then-evaluate jobs."""
    mapping = camera_mapping(sys.argv[1:], download=True)
    if mapping:
        argument = "--rename_map=" + json.dumps(mapping, separators=(",", ":"))
        sys.argv.append(argument)
        print(f"Restored checkpoint camera names: {argument}", flush=True)


def main():
    from lerobot.scripts.lerobot_eval import main as evaluate

    restore_camera_arguments()
    return evaluate()


if __name__ == "__main__":
    sys.exit(main())
