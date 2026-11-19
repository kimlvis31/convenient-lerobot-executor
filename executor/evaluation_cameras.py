"""Restore saved camera names when a native evaluation uses a checkpoint."""

from contextlib import contextmanager
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys


# Raw camera names declared by the corresponding LeRobot environment configs.
ENVIRONMENT_CAMERAS = {
    "libero": ("image", "image2"),
    "robocasa": ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"),
    "robotwin": ("head_camera", "left_camera", "right_camera"),
    "vlabench": ("image", "second_image", "wrist_image"),
}

# Verified camera positions for checkpoints that save an empty rename step.
# LIBERO maps the eye-in-hand camera to image2. The published RoboTwin
# LingBot policy uses its head and two wrist cameras in this order.
POLICY_CAMERA_ALIASES = {
    "libero": ("gaussian_actor", {"image": "image", "image2": "wrist_image"}),
    "robotwin": ("lingbot_vla_v2", {
        "head_camera": "cam_high", "left_camera": "cam_left_wrist", "right_camera": "cam_right_wrist",
    }),
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


def camera_mapping(arguments, *, download=False, model=None):
    """Return a verified map, {} if none applies, or None if the checkpoint is pending."""
    options = dict(arg[2:].split("=", 1) for arg in arguments if arg.startswith("--") and "=" in arg)
    kind = options.get("env.type")
    reference = options.get("policy.path")
    if kind not in ENVIRONMENT_CAMERAS or not reference:
        return {}
    aliases = POLICY_CAMERA_ALIASES.get(kind)
    if aliases and model is not None and model != aliases[0] and (kind, model) != ("robotwin", "smolvla"):
        return {}
    overrides = ("rename_map", "env.features", "env.features_map", "policy.input_features")
    if aliases:
        overrides += ("env.camera_name", "env.camera_names", "env.camera_name_mapping")
    if any(key == field or key.startswith(field + ".") for key in options
           for field in overrides):
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
    if not mapping and aliases and policy.get("type") == aliases[0]:
        expected = {f"observation.images.{src}": f"observation.images.{dst}" for src, dst in aliases[1].items()}
        if visual == set(expected.values()):
            mapping = expected
    if kind == "robotwin" and policy.get("type") == "smolvla":
        # RoboTwin datasets call the physical head and wrist cameras by their ALOHA names.
        # The SmolVLA processor maps those dataset names to its policy camera names.
        sources = {f"observation.images.{src}": f"observation.images.{alias}" for src, alias in aliases[1].items()}
        complete = set(mapping) == set(sources.values()) and all(isinstance(value, str) for value in mapping.values())
        if complete and set(mapping.values()) == visual:
            mapping = {src: mapping[alias] for src, alias in sources.items()}
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


def robotwin_directory():
    """Check the official simulator repository without importing its native extensions."""
    configured = os.environ.get("ROBOTWIN_ROOT")
    if configured:
        roots = [Path(configured).expanduser().resolve()]
    else:
        package = find_spec("envs")
        paths = package.submodule_search_locations if package else ()
        roots = [Path(path).resolve().parent for path in paths or ()]
    for root in roots:
        files = ("envs/_base_task.py", "task_config/demo_clean.yml", "assets/objects/objaverse/list.json")
        if all((root / name).is_file() for name in files) and any(root.glob("assets/embodiments/*/config.yml")):
            return root
    raise FileNotFoundError("Set ROBOTWIN_ROOT to the installed RoboTwin repository with its official task, robot and object assets.")


@contextmanager
def simulator_directory(arguments):
    split = json.loads(os.environ.get("CONVLEX_SIMULATOR_WORKER") or "{}")
    if "--env.type=robotwin" not in arguments or split.get("kind") == "robotwin":
        yield
        return
    # The simulator reads its assets relative to the working directory during import and rollout.
    root = robotwin_directory()
    previous = Path.cwd()
    os.chdir(root)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        os.chdir(previous)


def main():
    from lerobot.scripts.lerobot_eval import main as evaluate
    from robomme_split import simulator_context
    from simulator_worker import simulator_context as source_context

    restore_camera_arguments()
    with simulator_directory(sys.argv[1:]), simulator_context(), source_context():
        return evaluate()


if __name__ == "__main__":
    sys.exit(main())
