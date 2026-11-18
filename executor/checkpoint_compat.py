"""Prepare compatible, temporary copies of legacy native LeRobot checkpoints."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import re


LEGACY_POLICIES = frozenset({"diffusion", "vqbet"})
PROCESSORS = ("policy_preprocessor.json", "policy_postprocessor.json")
WEIGHTS = "model.safetensors"
RESULT_PREFIX = "MANAGER_CHECKPOINT "
NORMALIZER_PREFIXES = (
    "normalize_inputs.buffer_", "normalize_targets.buffer_", "unnormalize_outputs.buffer_",
)


def checkpoint_candidate(reference, model, stage=None):
    """Filter explicit native policy references without network or model imports."""
    molmo_eval = model == "molmoact2" and stage == "eval"
    wallx_eval = model == "wall_x" and stage == "eval"
    if model not in LEGACY_POLICIES and not (molmo_eval or wallx_eval):
        return False
    path = Path(reference).expanduser()
    if path.is_dir():
        if wallx_eval:
            from wallx_compat import is_legacy_processor

            try:
                processor = json.loads((path / PROCESSORS[0]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            return is_legacy_processor(processor)
        if molmo_eval:
            from molmo_compat import LEGACY_FIELDS

            config = _read_config(path / "config.json", model)
            return config is not None and LEGACY_FIELDS <= config.keys()
        return (path / "config.json").is_file() and (path / WEIGHTS).is_file() and not any(
            (path / name).exists() for name in PROCESSORS
        )
    return not path.is_absolute() and bool(re.fullmatch(r"[\w][\w.-]*/[\w][\w.-]*", str(reference), re.ASCII))


def validate_processor_states(reference):
    """Report absent state files explicitly listed by a local processor pipeline."""
    directory = Path(reference).expanduser()
    if not directory.is_dir():
        return
    missing = set()
    for name in PROCESSORS:
        pipeline = directory / name
        try:
            config = json.loads(pipeline.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            continue
        steps = config.get("steps") if isinstance(config, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            filename = step.get("state_file") if isinstance(step, dict) else None
            if isinstance(filename, str) and filename and not (directory / filename).is_file():
                missing.add(filename)
    if missing:
        names = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"Local checkpoint processor state is missing ({directory}): {names}. "
            "Download the missing files or use the checkpoint's Hugging Face repository ID."
        )


def _read_config(path, model):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("type") == model else None


def _has_legacy_buffers(keys):
    return any(key.startswith(NORMALIZER_PREFIXES) for key in keys)


def _local_source(directory, model):
    if not checkpoint_candidate(directory, model):
        return None
    config = _read_config(directory / "config.json", model)
    if config is None:
        return None
    from safetensors import safe_open

    weights = directory / WEIGHTS
    with safe_open(weights, framework="pt", device="cpu") as header:
        if _has_legacy_buffers(header.keys()):
            return config, weights
    return None


def _hub_source(reference, model, revision):
    from huggingface_hub import HfApi, constants, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled, SafetensorsParsingError
    from httpx import TransportError

    def cached_source():
        config_path = try_to_load_from_cache(reference, "config.json", revision=revision)
        if isinstance(config_path, str):
            return _local_source(Path(config_path).parent, model)
        return None

    if constants.HF_HUB_OFFLINE:
        return cached_source()

    api = HfApi()
    try:
        info = api.model_info(reference, revision=revision, timeout=10)
        names = {entry.rfilename for entry in info.siblings or ()}
        if not {"config.json", WEIGHTS} <= names or any(name in names for name in PROCESSORS):
            return None
        revision = info.sha or revision
        config_path = Path(hf_hub_download(reference, "config.json", revision=revision))
        config = _read_config(config_path, model)
        if config is None:
            return None
        if (config_path.parent / WEIGHTS).is_file():
            return _local_source(config_path.parent, model)
        header = api.parse_safetensors_file_metadata(reference, WEIGHTS, revision=revision)
        if not _has_legacy_buffers(header.tensors):
            return None
        return config, Path(hf_hub_download(reference, WEIGHTS, revision=revision))
    except (HfHubHTTPError, OfflineModeIsEnabled, SafetensorsParsingError, TransportError):
        # The ordinary LeRobot loader still receives the original reference if
        # this optional compatibility check cannot read Hub metadata.
        return cached_source()


def _check_normalization(config, stats):
    mapping = config.get("normalization_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Legacy checkpoint has no declared normalization mapping.")
    modes = {"IDENTITY": (), "MIN_MAX": ("min", "max"), "MEAN_STD": ("mean", "std")}
    for section in ("input_features", "output_features"):
        for name, feature in (config.get(section) or {}).items():
            mode = mapping.get(feature["type"], "IDENTITY")
            if mode not in modes or not all(key in stats.get(name, {}) for key in modes[mode]):
                raise ValueError(f"Legacy checkpoint has incomplete or unsupported normalization for {name}.")


def write_processor_view(destination, fetch, replacements):
    """Build a private checkpoint view while retaining the published weights and processor state."""
    destination = Path(destination)
    inputs = {}
    for name in ("config.json", *PROCESSORS):
        if name in replacements:
            document = replacements[name]
        else:
            source = fetch(name)
            inputs[name] = source
            document = json.loads(source.read_text(encoding="utf-8"))
        if name in PROCESSORS:
            for step in document["steps"]:
                state = step.get("state_file")
                if state is None:
                    continue
                if not isinstance(state, str):
                    raise ValueError(f"Invalid processor state filename: {state!r}")
                parts = PurePosixPath(state)
                if (parts.suffix != ".safetensors" or state == WEIGHTS or parts.is_absolute()
                        or ".." in parts.parts or chr(92) in state or ":" in state):
                    raise ValueError(f"Invalid processor state filename: {state!r}")
                inputs[state] = fetch(state)
    inputs[WEIGHTS] = fetch(WEIGHTS)
    destination.mkdir(mode=0o700)
    for name, document in replacements.items():
        (destination / name).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    for name, source in inputs.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = source.resolve(strict=True)
        try:
            os.link(source, target)
        except OSError:
            try:
                target.symlink_to(source)
            except OSError:
                shutil.copyfile(source, target)
    return destination


def prepare_checkpoint(reference, model, destination, revision=None, stage=None):
    """Return an adapted local path, or None when the original should be used."""
    if not checkpoint_candidate(reference, model, stage):
        return None
    if model == "wall_x":
        from wallx_compat import prepare_wallx_evaluation

        return prepare_wallx_evaluation(reference, destination, revision)
    if model == "molmoact2":
        from molmo_compat import prepare_molmo_evaluation

        return prepare_molmo_evaluation(reference, destination, revision)
    directory = Path(reference).expanduser()
    source = _local_source(directory, model) if directory.is_dir() else _hub_source(reference, model, revision)
    if source is None:
        return None

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import migrate_policy_normalization as migration
    from safetensors.torch import load_file, save_file

    config, weights = source
    state = load_file(str(weights), device="cpu")
    stats = migration.extract_normalization_stats(state)
    _check_normalization(config, stats)
    if model == "vqbet":
        # This older field has no matching parameter in the current VQ-BeT.
        config.pop("mlp_hidden_dim", None)

    destination = Path(destination)
    destination.mkdir(mode=0o700)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    # Decode the JSON with LeRobot's config loader. The legacy migrator passes
    # JSON lists straight to tuple fields and then fails to save the config.
    policy_config = PreTrainedConfig.from_pretrained(destination)
    for processor in make_pre_post_processors(policy_cfg=policy_config, dataset_stats=stats):
        processor.save_pretrained(destination)

    clean = migration.clean_state_dict(migration.remove_normalization_layers(state))
    save_file(clean, str(destination / WEIGHTS), metadata={"format": "pt"})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--model", required=True, choices=sorted(LEGACY_POLICIES | {"molmoact2", "wall_x"}))
    parser.add_argument("--stage", choices=["train", "eval"])
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_checkpoint(args.policy, args.model, args.output, args.revision, args.stage)
    print(RESULT_PREFIX + json.dumps(str(result) if result is not None else None))


if __name__ == "__main__":
    main()
