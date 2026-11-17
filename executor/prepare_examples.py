#!/usr/bin/env python3
"""Prepare the public TinyVLA exercise assets in the convLexecutor checkout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen

from runner import available_environments, conda_command, conda_executable, resolve_environment

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "datasets/example"
TINYVLA = REPO / "externals/TinyVLA"
SETTINGS = {"task": ["libero_object", 1], "updates": 4, "batch": 1,
            "lr": 0.0001, "seed": 0, "save_freq": 2}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def download(base_url, name, target, expected_size, expected_sha=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256() if expected_sha else None
    print(f"Downloading {name} ({expected_size:,} bytes)", flush=True)
    with urlopen(f"{base_url}/{name}", timeout=60) as response, target.open("xb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            if digest:
                digest.update(block)
    if target.stat().st_size != expected_size or (digest and digest.hexdigest() != expected_sha):
        raise ValueError(f"The downloaded file does not match the pinned release ({name}).")


def verify_base(path, spec):
    if path.stat().st_size != spec["size"]:
        raise ValueError(f"The model file has an unexpected size ({path}).")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"The model file does not match the pinned revision ({path}).")


def prepare_weights(workspace, spec):
    model = workspace / "models/Llava-Pythia-700M"
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"Model configuration not found ({model}).")
    url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}"
    for asset in (spec, spec["tokenizer"]):
        name = asset["file"]
        target = model / name
        if target.is_file():
            verify_base(target, asset)
            continue
        with tempfile.TemporaryDirectory(prefix=".prepare-model-", dir=model) as temporary:
            staging = Path(temporary) / name
            download(url, name, staging, asset["size"], asset["sha256"])
            # A hard link publishes the complete file without replacing another file.
            try:
                os.link(staging, target)
            except FileExistsError:
                verify_base(target, asset)
    print(f"Verified the base model at {model}", flush=True)
    return model


def publish_directory(source, target):
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output directory already exists ({target}).")
    source.rename(target)


def prepare_dataset(dataset, selection, files):
    if dataset.exists():
        if read_json(dataset / "selection.json") != selection:
            raise ValueError(f"Existing dataset has a different training split ({dataset}).")
        for name, size in files.items():
            if (dataset / name).stat().st_size != size:
                raise ValueError(f"Existing dataset is missing data from the pinned revision ({name}).")
        print(f"Using the example dataset at {dataset}", flush=True)
        return
    dataset.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{selection['dataset_repo']}/resolve/{selection['dataset_revision']}"
    with tempfile.TemporaryDirectory(prefix=".prepare-dataset-", dir=dataset.parent) as temporary:
        staging = Path(temporary) / "libero"
        staging.mkdir()
        for name, size in files.items():
            download(url, name, staging / name, size)
        write_json(staging / "selection.json", selection)
        publish_directory(staging, dataset)


def verify_resume(run_model, state, base, trainer, loader):
    program = """
from pathlib import Path
import hashlib
import sys
import torch

state, model, trainer, loader = map(Path, sys.argv[1:])
checkpoint = torch.load(state, map_location="cpu", weights_only=True)
metadata = checkpoint["metadata"]
files = sorted([*model.glob("*.safetensors"), *model.glob("*.json")])
expected_files = [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files]
valid = (checkpoint["step"] == 2 and checkpoint["optimizer"]["state"] and
         metadata["model_source"] == str(model) and metadata["model_files"] == expected_files and
         metadata["trainer_sha256"] == hashlib.sha256(trainer.read_bytes()).hexdigest() and
         metadata["loader_sha256"] == hashlib.sha256(loader.read_text(encoding="utf-8").encode()).hexdigest() and
         metadata["updates_per_task"] == 4)
if not valid:
    raise ValueError(f"The resume state does not match this installation: {state}")
"""
    run_model(["-B", "-c", program, *map(str, (state, base, trainer, loader))])


def verify_model(fixture, selection):
    if read_json(fixture / "selection.json") != selection:
        raise ValueError(f"The model fixture has a different training split ({fixture}).")
    model = fixture / "train/pretrained_model"
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors", "action_head.safetensors"):
        if (model / name).stat().st_size <= 0:
            raise ValueError(f"Model fixture file is empty ({model / name}).")


def prepare_checkpoints(root, checkpoint_root, dataset, selection, workspace, base, native, model, conda, model_env):
    trainer = workspace / "training/train_head.py"
    loader = workspace / "runtime/tinyvla_loader.py"
    trained = checkpoint_root / "tinyvla_checkpoint"
    resumable = checkpoint_root / "tinyvla_resume"
    state_name = "train/checkpoints/training_state.pt"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-examples-", dir=checkpoint_root) as temporary:
        stage = Path(temporary)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
               "HF_HOME": str(stage / "cache"), "HF_DATASETS_CACHE": str(stage / "cache/datasets")}

        def run_model(arguments):
            command = conda_command(conda, model_env, [model["python"], *arguments])
            subprocess.run(command, cwd=root, env=env, check=True)

        if trained.exists():
            verify_model(trained, selection)
        if resumable.exists():
            if read_json(resumable / "selection.json") != selection:
                raise ValueError(f"The resume fixture has a different training split ({resumable}).")
            saved = read_json(resumable / "settings.json")["train"]
            if any(saved.get(key) != value for key, value in SETTINGS.items()):
                raise ValueError(f"The resume fixture uses different training settings ({resumable}).")
            verify_resume(run_model, resumable / state_name, base, trainer, loader)
        if trained.exists() and resumable.exists():
            print("Using the existing verified example checkpoints", flush=True)
            return

        training_run = stage / "training"
        training_run.mkdir()
        output = training_run / "train"
        run_model(["-B", "-u", str(trainer), "--workspace", str(workspace), "--loader", str(loader),
                   "--data", str(dataset / "selection.json"), "--dataset-root", str(dataset),
                   "--dataset-python", native["python"], "--model", str(base), "--output", str(output),
                   "--libero-task", SETTINGS["task"][0], str(SETTINGS["task"][1]),
                   "--updates-per-task", str(SETTINGS["updates"]), "--batch_size", str(SETTINGS["batch"]),
                   "--lr", str(SETTINGS["lr"]), "--seed", str(SETTINGS["seed"]),
                   "--save-freq", str(SETTINGS["save_freq"]), "--keep-intermediate-checkpoints"])
        result = read_json(output / "training_result.json")
        if result["steps"] != 4 or result["resumed_from_step"] != 0:
            raise ValueError("Example training did not complete from step 0 to step 4.")

        if not trained.exists():
            fixture = stage / trained.name
            (fixture / "train").mkdir(parents=True)
            shutil.move(str(output / "pretrained_model"), str(fixture / "train/pretrained_model"))
            write_json(fixture / "selection.json", selection)
            verify_model(fixture, selection)
        if not resumable.exists():
            fixture = stage / resumable.name
            state = fixture / state_name
            state.parent.mkdir(parents=True)
            shutil.move(str(output / "checkpoints/training_state_000002.pt"), str(state))
            write_json(fixture / "selection.json", selection)
            write_json(fixture / "settings.json", {"schema": 1, "adapter": "tinyvla", "model": "tinyvla",
                       "train": {**SETTINGS, "dataset_root": str(dataset), "selection": str(resumable / "selection.json")}})
            verify_resume(run_model, state, base, trainer, loader)

        for target in (trained, resumable):
            fixture = stage / target.name
            if fixture.is_dir():
                publish_directory(fixture, target)
    print(f"Prepared example checkpoints under {checkpoint_root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", help="Path to the Conda executable")
    parser.add_argument("--native-env", default="lerobot")
    parser.add_argument("--model-env", help="TinyVLA Conda environment name")
    parser.add_argument("--checkpoint-root", type=Path, help="Optional staging location for the two checkpoint fixtures")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        root = REPO
        dataset = root / "datasets/example/libero"
        checkpoint_root = (args.checkpoint_root or root / "outputs/example").expanduser().resolve()
        selection = read_json(MANIFEST_DIR / "selection.json")
        assets = read_json(MANIFEST_DIR / "assets.json")
        print(f"Dataset: {selection['dataset_repo']} at {selection['dataset_revision']} → {dataset}")
        print(f"Base model: {assets['base_model']['repo']} at {assets['base_model']['revision']}")
        print(f"Checkpoints: step 2 resume state and step 4 model → {checkpoint_root}", flush=True)
        if args.dry_run:
            return 0

        conda = conda_executable(args.conda)
        names = available_environments(conda, root)
        model_env = args.model_env or ("lerobot_tinyvla" if "tinyvla_infer" not in names and "lerobot_tinyvla" in names
                                       else "tinyvla_infer")
        native = resolve_environment(conda, args.native_env, root)
        model = resolve_environment(conda, model_env, root)
        if native["packages"].get("lerobot") != "0.6.2":
            raise ValueError("Install LeRobot 0.6.2 in the native environment for this example.")
        if "torch" not in model["packages"]:
            raise ValueError(f"PyTorch was not found in {model_env}.")
        base = prepare_weights(TINYVLA, assets["base_model"])
        prepare_dataset(dataset, selection, assets["files"])
        prepare_checkpoints(root, checkpoint_root, dataset, selection, TINYVLA, base, native, model, conda, model_env)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Example preparation failed ({error})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
