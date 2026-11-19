"""Find local inputs and results without loading models or datasets."""

from dataclasses import dataclass, field
from itertools import chain
import os
from pathlib import Path

from adapters import native_peft_checkpoint, native_training_step, read_json, require, weights
from checkpoint_compat import legacy_tdmpc_config
from runner import run_log_directories, styled


@dataclass
class Inventory:
    datasets: list[tuple[Path, dict]] = field(default_factory=list)
    models: list[tuple[Path, str]] = field(default_factory=list)
    cached_models: dict[Path, str] = field(default_factory=dict)
    native_resume: list[Path] = field(default_factory=list)
    tiny_resume: list[Path] = field(default_factory=list)
    results: list[Path] = field(default_factory=list)
    settings: list[Path] = field(default_factory=list)


def metadata(path):
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def walk(root):
    for directory, dirs, filenames in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not name.startswith(".") and
                        name not in {"__pycache__", "videos", "worker_logs", "logs", "scripts"})
        yield Path(directory), set(filenames)


def cached_snapshots():
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    hf_home = Path(os.environ.get("HF_HOME") or cache_home / "huggingface")
    hub = Path(os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or hf_home / "hub")
    for snapshot in sorted(hub.expanduser().glob("models--*/snapshots/*")):
        if snapshot.is_dir():
            repo_id = snapshot.parents[1].name.removeprefix("models--").replace("--", "/")
            yield snapshot, f"{repo_id} @ {snapshot.name[:8]}"


def model_location(inventory, path):
    origin = inventory.cached_models.get(path)
    return f"[Hugging Face cache] {origin} · {path}" if origin else str(path)


def scan(root, external, log_root):
    result = Inventory()
    dataset_roots = (root / "datasets", root / "datasets/.hub")
    seen = set()
    for base in dataset_roots:
        for path, names in walk(base):
            if path.name != "meta" or "info.json" not in names:
                continue
            info = metadata(path / "info.json")
            dataset = path.parent.resolve()
            if info.get("features") and next((dataset / "data").rglob("*.parquet"), None) and dataset not in seen:
                selection = metadata(dataset / "selection.json").get("train_episodes")
                if isinstance(selection, list):
                    info["selected_episodes"] = len(selection)
                result.datasets.append((dataset, info))
                seen.add(dataset)

    checkpoints = set()
    local_models = ((root / "outputs", None), (external / "models", None))
    for base, origin in chain(local_models, cached_snapshots()):
        for path, names in walk(base):
            if origin is None:
                for filename in sorted({"eval_info.json", "training_result.json"} & names):
                    if metadata(path / filename):
                        result.results.append(path / filename)
            if origin is None and "training_state.pt" in names and (path.parent.parent / "selection.json").is_file():
                try:
                    if (path / "training_state.pt").stat().st_size:
                        result.tiny_resume.append(path.parent.parent)
                except OSError:
                    pass
            config = metadata(path / "config.json") if "config.json" in names else {}
            tiny = config.get("model_type") == "llava_pythia" or "action_head.safetensors" in names
            train = metadata(path / "train_config.json") if "train_config.json" in names else {}
            policy = train.get("policy")
            kind = policy.get("type") if isinstance(policy, dict) else None
            model = "tinyvla" if tiny else kind or config.get("type")
            if model is None and legacy_tdmpc_config(config) is not None:
                model = "tdmpc"
            if not isinstance(model, str):
                continue
            resolved = path.resolve()
            if resolved in checkpoints:
                continue
            if origin and tiny and not {"preprocessor_config.json", "tokenizer.json"} <= names:
                continue
            try:
                if not tiny and "adapter_config.json" in names:
                    if not native_peft_checkpoint(path):
                        continue
                    require(path / "config.json")
                elif tiny or {"model.safetensors", "model.safetensors.index.json"} & names:
                    weights(path)
                elif config and "pytorch_model.bin" in names:
                    require(path / "pytorch_model.bin")
                else:
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            checkpoints.add(resolved)
            result.models.append((resolved, model))
            if origin:
                result.cached_models[resolved] = origin
            if not tiny and train:
                try:
                    native_training_step(path)
                except (OSError, ValueError):
                    pass
                else:
                    result.native_resume.append(resolved)
                if origin is None:
                    latest = path.parent.parent / "last"
                    if not latest.is_dir() or path.parent.resolve() == latest.resolve():
                        result.results.append(path / "train_config.json")
    for directory in run_log_directories(root, log_root):
        imported = directory / "script.py"
        result.settings.append(imported if imported.is_file() else directory / "run.py")
    result.settings += sorted((root / "outputs/manager").glob("*/settings.json"), reverse=True)
    return result


def show(inventory, environments, limit=None):
    datasets = [f"{path} · {info['selected_episodes']} selected training episode(s)" if "selected_episodes" in info
                else f"{path} · metadata lists {info.get('total_episodes', '?')} episodes"
                for path, info in inventory.datasets]
    groups = (
        ("Conda Environments", sorted(environments)),
        ("Training Datasets", datasets),
        ("Models for Evaluation or Initialization", [f"{model} · {model_location(inventory, path)}" for path, model in inventory.models]),
        ("Checkpoints with Training State", [model_location(inventory, path) for path in inventory.native_resume + inventory.tiny_resume]),
        ("Training and Evaluation Results", [str(path) for path in inventory.results]),
        ("Previous Job Settings", [str(path) for path in inventory.settings]),
    )
    for group_index, (name, items) in enumerate(groups):
        if group_index:
            print()
        heading = styled(name, "1;36")
        count = styled(f"[{len(items)} found]", "33")
        print(f"{heading} {count}")
        for index, item in enumerate(items[:limit], 1):
            print(f"  {index}. {item}")
        if limit is not None and len(items) > limit:
            print("  Run with --inventory to show the full list.")
        if name == "Training Datasets":
            print("  Browse LeRobot datasets on Hugging Face · https://huggingface.co/datasets?author=lerobot")
