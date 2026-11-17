"""Find local inputs and results without loading models or datasets."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from adapters import read_json, weights
from runner import styled


@dataclass
class Inventory:
    datasets: list[tuple[Path, dict]] = field(default_factory=list)
    models: list[tuple[Path, str]] = field(default_factory=list)
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
    for base in (root / "outputs", external / "models"):
        for path, names in walk(base):
            for filename in sorted({"eval_info.json", "training_result.json"} & names):
                if metadata(path / filename):
                    result.results.append(path / filename)
            if "training_state.pt" in names and (path.parent.parent / "selection.json").is_file():
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
            if not isinstance(model, str) or path.resolve() in checkpoints:
                continue
            try:
                if tiny or {"model.safetensors", "model.safetensors.index.json"} & names:
                    weights(path)
                elif not config or not (path / "pytorch_model.bin").is_file():
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            checkpoints.add(path.resolve())
            result.models.append((path.resolve(), model))
            if not tiny and train:
                state = metadata(path.parent / "training_state/training_step.json")
                if type(state.get("step")) is int and state["step"] >= 0:
                    result.native_resume.append(path.resolve())
                latest = path.parent.parent / "last"
                if not latest.is_dir() or path.parent.resolve() == latest.resolve():
                    result.results.append(path / "train_config.json")
    for script in sorted(log_root.glob("*/run.py"), reverse=True):
        imported = script.with_name("script.py")
        result.settings.append(imported if imported.is_file() else script)
    result.settings += sorted((root / "outputs/manager").glob("*/settings.json"), reverse=True)
    return result


def show(inventory, environments, limit=None):
    datasets = [f"{path} · {info['selected_episodes']} selected training episode(s)" if "selected_episodes" in info
                else f"{path} · metadata lists {info.get('total_episodes', '?')} episodes"
                for path, info in inventory.datasets]
    groups = (
        ("Conda environments", sorted(environments)),
        ("Training datasets", datasets),
        ("Models for evaluation or initialization", [f"{model} · {path}" for path, model in inventory.models]),
        ("Checkpoints with training state", [str(path) for path in inventory.native_resume + inventory.tiny_resume]),
        ("Training and evaluation results", [str(path) for path in inventory.results]),
        ("Previous job settings", [str(path) for path in inventory.settings]),
    )
    for name, items in groups:
        heading = styled(name, "1;36")
        count = styled(f"[{len(items)} found]", "33")
        print(f"\n{heading} {count}")
        for index, item in enumerate(items[:limit], 1):
            print(f"  {index}. {item}")
        if limit is not None and len(items) > limit:
            print("  Run with --inventory to show the full list.")
