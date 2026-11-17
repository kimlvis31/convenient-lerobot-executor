"""Convert manager settings to the existing training and evaluation entry points."""

from dataclasses import dataclass, field
import importlib
import json
import math
from pathlib import Path
import re


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass
class Stage:
    name: str
    environment: str
    arguments: list[str]
    artifacts: tuple[Path, ...]
    units: int | None
    start: int | None = 0
    eval_batches: int = 0
    env: dict[str, str] = field(default_factory=dict)
    output: Path | None = None


def flag(name, value):
    if isinstance(value, (bool, list, dict)):
        value = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return f"--{name}={value}"


def additional_flags(tokens, owned, prefixes=()):
    result = []
    for token in tokens:
        name, equal, _ = token.removeprefix("--").partition("=")
        if not token.startswith("--") or not equal or not re.fullmatch(r"[\w.]+", name):
            raise ValueError(f"Use --name=value for extra options ({token}).")
        if name in owned or any(name.startswith(prefix) for prefix in prefixes):
            raise ValueError(f"The executor manages this option ({name}).")
        result.append(token)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(path):
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f"Required file is missing or empty ({path}).")
    return path


def native_checkpoint(path):
    path = path.resolve()
    for candidate in (path, path / "pretrained_model", path / "checkpoints/last/pretrained_model",
                      path / "train/checkpoints/last/pretrained_model"):
        if (candidate / "train_config.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"LeRobot training configuration not found ({path}).")


def weight_files(path):
    index = path / "model.safetensors.index.json"
    if index.is_file() and not (path / "model.safetensors").is_file():
        shards = set(read_json(index)["weight_map"].values())
        if not shards or any(Path(name).name != name for name in shards):
            raise ValueError(f"Invalid model weight index ({index}).")
        return tuple(require(path / name) for name in sorted(shards))
    return (require(path / "model.safetensors"),)


def weights(path):
    require(path / "config.json")
    weight_files(path)
    return path


class NativeAdapter:
    key = "native"
    label = "Native LeRobot policy"

    def __init__(self, settings, run_name, root, external, environments):
        self.settings = settings
        self.run_name = run_name
        self.root = root
        self.train_dir = root / "outputs/training" / run_name
        self.eval_dir = root / "outputs/evaluation" / run_name
        self.external = external
        self.environments = environments

    @property
    def environment(self):
        return self.environments.get(self.settings["model"], self.environments["native"])

    def train(self):
        cfg = self.settings["train"]
        output = self.train_dir / "train"
        steps = cfg["steps"]
        if type(steps) is not int or steps < 1:
            raise ValueError("Training steps must be a positive integer.")
        start = 0
        args = ["lerobot-train"]
        if cfg.get("resume"):
            model_dir = native_checkpoint(Path(cfg["resume"]).expanduser())
            old_cfg = read_json(model_dir / "train_config.json")
            old_model = old_cfg.get("policy", {}).get("type")
            if old_model and old_model != self.settings["model"]:
                raise ValueError(f"The checkpoint uses the {old_model} policy.")
            if (old_cfg.get("job") or {}).get("target") not in (None, "local"):
                raise ValueError("The checkpoint is configured for a remote job. Select a local checkpoint.")
            state = model_dir.parent / "training_state"
            if not state.is_dir():
                raise FileNotFoundError(f"Training state directory not found ({state}).")
            step_file = state / "training_step.json"
            start = read_json(step_file)["step"] if step_file.is_file() else None
            if start is not None and (type(start) is not int or not 0 <= start < steps):
                raise ValueError("The target step count must exceed the checkpoint step.")
            args += [flag("config_path", model_dir / "train_config.json"), "--resume=true"]
        else:
            args += [flag("dataset.repo_id", cfg["dataset"]), flag("batch_size", cfg["batch"]),
                     flag("policy.device", cfg["device"]), flag("seed", cfg["seed"])]
            if cfg.get("dataset_root"):
                args.append(flag("dataset.root", cfg["dataset_root"]))
            policy_key = "policy.path" if cfg.get("pretrained") else "policy.type"
            args.append(flag(policy_key, cfg.get("pretrained") or self.settings["model"]))
        args += [flag("output_dir", output), flag("job_name", self.run_name), flag("steps", steps),
                 flag("log_freq", min(track_frequency(steps), cfg["save_freq"])),
                 flag("save_freq", cfg["save_freq"]), "--save_checkpoint=true", "--policy.push_to_hub=false"]
        args += additional_flags(cfg.get("extra", []),
                                 {a[2:].split("=")[0] for a in args if a.startswith("--")} |
                                 {"config_path", "resume", "policy.path", "policy.type",
                                  "save_checkpoint_to_hub", "output_dir"}, ("job.",))
        expected = output / "checkpoints/last/pretrained_model/train_config.json"
        return Stage("train", self.environment, args, (expected,), steps, start, output=output)

    def eval_policy(self, after_training):
        model = (self.train_dir / "train/checkpoints/last/pretrained_model" if after_training
                 else self.settings["eval"]["policy"])
        return [flag("policy.path", model), flag("policy.device", self.settings["eval"]["device"])]

    def evaluate(self, after_training):
        cfg = self.settings["eval"]
        output = self.eval_dir
        kind, tasks = cfg["env"], cfg["tasks"]
        episodes, batch = cfg["episodes"], cfg["batch"]
        if type(episodes) is not int or type(batch) is not int or min(episodes, batch) < 1:
            raise ValueError("Episode count and evaluation batch size must be positive integers.")
        args = ["lerobot-eval", flag("env.type", kind), flag("env.task", ",".join(tasks)),
                flag("eval.n_episodes", episodes), flag("eval.batch_size", batch),
                flag("seed", cfg["seed"]), flag("output_dir", output)]
        units = None
        length = cfg.get("episode_length", 0)
        if type(length) is not int or length < 0:
            raise ValueError("Episode length must be a nonnegative integer.")
        if length:
            args.append(flag("env.episode_length", length))
        if kind == "libero":
            if not tasks or len(set(tasks)) != len(tasks):
                raise ValueError("Specify at least one suite and list each suite once.")
            ids = cfg.get("task_ids", [])
            known_suites = set(tasks) <= set(SUITES)
            if ids and (len(tasks) != 1 or len(set(ids)) != len(ids) or any(
                type(i) is not int or i < 0 or known_suites and i >= 10 for i in ids
            )):
                raise ValueError("Use unique, in-range task IDs from a single suite.")
            if ids:
                args.append(flag("env.task_ids", ids))
            args += ["--env.max_parallel_tasks=1", flag("env.control_mode", cfg["control"]),
                     "--env.obs_type=pixels_agent_pos", flag("env.observation_width", cfg["size"]),
                     flag("env.observation_height", cfg["size"])]
            if known_suites:
                units = episodes * (len(ids) or 10 * len(tasks))
        args += self.eval_policy(after_training)
        args += additional_flags(cfg.get("extra", []),
                                 {a[2:].split("=")[0] for a in args if a.startswith("--")} |
                                 {"config_path", "policy.type", "policy.path", "env.max_parallel_tasks",
                                  "env.task_ids", "env.episode_length", "eval.record_repo_id"})
        if kind == "libero":
            env = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
        else:
            env = {}
        return Stage("eval", self.environment, args, (output / "eval_info.json",), units,
                     eval_batches=(episodes + batch - 1) // batch, env=env, output=output)


class TinyVLAAdapter(NativeAdapter):
    key = "tinyvla"
    label = "TinyVLA"

    @property
    def environment(self):
        return self.environments["native"]

    @property
    def infer_env(self):
        return self.environments["tinyvla"]

    @property
    def base_model(self):
        return self.external / "models/Llava-Pythia-700M"

    def train(self):
        cfg = self.settings["train"]
        output = self.train_dir / "train"
        workspace = self.external
        require(workspace / "runtime/tinyvla_loader.py")
        loader = workspace / "runtime/tinyvla_loader.py"
        script = require(workspace / "training/train_head.py")
        require(workspace / "training/lerobot_data.py")
        if cfg.get("resume"):
            source = require(Path(cfg["resume"]).expanduser().resolve())
            origin = ["--resume-state", str(source)]
        elif cfg.get("warm_start"):
            source = weights(Path(cfg["warm_start"]).expanduser().resolve())
            require(source / "action_head.safetensors")
            origin = ["--init-checkpoint", str(source)]
        else:
            model = weights(Path(cfg["base_model"]).expanduser().resolve() if cfg.get("base_model") else self.base_model)
            origin = ["--model", str(model)]
        if cfg.get("selection"):
            require(Path(cfg["selection"]))
        selection = cfg.get("selection") or self.train_dir / "new_selection.json"
        task = cfg.get("task")
        if task and (len(task) != 2 or task[0] not in SUITES or type(task[1]) is not int or not 0 <= task[1] < 10):
            raise ValueError("Select a supported TinyVLA training suite and task ID.")
        for name in ("updates", "batch", "save_freq"):
            if type(cfg[name]) is not int or cfg[name] < 1:
                raise ValueError(f"Enter a positive integer for {name}.")
        if not math.isfinite(cfg["lr"]) or cfg["lr"] <= 0:
            raise ValueError("Learning rate must be positive and finite.")
        args = ["python", "-u", str(script), "--workspace", str(workspace), "--data", str(selection),
                "--loader", str(loader),
                "--output", str(output), "--dataset-python", f"@python:{self.environment}",
                "--dataset-root", cfg.get("dataset_root") or str(self.root / "datasets/libero"),
                "--batch_size", str(cfg["batch"]), "--lr", str(cfg["lr"]), "--seed", str(cfg["seed"]),
                "--updates-per-task", str(cfg["updates"]), "--save-freq", str(cfg["save_freq"]), *origin]
        if task:
            args.extend(("--libero-task", task[0], str(task[1])))
        return Stage("train", self.infer_env, args, (output / "training_result.json",
                     output / "pretrained_model/action_head.safetensors", output / "pretrained_model/model.safetensors",
                     output / "checkpoints/training_state.pt"), cfg["updates"] * (1 if task else 40),
                     None if cfg.get("resume") else 0, output=output)

    def eval_policy(self, after_training):
        cfg = self.settings["eval"]
        if cfg["env"] != "libero" or cfg["control"] != "relative" or cfg["size"] != 256:
            raise ValueError("TinyVLA evaluation requires LIBERO, relative control, and 256-pixel images.")
        loader = require(self.external / "runtime/tinyvla_loader.py")
        model = self.train_dir / "train/pretrained_model" if after_training else (
            Path(cfg["policy"]).expanduser().resolve() if cfg.get("policy") else self.base_model)
        if not after_training:
            weights(model)
        steps = cfg.get("action_steps", 16)
        if type(steps) is not int or not 1 <= steps <= 16:
            raise ValueError("TinyVLA action steps must be between 1 and 16.")
        return ["--policy.type=tinyvla", "--policy.device=cpu", "--policy.use_amp=false",
                flag("policy.workspace", self.external), flag("policy.step7_script", loader),
                flag("policy.model_path", model), flag("policy.inference_python", f"@python:{self.infer_env}"),
                flag("policy.worker_log_dir", self.eval_dir / "worker_logs"),
                flag("policy.seed", cfg["seed"]), flag("policy.n_action_steps", steps)]


def track_frequency(steps):
    return min(100, max(1, steps // 20))


def load_adapters(plugins):
    adapters = {cls.key: cls for cls in (NativeAdapter, TinyVLAAdapter)}
    for plugin in plugins:
        module, name = plugin.split(":", 1)
        cls = getattr(importlib.import_module(module), name)
        if not issubclass(cls, NativeAdapter) or cls.key in adapters:
            raise ValueError(f"The adapter must extend NativeAdapter and use a unique key ({plugin}).")
        adapters[cls.key] = cls
    return adapters
