"""Convert manager settings to the existing training and evaluation entry points."""

from dataclasses import dataclass, field
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

from training_preflight import native_training_preflight


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SINGLE_TASK_BENCHMARKS = {"pusht": "PushT-v0", "aloha": "AlohaInsertion-v0"}
LIBERO_BENCHMARKS = frozenset({"libero", "libero_plus"})
TASK_ID_BENCHMARKS = LIBERO_BENCHMARKS | {"robomme"}
EPISODE_LENGTH_BENCHMARKS = LIBERO_BENCHMARKS | SINGLE_TASK_BENCHMARKS.keys() | {
    "robocasa", "robomme", "robotwin", "vlabench", "isaaclab_arena",
}
TRAINING_SWITCHES = {
    "train_expert_only": "Action expert",
    "freeze_vision_encoder": "Vision encoder",
    "freeze_vision_backbone": "Vision backbone",
    "freeze_language_encoder": "Language encoder",
    "freeze_llm_backbone": "Language backbone",
    "freeze_embedding": "Embeddings",
    "freeze_vlm_embedding": "Vision-language embeddings",
    "freeze_video_expert": "Video expert",
    "freeze_qwen": "Qwen",
    "train_state_proj": "State projection",
    "train_policy_transformer": "Policy transformer",
    "train_soft_prompts": "Soft prompts",
    "tune_llm": "Language model",
    "tune_visual": "Vision model",
    "tune_projector": "Projector",
    "tune_diffusion_model": "Diffusion model",
    "tune_vlln": "Vision-language normalization",
    "finetune_language_model": "Language model",
    "finetune_vision_model": "Vision model",
    "finetune_action_head": "Action head",
    "unfreeze_vision_merger": "Vision merger",
    "unfreeze_lam_decoder": "Latent action decoder",
}
PEFT_POLICIES = frozenset({"smolvla", "pi0", "pi05", "lingbot_vla_v2", "flux3"})
SCOPE_OPTIONS = set(TRAINING_SWITCHES) | {
    "train_mode_vlm", "lora_rank", "lora_alpha", "lora_dropout",
    "tune_top_llm_layers", "unfreeze_llm_last_n_layers",
}


def is_policy_training_option(name):
    if name in SCOPE_OPTIONS or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        return False
    return name.startswith(("optimizer_", "scheduler_", "loss_", "freeze_", "tune_", "train_", "ema_")) or (
        name.endswith(("_dropout", "_loss_weight")) or name in {
            "dropout", "kl_weight", "use_ema", "gradient_checkpointing",
            "use_gradient_checkpointing", "weight_decay", "max_grad_norm",
        }
    )


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
    auto_flags: tuple[str, ...] = ()


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


def weights(path):
    require(path / "config.json")
    index = path / "model.safetensors.index.json"
    if index.is_file() and not (path / "model.safetensors").is_file():
        shards = set(read_json(index)["weight_map"].values())
        if not shards or any(Path(name).name != name for name in shards):
            raise ValueError(f"Invalid model weight index ({index}).")
        for name in shards:
            require(path / name)
    else:
        require(path / "model.safetensors")
    return path


def native_peft_checkpoint(path):
    path = Path(path).expanduser()
    adapter = path / "adapter_config.json"
    if not adapter.is_file():
        return False
    config = read_json(adapter)
    base_model = config.get("base_model_name_or_path") if isinstance(config, dict) else None
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError(f"PEFT adapter configuration does not identify its base model ({adapter}).")
    safe_weights = path / "adapter_model.safetensors"
    require(safe_weights if safe_weights.is_file() else path / "adapter_model.bin")
    return True


def validate_native_policy(path, model):
    config = Path(path).expanduser() / "config.json"
    if config.is_file():
        data = read_json(config)
        kind = data.get("type") if isinstance(data, dict) else None
        if kind and kind != model:
            raise ValueError(f"The checkpoint uses the {kind} policy, but the job selects {model}.")


def native_training_step(model_dir):
    state_dir = model_dir.parent / "training_state"
    step_file = require(state_dir / "training_step.json")
    metadata = read_json(step_file)
    step = metadata.get("step") if isinstance(metadata, dict) else None
    if type(step) is not int or step < 0:
        raise ValueError(f"Invalid checkpoint step ({step_file}).")
    require(state_dir / "rng_state.safetensors")

    def present(path):
        return path.is_file() and path.stat().st_size > 0

    optimizer_files = ("optimizer_state.safetensors", "optimizer_param_groups.json")
    optimizer_dirs = [path for path in state_dir.iterdir() if path.is_dir() and
                      any((path / name).exists() for name in optimizer_files)]
    if not all(present(state_dir / name) for name in optimizer_files):
        sharded = state_dir / "optimizer_0"
        shards_present = present(sharded / ".metadata") and any(present(path) for path in sharded.glob("*.distcp"))
        named_optimizers = optimizer_dirs and all(present(path / name) for path in optimizer_dirs
                                                  for name in optimizer_files)
        if not (shards_present or named_optimizers):
            raise FileNotFoundError(f"Complete optimizer state not found ({state_dir}).")
    return step


def training_flags(config, model):
    scope = config.get("policy_training", {})
    lora = config.get("lora")
    options = config.get("policy_options", {})
    if not isinstance(scope, dict) or not isinstance(options, dict):
        raise ValueError("policy_training and policy_options must be dictionaries.")
    if config.get("resume") and (scope or options or lora is not None):
        raise ValueError("Resumed jobs use the checkpoint's saved training settings.")

    args = []
    for name, value in scope.items():
        valid = name in TRAINING_SWITCHES and type(value) is bool
        if model == "groot" and name == "tune_top_llm_layers":
            valid = type(value) is int and value >= 0
        if model == "lawam" and name == "unfreeze_llm_last_n_layers":
            valid = type(value) is int and value >= -1
        if model == "molmoact2":
            valid = valid or name == "train_mode_vlm" and value in ("fft", "lora", "freeze")
            valid = valid or name in {"lora_rank", "lora_alpha"} and type(value) is int and value > 0
            valid = valid or name == "lora_dropout" and type(value) in (float, int) and 0 <= value <= 1
        if not valid:
            raise ValueError(f"Invalid policy training option ({name}).")
        args.append(flag(f"policy.{name}", value))

    for name, value in options.items():
        legacy = name in {"train_state_proj", "train_policy_transformer", "train_soft_prompts"} and type(value) is bool
        legacy = legacy or name == "tune_top_llm_layers" and type(value) is int and value >= 0
        if (not isinstance(name, str) or not (is_policy_training_option(name) or legacy) or name in scope
                or type(value) not in (bool, int, float, str)
                or type(value) is float and not math.isfinite(value)):
            raise ValueError(f"Invalid policy option ({name}).")
        args.append(flag(f"policy.{name}", value))

    if lora is None:
        return args
    if (not isinstance(lora, dict) or model not in PEFT_POLICIES
            or not config.get("pretrained") or scope):
        raise ValueError("LoRA requires a supported policy and a pretrained checkpoint, with its default scope.")
    if unknown := lora.keys() - {"rank", "alpha", "target_modules", "full_training_modules"}:
        raise ValueError(f"Unknown LoRA options: {', '.join(sorted(unknown))}.")
    args.append(flag("peft.method_type", "LORA"))
    for key, target in (("rank", "r"), ("alpha", "lora_alpha")):
        value = lora.get(key, 16)
        if type(value) is not int or value < 1:
            raise ValueError(f"LoRA {key} must be a positive integer.")
        args.append(flag(f"peft.{target}", value))
    for key in ("target_modules", "full_training_modules"):
        if key not in lora:
            continue
        value = lora[key]
        module_list = isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)
        if not module_list and not (key == "target_modules" and isinstance(value, str) and value.strip()):
            raise ValueError(f"Invalid LoRA {key}.")
        args.append(flag(f"peft.{key}", value))
    return args


def required_scope_flags(cfg, model, explicit):
    scope = cfg.get("policy_training", {})
    required = {}
    if model == "fineart_vla" and scope.get("train_expert_only") is True:
        required = {"policy.text_loss_weight": 0.0, "policy.enable_fast_action_loss": False}
    branches = ("finetune_language_model", "finetune_vision_model", "finetune_action_head")
    if model == "evo1" and all(name in scope for name in branches):
        required = {"policy.apply_training_stage_defaults": False,
                    "policy.finetune_vlm": scope[branches[0]] or scope[branches[1]]}
    additional = {}
    for name, value in required.items():
        if name in explicit:
            try:
                selected = json.loads(explicit[name])
            except ValueError:
                selected = explicit[name]
            if selected != value:
                raise ValueError(f"Selected training scope requires {flag(name, value)}.")
        else:
            additional[name] = value
    return additional


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
        extra = cfg.get("extra", [])
        automatic = {}
        if cfg.get("resume"):
            model_dir = native_checkpoint(Path(cfg["resume"]).expanduser())
            old_cfg = read_json(model_dir / "train_config.json")
            old_model = old_cfg.get("policy", {}).get("type")
            if old_model and old_model != self.settings["model"]:
                raise ValueError(f"The checkpoint uses the {old_model} policy.")
            if (old_cfg.get("job") or {}).get("target") not in (None, "local"):
                raise ValueError("The checkpoint is configured for a remote job. Select a local checkpoint.")
            start = native_training_step(model_dir)
            if start >= steps:
                raise ValueError("The target step count must exceed the checkpoint step.")
            args += [flag("config_path", model_dir / "train_config.json"), "--resume=true"]
        else:
            args += [flag("dataset.repo_id", cfg["dataset"]), flag("batch_size", cfg["batch"]),
                     flag("policy.device", cfg["device"]), flag("seed", cfg["seed"])]
            if cfg.get("dataset_root"):
                args.append(flag("dataset.root", cfg["dataset_root"]))
            episodes = cfg.get("episodes")
            explicit_episodes = any(arg.startswith("--dataset.episodes=") for arg in extra)
            if episodes is None and cfg.get("dataset_root") and not explicit_episodes:
                split = Path(cfg["dataset_root"]) / "selection.json"
                if split.is_file():
                    episodes = read_json(split).get("train_episodes")
                    if episodes is not None:
                        automatic["dataset.episodes"] = episodes
            if episodes is not None:
                if not isinstance(episodes, list) or not episodes or any(
                    type(episode) is not int or episode < 0 for episode in episodes
                ):
                    raise ValueError("Training episodes must be a nonempty list of nonnegative integers.")
                if "dataset.episodes" not in automatic:
                    args.append(flag("dataset.episodes", episodes))
            pretrained = cfg.get("pretrained")
            if pretrained:
                validate_native_policy(pretrained, self.settings["model"])
            policy_key = "policy.path" if pretrained else "policy.type"
            args.append(flag(policy_key, pretrained or self.settings["model"]))
        model = self.settings["model"]
        scope_flags = training_flags(cfg, model)
        explicit = dict(arg[2:].split("=", 1) for arg in (*scope_flags, *extra)
                        if arg.startswith("--") and "=" in arg)
        automatic.update(required_scope_flags(cfg, model, explicit))
        automatic.update(native_training_preflight(self.settings))
        args += scope_flags + [flag(name, value) for name, value in automatic.items()]
        save_checkpoint = self.settings.get("save_outputs", True) or "eval" in self.settings
        args += [flag("output_dir", output), flag("job_name", self.run_name), flag("steps", steps),
                 flag("log_freq", min(track_frequency(steps), cfg["save_freq"])),
                 flag("save_freq", cfg["save_freq"]), flag("save_checkpoint", save_checkpoint),
                 "--policy.push_to_hub=false"]
        args += additional_flags(extra,
                                 {a[2:].split("=")[0] for a in args if a.startswith("--")} |
                                 {"config_path", "resume", "policy.path", "policy.type",
                                  "save_checkpoint_to_hub", "output_dir"}, ("job.",))
        expected = (output / "checkpoints/last/pretrained_model/train_config.json",) if save_checkpoint else ()
        return Stage("train", self.environment, args, expected, steps, start,
                     output=output, auto_flags=tuple(automatic))

    def eval_policy(self, after_training):
        model = (self.train_dir / "train/checkpoints/last/pretrained_model" if after_training
                 else self.settings["eval"]["policy"])
        device = flag("policy.device", self.settings["eval"]["device"])
        if model is None:
            return [flag("policy.type", self.settings["model"]), device]
        if not after_training:
            validate_native_policy(model, self.settings["model"])
        return [flag("policy.path", model), device]

    def evaluate(self, after_training):
        cfg = self.settings["eval"]
        output = self.eval_dir
        kind, tasks = cfg["env"], cfg["tasks"]
        if not isinstance(tasks, (list, tuple)) or not tasks or any(
            not isinstance(task, str) or not task.strip() for task in tasks
        ):
            raise ValueError("Specify a nonempty list of evaluation tasks or suites.")
        tasks = [task.strip() for task in tasks]
        if kind in SINGLE_TASK_BENCHMARKS and (len(tasks) != 1 or "," in tasks[0]):
            raise ValueError(f"The {kind} benchmark supports one evaluation task per job.")
        episodes, batch = cfg["episodes"], cfg["batch"]
        if type(episodes) is not int or type(batch) is not int or min(episodes, batch) < 1:
            raise ValueError("Episode count and evaluation batch size must be positive integers.")
        parallel = cfg.get("parallel_tasks", 1)
        if type(parallel) is not int or parallel < 1:
            raise ValueError("Parallel evaluation tasks must be a positive integer.")
        if parallel > 1:
            if self.key != "tinyvla" or kind != "libero":
                raise ValueError("Parallel tasks are currently supported for TinyVLA LIBERO evaluation.")
            command = ["python", "-u", str(Path(__file__).with_name("parallel_eval.py").resolve())]
        else:
            command = ["lerobot-eval"]
        args = [*command, flag("env.type", kind), flag("env.task", ",".join(tasks)),
                flag("eval.n_episodes", episodes), flag("eval.batch_size", batch),
                flag("seed", cfg["seed"]), flag("output_dir", output)]
        units = episodes if kind in SINGLE_TASK_BENCHMARKS else None
        length = cfg.get("episode_length", 0)
        if type(length) is not int or length < 0:
            raise ValueError("Episode length must be a nonnegative integer.")
        if length:
            if kind == "metaworld":
                raise ValueError("The LeRobot Meta-World integration uses its default episode length.")
            args.append(flag("env.episode_length", length))
        ids = cfg.get("task_ids")
        if ids is None:
            ids = []
        if (not isinstance(ids, (list, tuple)) or any(type(i) is not int or i < 0 for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Use a list of unique, nonnegative task IDs.")
        if ids and kind not in TASK_ID_BENCHMARKS:
            raise ValueError(f"The {kind} benchmark does not support task IDs.")
        if kind in TASK_ID_BENCHMARKS and (len(set(tasks)) != len(tasks) or any("," in task for task in tasks)):
            raise ValueError("List each evaluation task or suite once.")
        if kind == "libero" and ids and (len(tasks) != 1 or tasks[0] in SUITES and max(ids) >= 10):
            raise ValueError("Use unique, in-range task IDs from a single suite.")
        if ids:
            args.append(flag("env.task_ids", list(ids)))

        if kind in LIBERO_BENCHMARKS:
            control, size = cfg.get("control", "relative"), cfg.get("size", 256)
            if control not in ("absolute", "relative"):
                raise ValueError("LIBERO control mode must be absolute or relative.")
            if type(size) is not int or size < 1:
                raise ValueError("LIBERO camera size must be a positive integer.")
            args += [flag("env.max_parallel_tasks", parallel), flag("env.control_mode", control),
                     "--env.obs_type=pixels_agent_pos", flag("env.observation_width", size),
                     flag("env.observation_height", size)]
            if ids or kind == "libero" and set(tasks) <= set(SUITES):
                units = episodes * len(tasks) * (len(ids) or 10)
        elif kind == "robomme":
            units = episodes * len(tasks) * (len(ids) or 1)
        args += self.eval_policy(after_training)
        args += additional_flags(cfg.get("extra", []),
                                 {a[2:].split("=")[0] for a in args if a.startswith("--")} |
                                 {"config_path", "policy.type", "policy.path", "env.max_parallel_tasks",
                                  "env.task_ids", "env.episode_length", "eval.record_repo_id", "eval.recording_repo_id"})
        automatic = ()
        if kind in SINGLE_TASK_BENCHMARKS and min(batch, episodes) > 1 and not any(
            argument.startswith("--eval.use_async_envs=") for argument in args
        ):
            # This still batches policy predictions. The subprocess Gym workers
            # in these benchmarks do not reliably register their environments.
            args.append("--eval.use_async_envs=false")
            automatic = ("eval.use_async_envs",)
        headless_mujoco = (kind in {"aloha", "robocasa", "vlabench"} and sys.platform == "linux"
                           and not (os.environ.get("DISPLAY") or os.environ.get("MUJOCO_GL")))
        env = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"} if kind in LIBERO_BENCHMARKS or headless_mujoco else {}
        environment = cfg.get("environment") or self.environment
        if not isinstance(environment, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", environment):
            raise ValueError("Invalid Conda evaluation environment name.")
        return Stage("eval", environment, args, (output / "eval_info.json",), units,
                     eval_batches=(episodes + batch - 1) // batch, env=env, output=output, auto_flags=automatic)


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
        if cfg.get("policy_training") or cfg.get("policy_options"):
            raise ValueError("TinyVLA supports action-head training with optional language-attention LoRA.")
        lora = cfg.get("lora")
        if lora is not None:
            if not isinstance(lora, dict) or set(lora) - {"rank", "alpha", "dropout"}:
                raise ValueError("TinyVLA LoRA accepts rank, alpha, and dropout.")
            rank, alpha, dropout = lora.get("rank", 16), lora.get("alpha", 16), lora.get("dropout", 0.0)
            if (type(rank) is not int or rank < 1 or type(alpha) is not int or alpha < 1
                    or type(dropout) not in (int, float) or not 0 <= dropout < 1):
                raise ValueError("Use positive TinyVLA LoRA rank and alpha, and dropout in [0, 1).")
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
        task = cfg.get("task")
        if task and (len(task) != 2 or task[0] not in SUITES or type(task[1]) is not int or not 0 <= task[1] < 10):
            raise ValueError("Select a supported TinyVLA training suite and task ID.")
        dataset = Path(cfg.get("dataset_root") or self.root / "datasets/libero").expanduser()
        selection = cfg.get("selection")
        local_selection = dataset / "selection.json"
        use_local_selection = not selection and local_selection.is_file()
        if use_local_selection:
            selection = local_selection
        if selection:
            split = read_json(require(Path(selection)))
            if not isinstance(split, dict):
                raise ValueError(f"Expected a training split dictionary ({selection}).")
            split_tasks = split.get("tasks")
            if use_local_selection and not split_tasks:
                raise ValueError(f"The local TinyVLA training split needs per-task metadata ({selection}).")
            if split_tasks:
                expected = {tuple(task)} if task else {(suite, i) for suite in SUITES for i in range(10)}
                try:
                    selected = {(row["suite"], row["task_id"]) for row in split_tasks.values()}
                except (AttributeError, KeyError, TypeError) as error:
                    raise ValueError(f"Invalid TinyVLA training split tasks ({selection}).") from error
                if selected != expected:
                    raise ValueError(f"The TinyVLA training split does not match the selected tasks ({selection}).")
        else:
            selection = self.train_dir / "new_selection.json"
        for name in ("updates", "batch", "save_freq"):
            if type(cfg[name]) is not int or cfg[name] < 1:
                raise ValueError(f"Enter a positive integer for {name}.")
        if not math.isfinite(cfg["lr"]) or cfg["lr"] <= 0:
            raise ValueError("Learning rate must be positive and finite.")
        args = ["python", "-u", str(script), "--workspace", str(workspace), flag("data", selection),
                "--loader", str(loader),
                "--output", str(output), "--dataset-python", f"@python:{self.environment}",
                "--dataset-root", str(dataset),
                "--batch_size", str(cfg["batch"]), "--lr", str(cfg["lr"]), "--seed", str(cfg["seed"]),
                "--updates-per-task", str(cfg["updates"]), "--save-freq", str(cfg["save_freq"]), *origin]
        if task:
            args.extend(("--libero-task", task[0], str(task[1])))
        if lora is not None:
            args.extend(("--lora-rank", str(rank), "--lora-alpha", str(alpha), "--lora-dropout", str(dropout)))
        return Stage("train", self.infer_env, args, (output / "training_result.json",
                     output / "pretrained_model/action_head.safetensors", output / "pretrained_model/model.safetensors",
                     output / "checkpoints/training_state.pt"), cfg["updates"] * (1 if task else 40),
                     None if cfg.get("resume") else 0, output=output,
                     auto_flags=("data",) if use_local_selection else ())

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
