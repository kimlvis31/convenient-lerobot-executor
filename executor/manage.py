#!/usr/bin/env python3
"""Interactive LeRobot training, evaluation, and resume manager."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from textwrap import indent
from uuid import uuid4

from adapters import SUITES, load_adapters, native_checkpoint, read_json
from inventory import model_location, scan, show
from questions import PreviousQuestion, QuestionDisplay, QuestionHistory, active_history, retry_question
from runner import ANSI, available_environments, conda_executable, execute, load_settings, native_policies, seconds, stage_title, styled


DEFAULT_ENVIRONMENTS = {"native": "lerobot", "tinyvla": "tinyvla_infer"}
# Representative checkpoints, usually using the Hub's Safetensors parameter total.
REFERENCE_MODEL_SIZES = {
    "act": 51_685_096,  # lerobot/act_aloha_sim_transfer_cube_human @ ba73b27
    "diffusion": 262_709_062,  # lerobot/diffusion_pusht @ 84a7c23
    "dm05": 5_157_859_728,  # lerobot/dm05_base @ 84ea03f
    "eo1": 3_771_607_072,  # lerobot/eo1-base @ bba306b
    "evo1": 770_000_000,  # LeRobot v0.6.0 release announcement
    "fastwam": 6_020_710_599,  # lerobot/fastwam_libero_uncond_2cam224 @ 53983e1
    "fineart_vla": 4_143_404_816,  # lerobot/FineART-VLA-subtask-midtrained @ b6cf4fa
    "flux3": 6_947_071_232,  # black-forest-labs/flux-3-action-so101 @ e2584b2
    "groot": 3_144_016_000,  # nvidia/GR00T-N1.7-3B @ 373d34e
    "lawam": 2_555_179_360,  # jialei02/lawam-pretrain-lerobot @ bc45090
    "lingbot_va": 5_088_872_670,  # lerobot/lingbot_va_base @ a9b1da7
    "lingbot_vla_v2": 6_375_907_511,  # lerobot/lingbot_vla_v2_base @ 32d0a32
    "longwam": 12_406_309_603,  # Efficient-Large-Model/Long-WAM-LIBERO-IDM-LeRobot @ 92f0910
    "molmoact2": 5_442_196_272,  # allenai/MolmoAct2-LIBERO-LeRobot @ f0c5a95
    "multi_task_dit": 450_000_000,  # LeRobot Multi-task DiT policy documentation
    "pi0": 3_501_372_176,  # lerobot/pi0_base @ 6e24539
    "pi05": 3_616_757_520,  # lerobot/pi05_base @ 22c30e5
    "pi0_fast": 2_923_335_408,  # lerobot/pi0fast-base @ 308eb84
    "smolvla": 450_046_176,  # lerobot/smolvla_base @ 5e8d12a
    "tinyvla": 740_388_103,  # Tensor count in the prepared example model, including its action head
    "vla_jepa": 3_081_498_247,  # lerobot/VLA-JEPA-LIBERO @ 735d9f6
    "vqbet": 39_532_013,  # lerobot/vqbet_pusht @ 15c2d0a
    "wall_oss_05": 4_162_219_008,  # lerobot/wall-oss-0.5 @ 03ab102
    "wall_x": 4_224_041_072,  # x-square-robot/wall-oss-flow @ 44e8276
    "xvla": 879_738_545,  # lerobot/xvla-libero @ 8e6aeef
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "executor/log"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"


def divider():
    print(styled("-" * shutil.get_terminal_size().columns, "90"), flush=True)


def section(title):
    divider()
    print(styled(f"<{title}>", "1;95"), flush=True)


def ask(question, default=None, optional=False, *, choices=(), blank_before=True):
    history = active_history.get()
    if history is not None and history.position < len(history.answers):
        answer = history.answers[history.position]
        history.position += 1
        return answer

    hint = styled(f"[Default: {default}]", "33") if not choices and default is not None else ""
    suffix = " " + hint if hint else ""
    if history is not None:
        if choices:
            choices = [*choices, "  B. Back"]
        else:
            suffix += " [B: Back]"
    prompt = "\n".join([*choices, styled(question, "1;36") + suffix])
    if blank_before:
        prompt = "\n" + prompt
    read = sys.stdout.read if isinstance(sys.stdout, QuestionDisplay) else input
    while True:
        value = read(f"{prompt}: ").strip()
        if not sys.stdin.isatty():
            print()
        if history is not None and value.casefold() == "b":
            raise PreviousQuestion
        if value or default is not None or optional:
            answer = value if value else default if default is not None else ""
            if history is not None:
                history.answers.append(answer)
                history.position += 1
            return answer


def number(question, default, minimum=1, maximum=None, *, choices=(), blank_before=True):
    while True:
        try:
            value = int(ask(question, default, choices=choices, blank_before=blank_before))
            if value >= minimum and (maximum is None or value <= maximum):
                return value
        except ValueError:
            pass
        retry_question()
        print("Enter an integer within the allowed range.")


def rate(question, default):
    while True:
        try:
            value = float(ask(question, default))
            if math.isfinite(value) and value > 0:
                return value
        except ValueError:
            pass
        retry_question()
        print("Enter a positive number.")


def choose(question, options, default=1, *, details=None, blank_before=True):
    choices = []
    labels = [ANSI.sub("", label) for _, label in options]
    detail_column = max((len(label) for (value, _), label in zip(options, labels)
                         if details and value in details), default=0)
    number_width = len(str(len(options))) if details else 1
    for index, ((value, label), plain_label) in enumerate(zip(options, labels), 1):
        row = f"  {index:>{number_width}}. {label}"
        if index == default:
            row = styled(ANSI.sub("", row), "33")
        if details and value in details:
            row += " " * (detail_column - len(plain_label) + 3) + styled(details[value], "90")
        if index == default:
            row += styled(" [Default]", "33")
        choices.append(row)
    position = number(question, default, maximum=len(options), choices=choices, blank_before=blank_before)
    return options[position - 1][0]


def path_value(value):
    return str(Path(value).expanduser().resolve()) if value else ""


def policy_value(value):
    if value.startswith(("~", "/", ".")) or Path(value).is_absolute() or Path(value).exists():
        return path_value(value)
    return value


def extra_flags():
    while True:
        try:
            return shlex.split(ask("Additional LeRobot Options (optional)", optional=True))
        except ValueError:
            retry_question()
            print("Check the quotation marks in the options.")


def select_path(question, options, default=None, optional=False, convert=path_value):
    if options:
        choice = choose(question, options + [(None, "Enter a path manually")])
        if choice is not None:
            return str(choice)
    return convert(ask(question, default, optional))


def choose_dataset(inventory, default=None, optional=False):
    return select_path("Training Dataset Directory", [(p, str(p)) for p, _ in inventory.datasets], default, optional)


def choose_model(inventory, model, optional=False):
    options = [(p, model_location(inventory, p)) for p, kind in inventory.models if kind == model]
    if optional:
        options.insert(0, ("", "Use the base model" if model == "tinyvla" else "Initialize a new model"))
    return select_path("Model Path or Repository Name" if model != "tinyvla" else "Model Directory", options,
                       optional=optional, convert=path_value if model == "tinyvla" else policy_value)


def tiny_resume(inventory):
    run = Path(select_path("Training Run to Resume", [(p, str(p)) for p in inventory.tiny_resume]))
    candidates = [directory / name for directory in (run / "executor", LOG_ROOT / run.name)
                  for name in ("script.py", "run.py")] + [run / "settings.json"]
    saved = next((path for path in candidates if path.is_file()), None)
    jobs = settings_jobs(load_settings(saved or ask("Saved Settings File for This Run")))
    if len(jobs) != 1:
        raise ValueError("Select settings for exactly one training run.")
    previous = jobs[0]
    if previous["adapter"] != "tinyvla" or "train" not in previous:
        raise ValueError("Select a TinyVLA training run directory.")
    previous_train = previous["train"]
    train = dict(previous_train)
    train.pop("warm_start", None)
    train.update(resume=str(run / "train/checkpoints/training_state.pt"),
                 selection=str(run / "selection.json"),
                 dataset_root=previous_train.get("dataset_root") or str(run / "dataset"))
    if not Path(train["dataset_root"]).is_dir():
        train["dataset_root"] = choose_dataset(inventory)
    return train


def tiny_train(warm, inventory, root):
    train = {}
    if warm:
        runs = [(p.parent.parent, str(p.parent.parent)) for p, kind in inventory.models if kind == "tinyvla"
                and (p / "action_head.safetensors").is_file() and (p.parent.parent / "selection.json").is_file()]
        run = Path(select_path("Existing TinyVLA Training Run", runs))
        train.update(warm_start=str(run / "train/pretrained_model"), selection=str(run / "selection.json"))
        default_dataset = str(run / "dataset") if (run / "dataset").is_dir() else None
        spec = read_json(Path(train["selection"]))
        tasks = list(spec.get("tasks", {}).values())
        if not tasks:
            raise ValueError("The existing selection file must include per-task training splits.")
        if len(tasks) == 1:
            train["task"] = [tasks[0]["suite"], tasks[0]["task_id"]]
        elif len(tasks) != 40:
            raise ValueError("The existing run must cover one task or all 40 tasks.")
    else:
        default_dataset = None
        suite = choose("Training Suite", [(name, name) for name in SUITES] + [("all", "All 40 tasks across four suites")])
        if suite != "all":
            train["task"] = [suite, number("Task ID", 1, 0, 9)]
        train["base_model"] = choose_model(inventory, "tinyvla", optional=True)
        train["selection"] = path_value(ask("Training Split File to Reuse (optional)", optional=True))
    train["dataset_root"] = choose_dataset(inventory, default_dataset or str(root / "datasets/libero"))
    train.update(updates=number("Training Steps per Task", 20), batch=number("Training Batch Size", 16),
                 lr=rate("Learning Rate", 0.0001), seed=number("Random Seed", 0, 0),
                 save_freq=number("Checkpoint Interval in Steps", 100))
    return train


def model_size(model):
    parameters = REFERENCE_MODEL_SIZES.get(model)
    if parameters is None:
        return "[N/A]"
    divisor, unit = (1_000_000_000, "B") if parameters >= 1_000_000_000 else (1_000_000, "M")
    amount = f"{parameters / divisor:.1f}".removesuffix(".0")
    return f"[{amount} {unit}]"


def choose_policy(adapters, policies):
    native_tag = "[LeRobot Native]"
    external_tag = "[External]"
    native = styled(native_tag, "38;5;75")
    external = styled(external_tag, "38;5;208") + " " * (len(native_tag) - len(external_tag))
    if not policies:
        print("\nCould not read the LeRobot model list. You can enter a model name manually.")
    entries = [(("native", model), f"{native} {model}") for model in sorted(policies)]
    entries.append((("native", ""), f"{native} Enter a model name manually"))
    entries += [((key, key), f"{external} {adapters[key].label}") for key in sorted(adapters) if key != "native"]
    sizes = {value: model_size(value[1]) for value, _ in entries if value[1]}
    default = next((index for index, (value, _) in enumerate(entries, 1) if value == ("native", "smolvla")), 1)
    backend, model = choose("Policy Model", entries, default=default, details=sizes)
    if not model:
        model = ask("LeRobot Policy Name", "smolvla")
    elif backend not in {"native", "tinyvla"}:
        model = ask("Model Name", model)
    return backend, model


def native_train(resume, inventory, model):
    if resume:
        models = {path: kind for path, kind in inventory.models}
        options = [(path, model_location(inventory, path)) for path in inventory.native_resume
                   if models.get(path, model) == model]
        path = select_path("Existing Run or Checkpoint Directory", options)
        checkpoint = native_checkpoint(Path(path))
        config = read_json(checkpoint / "train_config.json")
        checkpoint_model = config.get("policy", {}).get("type")
        if checkpoint_model and checkpoint_model != model:
            raise ValueError(f"The checkpoint uses {checkpoint_model}, but you selected {model}.")
        old_steps = config.get("steps", 1000)
        step_path = checkpoint.parent / "training_state/training_step.json"
        if step_path.is_file():
            old_steps = max(old_steps, read_json(step_path)["step"] + 1)
        train = {"resume": str(checkpoint)}
    else:
        old_steps = 1000
        options = [("", "Use a dataset repository")] + [(p, str(p)) for p, _ in inventory.datasets]
        dataset_root = select_path("Training Dataset", options, optional=True)
        info = next((info for p, info in inventory.datasets if str(p) == dataset_root), {})
        train = {"dataset": ask("Dataset Repository Name", info.get("repo_id")),
                 "dataset_root": dataset_root,
                 "pretrained": choose_model(inventory, model, optional=True),
                 "batch": number("Training Batch Size", 8), "device": ask("Training Device", "cuda"),
                 "seed": number("Random Seed", 0, 0)}
    train.update(steps=number("Target Training Step Count", old_steps),
                 save_freq=number("Checkpoint Interval in Steps", 100), extra=extra_flags())
    return train


def evaluation_suites(train, tiny):
    task = train.get("task") if tiny and train else None
    defaults = [task[0]] if task else list(SUITES) if tiny and train else ["libero_object"]
    options = [(name, name) for name in SUITES] + [
        ("all", "All four standard LIBERO suites"),
        ("manual", "Enter suite names manually"),
    ]
    default = SUITES.index(defaults[0]) + 1 if len(defaults) == 1 else len(SUITES) + 1
    choice = choose("Evaluation Suite", options, default=default)
    if choice == "all":
        return list(SUITES)
    if choice == "manual":
        names = ask("Evaluation Suites (comma-separated)", ",".join(defaults))
        return [name.strip() for name in names.split(",") if name.strip()]
    return [choice]


def eval_settings(model, tiny, train, inventory):
    if tiny:
        env_type = choose("Evaluation Benchmark", [("libero", "LIBERO (supported by TinyVLA)")])
    else:
        env_type = choose("Evaluation Benchmark", [
            ("libero", "LIBERO"),
            ("pusht", "PushT"),
            ("aloha", "ALOHA"),
            ("metaworld", "Meta-World"),
            ("manual", "Enter a LeRobot environment type"),
        ])
        if env_type == "manual":
            env_type = ask("LeRobot Environment Type")

    if env_type == "libero":
        tasks = evaluation_suites(train, tiny)
    else:
        default_task = {"pusht": "PushT-v0", "aloha": "AlohaInsertion-v0", "metaworld": "medium"}.get(env_type)
        tasks = [task.strip() for task in ask("Evaluation Tasks (comma-separated)", default_task).split(",")
                 if task.strip()]

    cfg = {"env": env_type, "tasks": tasks,
           "episodes": number("Episodes per Task", 5), "batch": number("Evaluation Batch Size", 1),
           "seed": number("Random Seed", 0, 0), "device": "cpu" if tiny else ask("Evaluation Device", "cuda")}
    if tiny:
        cfg["parallel_tasks"] = number("Parallel Evaluation Tasks", 1)
    if train is None:
        cfg["policy"] = choose_model(inventory, model, optional=tiny)
    if env_type == "libero":
        task = train.get("task") if tiny and train else None
        default_id = str(task[1]) if task and tasks == [task[0]] else "all"
        ids = ask("Task IDs (comma-separated, or all)", default_id) if len(tasks) == 1 else "all"
        cfg["task_ids"] = [int(value) for value in ids.split(",")] if ids != "all" else []
    if env_type in {"libero", "pusht", "aloha", "metaworld"}:
        cfg["episode_length"] = number("Episode Length (maximum control steps, 0 uses benchmark default)", 0, 0)
    if env_type == "libero":
        cfg.update(control="relative" if tiny else ask("Control Mode", "absolute" if model == "xvla" else "relative"),
                   size=256 if tiny else number("Camera Image Width and Height", 448 if model == "evo1" else 256))
    if tiny:
        cfg["action_steps"] = number("Actions to Execute per Prediction", 16, 1, 16)
    cfg["extra"] = extra_flags()
    return cfg


def script_paths():
    if not SCRIPT_ROOT.is_dir():
        return []
    queues = []
    for path in sorted(SCRIPT_ROOT.iterdir()):
        if not path.is_file() or path.name.startswith((".", "_")) or path.suffix not in {".py", ".json"}:
            continue
        try:
            jobs = settings_jobs(load_settings(path))
        except (OSError, SyntaxError, ValueError, KeyError, TypeError):
            continue
        if all({"adapter", "model"} <= job.keys() and {"train", "eval"} & job.keys() for job in jobs):
            queues.append(path)
    return queues


def run_in_use(run):
    try:
        status = read_json(run / "executor/status.json")
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True
    return not isinstance(status, dict) or status.get("status") not in ("completed", "failed", "interrupted")


def delete_result_runs(root, category, *, notice=print):
    label = {"evaluation": "Evaluation", "training": "Training"}[category]
    folder = root.resolve() / "outputs" / category
    if folder.resolve() != folder:
        raise ValueError(f"Refusing to clean a redirected result folder ({folder}).")
    if not folder.is_dir():
        notice(f"No {label} Run Folders Found.")
        return False

    selected = []
    protected = 0
    for run in sorted(folder.iterdir()):
        if run.name.startswith(".") or not run.is_dir():
            continue
        if run.resolve() != run or run_in_use(run):
            protected += 1
            continue
        selected.append((run, run.stat()))

    if protected:
        print(f"Protected Run Folders: {protected} (active, unreadable, or redirected).")
    if not selected:
        notice(f"No {label} Run Folders Available to Delete.")
        return False
    print(f"Result Folder: {folder}")
    if category == "training":
        print("This Also Deletes Checkpoints Stored in the Selected Run Folders.")
    count = len(selected)
    unit = "Run Folder" if count == 1 else "Run Folders"
    if ask(f"Permanently Delete {count} {label} {unit}? (Y/N)", "N").upper() != "Y":
        notice("Deletion Canceled.")
        return False

    deleted = 0
    for run, original in selected:
        try:
            if (folder.resolve() != folder or run.resolve() != run or not os.path.samestat(run.stat(), original)
                    or run_in_use(run)):
                print(f"Skipped Changed Run Folder: {run}")
                continue
            shutil.rmtree(run)
            deleted += 1
        except OSError as error:
            print(f"Could Not Delete {run} ({error}).", file=sys.stderr)
    notice(f"Deleted {deleted} of {count} {label} {unit}.")
    return deleted > 0


def wizard(adapters, inventory, root, conda=None, native_env="lerobot", *,
           external=None, environments=(), allow_delete=True):
    display = QuestionDisplay(sys.stdout, section)
    with redirect_stdout(display):
        while True:
            display.show()
            actions = [("new", "Configure a new job")]
            if inventory.settings:
                actions.append(("reuse", "Reuse settings from a previous run"))
            actions.append(("import", "Import a job queue script"))
            if inventory.results:
                actions.append(("results", "View existing results"))
            if allow_delete:
                actions.extend((("evaluation", "Delete all evaluation results"),
                                ("training", "Delete all training results")))
            actions.append(("quit", "Exit"))
            action = choose("What Would You Like to Do?", actions, blank_before=False)
            if action == "quit":
                return None

            history = QuestionHistory(display)
            try:
                if action == "new":
                    policies = native_policies(conda, native_env, root)
                    return configure_jobs(adapters, inventory, root, policies, display)
                if action == "reuse":
                    files = [(p, str(p)) for p in inventory.settings]
                    path = history.run(lambda: select_path("Settings File", files))
                    return load_settings(path)
                if action == "import":
                    scripts = [(p, p.name) for p in script_paths()]
                    path = history.run(lambda: select_path("Job Queue Script", scripts))
                    return load_settings(path)
                if action in {"evaluation", "training"}:
                    deleted = history.run(lambda: delete_result_runs(root, action, notice=display.notice))
                    if deleted:
                        inventory = scan(root, external or root / "externals/TinyVLA", root / "executor/log")
                        with redirect_stdout(display.stream):
                            section("Status")
                            show(inventory, environments, limit=5)
                    continue
                files = [(p, str(p)) for p in inventory.results]
                path = history.run(lambda: select_path("Result File", files))
            except PreviousQuestion:
                continue

            result = read_json(path)
            summary = result.get("overall") or {key: result[key] for key in
                        ("status", "steps", "batch_size", "resumed_from_step", "checkpoint") if key in result}
            display.notice(json.dumps(summary or result, ensure_ascii=False, indent=2))


def configure_jobs(adapters, inventory, root, policies, display):
    def configure():
        job = configure_job(adapters, inventory, root, policies)
        return job, ask("Add Another Job? (Y/N)", "N").upper() == "Y"

    jobs = []
    histories = [QuestionHistory(display)]
    while True:
        try:
            job, another = histories[-1].run(configure, job_number=len(histories))
        except PreviousQuestion:
            if not jobs:
                raise
            histories.pop()
            jobs.pop()
            histories[-1].rewind()
            continue
        jobs.append(job)
        if not another:
            return {"schema": 1, "jobs": jobs}
        histories.append(QuestionHistory(display))


def configure_job(adapters, inventory, root, policies):
    backend, model = choose_policy(adapters, policies)
    tiny = backend == "tinyvla"
    choices = [("train", "Start training"), ("eval", "Evaluate"), ("resume", "Resume training from a checkpoint"),
               ("both", "Train, then evaluate")]
    if tiny:
        choices.append(("warm", "Start training with existing action-head weights"))
    operation = choose("Job Type", choices)
    cfg = {"schema": 1, "adapter": backend, "model": model}
    if operation != "eval":
        print("\n" + styled("Training Setup", "1;36"))
        if tiny:
            print("\nTraining benchmark · LIBERO")
            cfg["train"] = tiny_resume(inventory) if operation == "resume" else tiny_train(operation == "warm", inventory, root)
        else:
            cfg["train"] = native_train(operation == "resume", inventory, model)
    evaluate = operation in ("eval", "both") or (
        operation in ("resume", "warm") and ask("Evaluate After Training? (Y/N)", "N").upper() == "Y"
    )
    if evaluate:
        print("\n" + styled("Evaluation Setup", "1;36"))
        cfg["eval"] = eval_settings(cfg["model"], tiny, cfg.get("train"), inventory)
    return cfg


def paths_for(args, cfg):
    root = Path(args.lerobot_root or cfg.get("lerobot_root") or PROJECT_ROOT).expanduser().resolve()
    if not root.is_dir():
        if args.lerobot_root:
            raise FileNotFoundError(f"Input base directory not found ({root}).")
        root = PROJECT_ROOT
    bundled = PROJECT_ROOT / "externals/TinyVLA"
    if args.tinyvla_root:
        return root, Path(args.tinyvla_root).expanduser().resolve()
    saved = cfg.get("tinyvla_root")
    if saved:
        configured = Path(saved).expanduser().resolve()
        legacy = configured.parent == root / "externals" and configured.name.lower() == "tinyvla"
        if configured.is_dir() and not legacy:
            return root, configured
    return root, bundled


def settings_jobs(settings):
    if "jobs" not in settings:
        return [settings]
    jobs = settings["jobs"]
    defaults = settings.get("defaults", {})
    if settings.get("schema", 1) != 1 or not isinstance(defaults, dict) or not isinstance(jobs, list) or not jobs:
        raise ValueError("Use schema 1 and include at least one job in the queue.")
    if not all(isinstance(job, dict) for job in jobs):
        raise ValueError("Each queue entry must be a dictionary.")
    return [{"schema": 1, **defaults, **job} for job in jobs]


def normalize_paths(cfg, root):
    def local(value):
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        project_path = path.parts[:1] in {("datasets",), ("outputs",), ("externals",)}
        return str(((PROJECT_ROOT if project_path else root) / path).resolve())

    def reference(value):
        resolved = local(value)
        return resolved if Path(value).expanduser().is_absolute() or value.startswith(".") or Path(resolved).is_dir() else value

    cfg = dict(cfg)
    if "train" in cfg:
        train = cfg["train"] = dict(cfg["train"])
        for key in ("dataset_root", "selection", "resume", "warm_start", "base_model"):
            if train.get(key):
                train[key] = local(train[key])
        if train.get("pretrained"):
            train["pretrained"] = reference(train["pretrained"])
    if "eval" in cfg:
        evaluation = cfg["eval"] = dict(cfg["eval"])
        if evaluation.get("policy"):
            evaluation["policy"] = (local if cfg["adapter"] == "tinyvla" else reference)(evaluation["policy"])
    return cfg


def plan_job(args, cfg, conda, detected):
    root, external = paths_for(args, cfg)
    plugins = args.plugin or cfg.get("plugins", [])
    adapters = load_adapters(plugins)
    if cfg.get("schema") != 1 or cfg.get("adapter") not in adapters or not {"train", "eval"} & cfg.keys():
        raise ValueError("Specify schema 1, a supported adapter, and a train or eval job.")
    if not isinstance(cfg["model"], str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", cfg["model"]):
        raise ValueError("Model names may contain letters, digits, periods, underscores, and hyphens.")
    cfg = normalize_paths(cfg, root)
    environments = {**DEFAULT_ENVIRONMENTS, **cfg.get("environments", {})}
    if args.env_map:
        environments.update(read_json(args.env_map))
    if PROJECT_ROOT not in detected:
        detected[PROJECT_ROOT] = available_environments(conda, PROJECT_ROOT)
    installed = detected[PROJECT_ROOT]
    if (installed and environments["tinyvla"] == "tinyvla_infer" and
            "tinyvla_infer" not in installed and "lerobot_tinyvla" in installed):
        environments["tinyvla"] = "lerobot_tinyvla"
    if any(not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", value)
           for value in environments.values()):
        raise ValueError("Invalid Conda environment name.")
    cfg.update(environments=environments, plugins=plugins, lerobot_root=str(root), tinyvla_root=str(external))
    name = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{cfg['model'][:50]}_{uuid4().hex[:8]}"
    adapter = adapters[cfg["adapter"]](cfg, name, PROJECT_ROOT, external, environments)
    stages = []
    if "train" in cfg:
        stages.append(adapter.train())
    if "eval" in cfg:
        stages.append(adapter.evaluate(after_training="train" in cfg))
    for stage in stages:
        stage.env.setdefault("HF_LEROBOT_HOME", str(PROJECT_ROOT / "datasets/.hub"))
    first = stages[0]
    output = first.output or PROJECT_ROOT / "outputs" / ("training" if first.name == "train" else "evaluation") / name
    run_dir = (output.parent if first.name == "train" and first.output else output) / "executor"
    return cfg, stages, run_dir, PROJECT_ROOT


def plan(args):
    source = args.settings or args.queue
    settings = load_settings(source) if source else None
    if args.queue and "jobs" not in settings:
        raise ValueError("Use --queue with a file containing a JOBS list.")
    try:
        conda = conda_executable(args.conda)
    except FileNotFoundError:
        conda = None
    detected = {}
    if args.inventory or settings is None:
        _, external = paths_for(args, settings or {})
        detected[PROJECT_ROOT] = available_environments(conda, PROJECT_ROOT)
        inventory = scan(PROJECT_ROOT, external, LOG_ROOT)
        section("Status")
        show(inventory, detected[PROJECT_ROOT], limit=None if args.inventory else 5)
        if args.inventory:
            return [], conda
        env_map = read_json(args.env_map) if args.env_map else {}
        native_env = env_map.get("native", DEFAULT_ENVIRONMENTS["native"])
        settings = wizard(load_adapters(args.plugin or []), inventory, PROJECT_ROOT, conda, native_env,
                          external=external, environments=detected[PROJECT_ROOT], allow_delete=not args.dry_run)
        if settings is None:
            return [], conda

    plans = []
    jobs = settings_jobs(settings)
    for index, cfg in enumerate(jobs, 1):
        try:
            plans.append(plan_job(args, cfg, conda, detected))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError) as error:
            raise ValueError(f"Invalid settings for job {index}/{len(jobs)} ({error})") from error
    return plans, conda


def command_preview(arguments):
    groups = []
    for argument in arguments:
        if not groups or argument.startswith("--"):
            groups.append([])
        groups[-1].append(argument)
    return " \\\n".join(f"    {shlex.join(group)}" for group in groups)


def job_title(index, cfg):
    title = styled(f"[Job {index}]", "1;36")
    return f"{title} {cfg.get('label') or cfg['model']}"


def show_settings(plans, show_commands):
    for index, (cfg, stages, _, _) in enumerate(plans, 1):
        if index > 1:
            print()
        print(job_title(index, cfg))
        for stage in stages:
            print("  " + stage_title(stage, cfg["model"]))
            if show_commands:
                print(command_preview(stage.arguments))
        evaluation = cfg.get("eval")
        if evaluation and evaluation["batch"] > evaluation["episodes"]:
            print(f"  Effective evaluation batch size: {evaluation['episodes']} "
                  f"({evaluation['batch']} requested, {evaluation['episodes']} episodes per task)")


def job_subheading(title):
    print(styled(f"  ({title})", "1;94"), flush=True)


def run_plans(plans, conda, verbose):
    for index, (cfg, stages, run_dir, root) in enumerate(plans, 1):
        if index > 1:
            print()
        print(job_title(index, cfg), flush=True)
        job_subheading("Status")
        execute(stages, cfg, run_dir, root, conda, verbose)
        show_job_result(stages)


def metric(stats, key):
    value = stats.get(key)
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def success_rate(stats):
    if not isinstance(stats, dict):
        return "N/A"
    successes, episodes = stats.get("n_success"), stats.get("n_episodes")
    if type(successes) is int and type(episodes) is int and episodes > 0 and 0 <= successes <= episodes:
        return f"{successes}/{episodes} ({100 * successes / episodes:.2f}%)"
    percent = metric(stats, "pc_success")
    return f"{percent:.2f}%" if percent is not None and percent <= 100 else "N/A"


def evaluation_summary(output):
    try:
        info = read_json(output / "eval_info.json")
    except (OSError, ValueError):
        info = {}
    if not isinstance(info, dict):
        info = {}
    overall = info.get("overall") or info.get("aggregated")
    if not isinstance(overall, dict):
        overall = {}
    total = metric(overall, "eval_s")
    average = metric(overall, "eval_ep_s")
    if average is None and total is not None and (episodes := metric(overall, "n_episodes")):
        average = total / episodes
    average_text = f"{average:.2f} s" if average is not None else "N/A"

    lines = [f"• Overall success rate: {success_rate(overall)}"]
    groups = info.get("per_group")
    if isinstance(groups, dict) and groups:
        lines.append("• Suite success rates:")
        lines.extend(f"  ◦ {name}: {success_rate(stats)}" for name, stats in groups.items())
    else:
        lines.append("• Suite success rates: N/A")
    lines.extend((f"• Total evaluation time: {seconds(total)}",
                  f"• Average time per episode: {average_text}", f"• Result path: {output}"))
    return "\n".join(lines)


def result_heading(in_job):
    if in_job:
        job_subheading("Result")
    else:
        section("Result")


def show_job_result(stages):
    result_heading(in_job=True)
    evaluations = [stage.output for stage in stages if stage.name == "eval" and stage.output is not None]
    if evaluations:
        print("\n\n".join(indent(evaluation_summary(output), "    ") for output in evaluations))
    else:
        print(styled("    • Job completed.", "32"))
        for stage in stages:
            if stage.output is not None:
                print(f"    • Result path: {stage.output}")
    divider()


def report_failure(message, in_job):
    result_heading(in_job)
    prefix = "    • " if in_job else ""
    print(f"{prefix}{message}", file=sys.stderr, flush=True)
    divider()


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LeRobot policies and external models")
    parser.add_argument("--lerobot-root", help="Base for older relative input paths outside datasets, outputs, and externals")
    parser.add_argument("--tinyvla-root", help="TinyVLA deployment directory")
    parser.add_argument("--conda", help="Path to the Conda executable")
    parser.add_argument("--env-map", type=Path, help="File mapping model names to Conda environments")
    parser.add_argument("--plugin", action="append", help="Trusted adapter in module:Class format")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--settings", type=Path, help="Load a settings dictionary or reuse a previous run")
    source.add_argument("--queue", type=Path, help="Load and run a job queue in file order")
    parser.add_argument("--inventory", action="store_true", help="List local datasets, checkpoints, and results")
    parser.add_argument("--dry-run", action="store_true", help="Show the execution plan without creating files")
    parser.add_argument("--verbose", action="store_true", help="Show detailed execution logs")
    args = parser.parse_args()
    executing = False
    try:
        plans, conda = plan(args)
        if not plans:
            return 0
        section("Settings")
        show_settings(plans, show_commands=not (args.settings or args.queue) or args.dry_run)
        if args.dry_run:
            return 0
        count = len(plans)
        job_label = "job" if count == 1 else "jobs"
        if not (args.settings or args.queue) and ask(f"Run {count} {job_label}? (Y/N)", "Y").upper() != "Y":
            section("Result")
            print("Run canceled.")
            divider()
            return 0
        conda = conda or conda_executable(args.conda)
        section("Execution")
        executing = True
        run_plans(plans, conda, args.verbose)
        return 0
    except (KeyboardInterrupt, EOFError):
        report_failure("Run interrupted.", in_job=executing)
        return 130
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, ImportError, SyntaxError,
            subprocess.TimeoutExpired) as error:
        report_failure(f"Run failed ({error})", in_job=executing)
        return 1


if __name__ == "__main__":
    sys.exit(main())
