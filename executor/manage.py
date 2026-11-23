#!/usr/bin/env python3
"""Interactive LeRobot training, evaluation, and resume manager."""

import argparse
from contextlib import ExitStack, nullcontext, redirect_stdout
from datetime import datetime, timezone
from itertools import product
import json
import math
import os
from pathlib import Path
from pprint import pformat
import re
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
from textwrap import indent
from uuid import uuid4

from adapters import (
    EPISODE_LENGTH_BENCHMARKS,
    EVALUATION_STEP_OPTIONS,
    LIBERO_BENCHMARKS,
    SINGLE_TASK_BENCHMARKS,
    SUITES,
    TASK_ID_BENCHMARKS,
    TRAINING_SWITCHES,
    load_adapters,
    native_checkpoint,
    native_peft_checkpoint,
    read_json,
    structured_policy_value,
)
from benchmarks import BENCHMARKS, DEPENDENCY_CONFLICT
from evaluation_capabilities import evaluation_notices, stock_evaluation_issue
from inventory import model_location, scan, show
from questions import PreviousQuestion, QuestionDisplay, QuestionHistory, active_history, retry_question
from runtime_setup import RuntimeSetupFailure
from runtime_activation import ActivationError, configure_clone_inheritance, prepare_clone_source
from training_preflight import native_training_issue
from runner import (ANSI, StageFailure, available_environments, conda_command, conda_executable, environment_catalog,
                    execute, interrupt_on_termination, load_settings, seconds, stage_failure_reason, stage_title, styled, write_json)


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
SOURCE_BENCHMARKS = {"robotwin", "robocasa", "vlabench", "libero_plus"}
HUB_PATH_VARIABLES = ("HF_HOME", "HF_HUB_CACHE", "HF_TOKEN_PATH", "HF_ASSETS_CACHE", "HF_XET_CACHE")
MODEL_CACHE_VARIABLES = (*HUB_PATH_VARIABLES, "TORCH_HOME", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE")


def simulator_entrypoint(stage):
    """Add simulator hooks without discarding another native evaluation helper."""
    helper = str(Path(__file__).with_name("evaluation_cameras.py").resolve())
    if stage.arguments[0] == "lerobot-eval":
        stage.arguments[:1] = ["python", "-u", helper]
        return
    first_flag = next((index for index, item in enumerate(stage.arguments) if item.startswith("--")),
                      len(stage.arguments))
    known = {str(Path(__file__).with_name(name).resolve())
             for name in ("evaluation_cameras.py", "pinned_policy.py", "parallel_eval.py")}
    if not any(item in known for item in stage.arguments[:first_flag]):
        raise RuntimeError("The simulator cannot be attached to an unknown evaluation entry point.")


def divider():
    print(styled("-" * shutil.get_terminal_size().columns, "90"), flush=True)


def section(title):
    divider()
    print(styled(f"<{title}>", "1;95"), flush=True)


def ask(question, default=None, optional=False, *, choices=(), blank_before=True):
    history = active_history.get()
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
    if history is not None:
        position = history.position
        if position < len(history.answers):
            answer = history.answers[position]
            history.position += 1
            if not history.replaying:
                history.marks[position] = history.display.position
                print(f"{prompt}: {answer}", flush=True)
            return answer
        history.replaying = False

    read = sys.stdout.read if isinstance(sys.stdout, QuestionDisplay) else input
    while True:
        mark = history.display.position if history is not None else None
        try:
            value = read(f"{prompt}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise
        if not sys.stdin.isatty():
            print()
        if history is not None and value.casefold() == "b":
            raise PreviousQuestion
        if value or default is not None or optional:
            answer = value if value else default if default is not None else ""
            if history is not None:
                history.answers.append(answer)
                history.marks.append(mark)
                history.position += 1
            return answer
        retry_question("Enter a response.", accepted=False)


def confirm(question, default):
    active = isinstance(sys.stdout, QuestionDisplay)
    display = sys.stdout if active else QuestionDisplay(sys.stdout, section)
    with redirect_stdout(display), (nullcontext() if active else display):
        while True:
            answer = ask(question, default).casefold()
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            retry_question("Enter Y or N.")


def number(question, default, minimum=1, maximum=None, *, choices=(), blank_before=True):
    while True:
        try:
            value = int(ask(question, default, choices=choices, blank_before=blank_before))
            if value >= minimum and (maximum is None or value <= maximum):
                return value
        except ValueError:
            pass
        retry_question("Enter an integer within the allowed range.")


def rate(question, default):
    while True:
        try:
            value = float(ask(question, default))
            if math.isfinite(value) and value > 0:
                return value
        except ValueError:
            pass
        retry_question("Enter a positive number.")


def choose(question, options, default=1, *, details=None, groups=None, unavailable=(), reasons=None, blank_before=True):
    choices = []
    labels = [ANSI.sub("", label) for _, label in options]
    columns = {}
    if details:
        for (value, _), label in zip(options, labels):
            if value in details:
                group = groups.get(value) if groups else None
                columns[group] = max(columns.get(group, 0), len(label))
    number_width = len(str(len(options))) if details else 1
    for index, ((value, label), plain_label) in enumerate(zip(options, labels), 1):
        row = f"  {index:>{number_width}}. {label}"
        if index == default:
            row = styled(ANSI.sub("", row), "33")
        if details and value in details:
            group = groups.get(value) if groups else None
            row += " " * (columns[group] - len(plain_label) + 3) + styled(details[value], "90")
        if value in unavailable:
            reason = (reasons or {}).get(value, "Not Installed")
            row = styled(ANSI.sub("", row) + f" <{reason}>", "2;90")
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
            retry_question("Check the quotation marks in the options.")


def select_path(question, options, default=None, optional=False, convert=path_value):
    if options:
        choice = choose(question, options + [(None, "Enter a path manually")])
        if choice is not None:
            return str(choice)
    return convert(ask(question, default, optional))


def choose_dataset(inventory, default=None, optional=False):
    return select_path("Training Dataset Directory", [(p, str(p)) for p, _ in inventory.datasets], default, optional)


def native_dataset(inventory):
    options = [("", "Use a dataset repository"), *((path, str(path)) for path, _ in inventory.datasets)]
    root = select_path("Training Dataset", options)
    if not root:
        return {"dataset": ask("Dataset Repository Name"), "dataset_root": ""}

    folder = Path(root)
    info = read_json(folder / "meta/info.json")
    split_file = folder / "selection.json"
    selection = read_json(split_file) if split_file.is_file() else {}
    name = re.sub(r"[^a-zA-Z0-9]+", "-", folder.name).strip("-")[:80].rstrip("-") or "dataset"
    repo = selection.get("dataset_repo") or info.get("repo_id") or f"local/{name}"
    return {"dataset": repo, "dataset_root": root}


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
    scope = choose("Training Scope", [
        ("head", "Train the action head"),
        ("lora", "Train the action head and language-attention LoRA"),
    ])
    if scope == "lora":
        train["lora"] = {"rank": number("LoRA Rank", 16), "alpha": number("LoRA Alpha", 16),
                         "dropout": number("LoRA Dropout Percent", 0, 0, 99) / 100}
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


def choose_policy(adapters, policies, inventory):
    native_tag = "[LeRobot Native]"
    external_tag = "[External]"
    native = styled(native_tag, "38;5;75")
    external = styled(external_tag, "38;5;208")
    if not policies:
        print("\nCould not read the LeRobot model list. You can enter a model name manually.")
    entries = [(("native", model), f"{native} {model}") for model in sorted(policies)]
    entries.append((("native", ""), f"{native} Enter a model name manually"))
    entries += [((key, key), f"{external} {adapters[key].label}") for key in sorted(adapters) if key != "native"]
    local = {kind for _, kind in inventory.models}
    reasons = {value: "No Local Checkpoint" for value, _ in entries if value[1] and value[1] not in local}
    sizes = {value: model_size(value[1]) for value, _ in entries if value[1]}
    groups = {value: "native" if value[0] == "native" else "external" for value, _ in entries}
    backend, model = choose("Policy Model", entries, default=None, details=sizes,
                            groups=groups, unavailable=reasons, reasons=reasons)
    if not model:
        model = ask("LeRobot Policy Name")
    elif backend not in {"native", "tinyvla"}:
        model = ask("Model Name")
    return backend, model


def component_scopes(model, fields, pretrained=None):
    if not fields:
        return []
    if "train_mode_vlm" in fields:
        choices = [
            ({"train_mode_vlm": "fft", "freeze_embedding": False}, "Action expert + full vision-language model"),
            ({"train_mode_vlm": "fft", "freeze_embedding": True}, "Action expert + vision-language model (embeddings frozen)"),
            ({"train_mode_vlm": "lora"}, "Action expert + vision-language LoRA"),
        ]
        checkpoint = Path(pretrained or ".") / "config.json"
        if not pretrained or (checkpoint.is_file() and read_json(checkpoint).get("action_mode", "continuous") == "continuous"):
            choices.append(({"train_mode_vlm": "freeze"}, "Action expert only"))
        if "freeze_embedding" not in fields:
            choices = [(scope, label) for scope, label in choices if not scope.get("freeze_embedding")]
            for scope, _ in choices:
                scope.pop("freeze_embedding", None)
        return choices

    partial = {"groot": "tune_llm", "lawam": "freeze_llm_backbone"}.get(model)
    fields = [name for name in fields if name in TRAINING_SWITCHES]
    combinations = []
    for values in product(*((False, True, "partial") if field == partial else (False, True) for field in fields)):
        if not fields:
            break
        scope = dict(zip(fields, values))
        expert = scope.get("train_expert_only")
        if expert and scope.get("freeze_vision_encoder") is False:
            continue
        if model in {"groot", "evo1"} and not any(values):
            continue
        if model == "lawam" and scope.get("freeze_vision_backbone") is False and scope.get("unfreeze_vision_merger") is False:
            continue

        if expert:
            title = "Action expert"
            if model == "fineart_vla":
                title += " (flow loss only)"
        elif "train_expert_only" in scope:
            title = ("Action expert + non-vision backbone" if scope.get("freeze_vision_encoder")
                     else "Action expert + full backbone")
        else:
            title = ""
        labels = [title] if title else []
        for field, value in scope.items():
            if field == "train_expert_only" or field == "freeze_vision_encoder" and "train_expert_only" in scope:
                continue
            if field == "unfreeze_vision_merger" and scope.get("freeze_vision_backbone") is False:
                continue
            component = TRAINING_SWITCHES[field]
            if value == "partial":
                labels.append(f"{component} top layers trained")
            else:
                trained = not value if field.startswith("freeze_") else value
                labels.append(f"{component} {'trained' if trained else 'frozen'}")
        combinations.append((scope, " · ".join(labels)))
    return combinations


def configure_training_scope(train, model, catalog):
    fields = catalog.get("training", {}).get(model, ())
    scopes = [({}, "Use policy or checkpoint defaults"), *component_scopes(model, fields, train.get("pretrained"))]
    peft = bool(train.get("pretrained") and model in catalog.get("peft_policies", ()))
    if len(scopes) == 1 and not peft:
        print("Training Scope · Policy or checkpoint defaults")
        return
    options = [(index, label) for index, (_, label) in enumerate(scopes)]
    if peft:
        options.append(("lora", "LoRA adapters with checkpoint training scope"))
    choice = choose("Training Scope", options)
    if choice != "lora":
        scope = scopes[choice][0]
        if model == "groot" and "tune_llm" in scope:
            language = scope["tune_llm"]
            scope["tune_llm"] = language is True
            scope["tune_top_llm_layers"] = number("Trainable Top Language Layers", 1) if language == "partial" else 0
        if model == "lawam" and "freeze_llm_backbone" in scope:
            language = scope["freeze_llm_backbone"]
            scope["freeze_llm_backbone"] = language is not False
            scope["unfreeze_llm_last_n_layers"] = number("Trainable Top Language Layers", 1) if language == "partial" else -1
        if scope.get("train_mode_vlm") == "lora":
            scope["lora_rank"] = number("LoRA Rank", 64)
            scope["lora_alpha"] = number("LoRA Alpha", 16)
            scope["lora_dropout"] = number("LoRA Dropout Percent", 5, 0, 100) / 100
        if scope:
            train["policy_training"] = scope
    else:
        lora = {"rank": number("LoRA Rank", 16), "alpha": number("LoRA Alpha", 16)}
        targets = ask("LoRA Target Modules (regular expression, blank uses model defaults)", optional=True)
        if targets:
            lora["target_modules"] = targets
        full = ask("Fully Train Modules (comma-separated, none, or blank for defaults)", optional=True)
        if full:
            lora["full_training_modules"] = [] if full.casefold() == "none" else [
                name.strip() for name in full.split(",") if name.strip()
            ]
        train["lora"] = lora


def configure_policy_options(train, options, *, question="Policy Training Options"):
    if not options:
        return
    # A saved policy can override the constructor defaults inspected in the
    # runtime catalog. Reading this local file must not download a Hub model.
    options = {name: dict(spec) for name, spec in options.items()}
    reference = train.get("pretrained")
    if reference and (Path(reference).expanduser() / "config.json").is_file():
        saved = read_json(Path(reference).expanduser() / "config.json")
        if isinstance(saved, dict):
            for name, spec in options.items():
                if name not in saved:
                    continue
                value = saved[name]
                if spec.get("value_type") in {"numeric_object", "numeric_sequence"}:
                    try:
                        spec["default"] = structured_policy_value(value, spec)
                    except ValueError:
                        pass
                    continue
                valid = value is None and spec.get("nullable")
                value_type = {"bool": bool, "int": int, "float": float, "str": str}.get(
                    spec.get("value_type"), type(spec["default"]))
                valid = valid or type(value) is value_type or value_type is float and type(value) is int
                if (valid and ("choices" not in spec or value in spec["choices"])
                        and (type(value) is not float or math.isfinite(value))):
                    spec["default"] = value
    values = {}
    inference_labels = {"n_action_steps": "Actions to Execute per Prediction", "num_steps": "Inference Steps",
                        "num_denoise_steps": "Denoising Steps", "dtype": "Model Data Type",
                        "use_amp": "Automatic Mixed Precision"}
    if question == "Policy Evaluation Options":
        inference_labels["use_amp"] = "CUDA Automatic Mixed Precision"
    labels = {name: options[name].get("label", inference_labels.get(name, name.replace("_", " ").title().replace("Lora", "LoRA")))
              for name in sorted(options)}
    while True:
        selection = choose(question, [("", "Finish / keep other defaults"), *[
            (name, f"{label} [{values.get(name, options[name]['default'])}]") for name, label in labels.items()
        ]])
        if not selection:
            for name, value in values.items():
                target = options[name].get("setting_group", "policy_options")
                if target not in {"policy_training", "policy_options"}:
                    raise ValueError(f"Unsupported configuration group ({target}).")
                train.setdefault(target, {})[name] = value
            return
        spec = options[selection]
        default = values.get(selection, spec["default"])
        if spec.get("value_type") in {"numeric_object", "numeric_sequence"}:
            while True:
                response = ask(labels[selection] + " (JSON)", json.dumps(default))
                try:
                    values[selection] = structured_policy_value(json.loads(response), spec)
                except (TypeError, ValueError):
                    retry_question("Enter a valid JSON value using the displayed numeric fields and dimensions.")
                    continue
                break
        elif "choices" in spec or type(default) is bool:
            allowed = spec.get("choices", [True, False])
            choices = [(value, "Automatic (null)" if value is None else str(value))
                       for value in allowed]
            values[selection] = choose(labels[selection], choices, default=allowed.index(default) + 1)
        else:
            while True:
                response = ask(labels[selection], "null" if default is None else default)
                value_type = {"int": int, "float": float, "str": str}.get(spec.get("value_type"), type(default))
                try:
                    value = (None if spec.get("nullable") and response.casefold() == "null"
                             else value_type(response))
                    if type(value) is float and not math.isfinite(value):
                        raise ValueError
                    if value is not None and ("minimum" in spec and value < spec["minimum"]
                                              or "maximum" in spec and value > spec["maximum"]):
                        raise ValueError
                    if selection in EVALUATION_STEP_OPTIONS and value is not None and (type(value) is not int or value < 1):
                        retry_question("Enter a positive inference step count, or null when supported.")
                        continue
                except (TypeError, ValueError):
                    suffix = " or null for automatic/default behavior" if spec.get("nullable") else ""
                    retry_question(f"Enter a valid {value_type.__name__} value{suffix}.")
                    continue
                values[selection] = value
                break


def native_train(resume, inventory, model, catalog=None):
    if resume:
        print("Training Scope · Restore checkpoint settings")
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
        train = {**native_dataset(inventory),
                 "pretrained": choose_model(inventory, model, optional=True),
                 "batch": number("Training Batch Size", 8), "device": ask("Training Device", "cuda"),
                 "seed": number("Random Seed", 0, 0)}
        configure_training_scope(train, model, catalog or {})
        policy_options = (catalog or {}).get("training_options", {}).get(model, {})
        if model == "fineart_vla" and train.get("policy_training", {}).get("train_expert_only"):
            policy_options = {key: value for key, value in policy_options.items() if key != "text_loss_weight"}
        configure_policy_options(train, policy_options)
    train.update(steps=number("Target Training Step Count", old_steps),
                 save_freq=number("Checkpoint Interval in Steps", 100), extra=extra_flags())
    return train


def evaluation_names(question, default=None):
    while True:
        names = [name.strip() for name in ask(question, default).split(",")]
        if all(names) and len(names) == len(set(names)):
            return names
        retry_question("Enter unique names separated by commas.")


def evaluation_task_ids(suite, default, env_type, *, allow_all=True):
    label = "Variant IDs per Suite" if env_type == "libero_plus" else "Task IDs"
    upper = 10 if env_type == "libero" and suite in SUITES else None
    hint = "comma-separated, or all" if allow_all else "comma-separated"
    while True:
        answer = ask(f"{label} ({hint})", default)
        if allow_all and answer.casefold() == "all":
            return []
        try:
            ids = [int(value) for value in answer.split(",")]
        except ValueError:
            ids = []
        if ids and len(ids) == len(set(ids)) and all(i >= 0 and (upper is None or i < upper) for i in ids):
            return ids
        retry_question(f"Enter unique, in-range {label.lower()}.")


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
        return evaluation_names("Evaluation Suites (comma-separated)", ",".join(defaults))
    return [choice]


def eval_settings(model, tiny, train, inventory, env_type, catalog=None):
    if env_type in LIBERO_BENCHMARKS:
        tasks = evaluation_suites(train, tiny)
    elif env_type in SINGLE_TASK_BENCHMARKS:
        while True:
            task = ask("Evaluation Task", SINGLE_TASK_BENCHMARKS[env_type])
            if "," not in task:
                tasks = [task]
                break
            retry_question("Enter one evaluation task.")
    else:
        default_task = {
            "metaworld": "medium", "robocasa": "CloseFridge", "robomme": "PickXtimes",
            "robotwin": "beat_block_hammer", "vlabench": "select_fruit",
        }.get(env_type)
        tasks = evaluation_names("Evaluation Tasks (comma-separated)", default_task)

    cfg = {"env": env_type, "tasks": tasks,
           "episodes": number("Episodes per Task", 5), "batch": number("Evaluation Batch Size", 1),
           "seed": number("Random Seed", 0, 0), "device": "cpu" if tiny else ask("Evaluation Device", "cuda")}
    if env_type not in SINGLE_TASK_BENCHMARKS:
        cfg["parallel_tasks"] = number("Parallel Evaluation Tasks", 1)
    if train is None:
        cfg["policy"] = choose_model(inventory, model, optional=tiny)
    if env_type in TASK_ID_BENCHMARKS:
        task = train.get("task") if tiny and train else None
        default_id = "all" if env_type == "libero" else "0"
        if task and tasks == [task[0]]:
            default_id = str(task[1])
        # RoboMME's upstream default without an ID filter is episode 0.
        if env_type == "libero" and len(tasks) != 1:
            cfg["task_ids"] = []
        elif env_type == "robomme":
            cfg["task_ids"] = evaluation_task_ids(tasks[0], default_id, env_type, allow_all=False)
        else:
            cfg["task_ids"] = evaluation_task_ids(tasks[0], default_id, env_type)
    if env_type in EPISODE_LENGTH_BENCHMARKS:
        cfg["episode_length"] = number("Episode Length (maximum control steps, 0 uses benchmark default)", 0, 0)
    if env_type == "robomme" and not tiny:
        cfg["robomme_runtime"] = choose("RoboMME Runtime", [
            ("auto", "Use a compatible combined environment, otherwise separate the simulator"),
            ("split", "Use a separate simulator environment"),
            ("shared", "Require a combined policy and simulator environment"),
        ])
        if cfg["robomme_runtime"] == "split":
            simulator = ask("Simulator Conda Name or Prefix (blank for automatic selection)", optional=True)
            if simulator:
                cfg["simulator_environment"] = simulator
    if env_type in LIBERO_BENCHMARKS:
        cfg.update(control="relative" if tiny else ask("Control Mode", "absolute" if model == "xvla" else "relative"),
                   size=256 if tiny else number("Camera Image Width and Height", 448 if model == "evo1" else 256))
    if tiny:
        cfg["action_steps"] = number("Actions to Execute per Prediction", 16, 1, 16)
    else:
        options = (catalog or {}).get("evaluation_options", {}).get(model, {})
        reference = cfg.get("policy") or (train or {}).get("pretrained")
        configured = {"pretrained": reference}
        configure_policy_options(configured, options, question="Policy Evaluation Options")
        if "policy_options" in configured:
            cfg["policy_options"] = configured["policy_options"]
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
        # A new run directory can appear before its status file is written.
        try:
            entries = {path.name for path in run.iterdir()}
        except OSError:
            return True
        return not entries or "executor" in entries
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
        notice(f"Protected Run Folders: {protected} (active, unreadable, or redirected).")
    if not selected:
        notice(f"No {label} Run Folders Available to Delete.")
        return False
    print(f"Result Folder: {folder}")
    if category == "training":
        print("This Also Deletes Checkpoints Stored in the Selected Run Folders.")
    count = len(selected)
    unit = "Run Folder" if count == 1 else "Run Folders"
    if not confirm(f"Permanently Delete {count} {label} {unit}? (Y/N)", "N"):
        notice("Deletion Canceled.")
        return False

    deleted = 0
    for run, original in selected:
        try:
            if (folder.resolve() != folder or run.resolve() != run or not os.path.samestat(run.stat(), original)
                    or run_in_use(run)):
                notice(f"Skipped Changed Run Folder: {run}")
                continue
            shutil.rmtree(run)
            deleted += 1
        except OSError as error:
            notice(f"Could Not Delete {run} ({error}).")
    notice(f"Deleted {deleted} of {count} {label} {unit}.")
    return deleted > 0


class LeRobotEnvironments:
    def __init__(self, conda, root, preferred, names):
        self.conda = conda
        self.root = root
        self.preferred = preferred
        self.names = set(names)
        self.catalogs = {}
        self.simulators = {}
        self.simulator_issues = {}
        self.package_reports = {}
        self.selection_reason = ""

    def simulator(self, name):
        if name not in self.simulators:
            script = str(Path(__file__).with_name("robomme_split.py"))
            selector = "--prefix" if Path(name).is_absolute() else "--name"
            command = [self.conda, "run", "--no-capture-output", selector, name, "python", "-B", script, "--inspect"]
            try:
                run = subprocess.run(command, cwd=self.root, capture_output=True, text=True, encoding="utf-8",
                                     errors="replace", timeout=75)
                entry = next((line for line in reversed(run.stdout.splitlines()) if line.startswith("ROBOMME_SIMULATOR ")), "")
                self.simulators[name] = json.loads(entry.partition(" ")[2]) if entry and run.returncode == 0 else None
                if not self.simulators[name]:
                    errors = [line for line in run.stderr.splitlines() if not line.startswith("ERROR conda.")]
                    self.simulator_issues[name] = errors[-1] if errors else "RoboMME runtime or assets missing."
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.simulators[name] = None
                self.simulator_issues[name] = str(error)
        return self.simulators[name]

    def prepare_robomme(self, stage, preferred, evaluation, *, model, peft, preview):
        from robomme_split import SIMULATOR_ENV

        pinned = bool(evaluation.get("environment"))
        name = evaluation.get("simulator_environment")
        mode = evaluation.get("robomme_runtime", "auto")
        catalogs = self.catalogs if preview else self.lerobot()
        shared = [key for key, catalog in catalogs.items()
                  if catalog.get("manual_status", {}).get("robomme", "unavailable") is None
                  and not catalog.get("runtime_issue") and (not pinned or key == preferred)]
        if mode == "shared" or (mode == "auto" and not name and shared):
            try:
                if not shared and not preview:
                    raise RuntimeError("No combined environment has a verified RoboMME simulator.")
                policy = self.prepare(preferred, model=model, benchmark="robomme", runtime="evaluation",
                                      peft=peft, pinned=pinned, preview=preview)
                return policy, ()
            except RuntimeError:
                if mode == "shared":
                    raise
        policy = self.prepare(preferred, model=model, runtime="evaluation", peft=peft, pinned=pinned, preview=preview)
        if preview:
            detail = (f"separate Conda environment {name}" if name else
                      "separate environment selected or prepared at launch")
            if mode == "auto" and not name:
                detail = "combined environment if compatible, otherwise " + detail
            return policy, (f"RoboMME simulator: {detail}.",
                            "Simulator packages: RoboMME, ManiSkill, NumPy, Gymnasium, SAPIEN, PyTorch, torchvision.")
        if not self.catalog(policy).get("split_robomme"):
            raise RuntimeError(f"Cannot verify the native RoboMME Gym adapter in policy environment {policy}. "
                               "Select a LeRobot version that provides RoboMME evaluation.")
        if name:
            worker = self.simulator(name)
            if not worker:
                raise RuntimeError(f"Cannot use RoboMME simulator {name}: "
                                   f"{self.simulator_issues.get(name, 'runtime or assets missing')}")
        else:
            worker = next((found for candidate in sorted(self.names)
                           if (found := self.simulator(candidate)) is not None), None)
            if not worker:
                from robomme_setup import prepare_simulator
                python, env = prepare_simulator(self.conda)
                probe = [python, "-B", str(Path(__file__).with_name("robomme_split.py")), "--versions"]
                result = subprocess.run(probe, env={**os.environ, **env}, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", check=True, timeout=30)
                worker = {"python": python, "env": env, "packages": json.loads(result.stdout)}
        simulator_entrypoint(stage)
        stage.env[SIMULATOR_ENV] = json.dumps({"python": worker["python"], "env": worker["env"]})
        notes = [f"RoboMME simulator: separate interpreter {worker['python']}."]
        if packages := worker.get("packages"):
            notes.append("Simulator packages: " + ", ".join(f"{k} {v}" for k, v in packages.items()) + ".")
        return policy, tuple(notes)

    def prepare_source(self, stage, preferred, evaluation, *, model, peft, preview):
        """Resolve a native policy and a reusable managed source simulator."""
        kind = evaluation["env"]
        pinned = bool(evaluation.get("environment"))
        catalogs = self.catalogs if preview else self.lerobot()
        shared = [name for name, catalog in catalogs.items()
                  if catalog.get("manual_status", {}).get(kind, "unavailable") is None
                  and not catalog.get("runtime_issue") and (not pinned or name == preferred)]
        if shared:
            # Reuse an existing verified combined installation before creating
            # any managed runtime, including a user's own source installation.
            try:
                target = self.prepare(preferred, model=model, benchmark=kind, runtime="evaluation",
                                      peft=peft, pinned=pinned, preview=preview)
                return target, ()
            except RuntimeError:
                pass
        policy = self.prepare(preferred, model=model, runtime="evaluation", peft=peft,
                              pinned=pinned, preview=preview)
        split = kind == "robotwin"
        if preview:
            role = "separate simulator" if split else "isolated compatible LeRobot"
            return policy, (f"{kind}: {role} environment and required source/assets prepared automatically at launch.",
                            "Existing compatible installations are checked before any download or compilation.")

        # Read the wrapper origin without importing simulator extensions. The
        # worker consumes exactly the source selected by this policy runtime.
        wrapper = {"libero_plus": "libero"}.get(kind, kind)
        probe = (
            "import hashlib,importlib.util,importlib.metadata,json,os,sys;from pathlib import Path;"
            "from huggingface_hub import constants as hub;"
            f"cache_paths={{name:str(getattr(hub,name)) for name in {HUB_PATH_VARIABLES!r} "
            "if getattr(hub,name,None) is not None};"
            "cache_paths['TORCH_HOME']=os.path.expanduser(os.environ.get('TORCH_HOME') or "
            "os.path.join(os.environ.get('XDG_CACHE_HOME') or str(Path.home()/'.cache'),'torch'));"
            "cache_paths.update({name:os.environ[name] for name in ('TRANSFORMERS_CACHE','HF_MODULES_CACHE') "
            "if os.environ.get(name)});"
            "p=Path(next(iter(importlib.util.find_spec('lerobot').submodule_search_locations)))"
            f"/'envs'/'{wrapper}.py';"
            "print('CONVLEX_NATIVE_SOURCE '+json.dumps({'source':str(p.resolve()),"
            "'base_prefix':sys.prefix,"
            "'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),"
            "'base_signature':hashlib.sha256(json.dumps(sorted((d.metadata.get('Name',''),d.version) "
            "for d in importlib.metadata.distributions())).encode()).hexdigest(),"
            "'hub_environment':cache_paths}))"
        )
        origin = subprocess.run(conda_command(self.conda, policy, ["python", "-B", "-c", probe]),
                                cwd=self.root, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=30)
        line = next((line for line in reversed(origin.stdout.splitlines())
                     if line.startswith("CONVLEX_NATIVE_SOURCE ")), "")
        if origin.returncode or not line:
            reason = origin.stderr.strip() or f"Native {kind} wrapper source was not found."
            raise RuntimeError(f"Cannot inspect {kind} in {policy}: {reason}")
        try:
            source = json.loads(line.partition(" ")[2])
            if not isinstance(source, dict) or set(source) != {"source", "source_sha256", "base_signature", "base_prefix", "hub_environment"}:
                raise ValueError("Unexpected native source report.")
            if not isinstance(source["base_prefix"], str) or not Path(source["base_prefix"]).is_absolute():
                raise ValueError("The native environment prefix must be an absolute path.")
            hub_environment = source["hub_environment"]
            if (not isinstance(hub_environment, dict) or set(hub_environment) - set(MODEL_CACHE_VARIABLES)
                    or any(not isinstance(value, str) or not value for value in hub_environment.values())):
                raise ValueError("The native Hub environment must contain only cache and credential file paths.")
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid {kind} wrapper source report in {policy}.") from error
        if split:
            from simulator_worker import validate_source
            validate_source({"kind": kind, **source})
        from runtime_setup import prepare_runtime
        options = {"base_environment": policy, **source}
        if kind == "robocasa":
            registries = next((item.partition("=")[2] for item in reversed(stage.arguments)
                               if item.startswith("--env.obj_registries=")), None)
            if registries is not None:
                options["obj_registries"] = registries
        prepared = prepare_runtime(kind, self.conda, options=options)
        stage.env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        if policy != preferred:
            # Dependency resolution may have selected/cloned a better policy
            # base. Reuse that base next time, never the simulator clone itself.
            evaluation["environment"] = policy
        worker_env = dict(prepared["env"])
        worker_env.setdefault("CONVLEX_SIMULATOR_ROOT", worker_env.get("ROBOTWIN_ROOT", str(prepared["root"])))
        notes = tuple(prepared.get("notes", ()))
        if split:
            from simulator_worker import SIMULATOR_ENV
            config = {"kind": kind, "python": str(prepared["python"]), "env": worker_env,
                      "recipe": prepared["recipe"], "source": source["source"],
                      "source_sha256": source["source_sha256"]}
            simulator_entrypoint(stage)
            stage.env[SIMULATOR_ENV] = json.dumps(config)
            notes = (f"{kind} simulator: separate interpreter {prepared['python']}; policy remains in {policy}.", *notes)
            return policy, notes
        target = str(prepared.get("prefix") or Path(prepared["python"]).parent.parent)
        # Managed simulator prefixes need their recorded source overrides.
        # Keep them out of the bare named-environment catalog: a later queued
        # job must pass through prepare_runtime to restore and verify that env.
        stage.env.update(worker_env)
        self.selection_reason = f"Managed {kind} runtime prepared from {policy}; verified before evaluation."
        return target, (f"{kind} simulator and native policy: isolated environment {target}.", *notes)

    def packages(self, name, model, runtime, benchmark=None, peft=False, tiny_root=None):
        key = (name, model, runtime, benchmark, peft, tiny_root)
        if key not in self.package_reports:
            groups = {}
            if self.conda and (not self.names or name in self.names or Path(name).is_absolute()):
                probe = ["python", "-B", str(Path(__file__).with_name("benchmarks.py")),
                         "--describe-stage", runtime, "--model-name", model]
                if benchmark:
                    probe += ["--benchmark-name", benchmark]
                if peft:
                    probe.append("--peft")
                if tiny_root:
                    probe += ["--tiny-root", str(tiny_root)]
                try:
                    proc = subprocess.run(conda_command(self.conda, name, probe), cwd=self.root,
                                          capture_output=True, text=True, encoding="utf-8",
                                          errors="replace", timeout=15, check=True)
                    for line in reversed(proc.stdout.splitlines()):
                        if line.startswith("MANAGER_REQUIREMENTS "):
                            groups = json.loads(line.removeprefix("MANAGER_REQUIREMENTS "))
                            break
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            self.package_reports[key] = groups
        return self.package_reports[key]

    def catalog(self, name, *, refresh=False):
        if refresh:
            self.package_reports = {key: value for key, value in self.package_reports.items() if key[0] != name}
        if refresh or name not in self.catalogs:
            self.catalogs[name] = environment_catalog(self.conda, name, self.root)
        return self.catalogs[name]

    def lerobot(self):
        names = [self.preferred, *sorted(self.names - {self.preferred})]
        return {name: catalog for name in names if (catalog := self.catalog(name))}

    @staticmethod
    def available(catalogs):
        locations = {}
        for name, catalog in catalogs.items():
            ready = {kind for kind, installed in catalog.get("benchmarks", {}).items() if installed}
            ready.update(catalog.get("plugins", ()))
            ready.update(kind for kind, issue in catalog.get("manual_status", {}).items() if issue is None)
            for kind in ready:
                locations.setdefault(kind, []).append(name)
        return locations

    def prepare(self, preferred, *, model, benchmark=None, tiny=False, pinned=False, peft=False, runtime=None,
                preview=False):
        prefix = "Pinned by eval.environment. " if pinned else ""
        self.selection_reason = prefix + "Configured environment; runtime readiness unverified."
        catalogs = ({name: entry for name, entry in self.catalogs.items() if entry}
                    if preview else self.lerobot())
        if not preview and preferred not in catalogs and (catalog := self.catalog(preferred)):
            catalogs[preferred] = catalog
        if not catalogs:
            return preferred
        unavailable = {name: entry["runtime_issue"] for name, entry in catalogs.items() if entry.get("runtime_issue")}
        catalogs = {name: entry for name, entry in catalogs.items() if name not in unavailable}
        if not catalogs:
            details = " ".join(f"{name}: {issue}" for name, issue in unavailable.items())
            raise RuntimeError(f"No usable LeRobot runtime. {details}")

        check_policy = tiny or any(model in entry.get("policies", {}) for entry in catalogs.values())
        check_benchmark = benchmark is not None and any(
            benchmark in entry.get("benchmarks", {}) or benchmark in entry.get("plugins", ())
            or benchmark in entry.get("manual_status", {})
            for entry in catalogs.values()
        )
        pinned_clones = set()
        if pinned and (source_prefix := catalogs.get(preferred, {}).get("prefix")):
            pinned_clones = {name for name, entry in catalogs.items()
                             if source_prefix in entry.get("clone_sources", ())}
        if pinned or (not check_policy and runtime != "dataset") or (benchmark is not None and not check_benchmark):
            if preferred in unavailable:
                raise RuntimeError(f"Cannot use Conda environment {preferred}: {unavailable[preferred]}")
            catalogs = {preferred: catalogs.get(preferred, {}),
                        **{name: catalogs[name] for name in pinned_clones}}

        def policy_ready(entry):
            if tiny:
                return entry.get("tinyvla_bridge", False)
            return not check_policy or entry.get("policies", {}).get(model, False)

        def benchmark_ready(entry):
            return (not check_benchmark or entry.get("benchmarks", {}).get(benchmark, False)
                    or benchmark in entry.get("plugins", ())
                    or (benchmark in entry.get("manual_status", {}) and entry["manual_status"][benchmark] is None))

        def runtime_ready(entry):
            return runtime is None or entry.get("runtime", {}).get(runtime, True)

        candidates = []
        for name, entry in catalogs.items():
            needs_policy = not policy_ready(entry)
            needs_benchmark = not benchmark_ready(entry)
            if needs_policy and (tiny or model not in entry.get("policy_installable", ())):
                continue
            if needs_benchmark and benchmark not in entry.get("benchmarks", {}):
                continue
            needs_peft = peft and not entry.get("peft_ready", False)
            if needs_peft and not entry.get("peft_installable", False):
                continue
            dependencies = []
            if needs_policy:
                dependencies.append(("--policy", model))
            if needs_benchmark:
                dependencies.append(("--install", benchmark))
            if needs_peft:
                dependencies.append(("--peft", None))
            if not runtime_ready(entry):
                dependencies.append(("--runtime", runtime))
            if name in pinned_clones and dependencies:
                # A pin permits reuse of a ready descendant, not installation
                # into unrelated or partially prepared environments.
                continue
            candidates.append((len(dependencies), name != preferred, name, dependencies))

        if not candidates:
            issue = catalogs.get(preferred, {}).get("policy_issues", {}).get(model)
            detail = f" ({issue})" if issue else ""
            issues = dict.fromkeys(entry.get("manual_status", {}).get(benchmark) for entry in catalogs.values())
            diagnostics = [reason for reason in issues if reason]
            if diagnostics:
                detail += " " + " ".join(diagnostics)
            stage = f"{model} with {benchmark}" if benchmark else model
            raise RuntimeError(f"No Conda environment can be prepared for {stage}{detail}.")
        process_options = {"cwd": self.root, "text": True, "encoding": "utf-8", "errors": "replace",
                           "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
        conflicts = []
        failures = []

        def verify(target, description):
            refreshed = self.catalog(target, refresh=True)
            if (refreshed and not refreshed.get("runtime_issue") and policy_ready(refreshed)
                    and benchmark_ready(refreshed) and runtime_ready(refreshed) and (not peft or refreshed.get("peft_ready"))):
                print(f"  Dependencies ready in {target}.", flush=True)
                return True
            issue = refreshed.get("runtime_issue") or refreshed.get("policy_issues", {}).get(model)
            detail = f" ({issue})" if issue else ""
            failures.append(f"Could not verify {description} in {target} after installation{detail}.")
            return False

        for _, _, source, dependencies in sorted(candidates):
            selection = (f"Verified managed clone of {preferred}" if source in pinned_clones else
                         "Preferred environment" if source == preferred else f"Alternative to {preferred}")
            if not dependencies:
                unknown = (runtime != "dataset" and not check_policy) or (benchmark and not check_benchmark)
                detail = ("policy or simulator support unverified"
                          if unknown or not catalogs.get(source) else "matches inspected runtime requirements")
                self.selection_reason = prefix + f"{selection}; {detail}."
                return source
            program = ["python", "-B", str(Path(__file__).with_name("benchmarks.py"))]
            names_to_install = [name or "LoRA" for _, name in dependencies]
            for option, name in dependencies:
                program.append(option)
                if name is not None:
                    program.append(name)
            description = " and ".join(names_to_install)
            if preview:
                self.selection_reason = prefix + f"{selection}; setup required for {description} at launch."
                return source
            print(f"  Preparing dependencies for {description} in {source}…", flush=True)
            result = subprocess.run(conda_command(self.conda, source, program), **process_options)
            if result.returncode:
                failures.append(result.stdout.strip() or f"Could not install {description} in {source}.")
                if result.returncode == DEPENDENCY_CONFLICT:
                    suffix = re.sub(r"[^a-zA-Z0-9_.-]", "_", "_".join(names_to_install))
                    conflicts.append((source, suffix, program, description))
            elif verify(source, description):
                self.selection_reason = prefix + f"Prepared and verified {description} in the selected environment."
                return source

        for source, suffix, program, description in conflicts:
            stem = f"{source}_{suffix}"
            target = stem
            index = 2
            names = self.names | available_environments(self.conda, self.root)
            while target in names:
                target = f"{stem}_{index}"
                index += 1
            print(f"  Dependency conflict in {source}. Creating {target}…", flush=True)
            try:
                prepare_clone_source(self.conda, source, self.root)
            except ActivationError as error:
                failures.append(str(error))
                continue
            clone = subprocess.run([self.conda, "create", "--yes", "--name", target, "--clone", source],
                                   **process_options)
            if clone.returncode:
                failures.append(clone.stdout.strip() or f"Could not clone {source} into {target}.")
                continue
            self.names.add(target)
            ready = False
            try:
                configure_clone_inheritance(self.conda, source, target, self.root)
                result = subprocess.run(conda_command(self.conda, target, [*program, "--isolated"]), **process_options)
                if result.returncode:
                    failures.append(result.stdout.strip() or f"Could not install {description} in {target}.")
                else:
                    ready = verify(target, description)
                if ready:
                    self.selection_reason = prefix + f"Cloned from {source} after a package conflict; verified {description}."
                    return target
            except (ActivationError, OSError, ValueError) as error:
                failures.append(str(error) if isinstance(error, ActivationError)
                                else "Could not configure runtime variables for the managed clone.")
            finally:
                if not ready:
                    removed = subprocess.run([self.conda, "env", "remove", "--yes", "--name", target],
                                             **process_options)
                    if removed.returncode:
                        failures.append(removed.stdout.strip() or f"Could not remove incomplete environment {target}.")
                    else:
                        self.names.discard(target)
                        self.catalogs.pop(target, None)
        raise RuntimeError(failures[-1])


def benchmark_options(kinds):
    labels = {**dict(BENCHMARKS), "libero_plus": "LIBERO-plus", "robotwin": "RoboTwin",
              "robocasa": "RoboCasa365", "vlabench": "VLABench", "robomme": "RoboMME"}
    return [(kind, labels.get(kind, kind.replace("_", " ").title()))
            for kind in [*(kind for kind in labels if kind in kinds), *sorted(kinds - labels.keys())]]


def policy_environments(environments, preferred, model, tiny=False):
    catalogs = environments.lerobot()
    if preferred not in catalogs:
        catalogs[preferred] = environments.catalog(preferred)
    ready = {
        name: catalog for name, catalog in catalogs.items()
        if (catalog.get("tinyvla_bridge") if tiny else catalog.get("policies", {}).get(model))
    }
    return catalogs, ready


def choose_evaluation_environment(environments, preferred, model, tiny):
    catalogs, eligible = policy_environments(environments, preferred, model, tiny)
    available = environments.available(eligible)
    installed = environments.available(catalogs)
    installable = {kind for catalog in catalogs.values() for kind in catalog.get("benchmarks", {})}
    manual = {kind for catalog in catalogs.values() for kind in catalog.get("manual_benchmarks", ())}
    options = ([("libero", "LIBERO (supported by TinyVLA)")] if tiny
               else benchmark_options(installable | installed.keys() | manual))
    task_adapters = {kind for kind, _ in options if stock_evaluation_issue(kind)} if not tiny else set()
    available = {kind: names for kind, names in available.items() if kind not in task_adapters}
    missing = {kind for kind, _ in options if kind not in available}
    reasons = {kind: ("Automatic Setup" if sys.platform == "linux" else "Requires Linux")
               if kind in SOURCE_BENCHMARKS or kind == "robomme" else "Manual Setup"
               for kind in missing & manual}
    reasons.update({kind: "Model Runtime Missing in " + ", ".join(installed[kind])
                    for kind in missing if kind in installed})
    reasons.update({kind: "Task Adapter Required" for kind in task_adapters})
    options = [(kind, f"{label} [{', '.join(available[kind])}]" if kind in available else label)
               for kind, label in options]
    if not tiny:
        options.append(("manual", "Enter a LeRobot environment type"))
    default = next((index for index, (kind, _) in enumerate(options, 1) if kind in available), None)
    kind = choose("Evaluation Benchmark", options, default=default, unavailable=missing, reasons=reasons)
    if kind == "manual":
        kind = ask("LeRobot Environment Type")
    task_issue = stock_evaluation_issue(kind) if not tiny else None
    if task_issue:
        print(task_issue)
    names = available.get(kind, [])
    if not task_issue and not names and kind in missing:
        if kind == "robomme" and not tiny:
            print("A separate RoboMME simulator can be selected or prepared automatically at launch.")
        elif kind in SOURCE_BENCHMARKS and not tiny:
            if sys.platform == "linux":
                print("The executor will prepare a compatible environment, source and required simulator assets at launch.")
            else:
                print("Automatic source simulator setup requires Linux.")
        elif kind in manual:
            print(f"{kind} requires separate simulator setup.")
            issue = catalogs.get(preferred, {}).get("manual_status", {}).get(kind)
            if issue:
                print(issue)
        else:
            print("The executor will check for installable dependencies when the job starts.")
    target = preferred if preferred in names or not names else names[0]
    print(f"Evaluation Conda Environment: {target}")
    return kind, target


def show_inventory(display, inventory, environments):
    display.clear(start=0)
    section("Status")
    show(inventory, environments, limit=5)
    display.mark()


def wizard(adapters, inventory, root, conda=None, native_env="lerobot", *,
           external=None, environments=(), env_map=None, allow_changes=True, environment_session=None):
    display = QuestionDisplay(sys.stdout, section)
    mapping = env_map or {}
    environments = environment_session or LeRobotEnvironments(conda, root, native_env, environments)
    catalog_for = environments.catalog

    def evaluation_environment(model, tiny):
        preferred = native_env if tiny else mapping.get(model, native_env)
        benchmark, selected = choose_evaluation_environment(environments, preferred, model, tiny)
        return benchmark, selected, False

    with redirect_stdout(display), display:
        show_inventory(display, inventory, environments.names)
        while True:
            display.show()
            actions = [("new", "Configure a new job")]
            if inventory.settings:
                actions.append(("reuse", "Reuse settings from a previous run"))
            actions.append(("import", "Import a job queue script"))
            if inventory.results:
                actions.append(("results", "View existing results"))
            if allow_changes:
                actions.extend((("evaluation", "Delete all evaluation results"),
                                ("training", "Delete all training results")))
            actions.append(("quit", "Exit"))
            action = choose("What Would You Like to Do?", actions, blank_before=False)
            if action == "quit":
                return None

            history = QuestionHistory(display)
            try:
                if action == "new":
                    catalogs = environments.lerobot()
                    policies = {model for details in catalogs.values() for model in details.get("policies", {})}

                    def training_catalog(model):
                        preferred = mapping.get(model, native_env)
                        details = catalog_for(preferred)
                        if details.get("policies", {}).get(model):
                            return details, None
                        for name, catalog in catalogs.items():
                            if catalog.get("policies", {}).get(model):
                                return catalog, name
                        return details, None

                    return configure_jobs(adapters, inventory, root, policies, display, training_catalog,
                                          evaluation_environment, evaluation_catalog=catalog_for)
                if action == "reuse":
                    files = [(p, str(p)) for p in inventory.settings]
                    path = history.run(lambda: select_path("Settings File", files))
                    return load_settings(path)
                if action == "import":
                    scripts = [(p, p.name) for p in script_paths()]
                    path = history.run(lambda: select_path("Job Queue Script", scripts))
                    return load_settings(path)
                if action in {"evaluation", "training"}:
                    messages = []
                    deleted = history.run(lambda: delete_result_runs(root, action, notice=messages.append))
                    if deleted:
                        inventory = scan(root, external or root / "externals/TinyVLA", root / "executor/log")
                        if display.terminal:
                            show_inventory(display, inventory, environments.names)
                    display.notice("\n".join(messages))
                    continue
                files = [(p, str(p)) for p in inventory.results]
                path = history.run(lambda: select_path("Result File", files))
                result = read_json(path)
                if not isinstance(result, dict):
                    raise ValueError("Expected a result dictionary.")
                summary = result.get("overall") or {key: result[key] for key in
                            ("status", "steps", "batch_size", "resumed_from_step", "checkpoint") if key in result}
                display.notice(json.dumps(summary or result, ensure_ascii=False, indent=2))
            except PreviousQuestion:
                continue
            except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
                display.notice(f"Could not complete the selection ({error}).")


def configure_jobs(adapters, inventory, root, policies, display, training_catalog, evaluation_environment,
                   *, evaluation_catalog=None):
    def configure():
        job = configure_job(adapters, inventory, root, policies, training_catalog, evaluation_environment,
                            evaluation_catalog=evaluation_catalog)
        return job, confirm("Add Another Job? (Y/N)", "N")

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


def configure_job(adapters, inventory, root, policies, training_catalog, evaluation_environment,
                  *, evaluation_catalog=None):
    backend, model = choose_policy(adapters, policies, inventory)
    tiny = backend == "tinyvla"
    catalog = {}
    operations = [
        ("eval", "Evaluation"),
        ("train", "Training"),
        ("both", "Training + Evaluation"),
        ("resume", "Training Continuation"),
        ("resume_eval", "Training Continuation + Evaluation"),
    ]
    if backend == "native" and (issue := native_training_issue(model)):
        print(styled(issue, "33"))
        operations = operations[:1]
    operation = choose("Job Type", operations)
    continuation = operation in {"resume", "resume_eval"}
    cfg = {"schema": 1, "adapter": backend, "model": model}
    if operation != "eval":
        print("\n" + styled("Training Setup", "1;36"))
        if tiny:
            print("\nTraining benchmark · LIBERO")
            method = choose("Continuation Method", [
                ("resume", "Resume from a training checkpoint"),
                ("warm", "Start training with existing action-head weights"),
            ]) if continuation else "new"
            if method == "resume":
                print("Training Scope · Restore checkpoint settings")
                cfg["train"] = tiny_resume(inventory)
            else:
                cfg["train"] = tiny_train(method == "warm", inventory, root)
        else:
            catalog, target = training_catalog(model)
            if target:
                print(f"Training Conda Environment: {target}")
                cfg["environments"] = {model: target}
            cfg["train"] = native_train(continuation, inventory, model, catalog)
    if operation in {"eval", "both", "resume_eval"}:
        print("\n" + styled("Evaluation Setup", "1;36"))
        benchmark, environment, override = evaluation_environment(model, tiny)
        if not tiny and evaluation_catalog is not None:
            catalog = evaluation_catalog(environment)
        cfg["eval"] = eval_settings(model, tiny, cfg.get("train"), inventory, benchmark, catalog)
        if override:
            cfg["eval"]["environment"] = environment
    cfg["save_outputs"] = confirm("Save Job Outputs? (Y/N)", "Y")
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


def plan_job(args, cfg, conda, detected, benchmark_sessions, temporary_output_directory=None):
    root, external = paths_for(args, cfg)
    plugins = args.plugin or cfg.get("plugins", [])
    adapters = load_adapters(plugins)
    if cfg.get("schema") != 1 or cfg.get("adapter") not in adapters or not {"train", "eval"} & cfg.keys():
        raise ValueError("Specify schema 1, a supported adapter, and a train or eval job.")
    if not isinstance(cfg["model"], str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", cfg["model"]):
        raise ValueError("Model names may contain letters, digits, periods, underscores, and hyphens.")
    save_outputs = cfg.get("save_outputs", True)
    if type(save_outputs) is not bool:
        raise ValueError("save_outputs must be true or false.")
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
    outputs = PROJECT_ROOT / "outputs"
    if not save_outputs:
        if args.dry_run:
            outputs = Path(gettempdir()) / f"convlexecutor-{name}"
        elif temporary_output_directory is not None:
            outputs = temporary_output_directory()
        else:
            raise ValueError("Temporary jobs require an output directory managed by the executor.")
        adapter.train_dir = outputs / "training" / name
        adapter.eval_dir = outputs / "evaluation" / name
    stages = []
    if "train" in cfg:
        stages.append(adapter.train())
    if "eval" in cfg:
        evaluate = cfg["eval"]
        if conda and not evaluate.get("environment") and cfg["adapter"] in {"native", "tinyvla"}:
            native = environments["native"]
            key = (PROJECT_ROOT, native)
            if key not in benchmark_sessions:
                benchmark_sessions[key] = LeRobotEnvironments(conda, PROJECT_ROOT, native, installed)
            session = benchmark_sessions[key]
            preferred = adapter.environment
            tiny = cfg["adapter"] == "tinyvla"
            # Queue jobs without an override follow the same installed-environment routing as the wizard.
            _, candidates = policy_environments(session, preferred, cfg["model"], tiny)
            ready = session.available(candidates).get(evaluate["env"], [])
            detected_target = ready[0] if ready and preferred not in ready else None
        else:
            detected_target = None
        stage = adapter.evaluate(after_training="train" in cfg)
        if detected_target:
            stage.environment = detected_target
        stages.append(stage)
    for stage in stages:
        if not save_outputs:
            paths = (*stage.artifacts, stage.output) if stage.output else stage.artifacts
            if any(not path.resolve().is_relative_to(outputs.resolve()) for path in paths):
                raise ValueError("This adapter must keep output paths inside the temporary job directory.")
        stage.env.setdefault("HF_LEROBOT_HOME", str(PROJECT_ROOT / "datasets/.hub"))
    first = stages[0]
    native = environments["native"]
    key = (PROJECT_ROOT, native)
    if key not in benchmark_sessions:
        benchmark_sessions[key] = LeRobotEnvironments(conda, PROJECT_ROOT, native, installed)
    prepare_job_environments(cfg, stages, PROJECT_ROOT, conda, benchmark_sessions, preview=True)
    output = first.output or outputs / ("training" if first.name == "train" else "evaluation") / name
    run_dir = (output.parent if first.name == "train" and first.output else output) / "executor"
    return cfg, stages, run_dir, PROJECT_ROOT


def plan(args, temporary_output_directory=None, sessions=None):
    source = args.settings or args.queue
    settings = load_settings(source) if source else None
    if args.queue and "jobs" not in settings:
        raise ValueError("Use --queue with a file containing a JOBS list.")
    try:
        conda = conda_executable(args.conda)
    except FileNotFoundError:
        conda = None
    detected = {}
    benchmark_sessions = {} if sessions is None else sessions
    if args.inventory or settings is None:
        _, external = paths_for(args, settings or {})
        detected[PROJECT_ROOT] = available_environments(conda, PROJECT_ROOT)
        inventory = scan(PROJECT_ROOT, external, LOG_ROOT)
        if args.inventory:
            section("Status")
            show(inventory, detected[PROJECT_ROOT])
            return [], conda
        env_map = read_json(args.env_map) if args.env_map else {}
        native_env = env_map.get("native", DEFAULT_ENVIRONMENTS["native"])
        session = LeRobotEnvironments(conda, PROJECT_ROOT, native_env, detected[PROJECT_ROOT])
        benchmark_sessions[(PROJECT_ROOT, native_env)] = session
        settings = wizard(load_adapters(args.plugin or []), inventory, PROJECT_ROOT, conda, native_env,
                          external=external, env_map=env_map, allow_changes=not args.dry_run,
                          environment_session=session)
        if settings is None:
            return [], conda

    plans = []
    jobs = settings_jobs(settings)
    for index, cfg in enumerate(jobs, 1):
        try:
            plans.append(plan_job(args, cfg, conda, detected, benchmark_sessions, temporary_output_directory))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError) as error:
            raise ValueError(f"Invalid settings for job {index}/{len(jobs)} ({error})") from error
    return plans, conda


def command_preview(arguments, auto_flags=()):
    groups = []
    for argument in arguments:
        if not groups or argument.startswith("--"):
            groups.append([])
        groups[-1].append(argument)
    commands, automatic = [], []
    for group in groups:
        command = shlex.join(group)
        if group[0].removeprefix("--").partition("=")[0] in auto_flags:
            automatic.append(styled(command, "32"))
        else:
            commands.append(command)
    commands.extend(automatic)
    return " \\\n".join(f"    {command}" for command in commands)


def job_title(index, cfg):
    title = styled(f"[Job {index}]", "1;36")
    return f"{title} {cfg.get('label') or cfg['model']}"


def hub_access_hints(model, stages):
    """Report known download gates without probing the network or the user's credentials."""
    paligemma = "google/paligemma-3b-pt-224"
    cosmos = "nvidia/Cosmos-Reason2-2B"
    flux_base = "black-forest-labs/flux-3-action-base"
    known = {paligemma, cosmos, flux_base, "black-forest-labs/flux-3-action-so101"}
    found = set()
    for stage in stages:
        flags = dict(token[2:].split("=", 1) for token in stage.arguments
                     if token.startswith("--policy.") and "=" in token)
        source = flags.get("policy.path")
        config = {}
        if source:
            path = Path(source).expanduser() / "config.json"
            if path.is_file():
                try:
                    config = read_json(path)
                except (OSError, ValueError):
                    pass
                if not isinstance(config, dict):
                    config = {}
        # Explicit flags take precedence over checkpoint metadata.
        fields = {**config, **{key.removeprefix("policy."): value for key, value in flags.items()}}
        candidates = [source]
        if model in {"pi0", "pi05", "pi0_fast", "fineart_vla"}:
            candidates.append(fields.get("text_tokenizer_name", paligemma))
        if model == "flux3":
            candidates.extend(fields.get(key, flux_base) for key in ("video_vae_id", "text_encoder_id"))
        if model == "groot" and (not source or str(source).startswith("nvidia/gr00t17-")
                                  or str(fields.get("base_model_path", "")).lower().startswith("nvidia/gr00t-n1.7")
                                  or str(fields.get("model_version", "")).lower() == "n1.7"):
            candidates.append(cosmos)
        for value in [*candidates, *fields.values()]:
            if isinstance(value, str) and value.split(":", 1)[0] in known:
                found.add(value.split(":", 1)[0])
    return sorted(found)


def show_settings(plans, show_commands):
    for index, (cfg, stages, _, _) in enumerate(plans, 1):
        if index > 1:
            print()
        print(job_title(index, cfg))
        if not cfg.get("save_outputs", True):
            print("  Job outputs: temporary")
        for stage in stages:
            print("  " + stage_title(stage, cfg["model"]))
            for line in stage.environment_notes:
                print(f"    {line}")
            if show_commands:
                print(command_preview(stage.arguments, stage.auto_flags))
            elif stage.auto_flags:
                automatic = [arg for arg in stage.arguments if arg.removeprefix("--").partition("=")[0] in stage.auto_flags]
                print(command_preview(automatic, stage.auto_flags))
        access = hub_access_hints(cfg["model"], stages) if cfg.get("adapter") == "native" else []
        if access:
            print("  Hugging Face access for downloads (if needed):")
            for repo in access:
                print(f"    https://huggingface.co/{repo}")
            print("    Use the same account as the executor to review the terms and request access.")
        evaluation = cfg.get("eval")
        if evaluation and evaluation["batch"] > evaluation["episodes"]:
            print(f"  Effective evaluation batch size: {evaluation['episodes']} "
                  f"({evaluation['batch']} requested, {evaluation['episodes']} episodes per task)")


def job_subheading(title):
    print(styled(f"  ({title})", "1;94"), flush=True)


def validate_model_training_fields(model, train, catalog, *, evaluation=False):
    """Validate imported component/option names against the selected runtime version."""
    if train.get("resume"):
        return
    declared = catalog.get("policy_fields", {}).get(model)
    if evaluation and model in catalog.get("evaluation_options", {}):
        declared = catalog["evaluation_options"][model]
    if declared is None:
        return  # Older or unavailable runtime catalogs cannot establish support.
    for field in ("policy_training", "policy_options"):
        options = train.get(field) or {}
        if isinstance(options, dict) and (unknown := options.keys() - set(declared)):
            raise ValueError(f"Policy {model} in the selected LeRobot runtime does not support "
                             f"{field} fields: {', '.join(sorted(unknown))}. "
                             "Remove these fields or use a runtime version that declares them.")
        if isinstance(options, dict):
            for name, value in options.items():
                spec = catalog.get("training_options", {}).get(model, {}).get(name, {})
                if spec.get("value_type") in {"numeric_object", "numeric_sequence"}:
                    try:
                        structured_policy_value(value, spec)
                    except ValueError as error:
                        raise ValueError(f"Invalid {model} {field}.{name}: {error}") from error


def prepare_job_environments(cfg, stages, root, conda, sessions, *, preview=False):
    backend = cfg["adapter"]
    if backend not in {"native", "tinyvla"}:
        for stage in stages:
            stage.environment_notes = (f"{stage.environment}: selected by the external adapter; requirements unverified.",)
        return
    native = cfg.get("environments", {}).get("native", DEFAULT_ENVIRONMENTS["native"])
    key = (root, native)
    if key not in sessions:
        sessions[key] = LeRobotEnvironments(conda, root, native, available_environments(conda, root))
    session = sessions[key]
    model = cfg["model"]
    train = cfg.get("train", {})
    uses_peft = False
    if backend == "native":
        source = train.get("pretrained") or cfg.get("eval", {}).get("policy")
        if train.get("resume"):
            checkpoint = native_checkpoint(Path(train["resume"]))
            saved = read_json(checkpoint / "train_config.json")
            uses_peft = bool(saved.get("peft") or (saved.get("policy") or {}).get("use_peft"))
            source = checkpoint
        explicit_peft = any(arg.startswith("--peft.") for arg in train.get("extra", ()))
        uses_peft = uses_peft or explicit_peft or train.get("lora") is not None or (
            train.get("policy_training", {}).get("train_mode_vlm") == "lora"
        ) or (
            model == "groot" and train.get("policy_training", {}).get("lora_rank", 0) > 0
        ) or bool(source and Path(source).is_dir() and native_peft_checkpoint(Path(source)))

    for stage in stages:
        notes = []
        tiny = backend == "tinyvla"
        tiny_train = tiny and stage.name == "train"
        evaluation = cfg["eval"] if stage.name == "eval" else {}
        benchmark = evaluation.get("env")
        if tiny:
            worker = cfg.get("environments", {}).get("tinyvla", DEFAULT_ENVIRONMENTS["tinyvla"])
            role = "training" if tiny_train else "inference"
            notes.append(f"TinyVLA {role}: {worker} (configured external model interpreter).")
            worker_packages = session.packages(worker, model, "tiny-model", tiny_root=cfg.get("tinyvla_root"))
            notes.extend(package_lines(worker_packages, "TinyVLA model: torch, torchvision, transformers, diffusers"))
        if tiny_train:
            dataset_python = stage.arguments.index("--dataset-python") + 1
            preferred = stage.arguments[dataset_python].removeprefix("@python:")
            runtime = "dataset"
        else:
            preferred = evaluation.get("environment") or (native if tiny else cfg.get("environments", {}).get(model, native))
            runtime = "evaluation" if evaluation else "training"
        if preview and conda and preferred not in session.catalogs and callable(getattr(session, 'catalog', None)):
            session.catalog(preferred)  # Read-only: imported queues need the same option/compatibility preview.
        try:
            simulator_detail = None
            if not tiny and benchmark == "robomme":
                target, simulator_detail = session.prepare_robomme(stage, preferred, evaluation, model=model,
                                                                  peft=uses_peft, preview=preview)
            elif not tiny and benchmark in SOURCE_BENCHMARKS:
                target, simulator_detail = session.prepare_source(stage, preferred, evaluation, model=model,
                                                                  peft=uses_peft, preview=preview)
            else:
                target = session.prepare(preferred, model=model, tiny=tiny and not tiny_train, benchmark=benchmark,
                                         pinned=bool(evaluation.get("environment")), peft=uses_peft,
                                         runtime=runtime, preview=preview)
            reason = session.selection_reason
        except RuntimeError as error:
            if not preview:
                raise
            target, reason = preferred, f"Cannot confirm readiness: {error}"
        role = "LeRobot dataset reader" if tiny_train else f"LeRobot {runtime}"
        if backend == "native" and stage.name == "train":
            validate_model_training_fields(model, train, session.catalogs.get(target, {}))
        elif backend == "native" and stage.name == "eval":
            validate_model_training_fields(model, {"policy_options": evaluation.get("policy_options", {})},
                                           session.catalogs.get(target, {}), evaluation=True)
        notes.append(f"{role}: {target}. {reason}")
        if simulator_detail:
            notes.extend(simulator_detail)
        groups = session.packages(target, model, runtime, None if simulator_detail else benchmark, uses_peft)
        if not tiny_train:
            key = "TinyVLA bridge" if tiny else "Policy requirements"
            if key not in groups:
                scope = "bridge" if tiny else "native policy"
                detail = "LeRobot core; no extra declared" if groups.get("LeRobot runtime") and session.catalogs.get(
                    target, {}).get("policies", {}).get(model) else "requirements unverified"
                notes.append(f"Model {model} ({scope}): {detail}.")
        if benchmark and not simulator_detail and f"Simulator {benchmark}" not in groups:
            notes.append(f"Simulator {benchmark}: package requirements unverified.")
        notes.extend(package_lines(groups, f"LeRobot {runtime} packages"))
        if backend == "native" and evaluation and not (cfg.get("train") and model == "fineart_vla"):
            notes.extend(evaluation_notices(model, evaluation))
        previous_notes = stage.environment_notes
        stage.environment_notes = tuple(notes)
        if not preview and previous_notes and previous_notes != stage.environment_notes:
            print(f"  Resolved {stage.name} environment:")
            for line in notes:
                print(f"    {line}")
        if tiny_train:
            stage.arguments[dataset_python] = f"@python:{target}"
        else:
            if not preview and target != preferred:
                if evaluation:
                    # Reusable settings describe the user's policy/base
                    # environment. The resolved prefix and source overrides are
                    # already recorded in run.py's stage provenance. Persisting
                    # a managed source prefix as the next clone's base would
                    # recursively create another runtime on every reuse.
                    if not (benchmark in SOURCE_BENCHMARKS and Path(target).is_absolute()):
                        evaluation["environment"] = target
                else:
                    cfg.setdefault("environments", {})[model] = target
            stage.environment = target


def package_lines(groups, fallback):
    if not groups:
        return [f"{fallback}: versions unverified."]
    return [f"{label}: {', '.join(packages)}" for label, packages in groups.items()]


def temporary_job_root(path, temporary_outputs):
    return next((root for root in temporary_outputs if path.is_relative_to(root)), None)


def record_preparation_failure(error, cfg, run_dir, root, conda, started):
    """Keep a retryable job and a concise diagnostic when no stage could start."""
    run_dir.mkdir(parents=True, exist_ok=False)
    details = str(error)
    # Conda's crash report dumps activation variables after this heading.
    details = re.split(r"(?im)^\s*environment variables:\s*$", details, maxsplit=1)[0]
    details = re.sub(r"\bhf_[A-Za-z0-9]{8,}\b", "[redacted]", details)
    details = re.sub(r"(?i)\bBearer\s+[^\s'\",;)\]]+", "Bearer [redacted]", details)
    logfile = run_dir / "setup.log"
    logfile.write_text(details + "\n", encoding="utf-8")
    reason = stage_failure_reason(logfile) or (
        error.reason if isinstance(error, RuntimeSetupFailure) else "Could not prepare the selected runtime.")
    settings_text = "JOBS = " + pformat([cfg], sort_dicts=False, width=110) + "\n"
    (run_dir / "script.py").write_text(settings_text, encoding="utf-8")
    retry = ('"""AI-written retry launcher: environment setup failed before any model stage ran."""\n\n' +
             settings_text + '\nif __name__ == "__main__":\n' +
             '    from pathlib import Path\n    import sys\n' +
             f'    sys.path.insert(0, {str(root / "executor")!r})\n' +
             '    from manage import main\n' +
             f'    sys.argv = ["manage.py", "--settings", str(Path(__file__).resolve()), "--conda", {str(conda)!r}]\n' +
             '    raise SystemExit(main())\n')
    (run_dir / "run.py").write_text(retry, encoding="utf-8")
    ended = datetime.now(timezone.utc)
    write_json(run_dir / "status.json", {"status": "failed", "phase": "environment_setup",
               "started_at": started.isoformat(), "ended_at": ended.isoformat(),
               "seconds": (ended - started).total_seconds(), "stages": [], "error": reason,
               "setup_log": str(logfile)})
    return RuntimeSetupFailure("Dependencies", reason, logfile)


def run_plans(plans, conda, verbose, temporary_outputs, sessions=None):
    sessions = {} if sessions is None else sessions
    for index, (cfg, stages, run_dir, root) in enumerate(plans, 1):
        if index > 1:
            print()
        print(job_title(index, cfg), flush=True)
        job_subheading("Status")
        started = datetime.now(timezone.utc)
        try:
            prepare_job_environments(cfg, stages, root, conda, sessions)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            raise record_preparation_failure(error, cfg, run_dir, root, conda, started) from error
        execute(stages, cfg, run_dir, root, conda, verbose)
        show_job_result(stages, save_outputs=cfg.get("save_outputs", True))
        if scratch := temporary_job_root(run_dir, temporary_outputs):
            temporary_outputs.pop(scratch).cleanup()


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


def evaluation_summary(output, *, save_outputs=True):
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
                  f"• Average time per episode: {average_text}"))
    if save_outputs:
        lines.append(f"• Result path: {output}")
    return "\n".join(lines)


def result_heading(in_job):
    if in_job:
        job_subheading("Result")
    else:
        section("Result")


def show_job_result(stages, *, save_outputs=True):
    result_heading(in_job=True)
    evaluations = [stage.output for stage in stages if stage.name == "eval" and stage.output is not None]
    if evaluations:
        summaries = (evaluation_summary(output, save_outputs=save_outputs) for output in evaluations)
        print("\n\n".join(indent(summary, "    ") for summary in summaries))
    else:
        print(styled("    • Job completed.", "32"))
        if save_outputs:
            for stage in stages:
                if stage.output is not None:
                    print(f"    • Result path: {stage.output}")
    if not save_outputs:
        print("    • Job outputs were not saved.")
    divider()


def report_failure(message, in_job, *, reason=None, logfile=None, temporary=False):
    result_heading(in_job)
    prefix = "    • " if in_job else ""
    print(f"{prefix}{message}", file=sys.stderr, flush=True)
    if reason:
        print(f"{prefix}Cause: {reason}", file=sys.stderr, flush=True)
    if logfile:
        print(f"{prefix}Full log: {logfile}", file=sys.stderr, flush=True)
    if temporary:
        print(f"{prefix}Job outputs were not saved.", file=sys.stderr, flush=True)
    divider()


@interrupt_on_termination()
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
    temporary_outputs = {}
    scratch_dirs = ExitStack()

    def make_temporary_output_directory():
        folder = TemporaryDirectory(prefix="convlexecutor-")
        path = Path(scratch_dirs.enter_context(folder)).resolve()
        temporary_outputs[path] = folder
        return path

    try:
        sessions = {}
        plans, conda = plan(args, make_temporary_output_directory, sessions)
        if not plans:
            return 0
        section("Settings")
        show_settings(plans, show_commands=not (args.settings or args.queue) or args.dry_run)
        if args.dry_run:
            return 0
        count = len(plans)
        job_label = "job" if count == 1 else "jobs"
        if not (args.settings or args.queue) and not confirm(f"Run {count} {job_label}? (Y/N)", "Y"):
            section("Result")
            print("Run canceled.")
            divider()
            return 0
        conda = conda or conda_executable(args.conda)
        section("Execution")
        executing = True
        run_plans(plans, conda, args.verbose, temporary_outputs, sessions)
        return 0
    except (KeyboardInterrupt, EOFError):
        report_failure("Run interrupted.", in_job=executing)
        return 130
    except StageFailure as error:
        temporary = temporary_job_root(error.logfile, temporary_outputs) is not None
        report_failure(f"Run failed ({error.description}).", in_job=executing,
                       reason=error.reason, logfile=None if temporary else error.logfile, temporary=temporary)
        return 1
    except RuntimeSetupFailure as error:
        temporary = bool(error.logfile and temporary_job_root(error.logfile, temporary_outputs) is not None)
        report_failure("Environment setup failed.", in_job=executing,
                       reason=error.reason, logfile=None if temporary else error.logfile, temporary=temporary)
        return 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, ImportError, SyntaxError,
            subprocess.SubprocessError) as error:
        report_failure(f"Run failed ({error})", in_job=executing)
        return 1
    finally:
        scratch_dirs.close()


if __name__ == "__main__":
    sys.exit(main())
