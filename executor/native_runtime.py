"""AI-written child-local native entrypoints with host-staged checkpoint loading."""

from contextlib import contextmanager, ExitStack
from functools import wraps
from inspect import signature
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


HELPERS = frozenset({"evaluation_cameras.py", "parallel_eval.py", "pinned_policy.py"})


def complete_checkpoint_policy(factory, policy, args, kwargs):
    from checkpoint_initialization import checkpoint_model_placement
    from molmo_compat import checkpoint_policy_factory

    with checkpoint_model_placement(factory, policy, args, kwargs):
        return checkpoint_policy_factory(factory, policy, args, kwargs)


def native_invocation(command):
    """Recognize only native console commands and this checkout's exact helpers."""
    if not command:
        return None
    if command[0] in {"lerobot-train", "lerobot-eval"}:
        return command[0], command[1:]
    if Path(command[0]).name not in {"python", "python3", "python.exe", Path(sys.executable).name}:
        return None
    index = 1
    while index < len(command) and command[index] in {"-B", "-u"}:
        index += 1
    if index == len(command):
        return None
    script = Path(command[index])
    if script.name not in HELPERS or script.resolve().parent != Path(__file__).resolve().parent:
        return None
    return str(script.resolve()), command[index + 1:]


@contextmanager
def host_staged_weights(pretrained):
    """Avoid materializing a second full checkpoint on CUDA or MPS.

    LeRobot retains its own load_model, strictness, shared-tensor handling,
    key reporting and final policy.to(device). Only the temporary file-read
    device changes in this child; existing model placement is untouched.
    """
    original = getattr(pretrained, "load_model_as_safetensor", None)
    if not callable(original):
        yield  # Other LeRobot versions may provide a different loading API.
        return

    @wraps(original)
    def load(model, filename, *args, **kwargs):
        # LeRobot supplies the read device by keyword. Preserve positional
        # calling conventions from other versions rather than guessing them.
        device = kwargs.get("device")
        if device is not None and str(device).split(":", 1)[0] in {"cuda", "mps"}:
            kwargs["device"] = "cpu"
            print("Loading native checkpoint tensors through host memory before device placement.", flush=True)
        return original(model, filename, *args, **kwargs)

    with patch.object(pretrained, "load_model_as_safetensor", load):
        yield


@contextmanager
def checkpoint_processors(evaluator):
    original = getattr(evaluator, "make_pre_post_processors", None)
    if not callable(original):
        yield
        return

    @wraps(original)
    def checked(*args, **kwargs):
        from molmo_compat import checkpoint_processor_factory

        return checkpoint_processor_factory(original, args, kwargs)

    with patch.object(evaluator, "make_pre_post_processors", checked):
        yield


@contextmanager
def evaluation_action_contract():
    """Check the native environment's declared policy action space before weights."""
    from lerobot.scripts import lerobot_eval
    from lerobot.policies.factory import env_to_policy_features
    from evaluation_capabilities import checkpoint_action_issue

    original = lerobot_eval.make_policy
    parameters = signature(original)

    @wraps(original)
    def checked(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        policy, environment = bound.arguments.get("cfg"), bound.arguments.get("env_cfg")
        if policy is not None and environment is not None:
            issue = checkpoint_action_issue(policy, env_to_policy_features(environment),
                                            getattr(environment, "type", type(environment).__name__))
            if issue:
                raise ValueError(issue)
        from checkpoint_compat import validate_evaluation_processor_files
        from importlib import import_module

        validate_evaluation_processor_files(policy, import_module('lerobot.policies.factory'))
        return complete_checkpoint_policy(original, policy, args, kwargs)

    with patch.object(lerobot_eval, "make_policy", checked), checkpoint_processors(lerobot_eval):
        yield


@contextmanager
def dataset_preflight(trainer):
    """Apply required settings immediately after native metadata resolution."""
    prepare = getattr(trainer, "make_train_eval_datasets", None)
    if not callable(prepare):
        yield
        return

    @wraps(prepare)
    def checked(cfg, *args, **kwargs):
        policy = getattr(cfg, "policy", None)
        if policy is None or getattr(cfg, "resume", False):
            return prepare(cfg, *args, **kwargs)
        from lerobot.datasets import factory as datasets
        from training_preflight import native_training_preflight
        import draccus

        original = getattr(datasets, "load_dataset_metadata", None)
        if not callable(original):
            return prepare(cfg, *args, **kwargs)

        @wraps(original)
        def metadata(*parameters, **options):
            result = original(*parameters, **options)
            if not getattr(result, "root", None) or getattr(result, "repo_id", None) != cfg.dataset.repo_id:
                return result
            extra = list(sys.argv[1:])
            if getattr(cfg, "rename_map", None):
                extra.append("--rename_map=" + json.dumps(cfg.rename_map))
            job = {"model": policy.type, "train": {"dataset": cfg.dataset.repo_id,
                   "dataset_root": str(result.root), "pretrained": str(policy.pretrained_path)
                   if getattr(policy, "pretrained_path", None) else None, "extra": extra}}
            for name, value in native_training_preflight(job, policy_config=draccus.encode(policy)).items():
                if name.startswith("policy."):
                    setattr(policy, name.removeprefix("policy."), value)
                else:
                    setattr(cfg, name, value)
                print(f"Automatic training option: --{name}=" + json.dumps(value), flush=True)
            return result

        with patch.object(datasets, "load_dataset_metadata", metadata):
            return prepare(cfg, *args, **kwargs)

    with patch.object(trainer, "make_train_eval_datasets", checked):
        yield


@contextmanager
def training_dataset_contract():
    """Resolve verified dataset-dependent defaults after a cold dataset download."""
    from lerobot.scripts import lerobot_train
    from training_preflight import native_training_preflight

    original = lerobot_train.make_policy
    parameters = signature(original)

    @wraps(original)
    def checked(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        policy, metadata = bound.arguments.get("cfg"), bound.arguments.get("ds_meta")
        molmo = getattr(policy, "type", None) == "molmoact2"
        exact_resume = "--resume=true" in sys.argv[1:]
        preserve_names = molmo and (exact_resume or any(argument.startswith(
            ("--policy.dataset_feature_names=", "--policy.dataset_feature_names.")) for argument in sys.argv[1:]))
        restored_names = getattr(policy, "dataset_feature_names", None) if preserve_names else None
        post_construction = {}
        if (getattr(policy, "type", None) in {"wall_oss_05", "molmoact2"} and metadata is not None
                and getattr(metadata, "root", None)):
            features = getattr(metadata, "features", {})
            mapping = getattr(policy, "camera_key_mapping", {})
            if getattr(policy, "type", None) == "molmoact2" or not mapping or not set(mapping) <= set(features):
                config = {"model": policy.type, "train": {"dataset": getattr(metadata, "repo_id", None),
                          "dataset_root": str(metadata.root), "extra": sys.argv[1:],
                          "pretrained": str(policy.pretrained_path) if getattr(policy, "pretrained_path", None) else None}}
                if exact_resume:
                    config["train"]["resume"] = True
                resolved = None
                if policy.type == "molmoact2":
                    import draccus

                    resolved = draccus.encode(policy)
                for name, value in native_training_preflight(config, policy_config=resolved).items():
                    if not name.startswith("policy."):
                        continue  # Root settings are applied before dataset preparation.
                    if name == "policy.dataset_feature_names":
                        post_construction[name] = value
                        continue
                    setattr(policy, name.removeprefix("policy."), value)
                    print(f"Automatic training option: --{name}=" + json.dumps(value), flush=True)
        result = complete_checkpoint_policy(original, policy, args, kwargs)
        if preserve_names:
            policy.dataset_feature_names = restored_names
            result.config.dataset_feature_names = restored_names
        else:
            for name, value in post_construction.items():
                setattr(policy, name.removeprefix("policy."), value)
                setattr(result.config, name.removeprefix("policy."), value)
                print(f"Automatic training option: --{name}=" + json.dumps(value), flush=True)
        return result

    with patch.object(lerobot_train, "make_policy", checked), checkpoint_processors(lerobot_train), dataset_preflight(lerobot_train):
        yield


def main(command=None):
    command = list(sys.argv[1:] if command is None else command)
    if command[:1] == ["--"]:
        command.pop(0)
    selected = native_invocation(command)
    if selected is None:
        raise ValueError("Expected a native LeRobot command or an executor evaluation helper.")
    entry, arguments = selected
    evaluation = entry == "lerobot-eval" or Path(entry).name in {"parallel_eval.py", "evaluation_cameras.py"}
    evaluation = evaluation or Path(entry).name == "pinned_policy.py" and arguments[:1] == ["eval"]
    # The image-library initialization must precede native extensions, as in
    # the original helper entrypoints. Importing policies can load those too.
    if evaluation:
        from evaluation_cameras import prepare_evaluation_runtime

        prepare_evaluation_runtime(arguments)
    from lerobot.policies import pretrained

    with ExitStack() as hooks:
        hooks.enter_context(patch.object(sys, "argv", [entry, *arguments]))
        hooks.enter_context(host_staged_weights(pretrained))
        if evaluation:
            hooks.enter_context(evaluation_action_contract())
            from evaluation_efficiency import standard_libero_rollouts
            from evaluation_video import bounded_evaluation_videos
            from lerobot.scripts import lerobot_eval

            hooks.enter_context(standard_libero_rollouts(lerobot_eval))
            hooks.enter_context(bounded_evaluation_videos(lerobot_eval))
        else:
            hooks.enter_context(training_dataset_contract())
        if entry == "lerobot-train":
            from lerobot.scripts.lerobot_train import main as execute

            return execute()
        if entry == "lerobot-eval":
            from lerobot.scripts.lerobot_eval import main as execute

            return execute()
        runpy.run_path(entry, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
