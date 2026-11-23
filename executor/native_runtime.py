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
        return original(*args, **kwargs)

    with patch.object(lerobot_eval, "make_policy", checked):
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
        if (getattr(policy, "type", None) == "wall_oss_05" and metadata is not None
                and getattr(metadata, "root", None)):
            features = getattr(metadata, "features", {})
            mapping = getattr(policy, "camera_key_mapping", {})
            if not mapping or not set(mapping) <= set(features):
                config = {"model": "wall_oss_05", "train": {"dataset": getattr(metadata, "repo_id", None),
                          "dataset_root": str(metadata.root), "extra": sys.argv[1:]}}
                for name, value in native_training_preflight(config).items():
                    setattr(policy, name.removeprefix("policy."), value)
                    print(f"Automatic training option: --{name}=" + json.dumps(value), flush=True)
        return original(*args, **kwargs)

    with patch.object(lerobot_train, "make_policy", checked):
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
            from lerobot.scripts import lerobot_eval

            hooks.enter_context(standard_libero_rollouts(lerobot_eval))
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
