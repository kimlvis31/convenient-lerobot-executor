"""Run the native LeRobot CLI with one immutable revision for a selected Hub policy."""

from contextlib import ExitStack
from functools import wraps
from pathlib import Path
import re
import sys
from unittest.mock import patch


REVISION_FLAG = "--policy.pretrained_revision="


def requested_revision(command):
    revisions = {item[len(REVISION_FLAG):] for item in command if item.startswith(REVISION_FLAG)}
    if len(revisions) > 1:
        raise ValueError("Conflicting --policy.pretrained_revision values. Select one Hub revision for the job.")
    return next(iter(revisions), None)


def pinned_command(command, stage, reference, revision, root):
    if revision is None:
        return command
    path = Path(reference).expanduser()
    if path.is_absolute() or (Path(root) / path).exists() or reference.startswith(("./", "../", "~", ".\\", "..\\")):
        return command
    if not revision.strip() or not re.fullmatch(r"[\w][\w.-]*(?:/[\w][\w.-]*)?", reference, re.ASCII):
        raise ValueError("--policy.pretrained_revision requires a Hugging Face repository ID and a nonempty revision.")
    entry = f"lerobot-{stage}"
    if entry in command:
        start = command.index(entry)
        end = start + 1
    else:
        camera_helper = str(Path(__file__).with_name("evaluation_cameras.py").resolve())
        if stage != "eval" or camera_helper not in command:
            raise ValueError(f"Unexpected LeRobot {stage} entry point.")
        end = command.index(camera_helper) + 1
        start = end - 3  # NativeAdapter emits python -u followed by the helper.
        if start < 0 or command[start:end - 1] != ["python", "-u"]:
            raise ValueError("Unexpected camera evaluation entry point.")
    return [*command[:start], "python", "-B", str(Path(__file__).resolve()), stage, *command[end:]]


def pinned_entrypoint(stage, reference, revision):
    from huggingface_hub import hf_hub_download
    from lerobot.configs import PreTrainedConfig

    snapshot = Path(hf_hub_download(reference, "config.json", revision=revision)).parent.name
    if not re.fullmatch(r"[0-9a-f]{40}", snapshot):
        raise RuntimeError(f"Could not resolve an immutable revision for {reference}@{revision}.")
    print(f"Using Hugging Face policy {reference}@{snapshot} (requested {revision}).", flush=True)

    def matches(value):
        return value is not None and Path(value).as_posix() == reference

    original_config = PreTrainedConfig.from_pretrained.__func__

    @wraps(original_config)
    def load_config(cls, pretrained_name_or_path, **kwargs):
        if matches(pretrained_name_or_path) and kwargs.get("revision") in (None, revision):
            kwargs["revision"] = snapshot
        return original_config(cls, pretrained_name_or_path, **kwargs)

    # The native CLI drops the revision on its initial config read; the native
    # evaluator also omits it when restoring processors. Limit the adjustments
    # to this child process and the exact policy selected for this job.
    with ExitStack() as hooks:
        cli = [REVISION_FLAG + snapshot if arg.startswith(REVISION_FLAG) else arg for arg in sys.argv]
        hooks.enter_context(patch.object(sys, "argv", cli))
        hooks.enter_context(patch.object(PreTrainedConfig, "from_pretrained", classmethod(load_config)))
        if stage == "eval":
            from lerobot.scripts import lerobot_eval

            from evaluation_cameras import restore_camera_arguments

            restore_camera_arguments()
            original_processors = lerobot_eval.make_pre_post_processors

            @wraps(original_processors)
            def load_processors(*args, **kwargs):
                if matches(kwargs.get("pretrained_path")) and kwargs.get("pretrained_revision") in (None, revision):
                    kwargs["pretrained_revision"] = snapshot
                return original_processors(*args, **kwargs)

            hooks.enter_context(patch.object(lerobot_eval, "make_pre_post_processors", load_processors))
            return lerobot_eval.main()
        if stage != "train":
            raise ValueError(f"Unsupported native stage: {stage}")
        from lerobot.policies import pretrained
        from lerobot.scripts import lerobot_train

        # Preserve the base commit in every saved PEFT adapter, including an
        # intermediate checkpoint if the training run is later interrupted.
        original_peft = getattr(pretrained, "get_peft_model", None)
        if original_peft is not None:
            @wraps(original_peft)
            def create_adapter(model, peft_config, *args, **kwargs):
                if matches(getattr(model.config, "pretrained_path", None)):
                    if kwargs.get("revision") in (None, revision) and peft_config.revision in (None, revision, snapshot):
                        kwargs["revision"] = snapshot
                return original_peft(model, peft_config, *args, **kwargs)

            hooks.enter_context(patch.object(pretrained, "get_peft_model", create_adapter))
        return lerobot_train.main()


def main():
    stage = sys.argv.pop(1)
    flags = dict(arg[2:].split("=", 1) for arg in sys.argv[1:] if arg.startswith("--") and "=" in arg)
    return pinned_entrypoint(stage, flags["policy.path"], requested_revision(sys.argv[1:]))


if __name__ == "__main__":
    main()
