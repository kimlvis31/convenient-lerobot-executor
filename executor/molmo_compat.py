"""Preserve full MolmoAct2 checkpoint topology across compatible native loaders."""

import inspect
import json
from dataclasses import fields
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch


LEGACY_FLAGS = ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only")
LEGACY_FIELDS = frozenset((*LEGACY_FLAGS, "model_dtype"))


def evaluation_config(config):
    """Migrate only the known full-weight topology, only for an incompatible loader."""
    if config.get("type") != "molmoact2" or not LEGACY_FIELDS <= config.keys():
        return None
    if any(config[key] is not False for key in LEGACY_FLAGS):
        raise ValueError("Legacy MolmoAct2 adapter or frozen-expert settings require their original LeRobot loader.")
    dtype = config["model_dtype"]
    if (dtype not in ("bfloat16", "float32") or config.get("dtype", dtype) != dtype
            or config.get("train_mode_vlm", "fft") != "fft"):
        raise ValueError("Conflicting legacy MolmoAct2 precision or training topology.")
    migrated = {key: value for key, value in config.items() if key not in LEGACY_FIELDS}
    return {**migrated, "dtype": dtype, "train_mode_vlm": "fft"}


def prepare_molmo_evaluation(reference, destination, revision=None):
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    supported = {field.name for field in fields(MolmoAct2Config)}
    if LEGACY_FIELDS & supported or not {"dtype", "train_mode_vlm"} <= supported:
        return None

    base = Path(reference).expanduser()
    local = base.is_dir()
    if local:
        config_path = base / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(reference, "config.json", revision=revision))
        # Hub resolves a branch or tag into an immutable snapshot directory.
        revision = config_path.parent.name
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not LEGACY_FIELDS <= source_config.keys():
        return None
    from draccus.utils import DecodingError
    from lerobot.configs import PreTrainedConfig

    try:
        PreTrainedConfig.from_pretrained(config_path.parent)
    except DecodingError:
        pass
    else:
        # Newer loaders may support old names without exposing dataclass fields.
        return None
    config = evaluation_config(source_config)
    if config is None:
        return None

    def fetch(name):
        if local:
            return base / name
        return Path(hf_hub_download(reference, name, revision=revision))

    from checkpoint_compat import write_processor_view

    return write_processor_view(destination, fetch, {"config.json": config})


def _complete_policy_checkpoint(config, reference=None):
    """Require a complete local full-policy snapshot before skipping base weights."""
    reference = reference if reference is not None else getattr(config, "pretrained_path", None)
    if not reference:
        return None
    directory = Path(reference).expanduser()
    if not directory.is_dir():
        return None
    try:
        saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if saved.get("type") != "molmoact2":
            return None
        single = directory / "model.safetensors"
        if single.is_file() and single.stat().st_size:
            return saved
        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index.get("weight_map", {}).values())
        if shards and all(isinstance(name, str) and (directory / name).resolve().is_relative_to(directory.resolve())
                          and (directory / name).is_file() and (directory / name).stat().st_size for name in shards):
            return saved
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def checkpoint_metadata_location(reference, *, revision=None, force_download=False, processors=False):
    """Resolve an immutable architecture/processor snapshot without model tensors."""
    directory = Path(reference).expanduser()
    if directory.is_dir():
        return str(directory)
    from huggingface_hub import hf_hub_download

    metadata = Path(hf_hub_download(reference, "config.json", revision=revision,
                                   force_download=force_download)).parent
    if processors:
        from huggingface_hub import snapshot_download

        snapshot_download(reference, revision=metadata.name, force_download=force_download,
                          allow_patterns=["*.json", "*.txt", "*.jinja", "*.jinja2", "tokenizer.model", "*.tiktoken"])
    print(f"MolmoAct2 {'processor files' if processors else 'architecture metadata'}: {metadata}", flush=True)
    return str(metadata)


def checkpoint_processor_factory(factory, args, kwargs):
    """Keep native processor behavior while excluding unused raw model weights."""
    bound = inspect.signature(factory).bind_partial(*args, **kwargs)
    config = bound.arguments.get("policy_cfg")
    if getattr(config, "type", None) != "molmoact2":
        return factory(*args, **kwargs)
    from lerobot.policies.molmoact2 import processor_molmoact2 as native

    if not callable(getattr(native, "_resolve_checkpoint_location", None)):
        return factory(*args, **kwargs)

    def location(reference, *, revision=None, force_download=False):
        # A full policy's constructor already resolved the raw architecture.
        # Its tokenizer/config must come from that same immutable snapshot.
        resolved = getattr(config, "_convlex_molmo_metadata", None)
        if resolved and resolved[:2] == (str(reference), revision):
            revision = Path(resolved[2]).name
        return checkpoint_metadata_location(reference, revision=revision, force_download=force_download,
                                            processors=True)

    with patch.object(native, "_resolve_checkpoint_location", location):
        return factory(*args, **kwargs)


def checkpoint_policy_factory(factory, config, args, kwargs):
    """Initialize architecture from metadata, then use native strict policy loading.

    A full LeRobot snapshot already contains the base weights. The native Molmo
    constructor otherwise downloads/loads another base and overwrites it with
    the policy snapshot. Fresh policies retain the original base initialization.
    New dense-to-LoRA training loads the dense state before inserting the native
    adapters; saved LoRA checkpoints keep their saved construction topology.
    """
    if getattr(config, "type", None) != "molmoact2":
        return factory(*args, **kwargs)
    try:
        call = inspect.signature(factory).bind_partial(*args, **kwargs).arguments
    except (TypeError, ValueError):
        return factory(*args, **kwargs)
    if call.get("defer_weight_load", False) or call.get("strict", True) is not True:
        # The optimization requires the factory to load all authoritative weights.
        return factory(*args, **kwargs)
    saved = _complete_policy_checkpoint(config, call.get("pretrained_path"))
    if saved is None:
        return factory(*args, **kwargs)
    from lerobot.policies.molmoact2 import modeling_molmoact2 as native
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            return factory(*args, **kwargs)

    hf_model = native.MolmoAct2ForConditionalGeneration
    policy_class = native.MolmoAct2Policy
    required = (getattr(native, "_resolve_checkpoint_location", None),
                getattr(native, "_strict_load_safetensors_weights", None),
                getattr(hf_model, "_from_config", None))
    if not all(callable(method) for method in required):
        return factory(*args, **kwargs)

    def metadata_location(reference, *, revision=None, force_download=False):
        metadata = checkpoint_metadata_location(reference, revision=revision, force_download=force_download)
        config._convlex_molmo_metadata = (str(reference), revision, metadata)
        return metadata

    def from_config(cls, reference, *, config, dtype=None, low_cpu_mem_usage=True, token=None):
        del reference, low_cpu_mem_usage, token
        with no_init_weights():
            return cls._from_config(config, dtype=dtype)

    transition = (getattr(config, "train_mode_vlm", None) == "lora"
                  and saved.get("train_mode_vlm", "fft") in ("fft", "freeze"))
    deferred_lora, deferred_compile = [], []
    with ExitStack() as hooks:
        hooks.enter_context(patch.object(native, "_resolve_checkpoint_location", metadata_location))
        hooks.enter_context(patch.object(hf_model, "from_pretrained", classmethod(from_config)))
        hooks.enter_context(patch.object(native, "_strict_load_safetensors_weights", lambda *_: None))
        if transition:
            lora, compile_model = policy_class._apply_lora_adapters, policy_class._apply_compile
            hooks.enter_context(patch.object(policy_class, "_apply_lora_adapters", lambda instance: deferred_lora.append(instance)))
            hooks.enter_context(patch.object(policy_class, "_apply_compile", lambda instance: deferred_compile.append(instance)))
        print("Initializing MolmoAct2 architecture from metadata; loading the full native policy checkpoint.", flush=True)
        policy = factory(*args, **kwargs)
        for instance in deferred_lora:
            training = instance.training
            lora(instance)
            instance.train(training)
        for instance in deferred_compile:
            compile_model(instance)
        return policy
