"""Inspect and install simulation dependencies using the selected LeRobot Python."""

import argparse
from dataclasses import fields, is_dataclass
from importlib import metadata, util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Literal, get_args, get_origin, get_type_hints

from adapters import PEFT_POLICIES, TRAINING_SWITCHES, is_policy_training_option


BENCHMARKS = (
    ("libero", "LIBERO"),
    ("pusht", "PushT"),
    ("aloha", "ALOHA"),
    ("metaworld", "Meta-World"),
)
BENCHMARK_DEPENDENCY_CONFLICT = 3
BENCHMARK_MODULES = {
    "libero": "libero", "pusht": "gym_pusht", "aloha": "gym_aloha", "metaworld": "metaworld",
}
POLICY_EXTRAS = {"pi0": "pi", "pi05": "pi", "pi0_fast": "pi", "wall_x": "wallx"}


def optional_requirements(distribution, extra):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pending = [canonicalize_name(extra)]
    visited = set()
    requirements = {}
    declared = [Requirement(item) for item in distribution.requires or ()]
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for requirement in declared:
            marker = requirement.marker
            if marker is None or marker.evaluate({"extra": ""}) or not marker.evaluate({"extra": name}):
                continue
            if canonicalize_name(requirement.name) == "lerobot":
                pending.extend(requirement.extras)
                continue
            # pip receives only the dependencies selected by the LeRobot extra.
            selected = Requirement(str(requirement))
            selected.marker = None
            requirements[str(selected)] = selected
    return list(requirements.values())


def installed_versions():
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version

    versions = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            try:
                versions.setdefault(canonicalize_name(name, validate=True), str(Version(distribution.version)))
            except (ValueError, InvalidVersion):
                continue
    return versions


def installed(requirement, versions):
    from packaging.utils import canonicalize_name

    version = versions.get(canonicalize_name(requirement.name))
    return version is not None and requirement.specifier.contains(version, prereleases=True)


def benchmark_extras(distribution, env_types):
    from packaging.utils import canonicalize_name

    extras = {canonicalize_name(name) for name in distribution.metadata.get_all("Provides-Extra") or ()}
    return {name: canonicalize_name(name) for name in env_types if canonicalize_name(name) in extras}


def local_benchmark_ready(name, requirements, versions):
    module = BENCHMARK_MODULES.get(name)
    return all(installed(req, versions) for req in requirements) and (
        module is None or util.find_spec(module) is not None
    )


def supported_here(name):
    return name != "libero" or sys.platform == "linux"


def policy_training_options(config):
    if not is_dataclass(config):
        return {}
    try:
        hints = get_type_hints(config)
    except (NameError, TypeError):
        hints = {}
    result = {}
    for item in fields(config):
        if not is_policy_training_option(item.name):
            continue
        hint = hints.get(item.name, item.type)
        default = item.default
        if get_origin(hint) is Literal and type(default) in (bool, int, float, str):
            values = [value for value in get_args(hint) if type(value) is type(default)]
            if values:
                result[item.name] = {"default": default, "choices": values}
        elif type(default) in (bool, int, float) or type(default) is str and hint in (str, "str"):
            result[item.name] = {"default": default}
    return result


def environment_catalog():
    from packaging.utils import canonicalize_name

    distribution = metadata.distribution("lerobot")
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.configs import EnvConfig
    from lerobot.policies.factory import get_policy_class
    import lerobot.policies
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    extras = set(distribution.metadata.get_all("Provides-Extra") or ())
    versions = installed_versions()
    policies, trainable, options, issues = {}, {}, {}, {}
    for name, config in PreTrainedConfig.get_known_choices().items():
        if config.__module__.startswith("lerobot.policies."):
            extra = POLICY_EXTRAS.get(name, canonicalize_name(name))
            missing = sorted({req.name for req in optional_requirements(distribution, extra)
                              if not installed(req, versions)}) if extra in extras else []
            if name == "flux3" and "natten" not in versions:
                missing.append("natten")
            if missing:
                issues[name] = f"Missing: {', '.join(missing)}"
            else:
                try:
                    get_policy_class(name)
                except ModuleNotFoundError as error:
                    issues[name] = f"Missing: {error.name}" if error.name else "Runtime Import Failed"
                except Exception:
                    issues[name] = "Runtime Import Failed"
            policies[name] = name not in issues
            editable = policy_training_options(config)
            if editable:
                options[name] = editable
            if is_dataclass(config):
                names = {item.name for item in fields(config)}
                switches = [key for key in TRAINING_SWITCHES if key in names]
                if name == "molmoact2" and {"train_mode_vlm", "lora_rank", "lora_alpha", "lora_dropout"} <= names:
                    switches.append("train_mode_vlm")
                if switches:
                    trainable[name] = switches

    env_types = EnvConfig.get_known_choices()
    installable = benchmark_extras(distribution, env_types)
    benchmarks = {name: local_benchmark_ready(name, optional_requirements(distribution, extra), versions)
                  for name, extra in installable.items() if supported_here(name)}
    # LeRobot imports installed third-party environment modules before parsing --env.type.
    plugins = sorted(name for name, config in env_types.items() if config.__module__.startswith("lerobot_env_"))
    manual = sorted(name for name, config in env_types.items()
                    if config.__module__.startswith("lerobot.envs.") and name not in installable
                    and name != "gym_manipulator")
    return {"policies": policies, "benchmarks": benchmarks, "plugins": plugins, "manual_benchmarks": manual,
            "tinyvla_bridge": util.find_spec("lerobot_policy_tinyvla") is not None,
            "peft_ready": "peft" in versions,
            "peft_policies": sorted(PEFT_POLICIES.intersection(policies)), "training": trainable,
            "policy_issues": issues, "training_options": options}


def install_benchmark(name, *, plugin=False, isolated=False):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    distribution = metadata.distribution("lerobot")
    versions = installed_versions()
    if plugin:
        requirement = Requirement(name)
        if not canonicalize_name(requirement.name).startswith("lerobot-env-"):
            raise ValueError("Use a LeRobot environment plugin package named lerobot-env-*.")
        requirements = [requirement]
    else:
        from lerobot.envs.configs import EnvConfig

        extra = benchmark_extras(distribution, EnvConfig.get_known_choices()).get(name)
        if extra is None:
            raise ValueError(f"LeRobot {distribution.version} does not provide the {name} benchmark extra.")
        if not supported_here(name):
            raise ValueError("LIBERO requires Linux.")
        requirements = optional_requirements(distribution, extra)
    needed = [str(req) for req in requirements if not installed(req, versions)]
    if not needed:
        print("Benchmark requirements are already installed.")
        return

    with tempfile.TemporaryDirectory(prefix="convlexecutor-benchmark-") as directory:
        constraints = Path(directory) / "installed.txt"
        core = {"lerobot", "numpy", "torch", "torchvision", "torchcodec", "transformers"}
        pins = ((name, version) for name, version in sorted(versions.items()) if not isolated or name in core)
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in pins), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-input", "--no-user",
             "--disable-pip-version-check", "--progress-bar", "off", "--constraint", str(constraints), *needed],
            text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode:
            return BENCHMARK_DEPENDENCY_CONFLICT if "ResolutionImpossible" in result.stdout else 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--catalog", action="store_true")
    operation.add_argument("--install")
    operation.add_argument("--plugin")
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    try:
        if args.catalog:
            print("MANAGER_CATALOG " + json.dumps(environment_catalog()))
        else:
            return install_benchmark(args.plugin or args.install, plugin=args.plugin is not None, isolated=args.isolated) or 0
    except (ImportError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Benchmark setup failed ({error}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
