"""Inspect and install simulation dependencies using the selected LeRobot Python."""

import argparse
from dataclasses import fields, is_dataclass
from importlib import metadata, util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from adapters import PEFT_POLICIES, TRAINING_SWITCHES


BENCHMARKS = (
    ("libero", "LIBERO"),
    ("pusht", "PushT"),
    ("aloha", "ALOHA"),
    ("metaworld", "Meta-World"),
)
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


def benchmark_extras(distribution):
    return set(distribution.metadata.get_all("Provides-Extra") or ()) & {name for name, _ in BENCHMARKS}


def supported_here(name):
    return name != "libero" or sys.platform == "linux"


def environment_catalog():
    from packaging.utils import canonicalize_name

    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.configs import EnvConfig
    import lerobot.policies
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    distribution = metadata.distribution("lerobot")
    extras = set(distribution.metadata.get_all("Provides-Extra") or ())
    versions = installed_versions()
    policies, trainable = {}, {}
    for name, config in PreTrainedConfig.get_known_choices().items():
        if config.__module__.startswith("lerobot.policies."):
            extra = POLICY_EXTRAS.get(name, canonicalize_name(name))
            policies[name] = extra not in extras or all(
                installed(req, versions) for req in optional_requirements(distribution, extra)
            )
            if is_dataclass(config):
                names = {item.name for item in fields(config)}
                switches = [key for key in TRAINING_SWITCHES if key in names]
                if name == "molmoact2" and {"train_mode_vlm", "lora_rank", "lora_alpha", "lora_dropout"} <= names:
                    switches.append("train_mode_vlm")
                if switches:
                    trainable[name] = switches

    benchmarks = {name: supported_here(name) and all(
        installed(req, versions) for req in optional_requirements(distribution, name)
    ) for name in benchmark_extras(distribution)}
    # LeRobot imports installed third-party environment modules before parsing --env.type.
    plugins = sorted(name for name, config in EnvConfig.get_known_choices().items()
                     if config.__module__.startswith("lerobot_env_"))
    return {"policies": policies, "benchmarks": benchmarks, "plugins": plugins,
            "tinyvla_bridge": util.find_spec("lerobot_policy_tinyvla") is not None,
            "peft_ready": "peft" in versions,
            "peft_policies": sorted(PEFT_POLICIES.intersection(policies)), "training": trainable}


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
        if name not in benchmark_extras(distribution):
            raise ValueError(f"LeRobot {distribution.version} does not provide the {name} benchmark extra.")
        if not supported_here(name):
            raise ValueError("LIBERO requires Linux.")
        requirements = optional_requirements(distribution, name)
    needed = [str(req) for req in requirements if not installed(req, versions)]
    if not needed:
        print("Benchmark requirements are already installed.")
        return

    with tempfile.TemporaryDirectory(prefix="convlexecutor-benchmark-") as directory:
        constraints = Path(directory) / "installed.txt"
        core = {"lerobot", "numpy", "torch", "torchvision", "torchcodec", "transformers"}
        pins = ((name, version) for name, version in sorted(versions.items()) if not isolated or name in core)
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in pins), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-input", "--no-user",
                        "--disable-pip-version-check", "--progress-bar", "off",
                        "--constraint", str(constraints), *needed], check=True)


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
            install_benchmark(args.plugin or args.install, plugin=args.plugin is not None, isolated=args.isolated)
    except (ImportError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Benchmark setup failed ({error}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
