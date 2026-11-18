"""Inspect and install policy and simulation dependencies in the selected LeRobot environment."""

import argparse
from dataclasses import fields, is_dataclass
from importlib import metadata, util
from importlib.machinery import PathFinder
import json
import os
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
DEPENDENCY_CONFLICT = 3
LEROBOT_RUNTIME_EXTRAS = ("training", "evaluation", "dataset")
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


def policy_requirements(distribution, name):
    from packaging.utils import canonicalize_name

    extra = POLICY_EXTRAS.get(name, canonicalize_name(name))
    extras = {canonicalize_name(value) for value in distribution.metadata.get_all("Provides-Extra") or ()}
    return optional_requirements(distribution, extra) if extra in extras else []


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


def libero_plus_issue():
    """Probe the effective fork without importing LIBERO (which can prompt or write a config)."""
    outer = util.find_spec("libero")
    paths = outer.submodule_search_locations if outer else None
    inner = PathFinder.find_spec("libero.libero", paths) if paths else None
    roots = inner.submodule_search_locations if inner else None
    root = Path(next(iter(roots))) if roots else None
    if root is None or not (root / "benchmark/task_classification.json").is_file():
        return "The active libero package is not the LIBERO-plus fork."

    settings = Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser() / "config.yaml"
    try:
        import yaml

        config = yaml.safe_load(settings.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError) as error:
        return f"Cannot read {settings}: {error}"
    except yaml.YAMLError as error:
        return f"Invalid LIBERO config {settings}: {error}"
    if not isinstance(config, dict):
        return f"Invalid LIBERO config: {settings}"
    for key in ("bddl_files", "init_states"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_dir():
            return f"Set {key} to a LIBERO-plus directory in {settings}."
    if not (Path(config["init_states"]) / "libero_newobj").is_dir():
        return f"The init_states directory in {settings} does not contain the LIBERO-plus variants."
    if not (root / "assets/scenes/libero_tabletop_base_style.xml").is_file():
        return f"LIBERO-plus scene assets are missing under {root / 'assets'}."
    try:
        from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
        from skimage.filters import gaussian
        from wand.image import Image

        # Importing the Python package alone does not load the ImageMagick shared library.
        with Image(width=1, height=1) as image:
            image.flip()
    except Exception as error:
        # An incompatible native extension can also raise a ValueError or AttributeError.
        return f"LIBERO-plus simulator dependency failed: {error}"
    return None


# These source-based environments require simulator assets in addition to Python modules.
# The probe covers shared runtime assets. Individual scene assets are checked by the simulator.
SIMULATOR_REQUIREMENTS = {
    "robomme": (("robomme", "sapien", "mani_skill"), ("env_metadata/test/record_dataset_*_metadata.json",)),
    "robocasa": (("robocasa", "robosuite", "mujoco"),
                 ("wrappers/gym_wrapper.py", "models/assets/fixtures/*/*/model.xml",
                  "models/assets/objects/lightwheel/*/*/model.xml", "models/assets/textures/*/*",
                  "models/assets/generative_textures/*/*")),
    "vlabench": (("VLABench", "dm_control", "open3d"),
                 ("configs/robot_config.json", "assets/scenes/default/empty.xml", "assets/obj/*")),
}


def simulator_runtime_issue(benchmark):
    modules, assets = SIMULATOR_REQUIREMENTS[benchmark]
    specs = {name: util.find_spec(name) for name in modules}
    missing = [name for name, spec in specs.items() if spec is None]
    if missing:
        return "Missing simulator modules: " + ", ".join(missing) + "."
    roots = specs[modules[0]].submodule_search_locations
    if not roots:
        return f"Cannot locate the {modules[0]} simulator package."
    root = Path(next(iter(roots)))
    # Upstream VLABench sets its own package root when imported on Linux.
    if benchmark == "vlabench" and sys.platform != "linux":
        root = Path(os.environ.get("VLABENCH_ROOT", root))
    for pattern in assets:
        if not any(root.glob(pattern)):
            return f"Missing simulator files under {root}: {pattern}."
    return None


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
    distribution = metadata.distribution("lerobot")
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs.configs import EnvConfig
    from lerobot.policies.factory import get_policy_class
    import lerobot.policies
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    versions = installed_versions()
    policies, trainable, options, issues, policy_installable = {}, {}, {}, {}, []
    for name, config in PreTrainedConfig.get_known_choices().items():
        if config.__module__.startswith("lerobot.policies."):
            missing = sorted({req.name for req in policy_requirements(distribution, name)
                              if not installed(req, versions)})
            if name == "flux3" and "natten" not in versions:
                if "natten" not in missing:
                    missing.append("natten")
            elif missing:
                policy_installable.append(name)
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
    # The extra alone cannot install the native image library or perturbation assets.
    # Keep this separate from pip-installable benchmarks to protect stock LIBERO.
    manual_status = {}
    if "libero_plus" in env_types and env_types["libero_plus"].__module__.startswith("lerobot.envs."):
        benchmarks.pop("libero_plus", None)
        if "libero_plus" not in manual:
            manual.append("libero_plus")
        manual_status["libero_plus"] = libero_plus_issue()
    for name in SIMULATOR_REQUIREMENTS.keys() & set(manual):
        manual_status[name] = simulator_runtime_issue(name)
    peft_requirements = optional_requirements(distribution, "peft")
    runtime = {extra: all(installed(req, versions) for req in optional_requirements(distribution, extra))
               for extra in LEROBOT_RUNTIME_EXTRAS}
    return {"policies": policies, "benchmarks": benchmarks, "plugins": plugins, "manual_benchmarks": manual,
            "manual_status": manual_status, "runtime": runtime,
            "tinyvla_bridge": util.find_spec("lerobot_policy_tinyvla") is not None,
            "peft_ready": "peft" in versions and all(installed(req, versions) for req in peft_requirements),
            "peft_installable": bool(peft_requirements),
            "peft_policies": sorted(PEFT_POLICIES.intersection(policies)), "training": trainable,
            "policy_issues": issues, "policy_installable": sorted(policy_installable), "training_options": options}


def install_dependencies(*, benchmark=None, policy=None, plugin=None, peft=False, runtime=None, isolated=False):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    distribution = metadata.distribution("lerobot")
    versions = installed_versions()
    requirements = []
    if plugin:
        requirement = Requirement(plugin)
        if not canonicalize_name(requirement.name).startswith("lerobot-env-"):
            raise ValueError("Use a LeRobot environment plugin package named lerobot-env-*.")
        requirements.append(requirement)
    if benchmark:
        from lerobot.envs.configs import EnvConfig

        extra = benchmark_extras(distribution, EnvConfig.get_known_choices()).get(benchmark)
        if extra is None:
            raise ValueError(f"LeRobot {distribution.version} does not provide the {benchmark} benchmark extra.")
        if not supported_here(benchmark):
            raise ValueError("LIBERO requires Linux.")
        requirements.extend(optional_requirements(distribution, extra))
    if policy:
        from lerobot.configs import PreTrainedConfig
        import lerobot.policies

        config = PreTrainedConfig.get_known_choices().get(policy)
        declared = policy_requirements(distribution, policy)
        if config is None or not config.__module__.startswith("lerobot.policies.") or not declared:
            raise ValueError(f"LeRobot {distribution.version} does not declare installable dependencies for {policy}.")
        requirements.extend(declared)
    if runtime:
        if runtime not in LEROBOT_RUNTIME_EXTRAS:
            raise ValueError(f"Unknown LeRobot runtime extra: {runtime}.")
        requirements.extend(optional_requirements(distribution, runtime))
    if peft:
        declared = optional_requirements(distribution, "peft")
        if not declared:
            raise ValueError(f"LeRobot {distribution.version} does not declare LoRA dependencies.")
        requirements.extend(declared)
    needed = sorted({str(req) for req in requirements if not installed(req, versions)})
    if not needed:
        print("Requested dependencies are already installed.")
        return 0

    with tempfile.TemporaryDirectory(prefix="convlexecutor-dependencies-") as directory:
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
            return DEPENDENCY_CONFLICT if "ResolutionImpossible" in result.stdout else 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--install", help="Install the declared dependencies for a benchmark")
    parser.add_argument("--policy", help="Install the declared dependencies for a native policy")
    parser.add_argument("--plugin", help="Install a named LeRobot environment plugin")
    parser.add_argument("--peft", action="store_true", help="Install the declared LoRA dependencies")
    parser.add_argument("--runtime", choices=LEROBOT_RUNTIME_EXTRAS, help="Install the declared LeRobot runtime extra")
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    if sum(map(bool, (args.catalog, args.install or args.policy or args.peft or args.runtime, args.plugin))) != 1:
        parser.error("Choose --catalog, --install/--policy/--peft/--runtime, or --plugin.")
    try:
        if args.catalog:
            print("MANAGER_CATALOG " + json.dumps(environment_catalog()))
        else:
            return install_dependencies(benchmark=args.install, policy=args.policy, plugin=args.plugin,
                                        peft=args.peft, runtime=args.runtime, isolated=args.isolated)
    except (ImportError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Dependency setup failed ({error}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
