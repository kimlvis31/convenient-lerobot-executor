"""AI-written Conda hooks that inherit source variables only in child memory.

The generated hooks contain paths and code, never variable values. This helper
reads source conda-meta/state only when the managed environment is activated.
Its safely quoted shell output is consumed directly by the activating shell.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys


UNSET_SENTINEL = "***unset***"  # conda.base.constants.CONDA_ENV_VARS_UNSET_VAR
VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
SCOPE = re.compile(r"[0-9a-f]{16}\Z")
MODEL_CACHE_VARIABLES = {
    "HF_HOME", "HF_HUB_CACHE", "HF_TOKEN_PATH", "HF_ASSETS_CACHE", "HF_XET_CACHE",
    "TORCH_HOME", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_LEROBOT_HOME",
}
PROTECTED = MODEL_CACHE_VARIABLES | {
    "PATH", "PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
    "MPLCONFIGDIR", "CUDA_HOME", "CUDA_PATH", "CUDACXX", "CC", "CXX", "NVCC",
    "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VULKAN_SDK", "ROBOTWIN_ROOT", "VLABENCH_ROOT",
    "MUJOCO_GL", "PYOPENGL_PLATFORM", "MS_ASSET_DIR",
}


class ActivationError(RuntimeError):
    """A safe diagnostic without configuration values or parsing fragments."""


def protected(name):
    return name in PROTECTED or name.startswith(("CONDA_", "_CE_", "XDG_", "LD_", "CONVLEX_", "__CONVLEX_"))


def marker(scope):
    if not isinstance(scope, str) or not SCOPE.fullmatch(scope):
        raise ActivationError("Invalid managed environment activation scope.")
    return "__CONVLEX_SOURCE_VARS_" + scope.upper()


def source_variables(base_prefix):
    """Read only this source prefix's Conda env_vars, without exposing values."""
    try:
        base = Path(base_prefix).expanduser()
        if not base.is_absolute() or not base.is_dir():
            raise ValueError("Source prefix must be absolute.")
        path = base / "conda-meta/state"
        if not path.is_file():
            return {}
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("Source state is too large.")
        state = json.loads(path.read_text(encoding="utf-8"))
        variables = state.get("env_vars", {}) if isinstance(state, dict) else None
        if not isinstance(variables, dict):
            raise ValueError("Invalid source state.")
        result = {}
        for name, value in variables.items():
            if not isinstance(name, str) or not VARIABLE.fullmatch(name):
                raise ValueError("Invalid source variable name.")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("Invalid source variable value.")
            if value != UNSET_SENTINEL and (not protected(name) or name == "LD_LIBRARY_PATH"):
                result[name] = value
        return result
    except (OSError, UnicodeError, ValueError, TypeError):
        raise ActivationError("Cannot read the source Conda environment variable configuration.") from None


def activation_commands(base_prefix, scope, environment=None, *, prefix=None):
    environment = os.environ if environment is None else environment
    key = marker(scope)
    if key in environment:
        return ""  # Preserve the original backup across duplicate activation.
    variables = source_variables(base_prefix)
    source_libraries = variables.pop("LD_LIBRARY_PATH", None)
    if source_libraries is not None and prefix is not None:
        target = Path(prefix).expanduser()
        if not target.is_absolute():
            raise ActivationError("The managed library prefix must be absolute.")
        # Keep the target's native ABI ahead of source-only extensions. All
        # values and the prior exact value remain in process memory only.
        paths = [str(target / "lib"), *environment.get("LD_LIBRARY_PATH", "").split(os.pathsep),
                 *source_libraries.split(os.pathsep)]
        variables["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    if not variables:
        return ""
    backup = {name: environment.get(name) for name in variables}
    encoded = json.dumps({"version": 1, "values": backup}, separators=(",", ":"))
    lines = [f"export {key}={shlex.quote(encoded)}"]
    lines.extend(f"export {name}={shlex.quote(value)}" for name, value in sorted(variables.items()))
    return "\n".join(lines) + "\n"


def deactivation_commands(scope, environment=None):
    environment = os.environ if environment is None else environment
    key = marker(scope)
    if key not in environment:
        return ""
    try:
        saved = json.loads(environment[key])
        if not isinstance(saved, dict) or set(saved) != {"version", "values"} or saved["version"] != 1:
            raise ValueError("Unknown activation backup.")
        values = saved["values"]
        if not isinstance(values, dict):
            raise ValueError("Invalid activation backup.")
        lines = []
        for name, value in sorted(values.items()):
            if (not isinstance(name, str) or not VARIABLE.fullmatch(name)
                    or (protected(name) and name != "LD_LIBRARY_PATH")):
                raise ValueError("Invalid backup variable name.")
            if value is not None and (not isinstance(value, str) or "\x00" in value):
                raise ValueError("Invalid backup value.")
            lines.append(f"unset {name}" if value is None else f"export {name}={shlex.quote(value)}")
        lines.append(f"unset {key}")
        return "\n".join(lines) + "\n"
    except (TypeError, ValueError):
        raise ActivationError("Cannot restore inherited Conda environment variables.") from None


def _atomic_text(path, content, root):
    if not path.resolve().is_relative_to(root) or not path.parent.resolve().is_relative_to(root):
        raise ValueError("Activation hooks must stay inside the managed build.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.is_symlink():
        raise ValueError("Refusing a linked activation staging file.")
    pending.write_text(content, encoding="utf-8")
    pending.chmod(0o700)
    pending.replace(path)


def install_activation(root, prefix, base_prefix):
    """Install value-free hooks and a standalone copy in the owned build."""
    root, prefix, base = Path(root).resolve(), Path(prefix).resolve(), Path(base_prefix).expanduser()
    if (not prefix.is_relative_to(root) or prefix == root or not base.is_absolute()
            or not base.is_dir() or base.resolve() == prefix):
        raise ValueError("Managed activation needs a distinct absolute source prefix and an owned target prefix.")
    base = base.resolve()
    scope = hashlib.sha256(str(prefix).encode()).hexdigest()[:16]
    helper = root / "runtime_activation.py"
    _atomic_text(helper, Path(__file__).read_text(encoding="utf-8"), root)
    variable = "__convlex_activation_" + scope
    status = "__convlex_activation_status_" + scope
    for operation, directory in (("activate", "activate.d"), ("deactivate", "deactivate.d")):
        command = [str(prefix / "bin/python"), str(helper), operation, "--scope", scope]
        if operation == "activate":
            command += ["--base-prefix", str(base), "--target-prefix", str(prefix)]
        # The substitution and eval are both quoted. Values exist only in the
        # activating shell and its children; no secret is written to this file.
        text = ("# Managed source environment variables; generated by convLexecutor.\n"
                f"{variable}=\"$({shlex.join(command)})\"\n"
                f"{status}=$?\n"
                f"if [ \"${{{status}}}\" -ne 0 ]; then\n"
                f"    unset {variable} {status}\n"
                "    return 1\n"
                "fi\n"
                f"eval \"${{{variable}}}\"\n"
                f"unset {variable} {status}\n")
        _atomic_text(prefix / "etc/conda" / directory / "convlex_source_vars.sh", text, root)
    return {"base_prefix": str(base), "helper": str(helper), "scope": scope}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("activate", "deactivate"))
    parser.add_argument("--scope", required=True)
    parser.add_argument("--base-prefix")
    parser.add_argument("--target-prefix")
    args = parser.parse_args(argv)
    try:
        if args.operation == "activate":
            if not args.base_prefix:
                raise ActivationError("A source Conda prefix is required for activation.")
            commands = activation_commands(args.base_prefix, args.scope, prefix=args.target_prefix)
        else:
            commands = deactivation_commands(args.scope)
    except ActivationError as error:
        print(f"Conda activation failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(commands)
    return 0


if __name__ == "__main__":
    sys.exit(main())
