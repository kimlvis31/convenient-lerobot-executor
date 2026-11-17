"""Read the installed entry points and package versions in a selected Conda environment."""

from importlib import metadata, util
import json
from pathlib import Path
import shutil
import sys


PACKAGE_NAMES = {"lerobot", "lerobot-policy-tinyvla", "torch", "transformers", "diffusers", "safetensors"}


def describe_environment():
    packages, entrypoints = {}, {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "").lower().replace("_", "-")
        if name in PACKAGE_NAMES:
            packages[name] = dist.version
        if name == "lerobot":
            for entry in dist.entry_points:
                if entry.group != "console_scripts" or entry.name not in {"lerobot-train", "lerobot-eval"}:
                    continue
                spec = util.find_spec(entry.module)
                executable = shutil.which(entry.name)
                if executable and not Path(executable).resolve().is_relative_to(Path(sys.prefix).resolve()):
                    executable = None
                entrypoints[entry.name] = {"launcher": executable, "source": spec.origin if spec else None}
    bridge = util.find_spec("lerobot_policy_tinyvla")
    return {"python": sys.executable, "version": sys.version.split()[0], "packages": packages,
            "entrypoints": entrypoints, "tinyvla_bridge": bridge.origin if bridge else None}


if __name__ == "__main__":
    print("MANAGER_ENV " + json.dumps(describe_environment()))
