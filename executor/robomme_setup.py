"""Provision the simulator side of a split RoboMME evaluation on Linux."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import mkdtemp
from xml.etree import ElementTree


# The LeRobot benchmark image pins RoboMME. Its pyproject pins this ManiSkill fork.
ROBOMME = "53673b3c94a723d757e9c235f1961e398950e5e3"
MANISKILL = "07be6fbc66350ddca200abfb0a11b692f078f7fd"
RECIPE = "robomme-" + ROBOMME[:8] + "-" + MANISKILL[:8] + "-2"
REQUIREMENTS = (
    f"robomme @ https://codeload.github.com/RoboMME/robomme_benchmark/tar.gz/{ROBOMME}",
    f"mani-skill @ https://codeload.github.com/YinpeiDai/ManiSkill/tar.gz/{MANISKILL}",
    "numpy==1.26.4", "gymnasium==0.29.1", "sapien==3.0.2", "scipy==1.15.3", "matplotlib",
)


def torch_index():
    """Select an official CUDA wheel supported by the active NVIDIA driver."""
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True)
    available = ElementTree.fromstring(result.stdout).findtext("cuda_version")
    try:
        version = tuple(int(value) for value in available.split(".")[:2])
    except (AttributeError, ValueError) as error:
        raise RuntimeError("Cannot determine the CUDA runtime supported by the NVIDIA driver.") from error
    # PyTorch's official 2.9.1 wheels include CUDA 12.6, 12.8 and 13.0.
    # Use the 12.x line for the simulator when the current driver supports it.
    if version >= (12, 8):
        return "https://download.pytorch.org/whl/cu128"
    if version >= (12, 6):
        return "https://download.pytorch.org/whl/cu126"
    raise RuntimeError("The pinned RoboMME runtime needs a CUDA 12.6-compatible NVIDIA driver.")


def check_simulator(physics):
    import numpy as np
    import robomme
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    if np.__version__ != "1.26.4":
        raise RuntimeError(f"The managed RoboMME simulator requires NumPy 1.26.4, found {np.__version__}.")
    metadata = Path(robomme.__file__).parent / "env_metadata/test/record_dataset_PickXtimes_metadata.json"
    if not metadata.is_file():
        raise RuntimeError(f"Missing RoboMME episode metadata: {metadata}")
    if not physics:
        return
    builder = BenchmarkEnvBuilder(env_id="PickXtimes", dataset="test", action_space="joint_angle",
                                  gui_render=False, max_steps=1)
    env = builder.make_env_for_episode(episode_idx=0, max_steps=1)
    try:
        env.reset()
        obs, *_ = env.step(np.zeros(8, dtype=np.float32))
        for name in ("front_rgb_list", "wrist_rgb_list"):
            image = obs[name][-1] if isinstance(obs[name], list) else obs[name]
            if np.asarray(image).shape != (256, 256, 3):
                raise RuntimeError(f"RoboMME returned an unexpected camera shape for {name}.")
    finally:
        env.close()


def simulator_environment(prefix, folder):
    return {
        "CONDA_PREFIX": str(prefix),
        "PATH": str(prefix / "bin") + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "MS_ASSET_DIR": os.environ.get("MS_ASSET_DIR") or str(folder / "assets"),
    }


def prepare_simulator(conda):
    """Return (Python executable, environment overrides) for a verified simulator."""
    if sys.platform != "linux":
        raise RuntimeError("RoboMME simulator setup requires Linux and a working Vulkan renderer.")
    import fcntl

    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "convlexecutor" / "simulators"
    folder = Path(os.environ.get("CONVLEX_ROBOMME_CACHE") or cache).expanduser().resolve() / RECIPE
    folder.mkdir(parents=True, exist_ok=True)
    ready = folder / "ready"
    script = str(Path(__file__).resolve())
    with (folder / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ready.is_file():
            name = ready.read_text(encoding="utf-8").strip()
            if Path(name).name == name and name.startswith("conda-"):
                prefix = folder / name
                python = prefix / "bin/python"
                worker_env = simulator_environment(prefix, folder)
                if python.is_file() and subprocess.run(
                    [str(python), "-B", script, "--check"], env={**os.environ, **worker_env},
                ).returncode == 0:
                    print(f"  Reusing verified RoboMME simulator environment: {prefix}", flush=True)
                    return str(python), worker_env

        # Conda embeds the absolute prefix in installed binaries. Publish a small
        # pointer only after verification, leaving the prefix itself in place.
        prefix = Path(mkdtemp(prefix="conda-", dir=folder))
        python = str(prefix / "bin/python")
        worker_env = simulator_environment(prefix, folder)
        env = {**os.environ, **worker_env}
        try:
            print(f"  Creating a separate RoboMME simulator environment: {prefix}", flush=True)
            index = torch_index()
            subprocess.run([str(conda), "create", "--yes", "--prefix", str(prefix), "python=3.12", "pip"],
                           check=True)
            subprocess.run([python, "-m", "pip", "install", "--no-input", "--no-cache-dir",
                            "--disable-pip-version-check", "--index-url", index,
                            "torch==2.9.1", "torchvision==0.24.1"], env=env, check=True)
            subprocess.run([python, "-m", "pip", "install", "--no-input", "--no-cache-dir",
                            "--disable-pip-version-check", *REQUIREMENTS], env=env, check=True)
            subprocess.run([python, "-m", "pip", "check"], env=env, check=True)
            print("  Checking RoboMME cameras and a simulation step...", flush=True)
            subprocess.run([python, "-B", script, "--probe"], env=env, check=True)
            pending = ready.with_suffix(".tmp")
            pending.write_text(prefix.name + "\n", encoding="utf-8")
            pending.replace(ready)
        except BaseException:
            shutil.rmtree(prefix)
            raise
        return python, worker_env


if __name__ == "__main__":
    if sys.argv[1:] not in (["--check"], ["--probe"]):
        raise SystemExit("Expected --check or --probe.")
    check_simulator(sys.argv[1] == "--probe")
