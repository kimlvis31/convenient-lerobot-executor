"""Run the native LeRobot evaluator with RoboMME physics in a separate Python."""

import io
import json
import os
from functools import partial
from pathlib import Path
import random
import struct
import subprocess
import sys
import threading
import types
from contextlib import contextmanager


CAMERA_AND_STATE = ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list")
SIMULATOR_ENV = "CONVLEX_ROBOMME_WORKER"
MAX_MESSAGE = 32 * 1024 * 1024
BUILDER_LOCK = threading.RLock()


def send(stream, message, **arrays):
    import numpy as np

    payload = io.BytesIO()
    np.savez(payload, metadata=np.asarray(json.dumps(message)), **arrays)
    size = payload.tell()
    if size > MAX_MESSAGE:
        raise ValueError("RoboMME worker message exceeds the size limit.")
    stream.write(struct.pack("!I", size))
    stream.write(payload.getbuffer())
    stream.flush()


def receive(stream):
    import numpy as np

    header = stream.read(4)
    if not header:
        raise EOFError("The RoboMME worker connection closed.")
    if len(header) != 4:
        raise EOFError("Incomplete RoboMME worker message header.")
    size, = struct.unpack("!I", header)
    if size > MAX_MESSAGE:
        raise ValueError("Invalid RoboMME worker message length.")
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("Incomplete RoboMME worker message.")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"])), {key: archive[key] for key in archive.files if key != "metadata"}


def serve(reader, writer):
    import numpy as np
    import torch
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    config, _ = receive(reader)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    builder = BenchmarkEnvBuilder(**config["builder"])
    env = builder.make_env_for_episode(**config["episode"])
    try:
        while True:
            try:
                operation, arrays = receive(reader)
            except EOFError:
                break
            if operation == "reset":
                obs, info = env.reset()
                outcome = []
            elif operation == "step":
                obs, reward, terminated, truncated, info = env.step(arrays["action"])
                outcome = [float(reward), bool(terminated), bool(truncated)]
            else:
                raise ValueError(f"Unknown RoboMME operation: {operation}")
            # The official Gym adapter consumes the latest image/joint/gripper sample.
            samples = {key: np.asarray(obs[key][-1] if isinstance(obs[key], list) else obs[key]) for key in CAMERA_AND_STATE}
            send(writer, {"outcome": outcome, "info": {"task_goal": info.get("task_goal", ""),
                                                        "status": info.get("status", "ongoing")}}, **samples)
    finally:
        env.close()


class SimulatorEpisode:
    def __init__(self, config, builder, episode, seed):
        self.process = subprocess.Popen([config["python"], "-B", str(Path(__file__).resolve()), "--worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        env={**os.environ, **config["env"]})
        try:
            send(self.process.stdin, {"builder": builder, "episode": episode, "seed": seed})
        except BaseException:
            self.close()
            raise

    def call(self, operation, **arrays):
        try:
            send(self.process.stdin, operation, **arrays)
            response, observations = receive(self.process.stdout)
            if "error" in response:
                raise RuntimeError(f"RoboMME simulator failed: {response['error']}")
            return (observations, *response["outcome"], response["info"])
        except (EOFError, BrokenPipeError) as error:
            raise RuntimeError(f"RoboMME simulator exited unexpectedly (code {self.process.poll()}).") from error

    def reset(self):
        return self.call("reset")

    def step(self, action):
        return self.call("step", action=action)

    def close(self):
        process = self.process
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            process.stdout.close()


class SimulatorBuilder:
    def __init__(self, config, seed, **builder):
        self.config = config
        self.builder = builder
        self.seeds = random.Random(seed)
        self.seed = seed

    def make_env_for_episode(self, **episode):
        return SimulatorEpisode(self.config, self.builder, episode, self.seed)


def make_split_env(factory, config, seed):
    # The official Gym wrapper imports the builder lazily inside its constructor.
    # Install this substitute only for construction, also in spawned Gym workers.
    name = "robomme.env_record_wrapper"
    bridge = types.ModuleType(name)
    bridge.BenchmarkEnvBuilder = partial(SimulatorBuilder, config, seed)
    with BUILDER_LOCK:
        previous = sys.modules.get(name)
        sys.modules[name] = bridge
        try:
            env = factory()
        finally:
            if previous is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous
    builder = env._builder
    original_reset = env.reset

    def reset(*, seed=None, options=None):
        if seed is not None:
            builder.seeds.seed(seed)
        builder.seed = seed if seed is not None else builder.seeds.randrange(2**32)
        return original_reset(seed=seed, options=options)

    env.reset = reset
    return env


@contextmanager
def simulator_context():
    if SIMULATOR_ENV not in os.environ or "--env.type=robomme" not in sys.argv:
        yield
        return
    from lerobot.envs import robomme

    config = json.loads(os.environ[SIMULATOR_ENV])
    seed = next((int(arg.partition("=")[2]) for arg in sys.argv[1:] if arg.startswith("--seed=")), 0)
    factory = getattr(robomme, "_make_env_fns", None)
    if factory is None:
        raise RuntimeError("This LeRobot version does not expose the supported RoboMME Gym environment factory.")

    def factories(**kwargs):
        return [partial(make_split_env, fn, config, seed) for fn in factory(**kwargs)]

    robomme._make_env_fns = factories
    try:
        yield
    finally:
        robomme._make_env_fns = factory


def inspect_simulator():
    from importlib.util import find_spec
    if find_spec("robomme") is None:
        return 1
    from benchmarks import simulator_runtime_issue
    if issue := simulator_runtime_issue("robomme"):
        print(issue, file=sys.stderr)
        return 1
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    keys = ("CONDA_PREFIX", "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "MS_ASSET_DIR", "VK_ICD_FILENAMES",
            "VK_DRIVER_FILES", "PYTHONNOUSERSITE")
    env = {key: os.environ[key] for key in keys if key in os.environ}
    # An absent PYTHONPATH in the simulator must not inherit the policy's imports.
    env.setdefault("PYTHONPATH", "")
    print("ROBOMME_SIMULATOR " + json.dumps({"python": sys.executable, "env": env, "packages": simulator_versions()}))
    return 0


def simulator_versions():
    from importlib import metadata
    versions = {}
    for name in ("robomme", "mani-skill", "numpy", "gymnasium", "sapien", "torch", "torchvision"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            if name in ("robomme", "mani-skill"):
                versions[name] = "source installation"
    return versions


if __name__ == "__main__":
    if sys.argv[1:] == ["--versions"]:
        print(json.dumps(simulator_versions()))
    elif sys.argv[1:] == ["--inspect"]:
        sys.exit(inspect_simulator())
    elif sys.argv[1:] == ["--worker"]:
        # Native simulator extensions may write directly to fd 1. Keep their logs
        # on stderr and reserve the original stdout pipe for framed messages.
        with os.fdopen(os.dup(1), "wb") as writer:
            os.dup2(2, 1)
            try:
                serve(sys.stdin.buffer, writer)
            except Exception as error:
                try:
                    send(writer, {"error": f"{type(error).__name__}: {error}"})
                except BrokenPipeError:
                    pass
                raise
    else:
        sys.exit("Expected --inspect or --worker.")
