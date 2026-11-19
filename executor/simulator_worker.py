"""AI-written bridge for source simulators that reuses native Gym evaluation.

The transport contains JSON and numeric NumPy arrays only. RoboTwin's official
Gym wrapper is read from the selected policy installation, never modified.
"""

import argparse
import ast
import atexit
from contextlib import contextmanager
from functools import partial
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import traceback
import types


SIMULATOR_ENV = "CONVLEX_SIMULATOR_WORKER"
ROOT_ENV = "CONVLEX_SIMULATOR_ROOT"
MAX_MESSAGE = 32 * 1024 * 1024
# LeRobot 0.6.2, revision 145cc7ea53e5864bd5efec4e9bb70ee799e47bd8.
# Add another source only after checking its wrapper API and physical probe.
ROBOTWIN_SOURCE = "8789d32d745eedf9aa2ea041bcb44ac1a2995bf1eccbdabc62c9c30d05104bb9"
SOURCE_REVISION = "145cc7ea53e5864bd5efec4e9bb70ee799e47bd8"
SOURCE_URL = "https://github.com/huggingface/lerobot/blob/" + SOURCE_REVISION + "/src/lerobot/envs/robotwin.py"


def encode(value, arrays):
    """Represent nested values unambiguously, rejecting object arrays and pickle."""
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in "biufcSU":
            raise TypeError(f"Unsupported simulator array dtype: {value.dtype}")
        key = "array_" + str(len(arrays))
        arrays[key] = value
        return ["array", key]
    if isinstance(value, np.generic):
        return encode(value.item(), arrays)
    if value is None or type(value) in (bool, int, float, str):
        return ["value", value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Simulator dictionaries require string keys.")
        return ["dict", [[key, encode(item, arrays)] for key, item in value.items()]]
    if isinstance(value, (tuple, list)):
        return ["tuple" if isinstance(value, tuple) else "list", [encode(item, arrays) for item in value]]
    raise TypeError(f"Unsupported simulator message value: {type(value).__name__}")


def decode(value, arrays):
    kind, item = value
    if kind == "value":
        if item is not None and type(item) not in (bool, int, float, str):
            raise ValueError("Invalid simulator primitive.")
        return item
    if kind == "array":
        return arrays[item]
    if kind == "dict":
        return {key: decode(child, arrays) for key, child in item}
    if kind in {"tuple", "list"}:
        children = [decode(child, arrays) for child in item]
        return tuple(children) if kind == "tuple" else children
    raise ValueError(f"Invalid simulator message type: {kind}")


def send(stream, value):
    import numpy as np

    arrays = {}
    metadata = encode(value, arrays)
    payload = io.BytesIO()
    np.savez(payload, metadata=np.asarray(json.dumps(metadata)), **arrays)
    if payload.tell() > MAX_MESSAGE:
        raise ValueError("Simulator worker message exceeds the size limit.")
    stream.write(struct.pack("!I", payload.tell()))
    stream.write(payload.getbuffer())
    stream.flush()


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("The simulator worker connection closed.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive(stream):
    import numpy as np

    size, = struct.unpack("!I", read_exact(stream, 4))
    if size <= 0 or size > MAX_MESSAGE:
        raise ValueError("Invalid simulator worker message length.")
    payload = read_exact(stream, size)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "metadata"}
        return decode(json.loads(str(archive["metadata"])), arrays)


def space_spec(space):
    """Use a small data-only schema instead of pickling Gym space objects."""
    from gymnasium import spaces

    if isinstance(space, spaces.Box):
        return {"type": "Box", "low": space.low, "high": space.high, "dtype": str(space.dtype)}
    if isinstance(space, spaces.Dict):
        return {"type": "Dict", "spaces": [(key, space_spec(value)) for key, value in space.spaces.items()]}
    if isinstance(space, spaces.Tuple):
        return {"type": "Tuple", "spaces": [space_spec(value) for value in space.spaces]}
    if isinstance(space, spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n), "start": int(space.start)}
    if isinstance(space, spaces.MultiDiscrete):
        return {"type": "MultiDiscrete", "nvec": space.nvec, "start": space.start, "dtype": str(space.dtype)}
    if isinstance(space, spaces.MultiBinary):
        return {"type": "MultiBinary", "n": space.n}
    if isinstance(space, spaces.Text):
        return {"type": "Text", "min_length": space.min_length, "max_length": space.max_length,
                "charset": "".join(sorted(space.character_set))}
    raise TypeError(f"Unsupported simulator Gym space: {type(space).__name__}")


def restore_space(spec):
    from gymnasium import spaces

    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "Dict":
        return spaces.Dict([(key, restore_space(value)) for key, value in spec["spaces"]])
    if kind == "Tuple":
        return spaces.Tuple([restore_space(value) for value in spec["spaces"]])
    if kind not in {"Box", "Discrete", "MultiDiscrete", "MultiBinary", "Text"}:
        raise ValueError(f"Unsupported simulator Gym space: {kind}")
    return getattr(spaces, kind)(**spec)


def validate_source(config):
    if config.get("kind") != "robotwin":
        raise RuntimeError(f"No split simulator wrapper is registered for {config.get('kind')!r}.")
    path = Path(config.get("source", "")).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError("Select the absolute official LeRobot RoboTwin wrapper source path.")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != config.get("source_sha256"):
        raise RuntimeError("The official RoboTwin wrapper changed after environment selection; refresh the job.")
    if digest != ROBOTWIN_SOURCE:
        raise RuntimeError("This LeRobot RoboTwin wrapper version has not been verified for split simulation. "
                           f"Use revision {SOURCE_REVISION}; source: {SOURCE_URL}")
    return path, content


def load_robotwin(config):
    """Read the pinned native wrapper, omitting only three unrelated imports.

    RobotObservation is an annotation alias. _scipy_available is evaluated in
    this interpreter. _LazyAsyncVectorEnv is used only by the policy-side vector
    factory; the worker creates a single native RoboTwinEnv. All wrapper/helper
    bodies and their license remain the original source, checked by SHA-256.
    """
    path, content = validate_source(config)
    tree = ast.parse(content, filename=str(path))
    omitted = {
        ("lerobot.lerobot_types", 0, "RobotObservation"),
        ("lerobot.utils.import_utils", 0, "_scipy_available"),
        ("utils", 1, "_LazyAsyncVectorEnv"),
    }
    found = set()
    body = []
    for node in tree.body:
        key = (node.module, node.level, node.names[0].name) if isinstance(node, ast.ImportFrom) and len(node.names) == 1 else None
        if key in omitted:
            if node.names[0].asname is not None or key in found:
                raise RuntimeError("The official RoboTwin wrapper import contract changed.")
            found.add(key)
        else:
            body.append(node)
    if found != omitted:
        raise RuntimeError("The official RoboTwin wrapper import contract changed.")
    tree.body = body
    module = types.ModuleType("_convlex_native_robotwin")
    module.__file__ = str(path)
    module.__dict__.update(RobotObservation=dict, _scipy_available=importlib.util.find_spec("scipy") is not None)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    env = getattr(module, "RoboTwinEnv", None)
    required = ("reset", "step", "render", "close")
    if env is None or not all(callable(getattr(env, name, None)) for name in required):
        raise RuntimeError("The official RoboTwin Gym wrapper API changed.")
    return module


@contextmanager
def source_directory(config):
    kind = config.get("kind")
    configured = os.environ.get(ROOT_ENV) or (os.environ.get("ROBOTWIN_ROOT") if kind == "robotwin" else None)
    if not configured:
        raise RuntimeError(f"The {kind} simulator source root is not configured.")
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"The simulator source directory does not exist: {root}")
    previous = Path.cwd()
    sys.path.insert(0, str(root))
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(previous)
        sys.path.remove(str(root))


def environment_details(env):
    return {"observation_space": space_spec(env.observation_space), "action_space": space_spec(env.action_space),
            "metadata": env.metadata, "render_mode": env.render_mode,
            "task": env.task, "task_description": env.task_description,
            "_max_episode_steps": env._max_episode_steps}


def serve(reader, writer):
    config = receive(reader)
    with source_directory(config):
        module = load_robotwin(config)
        env = module.RoboTwinEnv(**config["kwargs"])
        try:
            send(writer, {"result": environment_details(env)})
            while True:
                try:
                    request = receive(reader)
                except EOFError:
                    break
                operation = request.get("operation")
                if operation == "close":
                    break
                if operation == "reset":
                    result = env.reset(**request.get("kwargs", {}))
                elif operation == "step":
                    result = env.step(request["action"])
                elif operation == "render":
                    result = env.render()
                else:
                    raise ValueError(f"Unknown simulator operation: {operation}")
                send(writer, {"result": result, "task_description": env.task_description})
        finally:
            env.close()


def close_process(process):
    """Reap the worker and only the process group created for this worker."""
    try:
        if process.stdin:
            process.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
    finally:
        if process.stdout:
            process.stdout.close()


class SimulatorProxy:
    """One lazily initialized simulator, also usable inside spawned Gym workers."""

    def __init__(self, config, kwargs):
        self._closed = False
        self._lock = threading.RLock()
        self.process = subprocess.Popen([config["python"], "-B", str(Path(__file__).resolve()), "--worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        env={**os.environ, **config.get("env", {})},
                                        start_new_session=os.name != "nt")
        self._finalize = partial(close_process, self.process)
        atexit.register(self._finalize)
        try:
            send(self.process.stdin, {**config, "kwargs": kwargs})
            details = self._response()
            self.observation_space = restore_space(details.pop("observation_space"))
            self.action_space = restore_space(details.pop("action_space"))
            self.__dict__.update(details)
        except BaseException:
            self.close()
            raise

    def _response(self):
        try:
            response = receive(self.process.stdout)
        except (EOFError, BrokenPipeError) as error:
            raise RuntimeError(f"The simulator worker exited unexpectedly (code {self.process.poll()}). "
                               "See the stage log for the simulator traceback.") from error
        if "error" in response:
            raise RuntimeError("Simulator failed: " + response["error"])
        if "task_description" in response:
            self.task_description = response["task_description"]
        return response["result"]

    def call(self, operation, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError("The simulator worker is closed.")
            try:
                send(self.process.stdin, {"operation": operation, **kwargs})
                return self._response()
            except BaseException:
                self.close()
                raise

    def reset(self, *, seed=None, options=None):
        return self.call("reset", kwargs={"seed": seed, "options": options})

    def step(self, action):
        return self.call("step", action=action)

    def render(self):
        return self.call("render")

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                atexit.unregister(self._finalize)
                close_process(self.process)


def make_proxy(config, kwargs):
    import gymnasium as gym

    # No simulator or Gym dependency is imported while the manager scans jobs.
    # This top-level factory is serializable by multiprocessing's spawn context.
    proxy = type("SourceSimulatorEnv", (SimulatorProxy, gym.Env), {})
    return proxy(config, kwargs)


def split_factories(config, **kwargs):
    names = ("task_name", "n_envs", "camera_names", "observation_height", "observation_width", "episode_length")
    if any(name not in kwargs for name in names) or set(kwargs) - {*names, "action_mode"}:
        raise RuntimeError("The native RoboTwin vector factory API changed; update the simulator integration.")
    return [partial(make_proxy, config, {**kwargs, "episode_index": index}) for index in range(kwargs["n_envs"])]


@contextmanager
def simulator_context():
    raw = os.environ.get(SIMULATOR_ENV)
    if not raw:
        yield
        return
    config = json.loads(raw)
    if config.get("kind") != "robotwin" or "--env.type=robotwin" not in sys.argv:
        raise RuntimeError("The configured split simulator does not match this evaluation benchmark.")
    validate_source(config)
    from lerobot.envs import robotwin

    actual = Path(robotwin.__file__).resolve()
    if actual != Path(config["source"]).resolve() or not callable(getattr(robotwin, "_make_env_fns", None)):
        raise RuntimeError("The selected policy environment's RoboTwin wrapper changed; refresh the job.")
    previous = robotwin._make_env_fns
    robotwin._make_env_fns = partial(split_factories, config)
    try:
        yield
    finally:
        robotwin._make_env_fns = previous


def parent_death_signal():
    """Linux workers terminate even if their owning Gym process is killed."""
    if sys.platform == "linux":
        import ctypes

        parent = os.getppid()
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "Cannot bind simulator worker lifetime to its parent.")
        if os.getppid() != parent:
            raise SystemExit(130)


def inspect_or_probe(config, physics=False):
    import numpy as np

    with source_directory(config):
        module = load_robotwin(config)
        env = module.RoboTwinEnv(task_name="beat_block_hammer", episode_length=2)
        try:
            environment_details(env)
            result = {"kind": config["kind"], "source_sha256": config["source_sha256"],
                      "source_url": SOURCE_URL, "numpy": np.__version__, "action_shape": list(env.action_space.shape)}
            if physics:
                obs, _ = env.reset(seed=0)
                obs, reward, terminated, truncated, info = env.step(np.asarray(obs["agent_pos"], dtype=np.float32))
                for name in module.ROBOTWIN_CAMERA_NAMES:
                    image = np.asarray(obs["pixels"][name])
                    if image.shape != (240, 320, 3) or image.dtype != np.uint8 or not image.any():
                        raise RuntimeError(f"RoboTwin camera {name} returned invalid image data: {image.shape}.")
                state = np.asarray(obs["agent_pos"])
                if state.shape != (14,) or not np.isfinite(state).all():
                    raise RuntimeError("RoboTwin state must contain fourteen finite joint positions.")
                result.update(physics_steps=1, cameras=list(obs["pixels"]), state_shape=list(obs["agent_pos"].shape))
            print(("CONVLEX_SIMULATOR_PROBE " if physics else "CONVLEX_SIMULATOR_INSPECT ") + json.dumps(result))
            return result
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worker", action="store_true")
    modes.add_argument("--inspect", action="store_true")
    modes.add_argument("--probe", action="store_true")
    parser.add_argument("--kind", default="robotwin")
    parser.add_argument("--source")
    parser.add_argument("--source-sha256")
    options = parser.parse_args()
    if options.worker:
        parent_death_signal()
        # Native simulator code can write directly to stdout. Reserve the old fd
        # for framed messages and keep every native/Python log on stderr.
        with os.fdopen(os.dup(1), "wb") as writer:
            os.dup2(2, 1)
            try:
                serve(sys.stdin.buffer, writer)
            except BaseException as error:
                if not isinstance(error, (KeyboardInterrupt, SystemExit)):
                    try:
                        send(writer, {"error": f"{type(error).__name__}: {error}"})
                    except (BrokenPipeError, OSError):
                        pass
                traceback.print_exc(file=sys.stderr)
                return 130 if isinstance(error, KeyboardInterrupt) else 1
    else:
        config = {"kind": options.kind, "source": options.source, "source_sha256": options.source_sha256}
        inspect_or_probe(config, options.probe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
