import atexit
from collections import deque
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_tinyvla import DEFAULT_WORKSPACE, TinyVLAConfig
from .worker_builder import build_worker


class TinyVLAPolicy(PreTrainedPolicy):
    config_class = TinyVLAConfig
    name = "tinyvla"

    def __init__(self, config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self._action_queue = deque()
        self._lock = threading.Lock()
        self._process = None
        self._log = None
        self._channel = None
        self._worker_dir = None
        self._request_id = 0
        self._last_task = None
        self._failed = False
        atexit.register(self.close)

    def reset(self):
        self._action_queue.clear()
        self._last_task = None

    def get_optim_params(self):
        raise NotImplementedError("This adapter supports inference only.")

    def forward(self, batch, **kwargs):
        raise NotImplementedError("This adapter supports inference only.")

    def _ensure_worker(self):
        if self._failed:
            raise RuntimeError("TinyVLA worker failed earlier; restart this evaluation.")
        if self._process is not None:
            if self._process.poll() is not None:
                raise RuntimeError(f"TinyVLA process stopped. See {self._channel / 'worker.log'}")
            return
        workspace = Path(self.config.workspace).expanduser().resolve()
        if self.config.workspace == DEFAULT_WORKSPACE and not workspace.is_dir():
            for name in ("tinyVLA", "tinyvla"):
                legacy = workspace.with_name(name)
                if legacy.is_dir():
                    workspace = legacy
                    break
        step7 = (Path(self.config.step7_script).expanduser().resolve() if self.config.step7_script
                 else workspace / "runtime/tinyvla_loader.py")
        if self.config.inference_python:
            inference_python = Path(self.config.inference_python).expanduser()
        else:
            envs = Path(sys.prefix).parent
            candidates = [envs / name / "bin/python" for name in ("tinyvla_infer", "lerobot_tinyvla")]
            inference_python = next((path for path in candidates if path.is_file()), candidates[0])
        if not inference_python.is_file():
            raise FileNotFoundError(f"TinyVLA interpreter not found: {inference_python}")
        generated, source_hash = build_worker(step7.read_text(encoding="utf-8"))
        logs = (Path(self.config.worker_log_dir).expanduser().resolve()
                if self.config.worker_log_dir else workspace / "logs")
        logs.mkdir(parents=True, exist_ok=True)
        self._channel = Path(tempfile.mkdtemp(prefix="native_tinyvla_", dir=logs))
        self._worker_dir = tempfile.TemporaryDirectory(prefix="tinyvla_worker_")
        worker_script = Path(self._worker_dir.name) / "worker.py"
        worker_script.write_text(generated, encoding="utf-8")
        (self._channel / "adapter.json").write_text(json.dumps({
            "adapter_version": "0.2.0", "step7": str(step7), "step7_sha256": source_hash,
            "worker_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            "workspace": str(workspace), "inference_python": str(inference_python),
            "seed": self.config.seed, "n_action_steps": self.config.n_action_steps,
            "image_pipeline": "LeRobot LIBERO rotation once; CHW float to HWC uint8; TinyVLA processor once",
            "state_scaling": "identity", "action_scaling": "identity then clamp [-1,1]",
            "gripper_sign_change": False, "supports_batched_inference": True,
        }, indent=2), encoding="utf-8")
        self._log = (self._channel / "worker.log").open("w", encoding="utf-8")
        command = [
            str(inference_python), "-u", str(worker_script), "--workspace", str(workspace),
            "--output", str(self._channel / "ready.json"), "--seed", str(self.config.seed),
        ]
        if self.config.model_path:
            command.extend(("--model", str(Path(self.config.model_path).expanduser().resolve())))
        self._process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT)
        logging.warning("TinyVLA loading once. Log: %s", self._channel / "worker.log")
        started = time.monotonic()
        next_update = started + 30
        while not (self._channel / "ready.json").exists():
            if self._process.poll() is not None:
                raise RuntimeError(f"TinyVLA loading failed. See {self._channel / 'worker.log'}")
            if time.monotonic() - started > self.config.startup_timeout:
                raise TimeoutError(f"TinyVLA startup timed out. See {self._channel / 'worker.log'}")
            if time.monotonic() > next_update:
                logging.warning("TinyVLA still loading. Log: %s", self._channel / "worker.log")
                next_update += 30
            time.sleep(0.1)
        logging.warning("TINYVLA_READY: %s", self._channel)

    @staticmethod
    def _instructions(batch, size):
        value = batch.get("task")
        if isinstance(value, str):
            value = (value,) * size
        if not isinstance(value, (list, tuple)) or len(value) != size or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("Expected one nonempty task instruction per batch element.")
        return tuple(value)

    @staticmethod
    def _batch_size(state):
        if state.ndim != 2 or not state.shape[0] or state.shape[1] != 8:
            raise ValueError(f"Expected state shape (batch, 8), got {state.shape}.")
        return state.shape[0]

    @staticmethod
    def _camera(batch, key, size):
        image = batch[key].detach().to("cpu")
        if image.ndim != 4 or image.shape[0] != size or image.shape[1] != 3:
            raise ValueError(f"Expected {size} channel-first images: {key}, {image.shape}")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"Expected finite image values in [0,1]: {key}")
        # No rotation here: LeRobot's LIBERO environment processor already did it.
        return (image.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).numpy()

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Use env.max_parallel_tasks=1 with this adapter.")
        try:
            state = batch["observation.state"].detach().to("cpu", dtype=torch.float32).numpy()
            size = self._batch_size(state)
            instructions = self._instructions(batch, size)
            if not np.isfinite(state).all():
                raise ValueError("Robot state must contain only finite values.")
            external = self._camera(batch, "observation.images.image", size)
            wrist = self._camera(batch, "observation.images.image2", size)
            self._ensure_worker()
            self._request_id += 1
            temporary = self._channel / "input.tmp.npz"
            np.savez(temporary, external=external, wrist=wrist, state=state)
            temporary.replace(self._channel / "input.npz")
            temporary = self._channel / "request.tmp"
            temporary.write_text(json.dumps({"id": self._request_id, "instructions": instructions}),
                                 encoding="utf-8")
            temporary.replace(self._channel / "request.json")
            started = time.monotonic()
            while True:
                response_path = self._channel / "response.json"
                if response_path.exists():
                    response = json.loads(response_path.read_text(encoding="utf-8"))
                    if response["id"] == self._request_id:
                        if not response["ok"]:
                            raise RuntimeError(response["error"])
                        break
                if self._process.poll() is not None:
                    raise RuntimeError(f"TinyVLA process stopped. See {self._channel / 'worker.log'}")
                if time.monotonic() - started > self.config.inference_timeout:
                    raise TimeoutError(f"TinyVLA inference timed out. See {self._channel / 'worker.log'}")
                time.sleep(0.01)
            actions = torch.tensor(response["actions"], dtype=torch.float32)
            if tuple(actions.shape) != (size, 16, 7) or not torch.isfinite(actions).all():
                raise RuntimeError("TinyVLA returned invalid actions.")
            with (self._channel / "requests.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"id": self._request_id, "instructions": instructions, "batch_size": size,
                                      "seconds": response["seconds"],
                                      "clipped_values": int((actions.abs() > 1).sum())}) + "\n")
            return actions.clamp(-1, 1).to(self._device_anchor.device)
        except Exception:
            self._failed = True
            self._action_queue.clear()
            self.close()
            raise
        finally:
            self._lock.release()

    @torch.no_grad()
    def select_action(self, batch, **kwargs):
        size = self._batch_size(batch["observation.state"])
        instructions = self._instructions(batch, size)
        if instructions != self._last_task:
            self._action_queue.clear()
            self._last_task = instructions
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)
            self._action_queue.extend(chunk[:, :self.config.n_action_steps].unbind(dim=1))
        return self._action_queue.popleft()

    def close(self):
        process = self._process
        if process is not None and process.poll() is None:
            if self._channel is not None:
                (self._channel / "stop").touch()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._worker_dir is not None:
            self._worker_dir.cleanup()
            self._worker_dir = None
