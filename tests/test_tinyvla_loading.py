"""Standalone head-loading diagnostics must describe the active head honestly."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "externals/TinyVLA/runtime/tinyvla_loader.py"
SPEC = importlib.util.spec_from_file_location("tinyvla_loading_diagnostics", SOURCE)
LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)
WORKER_SOURCE = SOURCE.parents[1] / "lerobot_policy_tinyvla/src/lerobot_policy_tinyvla/worker_builder.py"
WORKER_SPEC = importlib.util.spec_from_file_location("tinyvla_worker_loading", WORKER_SOURCE)
WORKER = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(WORKER)


class TinyVLALoadingTests(unittest.TestCase):
    head = ("embed_out.down_modules.0.weight", "embed_out.final_conv.0.weight")

    def test_checkpoint_head_is_loaded_even_if_an_unused_old_language_head_is_present(self):
        info = {"missing_keys": [], "mismatched_keys": [], "unexpected_keys": ["embed_out.weight"]}
        result = LOADER.action_head_loading((*self.head, "gpt_neox.layers.0.weight"), info)
        self.assertEqual(result["status"], "loaded_from_checkpoint")
        self.assertEqual(result["expected_parameter_keys"], 2)
        self.assertEqual(result["newly_initialized_keys"], [])
        self.assertNotIn("trained", result["status"])

    def test_a_fully_new_diffusion_head_includes_missing_and_mismatched_parameters(self):
        info = {"missing_keys": [self.head[0]], "mismatched_keys": [(self.head[1], (10, 20), (3, 4))]}
        result = LOADER.action_head_loading(self.head, info)
        self.assertEqual(result["status"], "new_random_initialization")
        self.assertEqual(set(result["newly_initialized_keys"]), set(self.head))

    def test_a_partially_restored_head_is_not_reported_as_fully_loaded(self):
        info = {"missing_keys": [self.head[0]], "unexpected_keys": ["unused.weight"]}
        result = LOADER.action_head_loading(self.head, info)
        self.assertEqual(result["status"], "partially_initialized")
        self.assertEqual(result["newly_initialized_keys"], [self.head[0]])

    def test_other_loading_diagnostics_do_not_change_active_head_classification(self):
        result = LOADER.action_head_loading(self.head, {
            "missing_keys": ["gpt_neox.some_other.weight", "embed_out.weight"],
            "mismatched_keys": [["embed_out.weight", [50000, 1024], [14, 1024]]],
        })
        self.assertEqual(result["status"], "loaded_from_checkpoint")

    def test_a_model_without_an_identified_action_head_has_unknown_status(self):
        result = LOADER.action_head_loading(["gpt_neox.layer.weight"], {})
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["expected_parameter_keys"], 0)

    def worker_ready(self, folder, *, loading=None, sidecar=False, matching=True):
        root = Path(folder)
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (root / "stop").touch()

        class Tensor:
            dtype = "float32"

            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def to(self, dtype):
                return self

        parameters = {"embed_out.actual.weight": Tensor(1)}
        model = SimpleNamespace(named_parameters=lambda: parameters.items(),
                                noise_scheduler=SimpleNamespace(alphas_cumprod=SimpleNamespace(dtype="float32")))
        scope = {
            "args": SimpleNamespace(output=root / "ready.json", model=model_dir, repo=root, seed=0),
            "model": model, "info": {}, "revision": lambda _: "verified-source",
            "json": json,
            "torch": SimpleNamespace(float32="float32", equal=lambda a, b: a.value == b.value,
                                     isfinite=lambda _: SimpleNamespace(all=lambda: True)),
        }
        if loading is not None:
            scope["head_loading"] = loading
        if sidecar:
            (model_dir / "action_head.safetensors").touch()
        modules = {"numpy": SimpleNamespace(),
                   "safetensors.torch": SimpleNamespace(load_file=lambda _: {
                       "embed_out.actual.weight": Tensor(1 if matching else 2)})}
        body = textwrap.dedent(WORKER.TAIL[:WORKER.TAIL.index("\n\nif __name__")])
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(body, "generated-worker-tail", "exec"), scope)
        return json.loads((root / "ready.json").read_text())

    def test_generated_worker_preserves_loaded_head_diagnostics_without_a_sidecar(self):
        diagnostics = {"status": "loaded_from_checkpoint", "newly_initialized_keys": []}
        with tempfile.TemporaryDirectory() as folder:
            ready = self.worker_ready(folder, loading=diagnostics)
        self.assertEqual(ready["action_head"], "loaded_from_checkpoint")
        self.assertEqual(ready["action_head_loading"], diagnostics)

    def test_legacy_custom_loader_without_diagnostics_is_unknown_instead_of_random(self):
        with tempfile.TemporaryDirectory() as folder:
            ready = self.worker_ready(folder)
        self.assertEqual(ready["action_head"], "unknown")

    def test_a_sidecar_still_requires_matching_actual_parameters_before_readiness(self):
        with tempfile.TemporaryDirectory() as folder:
            ready = self.worker_ready(folder, sidecar=True)
        self.assertEqual(ready["action_head"], "verified_checkpoint")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "tensor did not reload"):
                self.worker_ready(folder, sidecar=True, matching=False)
            self.assertFalse((Path(folder) / "ready.json").exists())

    def test_actual_candidate_loader_still_builds_the_worker_program(self):
        generated, digest = WORKER.build_worker(SOURCE.read_text())
        compile(generated, "generated-tinyvla-worker", "exec")
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
