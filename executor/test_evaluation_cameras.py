"""Regression coverage for applying saved camera semantics in native evaluation."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from adapters import NativeAdapter
from evaluation_cameras import main


class EvaluationCameraTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.model = self.root / "model"
        self.model.mkdir()

    def checkpoint(self, sources, targets):
        self.mapping = {f"observation.images.{src}": f"observation.images.{dst}" for src, dst in zip(sources, targets)}
        features = {name: {"type": "VISUAL"} for name in self.mapping.values()}
        (self.model / "config.json").write_text(json.dumps({"type": "smolvla", "input_features": features}))
        processor = {"steps": [{"registry_name": "rename_observations_processor", "config": {"rename_map": self.mapping}}]}
        (self.model / "policy_preprocessor.json").write_text(json.dumps(processor))

    def stage(self, kind, extra=(), after_training=False):
        eval_config = {"env": kind, "tasks": ["task"], "policy": str(self.model), "device": "cpu",
                       "episodes": 1, "batch": 1, "seed": 0, "extra": list(extra)}
        manager = NativeAdapter({"model": "smolvla", "eval": eval_config}, "example", self.root, self.root,
                                {"native": "lerobot"})
        return manager.evaluate(after_training)

    def test_checkpoint_mapping_is_used_for_matching_environment_only(self):
        for kind, names in (("robocasa", ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")),
                            ("vlabench", ("image", "second_image", "wrist_image"))):
            with self.subTest(kind=kind):
                self.checkpoint(names, ("camera1", "camera2", "camera3"))
                stage = self.stage(kind)
                flag = next(arg for arg in stage.arguments if arg.startswith("--rename_map="))
                self.assertEqual(json.loads(flag.partition("=")[2]), self.mapping)
                self.assertIn("rename_map", stage.auto_flags)
                self.assertEqual(stage.arguments[0], "lerobot-eval")
                other = self.stage("vlabench" if kind == "robocasa" else "robocasa")
                self.assertNotIn("rename_map", other.auto_flags)

    def test_explicit_empty_map_custom_features_and_matched_names_are_preserved(self):
        names = ("image", "second_image", "wrist_image")
        self.checkpoint(names, ("camera1", "camera2", "camera3"))
        for override in ("--rename_map={}", "--rename_map.custom=custom", "--env.features_map={}",
                         "--policy.input_features={}"):
            with self.subTest(override=override):
                stage = self.stage("vlabench", [override])
                self.assertIn(override, stage.arguments)
                self.assertNotIn("rename_map", stage.auto_flags)
        self.checkpoint(names, names)
        self.assertNotIn("rename_map", self.stage("vlabench").auto_flags)

    def test_ambiguous_and_incomplete_maps_are_ignored(self):
        names = ("image", "second_image", "wrist_image")
        self.checkpoint(names, ("camera1", "camera2", "camera3"))
        path = self.model / "policy_preprocessor.json"
        for targets in (("camera1", "camera1", "camera3"), ("camera1", "camera2", "unknown")):
            step = json.loads(path.read_text())["steps"][0]
            step["config"]["rename_map"] = dict(zip(self.mapping, (f"observation.images.{key}" for key in targets)))
            path.write_text(json.dumps({"steps": [step]}))
            self.assertNotIn("rename_map", self.stage("vlabench").auto_flags)

    def test_after_training_has_a_deferred_lookup_and_respects_manual_override(self):
        stage = self.stage("vlabench", after_training=True)
        self.assertEqual(Path(stage.arguments[2]).name, "evaluation_cameras.py")
        self.assertEqual(self.stage("vlabench", ["--rename_map={}"], after_training=True).arguments[0], "lerobot-eval")
        self.assertEqual(self.stage("robomme", after_training=True).arguments[0], "lerobot-eval")

    def test_runtime_looks_up_a_new_checkpoint_before_the_standard_evaluator(self):
        self.checkpoint(("image", "second_image", "wrist_image"), ("camera1", "camera2", "camera3"))
        invoked = []
        endpoint = ModuleType("lerobot.scripts.lerobot_eval")
        endpoint.main = lambda: invoked.append(list(sys.argv))
        args = ["evaluation_cameras.py", "--env.type=vlabench", "--policy.path=org/model"]
        with patch.dict(sys.modules, {endpoint.__name__: endpoint}), patch.object(sys, "argv", args), \
             patch("evaluation_cameras.checkpoint_directory", return_value=self.model) as resolve:
            main()
        resolve.assert_called_once_with("org/model", None, download=True)
        self.assertEqual(json.loads(invoked[0][-1].partition("=")[2]), self.mapping)


if __name__ == "__main__":
    unittest.main()
