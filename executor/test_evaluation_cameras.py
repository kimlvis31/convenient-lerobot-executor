"""Regression coverage for applying saved camera semantics in native evaluation."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from adapters import NativeAdapter
from evaluation_cameras import main, simulator_directory


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

    def stage(self, kind, extra=(), after_training=False, model="smolvla"):
        eval_config = {"env": kind, "tasks": ["task"], "policy": str(self.model), "device": "cpu",
                       "episodes": 1, "batch": 1, "seed": 0, "extra": list(extra)}
        manager = NativeAdapter({"model": model, "eval": eval_config}, "example", self.root, self.root,
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

    def test_empty_saved_maps_use_only_documented_policy_camera_positions(self):
        cases = [
            ("libero", "gaussian_actor", ("image", "image2"), ("image", "wrist_image")),
            ("robotwin", "lingbot_vla_v2", ("head_camera", "left_camera", "right_camera"),
             ("cam_high", "cam_left_wrist", "cam_right_wrist")),
        ]
        for kind, model, sources, targets in cases:
            with self.subTest(kind=kind):
                self.checkpoint(sources, targets)
                config = self.model / "config.json"
                policy = dict(json.loads(config.read_text()), type=model)
                config.write_text(json.dumps(policy))
                processor = self.model / "policy_preprocessor.json"
                saved = json.loads(processor.read_text())
                saved["steps"][0]["config"]["rename_map"] = {}
                processor.write_text(json.dumps(saved))
                stage = self.stage(kind, model=model)
                expected = {k: v for k, v in self.mapping.items() if k != v}
                argument = next(arg for arg in stage.arguments if arg.startswith("--rename_map="))
                self.assertEqual(json.loads(argument.partition("=")[2]), expected)
                self.assertIn("rename_map", stage.auto_flags)
                for custom in ("--rename_map={}", "--env.camera_names=head_camera", "--env.camera_name_mapping={}"):
                    self.assertNotIn("rename_map", self.stage(kind, [custom], model=model).auto_flags)
                policy["input_features"].pop(next(iter(expected.values())))
                config.write_text(json.dumps(policy))
                self.assertNotIn("rename_map", self.stage(kind, model=model).auto_flags)

    def test_robotwin_smol_composes_physical_positions_with_saved_dataset_names(self):
        aliases = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        self.checkpoint(aliases, ("camera1", "camera2", "camera3"))
        stage = self.stage("robotwin")
        expected = dict(zip(("observation.images." + name for name in ("head_camera", "left_camera", "right_camera")),
                            self.mapping.values()))
        option = next(arg for arg in stage.arguments if arg.startswith("--rename_map="))
        self.assertEqual(json.loads(option.partition("=")[2]), expected)
        self.assertEqual(Path(stage.arguments[2]).name, "evaluation_cameras.py")
        self.assertNotIn("rename_map", self.stage("robotwin", ["--rename_map={}"]).auto_flags)
        self.assertNotIn("rename_map", self.stage("robotwin", ["--env.camera_names=head_camera"]).auto_flags)
        self.checkpoint(("cam_high", "cam_right_wrist", "unrelated"), ("camera1", "camera2", "camera3"))
        self.assertNotIn("rename_map", self.stage("robotwin").auto_flags)

    def test_robotwin_relative_assets_are_found_and_cwd_and_import_path_are_restored(self):
        previous, paths = Path.cwd(), sys.path.copy()
        files = ("envs/_base_task.py", "task_config/demo_clean.yml", "assets/objects/objaverse/list.json",
                 "assets/embodiments/aloha/config.yml")
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True)
            path.touch()
        with patch.dict("os.environ", {"ROBOTWIN_ROOT": str(self.root)}):
            with self.assertRaisesRegex(RuntimeError, "failed rollout"):
                with simulator_directory(["--env.type=robotwin"]):
                    self.assertTrue(Path(files[2]).is_file())
                    self.assertEqual(sys.path[0], str(self.root.resolve()))
                    raise RuntimeError("failed rollout")
            self.assertEqual((Path.cwd(), sys.path), (previous, paths))
            (self.root / files[2]).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "ROBOTWIN_ROOT"), simulator_directory(["--env.type=robotwin"]):
                pass
            with simulator_directory(["--env.type=libero"]):
                self.assertEqual(Path.cwd(), previous)

    def test_robotwin_total_counts_the_unique_names_in_the_upstream_task_dictionary(self):
        self.checkpoint(("cam_high", "cam_left_wrist", "cam_right_wrist"), ("camera1", "camera2", "camera3"))
        config = {"eval": {"env": "robotwin", "tasks": ["beat_block_hammer", "click_bell, beat_block_hammer"],
                           "batch": 2, "episodes": 3, "seed": 0, "device": "cpu", "policy": str(self.model)},
                  "model": "smolvla", "adapter": "native"}
        adapter = NativeAdapter(config, "test", self.root, None, {"native": "lerobot"})
        stage = adapter.evaluate(False)
        self.assertEqual(stage.units, 6)
        self.assertEqual(stage.eval_batches, 2)

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
