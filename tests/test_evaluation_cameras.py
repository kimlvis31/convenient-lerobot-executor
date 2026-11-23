"""Regression coverage for applying saved camera semantics in native evaluation."""

import builtins
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from adapters import NativeAdapter
from evaluation_cameras import main, observation_arguments, prepare_evaluation_runtime, simulator_directory


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

    def libero_plus_checkpoint(self, *, count=2, extra_camera=None, model="smolvla"):
        self.checkpoint(("front", "wrist"), ("camera1", "camera2"))
        path = self.model / "config.json"
        config = json.loads(path.read_text())
        config.update(type=model, empty_cameras=count)
        for name in ("empty_camera_0", "empty_camera_1", extra_camera):
            if name:
                config["input_features"]["observation.images." + name] = {"type": "VISUAL"}
        path.write_text(json.dumps(config))
        return path

    def test_libero_plus_smol_maps_saved_front_and_wrist_without_turning_virtual_slots_into_images(self):
        config = self.libero_plus_checkpoint()
        original = config.read_bytes()
        stage = self.stage("libero_plus")
        option = next(arg for arg in stage.arguments if arg.startswith("--rename_map="))
        self.assertEqual(json.loads(option.partition("=")[2]), {
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        })
        self.assertEqual(Path(stage.arguments[2]).name, "evaluation_cameras.py")
        self.assertIn("rename_map", stage.auto_flags)
        self.assertEqual(config.read_bytes(), original)
        self.assertFalse(any(arg.startswith("--policy.empty_cameras=") for arg in stage.arguments))
        for override in ("--rename_map={}", "--env.features_map={}", "--policy.input_features={}",
                         "--env.camera_name_mapping={}"):
            with self.subTest(override=override):
                manual = self.stage("libero_plus", [override])
                self.assertIn(override, manual.arguments)
                self.assertNotIn("rename_map", manual.auto_flags)

    def test_only_smol_declared_canonical_virtual_slots_are_excluded_from_physical_matching(self):
        cases = [(None, None, "smolvla"), (0, None, "smolvla"), (-1, None, "smolvla"),
                 (True, None, "smolvla"), ("2", None, "smolvla"), (1, None, "smolvla"),
                 (2, "empty_camera_2", "smolvla"), (2, "empty_camera_01", "smolvla"),
                 (2, "third_camera", "smolvla"), (2, None, "act")]
        for count, extra, model in cases:
            with self.subTest(count=count, extra=extra, model=model):
                self.libero_plus_checkpoint(count=count, extra_camera=extra, model=model)
                self.assertNotIn("rename_map", self.stage("libero_plus", model=model).auto_flags)

    def test_libero_plus_aliases_require_a_complete_unambiguous_saved_real_camera_mapping(self):
        alternatives = ({}, {"front": "camera1"}, {"front": "camera1", "wrist": "camera1"},
                        {"front": "empty_camera_0", "wrist": "camera2"},
                        {"unverified_front": "camera1", "wrist": "camera2"})
        for saved in alternatives:
            with self.subTest(saved=saved):
                self.libero_plus_checkpoint()
                path = self.model / "policy_preprocessor.json"
                content = json.loads(path.read_text())
                content["steps"][0]["config"]["rename_map"] = {
                    "observation.images." + source: "observation.images." + target
                    for source, target in saved.items()
                }
                path.write_text(json.dumps(content))
                self.assertNotIn("rename_map", self.stage("libero_plus").auto_flags)

    def test_smol_virtual_slots_keep_their_masks_when_physical_names_already_match(self):
        names = ("image", "image2")
        identity = {"observation.images." + name: "observation.images." + name for name in names}
        for saved in ({}, identity):
            with self.subTest(saved=saved):
                self.checkpoint(names, names)
                config_path = self.model / "config.json"
                policy = json.loads(config_path.read_text())
                policy["empty_cameras"] = 2
                policy["input_features"].update({
                    "observation.images.empty_camera_0": {"type": "VISUAL"},
                    "observation.images.empty_camera_1": {"type": "VISUAL"},
                })
                config_path.write_text(json.dumps(policy))
                original = config_path.read_bytes()
                processor_path = self.model / "policy_preprocessor.json"
                processor = json.loads(processor_path.read_text())
                processor["steps"][0]["config"]["rename_map"] = saved
                processor_path.write_text(json.dumps(processor))
                stage = self.stage("libero_plus")
                option = next(arg for arg in stage.arguments if arg.startswith("--rename_map="))
                self.assertEqual(json.loads(option.partition("=")[2]), identity)
                self.assertEqual(config_path.read_bytes(), original)
                for override in ("--rename_map={}", "--env.features={}", "--policy.empty_cameras=0"):
                    self.assertNotIn("rename_map", self.stage("libero_plus", [override]).auto_flags)

    def test_identity_recovery_requires_virtual_slots_and_exact_verified_physical_features(self):
        names = ("image", "image2")
        for count, slots, model, saved in (
            (0, (), "smolvla", {}), (2, (), "smolvla", {}),
            (1, ("empty_camera_0", "empty_camera_1"), "smolvla", {}),
            (2, ("empty_camera_0", "unknown_camera"), "smolvla", {}),
            (2, ("empty_camera_0",), "act", {}),
            (2, ("empty_camera_0",), "smolvla", {"observation.images.image": "observation.images.image"}),
        ):
            with self.subTest(count=count, slots=slots, model=model, saved=saved):
                self.checkpoint(names, names)
                config_path = self.model / "config.json"
                policy = json.loads(config_path.read_text())
                policy.update(type=model, empty_cameras=count)
                policy["input_features"].update({"observation.images." + name: {"type": "VISUAL"}
                                                  for name in slots})
                config_path.write_text(json.dumps(policy))
                processor_path = self.model / "policy_preprocessor.json"
                processor = json.loads(processor_path.read_text())
                processor["steps"][0]["config"]["rename_map"] = saved
                processor_path.write_text(json.dumps(processor))
                self.assertNotIn("rename_map", self.stage("libero_plus", model=model).auto_flags)

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

    def test_libero_plus_always_uses_runtime_initialization_even_without_camera_renaming(self):
        self.checkpoint(("image", "image2"), ("image", "image2"))
        for overrides in ((), ("--rename_map={}",)):
            with self.subTest(overrides=overrides):
                stage = self.stage("libero_plus", overrides)
                self.assertEqual(Path(stage.arguments[2]).name, "evaluation_cameras.py")
                self.assertNotIn("rename_map", stage.auto_flags)
        self.assertEqual(Path(self.stage("libero_plus", after_training=True).arguments[2]).name,
                         "evaluation_cameras.py")

    def test_other_benchmarks_do_not_require_or_import_wand(self):
        original = builtins.__import__

        def without_wand(name, *args, **kwargs):
            if name.startswith("wand"):
                raise AssertionError("An unrelated benchmark must not load ImageMagick")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_wand):
            for kind in ("libero", "robotwin", "robocasa", "vlabench"):
                prepare_evaluation_runtime([f"--env.type={kind}"])

    def test_libero_plus_initialization_failure_preserves_the_native_error(self):
        image = ModuleType("wand.image")
        image.Image = lambda **_: (_ for _ in ()).throw(OSError("missing native JPEG symbol"))
        with patch.dict(sys.modules, {"wand.image": image}):
            with self.assertRaisesRegex(RuntimeError, "ImageMagick initialization failed.*missing native JPEG symbol"):
                prepare_evaluation_runtime(["--env.type=libero_plus"])

    def test_standard_and_parallel_entrypoints_initialize_before_importing_evaluator_extensions(self):
        from parallel_eval import main as parallel_main

        for entrypoint in (main, parallel_main):
            with self.subTest(entrypoint=entrypoint.__module__):
                events = []

                class Image:
                    def __init__(self, **kwargs):
                        events.append(("image", kwargs))

                    def __enter__(self):
                        return self

                    def __exit__(self, *error):
                        events.append(("image-close",))

                image = ModuleType("wand.image")
                image.Image = Image
                evaluator = ModuleType("lerobot.scripts.lerobot_eval")
                evaluator.main = lambda: events.append(("evaluate",)) or 29
                scripts = ModuleType("lerobot.scripts")
                scripts.lerobot_eval = evaluator
                original = builtins.__import__

                def ordered_import(name, *args, **kwargs):
                    if name in ("lerobot.scripts", evaluator.__name__):
                        self.assertIn(("image-close",), events,
                                      "Native evaluator extensions must load after ImageMagick")
                        events.append(("import-evaluator",))
                    return original(name, *args, **kwargs)

                with patch.dict(sys.modules, {image.__name__: image, evaluator.__name__: evaluator,
                                              scripts.__name__: scripts}), \
                     patch.object(sys, "argv", ["evaluation", "--env.type=libero_plus"]), \
                     patch("builtins.__import__", side_effect=ordered_import), \
                     patch("parallel_eval.track_parallel_rollouts"):
                    self.assertEqual(entrypoint(), 29)
                self.assertEqual(events[:2], [("image", {"width": 1, "height": 1}), ("image-close",)])
                self.assertEqual(events[-1], ("evaluate",))

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

    def test_pusht_keypoint_observations_follow_checkpoint_features_and_respect_overrides(self):
        config = {"type": "tdmpc", "input_features": {
            "observation.state": {"type": "STATE", "shape": [2]},
            "observation.environment_state": {"type": "ENV", "shape": [16]},
        }}
        path = self.model / "config.json"
        path.write_text(json.dumps(config))
        original = path.read_bytes()
        stage = self.stage("pusht", model="tdmpc")
        self.assertIn("--env.obs_type=environment_state_agent_pos", stage.arguments)
        self.assertIn("env.obs_type", stage.auto_flags)
        self.assertEqual(path.read_bytes(), original)
        for override in ("--env.obs_type=pixels_agent_pos", "--env.features={}",
                         "--policy.input_features={}", "--rename_map={}"):
            with self.subTest(override=override):
                explicit = self.stage("pusht", [override], model="tdmpc")
                self.assertIn(override, explicit.arguments)
                self.assertNotIn("env.obs_type", explicit.auto_flags)
        config["input_features"]["observation.images.top"] = {"type": "VISUAL", "shape": [3, 96, 96]}
        path.write_text(json.dumps(config))
        self.assertNotIn("env.obs_type", self.stage("pusht", model="tdmpc").auto_flags)
        del config["input_features"]["observation.images.top"]
        config["input_features"]["observation.environment_state"]["shape"] = [24]
        path.write_text(json.dumps(config))
        self.assertNotIn("env.obs_type", self.stage("pusht", model="tdmpc").auto_flags)
        legacy = {"cem_iterations": 6, "q_ensemble_size": 5, "n_gaussian_samples": 512,
                  "input_shapes": {"observation.state": [2], "observation.environment_state": [16]},
                  "input_normalization_modes": {"observation.state": "min_max", "observation.environment_state": "min_max"},
                  "output_shapes": {"action": [2]}, "output_normalization_modes": {"action": "min_max"}}
        path.write_text(json.dumps(legacy))
        self.assertIn("env.obs_type", self.stage("pusht", model="tdmpc").auto_flags)

    def test_pusht_pending_checkpoint_is_resolved_at_runtime_without_a_processor_file(self):
        config = {"type": "tdmpc", "input_features": {
            "observation.state": {"type": "STATE", "shape": [2]},
            "observation.environment_state": {"type": "ENV", "shape": [16]},
        }}
        (self.model / "config.json").write_text(json.dumps(config))
        args = ["evaluate", "--env.type=pusht", "--policy.path=org/keypoints"]
        with patch("evaluation_cameras.checkpoint_directory", return_value=None):
            self.assertIsNone(observation_arguments(args[1:]))
        with patch("adapters.observation_arguments", return_value=None):
            stage = self.stage("pusht", model="tdmpc")
            self.assertEqual(Path(stage.arguments[2]).name, "evaluation_cameras.py")
        invoked = []
        endpoint = ModuleType("lerobot.scripts.lerobot_eval")
        endpoint.main = lambda: invoked.append(list(sys.argv))
        with patch.dict(sys.modules, {endpoint.__name__: endpoint}), patch.object(sys, "argv", args), \
             patch("evaluation_cameras.checkpoint_directory", return_value=self.model) as resolve:
            main()
        resolve.assert_called_once_with("org/keypoints", None, download=True, filename="config.json")
        self.assertIn("--env.obs_type=environment_state_agent_pos", invoked[0])

    def test_partial_episode_batch_uses_shared_video_and_progress_limits_for_one_task(self):
        for kind in ("pusht", "libero"):
            with self.subTest(kind=kind):
                self.checkpoint(("image", "image2"), ("image", "image2"))
                cfg = {"env": kind, "tasks": ["task"], "policy": str(self.model), "device": "cpu",
                       "episodes": 3, "batch": 2, "parallel_tasks": 1, "seed": 0}
                manager = NativeAdapter({"model": "smolvla", "eval": cfg}, "example", self.root,
                                        self.root, {"native": "lerobot"})
                stage = manager.evaluate(False)
                self.assertEqual(Path(stage.arguments[2]).name, "parallel_eval.py")
                self.assertIn("--env.max_parallel_tasks=1", stage.arguments)
                cfg["episodes"] = 4
                self.assertEqual(manager.evaluate(False).arguments[0], "lerobot-eval")


if __name__ == "__main__":
    unittest.main()
