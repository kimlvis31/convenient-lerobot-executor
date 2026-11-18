"""Regression tests for manual simulator detection and environment selection."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from benchmarks import libero_plus_issue, simulator_runtime_issue
from manage import LeRobotEnvironments


class LiberoPlusRuntimeTests(unittest.TestCase):
    def test_stock_libero_is_not_a_prepared_plus_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            package = SimpleNamespace(submodule_search_locations=[folder])
            with patch("benchmarks.util.find_spec", return_value=package), \
                 patch("benchmarks.PathFinder.find_spec", return_value=package):
                self.assertIn("not the LIBERO-plus fork", libero_plus_issue())

    def test_configuration_and_loaded_native_library_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "benchmark").mkdir()
            (root / "benchmark/task_classification.json").touch()
            settings = root / "config.yaml"
            package = SimpleNamespace(submodule_search_locations=[folder])
            config = {"init_states": str(root / "init"), "bddl_files": str(root / "bddl")}
            with patch("benchmarks.util.find_spec", return_value=package), \
                 patch("benchmarks.PathFinder.find_spec", return_value=package), \
                 patch.dict("os.environ", {"LIBERO_CONFIG_PATH": folder}), \
                 patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=lambda _: config)}):
                self.assertIn("config.yaml", libero_plus_issue())
                self.assertFalse(settings.exists())
                settings.write_text("config", encoding="utf-8")
                (root / "init/libero_newobj").mkdir(parents=True)
                (root / "bddl").mkdir()
                self.assertIn("scene assets", libero_plus_issue())
                (root / "assets/scenes").mkdir(parents=True)
                (root / "assets/scenes/libero_tabletop_base_style.xml").touch()
                factory = MagicMock(side_effect=OSError("libMagickWand missing"))
                modules = {
                    "wand.image": SimpleNamespace(Image=factory),
                    "skimage.filters": SimpleNamespace(gaussian=lambda: None),
                    "robosuite.environments.manipulation.single_arm_env": SimpleNamespace(SingleArmEnv=object),
                }
                with patch.dict(sys.modules, modules):
                    self.assertIn("libMagickWand missing", libero_plus_issue())
                    factory.side_effect = None
                    self.assertIsNone(libero_plus_issue())
                    factory.return_value.__enter__.return_value.flip.assert_called_once()

    def test_robomme_requires_the_runtime_and_episode_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = SimpleNamespace(submodule_search_locations=[folder])
            with patch("benchmarks.util.find_spec", side_effect=lambda name: None if name == "sapien" else spec):
                self.assertIn("sapien", simulator_runtime_issue("robomme"))
            with patch("benchmarks.util.find_spec", return_value=spec):
                self.assertIn("env_metadata", simulator_runtime_issue("robomme"))
                path = Path(folder) / "env_metadata/test/record_dataset_PickXtimes_metadata.json"
                path.parent.mkdir(parents=True)
                path.touch()
                self.assertIsNone(simulator_runtime_issue("robomme"))

    def test_prepared_manual_runtime_routes_without_installing_packages(self):
        issue = "The active libero package is not the LIBERO-plus fork."
        catalogs = {
            "stock": {"policies": {"act": True}, "benchmarks": {"libero": True},
                      "manual_status": {"libero_plus": issue}},
            "fork": {"policies": {"act": True}, "manual_status": {"libero_plus": None}},
        }
        envs = LeRobotEnvironments("conda", Path("."), "stock", set(catalogs))
        envs.catalogs = catalogs
        self.assertEqual(envs.available(catalogs)["libero_plus"], ["fork"])
        with patch("manage.subprocess.run") as run:
            self.assertEqual(envs.prepare("stock", model="act", benchmark="libero_plus"), "fork")
            with self.assertRaisesRegex(RuntimeError, "not the LIBERO-plus fork"):
                envs.prepare("stock", model="act", benchmark="libero_plus", pinned=True)
            run.assert_not_called()
