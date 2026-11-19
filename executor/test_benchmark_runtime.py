"""Regression tests for manual simulator detection and environment selection."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from benchmarks import data_runtime_issue, environment_catalog, install_dependencies, libero_plus_issue, maniskill_numpy_issue, simulator_runtime_issue
from manage import LeRobotEnvironments
from packaging.requirements import Requirement


class LiberoPlusRuntimeTests(unittest.TestCase):
    def test_flux_extension_is_installable_and_not_reported_ready_when_its_import_fails(self):
        config = SimpleNamespace(__module__="lerobot.policies.flux3.configuration_flux3")
        root = SimpleNamespace()
        policies = SimpleNamespace()
        modules = {
            "lerobot": root, "lerobot.policies": policies,
            "lerobot.configs": SimpleNamespace(PreTrainedConfig=SimpleNamespace(get_known_choices=lambda: {"flux3": config})),
            "lerobot.envs.configs": SimpleNamespace(EnvConfig=SimpleNamespace(get_known_choices=lambda: {})),
            "lerobot.policies.factory": SimpleNamespace(get_policy_class=MagicMock()),
            "lerobot.utils.import_utils": SimpleNamespace(register_third_party_plugins=lambda: None),
        }
        for detail in ("Missing or outdated: natten>=0.21.6", "natten runtime import failed: undefined symbol"):
            with self.subTest(detail=detail), patch.dict(sys.modules, modules), \
                 patch("benchmarks.metadata.distribution"), patch("benchmarks.data_runtime_issue", return_value=None), \
                 patch("benchmarks.installed_versions", return_value={"transformers": "5.5.4"}), \
                 patch("benchmarks.optional_requirements", return_value=[]), \
                 patch("benchmarks.policy_requirements", return_value=[Requirement("transformers>=5")]), \
                 patch("benchmarks.extension_problems", return_value={"natten": detail}), \
                 patch("benchmarks.extension_runtime_issue", return_value=None), \
                 patch("benchmarks.util.find_spec", return_value=None):
                result = environment_catalog()
            self.assertFalse(result["policies"]["flux3"])
            self.assertIn("flux3", result["policy_installable"])
            self.assertIn(detail, result["policy_issues"]["flux3"])

    def test_confirmed_flux_install_includes_registered_kernel_outside_pyproject_extra(self):
        modules = {"lerobot": SimpleNamespace(), "lerobot.policies": SimpleNamespace(),
                   "lerobot.configs": SimpleNamespace(PreTrainedConfig=SimpleNamespace(get_known_choices=lambda: {
                       "flux3": SimpleNamespace(__module__="lerobot.policies.flux3.configuration_flux3")}))}
        with patch.dict(sys.modules, modules), patch("benchmarks.metadata.distribution"), \
             patch("benchmarks.installed_versions", return_value={"torch": "2.11.0"}), \
             patch("benchmarks.policy_requirements", return_value=[]), \
             patch("benchmarks.extension_problems", return_value={"natten": "missing"}), \
             patch("benchmarks.install_extensions", return_value=[]) as install, \
             patch("benchmarks.subprocess.run") as process:
            self.assertEqual(install_dependencies(policy="flux3"), 0)
        install.assert_called_once_with("flux3", {"torch": "2.11.0"}, [])
        process.assert_not_called()

    def test_robotwin_rejects_the_active_numpy_when_it_conflicts_with_mplib(self):
        with patch("benchmarks.util.find_spec", return_value=object()), \
             patch("benchmarks.robotwin_directory", return_value=Path("/installed")), \
             patch("benchmarks.metadata.requires", return_value=["numpy<2.0", "toppra>=0.4"]):
            for version, expected in (("1.26.4", None), ("2.5.5", "RoboTwin MPlib requires numpy<2.0; NumPy 2.5.5 is active.")):
                with self.subTest(version=version), patch.dict(sys.modules, {"numpy": SimpleNamespace(__version__=version)}):
                    self.assertEqual(simulator_runtime_issue("robotwin"), expected)

    def test_missing_arrow_binary_is_reported_before_launch(self):
        issue = "PyArrow cannot load in this interpreter"
        with patch("benchmarks.util.find_spec", return_value=object()), patch.dict(sys.modules, {"pyarrow": None}):
            self.assertIn(issue, data_runtime_issue())
            with patch("benchmarks.metadata.distribution"):
                self.assertIn(issue, environment_catalog()["runtime_issue"])
                with patch("benchmarks.installed_versions", return_value={}):
                    with self.assertRaisesRegex(ValueError, issue):
                        install_dependencies(runtime="evaluation")
        envs = LeRobotEnvironments("conda", Path("."), "broken", {"broken", "ready"})
        envs.catalogs = {"broken": {"runtime_issue": issue},
                         "ready": {"policies": {"smolvla": True}, "manual_status": {"robomme": None}}}
        with patch("manage.subprocess.run") as run:
            self.assertEqual(envs.prepare("broken", model="smolvla", benchmark="robomme", runtime="evaluation"), "ready")
            with self.assertRaisesRegex(RuntimeError, issue):
                envs.prepare("broken", model="smolvla", benchmark="robomme", pinned=True, runtime="evaluation")
            run.assert_not_called()

    def test_maniskill_checks_the_effective_numpy_against_its_own_declaration(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "mani_skill"
            source.mkdir()
            (source.parent / "setup.py").write_text("install_requires = ['numpy>=1.22,<2']")
            spec = SimpleNamespace(submodule_search_locations=[str(source)])
            for version, incompatible in (("1.26.4", False), ("2.2.5", True)):
                with self.subTest(version=version), patch.dict(sys.modules, {"numpy": SimpleNamespace(__version__=version)}):
                    self.assertEqual(bool(maniskill_numpy_issue(spec)), incompatible)

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
