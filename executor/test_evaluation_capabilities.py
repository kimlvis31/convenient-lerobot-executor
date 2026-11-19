"""Ensure simulator transports are not mistaken for task evaluation contracts."""

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from adapters import NativeAdapter, Stage
from evaluation_capabilities import (
    FINEART_REPOSITORY,
    UNITREE_REPOSITORY,
    UNITREE_REVISION,
    UNITREE_SOURCE,
    evaluation_notices,
    stock_evaluation_issue,
)
from manage import choose_evaluation_environment, prepare_job_environments, show_settings


class EvaluationCapabilityTests(unittest.TestCase):
    def settings(self, *, kind="unitree_g1_mujoco", extra=()):
        return {
            "adapter": "native", "model": "act",
            "eval": {"env": kind, "tasks": ["default"], "episodes": 1, "batch": 1,
                     "seed": 0, "device": "cpu", "policy": None, "extra": list(extra)},
        }

    def test_stock_transport_cannot_launch_policy_or_dependency_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            adapter = NativeAdapter(self.settings(), "run", root, root / "external", {"native": "lerobot"})
            with patch.object(adapter, "eval_policy") as policy, patch("manage.subprocess.run") as process:
                with self.assertRaisesRegex(ValueError, "Task Adapter Required") as error:
                    adapter.evaluate(False)
            self.assertIn("does not apply", str(error.exception))
            self.assertIn("episode termination", str(error.exception))
            self.assertIn(UNITREE_SOURCE, str(error.exception))
            policy.assert_not_called()
            process.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_pinning_the_stock_source_does_not_add_a_controller(self):
        for reference in (UNITREE_REPOSITORY, UNITREE_REPOSITORY + f"@{UNITREE_REVISION}:env.py",
                          UNITREE_REPOSITORY + f"@{UNITREE_REVISION[:7]}:env.py",
                          UNITREE_REPOSITORY + "@main:env.py",
                          UNITREE_REPOSITORY + ":env.py"):
            with self.subTest(reference=reference):
                self.assertIsNotNone(stock_evaluation_issue("unitree_g1_mujoco", [f"--env.hub_path={reference}"]))

    def test_unknown_explicit_unitree_revision_is_not_blocked_by_old_inspection(self):
        revision = "a" * 40
        extra = [f"--env.hub_path={UNITREE_REPOSITORY}@{revision}:env.py"]
        self.assertIsNone(stock_evaluation_issue("unitree_g1_mujoco", extra))
        notice = evaluation_notices("act", {"env": "unitree_g1_mujoco", "extra": extra})
        self.assertIn("capability unverified", notice[0])
        self.assertIn(revision, notice[0])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = self.settings(extra=extra)
            adapter = NativeAdapter(cfg, "run", root, root, {"native": "lerobot"})
            with patch.object(adapter, "eval_policy", return_value=[]):
                self.assertIn(extra[0], adapter.evaluate(False).arguments)

    def test_existing_custom_task_hook_and_plugin_types_remain_usable(self):
        for kind, extra in (("unitree_g1_mujoco", ["--env.hub_path=owner/g1-task@version:env.py"]),
                            ("unitree_g1_mujoco", [f"--env.hub_path={UNITREE_REPOSITORY}@version:task_adapter.py"]),
                            ("unitree_g1_mujoco_task", []), ("libero", [])):
            with self.subTest(kind=kind, extra=extra):
                self.assertIsNone(stock_evaluation_issue(kind, extra))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = self.settings(extra=["--env.hub_path=owner/g1-task@version:env.py"])
            adapter = NativeAdapter(cfg, "run", root, root, {"native": "lerobot"})
            with patch.object(adapter, "eval_policy", return_value=["--policy.type=act"]):
                stage = adapter.evaluate(False)
            self.assertIn("--env.hub_path=owner/g1-task@version:env.py", stage.arguments)

            class CustomAdapter(NativeAdapter):
                key = "custom-task-adapter"

            custom = CustomAdapter(self.settings(), "run", root, root, {"native": "lerobot"})
            with patch.object(custom, "eval_policy", return_value=[]):
                self.assertEqual(custom.evaluate(False).name, "eval")

    def test_menu_distinguishes_task_contract_from_missing_packages(self):
        catalogs = {"lerobot": {"manual_benchmarks": ["unitree_g1_mujoco"], "policies": {"act": True}}}
        session = SimpleNamespace(available=MagicMock(return_value={"unitree_g1_mujoco": ["lerobot"]}))
        output = io.StringIO()
        with patch("manage.policy_environments", return_value=(catalogs, catalogs)), \
             patch("sys.stdin", io.StringIO("1\n")), contextlib.redirect_stdout(output):
            self.assertEqual(choose_evaluation_environment(session, "lerobot", "act", False),
                             ("unitree_g1_mujoco", "lerobot"))
        text = output.getvalue()
        self.assertIn("<Task Adapter Required>", text)
        self.assertNotIn("Not Installed", text)
        self.assertNotIn("[Default]", text)
        self.assertIn("Data Distribution Service", text)
        self.assertNotIn("separate simulator setup", text)

    def test_public_fineart_checkpoint_warns_about_semantics_without_a_model_wide_ban(self):
        for source in (FINEART_REPOSITORY, FINEART_REPOSITORY + "@version",
                       str(Path("hub") / "models--lerobot--FineART-VLA-subtask-midtrained" / "snapshots/hash")):
            with self.subTest(source=source):
                notes = evaluation_notices("fineart_vla", {"env": "aloha", "policy": source})
                self.assertEqual(len(notes), 1)
                self.assertIn("five arm joints", notes[0])
                self.assertIn("six arm joints", notes[0])
                self.assertIn("camera renaming alone", notes[0])
        for model, evaluation in (("fineart_vla", {"env": "aloha", "policy": "/outputs/fineart-adapted"}),
                                  ("fineart_vla", {"env": "aloha", "policy": FINEART_REPOSITORY,
                                                   "extra": ["--env.hub_path=owner/matching-task"]}),
                                  ("fineart_vla", {"env": "matching_task", "policy": FINEART_REPOSITORY}),
                                  ("act", {"env": "aloha", "policy": FINEART_REPOSITORY})):
            self.assertEqual(evaluation_notices(model, evaluation), ())

    def test_checkpoint_notice_survives_settings_preparation(self):
        root = Path("project")
        cfg = self.settings(kind="aloha")
        cfg["model"] = "fineart_vla"
        cfg["eval"]["policy"] = FINEART_REPOSITORY
        stage = Stage("eval", "lerobot", ["lerobot-eval"], (), 1)
        session = SimpleNamespace(
            prepare=MagicMock(return_value="lerobot"), selection_reason="Existing runtime.",
            packages=MagicMock(return_value={"LeRobot runtime": ["lerobot", "torch"]}),
            catalogs={"lerobot": {"policies": {"fineart_vla": True}}},
        )
        prepare_job_environments(cfg, [stage], root, None, {(root, "lerobot"): session}, preview=True)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            show_settings([(cfg, [stage], root / "outputs/run", root)], False)
        self.assertIn("Checkpoint compatibility:", output.getvalue())
        self.assertIn("Equal 14-value shapes", output.getvalue())


if __name__ == "__main__":
    unittest.main()
