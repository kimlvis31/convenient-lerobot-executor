"""Regression tests for split RoboMME environment routing and local IPC."""

import io
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from adapters import Stage
from manage import LeRobotEnvironments
from robomme_split import CAMERA_AND_STATE, SIMULATOR_ENV, make_split_env, receive, send, serve


class SplitRoboMMETests(unittest.TestCase):
    def setUp(self):
        self.session = LeRobotEnvironments("conda", Path("."), "policy", {"policy", "sim"})
        self.stage = Stage("eval", "policy", ["lerobot-eval", "--env.type=robomme"], (), 1)
        self.session.catalogs = {"policy": {"policies": {"smolvla": True}, "split_robomme": True,
                                             "manual_status": {"robomme": "simulator not installed"}}, "sim": {}}

    def select(self, evaluation, preview=False):
        return self.session.prepare_robomme(self.stage, "policy", evaluation, model="smolvla", peft=False, preview=preview)

    def test_shared_ready_environment_stays_on_native_entrypoint(self):
        self.session.catalogs["policy"]["manual_status"]["robomme"] = None
        with patch.object(self.session, "simulator", side_effect=AssertionError("unexpected probe")):
            self.assertEqual(self.select({}), ("policy", ()))
        self.assertEqual(self.stage.arguments[0], "lerobot-eval")
        self.assertFalse(self.stage.env)

    def test_preview_never_starts_or_provisions_a_simulator(self):
        with patch.object(self.session, "lerobot", side_effect=AssertionError("unexpected probe")), \
             patch.object(self.session, "simulator", side_effect=AssertionError("unexpected probe")):
            policy, note = self.select({}, preview=True)
        self.assertEqual(policy, "policy")
        self.assertIn("at launch", " ".join(note))
        self.assertNotIn(SIMULATOR_ENV, self.stage.env)

    def test_explicit_existing_environment_and_missing_selection(self):
        worker = {"python": "/sim/bin/python", "env": {"PYTHONPATH": ""}}
        with patch.object(self.session, "simulator", return_value=worker) as inspect:
            policy, note = self.select({"simulator_environment": "sim"})
        inspect.assert_called_once_with("sim")
        self.assertEqual(policy, "policy")
        self.assertIn(worker["python"], " ".join(note))
        self.assertEqual(json.loads(self.stage.env[SIMULATOR_ENV]), worker)
        self.assertEqual(Path(self.stage.arguments[2]).name, "evaluation_cameras.py")
        with patch.object(self.session, "simulator", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requested|Cannot use RoboMME simulator"):
                self.select({"simulator_environment": "absent"})

    def test_automatic_provisioning_needs_a_selected_policy(self):
        with patch.object(self.session, "simulator", return_value=None), \
             patch("manage.subprocess.run", return_value=SimpleNamespace(stdout='{"numpy":"1.26.4"}')), \
             patch.dict(sys.modules, {"robomme_setup": SimpleNamespace(prepare_simulator=lambda conda: ("/new/python", {}))}):
            _, note = self.select({})
        self.assertIn("/new/python", " ".join(note))
        self.assertIn("numpy 1.26.4", " ".join(note))
        self.assertEqual(json.loads(self.stage.env[SIMULATOR_ENV])["python"], "/new/python")
        with self.assertRaisesRegex(RuntimeError, "combined environment"):
            self.select({"robomme_runtime": "shared"})

    def test_missing_policy_integration_fails_before_provisioning(self):
        self.session.catalogs["policy"]["split_robomme"] = False
        with patch.object(self.session, "simulator", side_effect=AssertionError("unexpected probe")):
            with self.assertRaisesRegex(RuntimeError, "Gym adapter"):
                self.select({})

    def test_builder_scoped_to_constructor_and_reset_seed_forwarded(self):
        original = object()
        def create():
            from robomme.env_record_wrapper import BenchmarkEnvBuilder
            return SimpleNamespace(_builder=BenchmarkEnvBuilder(), reset=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {"robomme.env_record_wrapper": original}):
            env = make_split_env(create, {}, 0)
            self.assertIs(sys.modules["robomme.env_record_wrapper"], original)
            self.assertEqual(env.reset(seed=17), {"seed": 17, "options": None})
            self.assertEqual(env._builder.seed, 17)
            env.reset()
            next_seed = env._builder.seed
            env.reset(seed=17)
            env.reset()
            self.assertEqual(env._builder.seed, next_seed)

    @unittest.skipUnless(find_spec("numpy"), "NumPy is required for the simulator transport")
    def test_reset_step_preserve_camera_data_and_close_on_eof(self):
        import numpy as np
        images = {key: [np.arange(5, dtype="uint8")] for key in CAMERA_AND_STATE}
        class Episode:
            closed = False
            def reset(self):
                return images, {"task_goal": ["pick up the cube"]}
            def step(self, action):
                return images, float(action[0]), False, True, {"status": "success"}
            def close(self):
                self.closed = True
        episode = Episode()
        builder = SimpleNamespace(make_env_for_episode=lambda **kwargs: episode)
        reader, writer = io.BytesIO(), io.BytesIO()
        send(reader, {"seed": 17, "builder": {}, "episode": {}})
        send(reader, "reset")
        send(reader, "step", action=np.ones(8))
        reader.seek(0)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(manual_seed=lambda seed: None),
                                      "robomme.env_record_wrapper": SimpleNamespace(BenchmarkEnvBuilder=lambda: builder)}):
            serve(reader, writer)
        writer.seek(0)
        response, arrays = receive(writer)
        self.assertEqual(response["info"]["task_goal"], ["pick up the cube"])
        for key in CAMERA_AND_STATE:
            np.testing.assert_array_equal(arrays[key], images[key][-1])
        self.assertEqual(receive(writer)[0]["outcome"], [1.0, False, True])
        self.assertTrue(episode.closed)
