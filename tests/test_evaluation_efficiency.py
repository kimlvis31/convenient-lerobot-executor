"""AI-written semantic checks for fixed language and render-frame reuse."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from evaluation_efficiency import (existing_observations, observation_render_supported,
                                   standard_libero_camera, standard_libero_rollouts)

try:
    import numpy as np
except ImportError:
    np = None


class ReferenceRenderer:
    def render(self):
        self._ensure_env()
        raw_obs = self._env.env._get_observations()
        pixels = self._format_raw_obs(raw_obs)["pixels"]
        image = next(iter(pixels.values()))
        image = image[::-1, ::-1]
        return image


class FakeVector:
    num_envs = 2

    def __init__(self, label="task", sync=False):
        self.label, self.generation = label, 0
        self.calls = Counter()
        self.received = None
        self.current = None
        if sync:
            self.envs = [SimpleNamespace(render=lambda i=i: self.native_frames()[i]) for i in range(2)]

    def observation(self):
        if np is None:
            return {}
        front = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3) + self.generation
        self.current = {"pixels": {"a_hand": front + 60, "z_front": front}}
        return self.current

    def reset(self, *args, **kwargs):
        self.generation += 1
        self.received = (args, kwargs)
        return self.observation(), {"reset": True}

    def step(self, actions):
        self.received = actions
        self.generation += 1
        return self.observation(), 0, False, False, {}

    def native_frames(self):
        return tuple(frame[::-1, ::-1].copy() for frame in self.current["pixels"]["z_front"])

    def call(self, name, *args, **kwargs):
        self.calls[name] += 1
        if name == "task_description" and not args and not kwargs:
            return tuple(f"{self.label}-{self.generation}-{i}" for i in range(self.num_envs))
        if name == "render":
            return self.native_frames()
        return name, args, kwargs


class EvaluationEfficiencyTests(unittest.TestCase):
    def test_language_is_requested_once_and_reset_refreshes_it(self):
        env = FakeVector()
        with existing_observations(env, None):
            env.reset(seed=[4, 5], options={"new_rollout": True})
            first = env.call("task_description")
            self.assertEqual(first, env.call("task_description"))
            self.assertEqual(env.calls["task_description"], 1)
            env.reset(seed=[6, 7])
            self.assertNotEqual(first, env.call("task_description"))
            self.assertEqual(env.calls["task_description"], 2)
            self.assertEqual(env.received, ((), {"seed": [6, 7]}))

    def test_other_methods_and_parameterized_calls_are_not_cached(self):
        env = FakeVector()
        with existing_observations(env, None):
            for _ in range(2):
                self.assertEqual(env.call("other", 8, name2="x"), ("other", (8,), {"name2": "x"}))
                env.call("task_description", mode="custom")
        self.assertEqual(env.calls, {"other": 2, "task_description": 2})

    def test_mutable_or_nonstring_language_replies_retain_native_behavior(self):
        for result in (["mutable"], (None,), {"task": "value"}):
            env = FakeVector()
            count = []
            env.call = lambda *a, **k: count.append(a) or result
            with existing_observations(env, None):
                self.assertIs(env.call("task_description"), result)
                self.assertIs(env.call("task_description"), result)
            self.assertEqual(len(count), 2)

    def test_exception_restores_original_instance_methods(self):
        env = FakeVector()
        original = {name: getattr(env, name) for name in ("call", "reset", "step")}
        with self.assertRaisesRegex(RuntimeError, "rollout failed"):
            with existing_observations(env, None):
                env.reset()
                env.call("task_description")
                raise RuntimeError("rollout failed")
        for name, method in original.items():
            self.assertEqual(getattr(env, name), method)
            self.assertNotIn(name, vars(env))

    @unittest.skipIf(np is None, "NumPy is unavailable")
    def test_render_pixels_match_native_flip_and_do_not_mutate_observations(self):
        env = FakeVector()
        with existing_observations(env, "z_front"):
            observation, _ = env.reset()
            expected = env.native_frames()
            frames = env.call("render")
            for actual, native in zip(frames, expected):
                np.testing.assert_array_equal(actual, native)
                self.assertTrue(actual.flags.c_contiguous)
            frames[0].fill(0)
            np.testing.assert_array_equal(observation["pixels"]["z_front"][0], np.arange(18).reshape(2,3,3)+1)
            self.assertEqual(env.calls["render"], 0)
            actions = np.zeros((2, 7))
            env.step(actions)
            self.assertIs(env.received, actions)
            np.testing.assert_array_equal(env.call("render")[0], env.native_frames()[0])
            self.assertFalse(np.array_equal(expected[0], env.call("render")[0]))

    @unittest.skipIf(np is None, "NumPy is unavailable")
    def test_missing_or_nonuint8_camera_falls_back_to_native_render(self):
        for key in (None, "missing", "z_front"):
            env = FakeVector()
            with existing_observations(env, key):
                env.reset()
                if key == "z_front":
                    # Change the next step's source dtype before the wrapper sees it.
                    original = env.observation
                    env.observation = lambda: {"pixels": {"z_front": original()["pixels"]["z_front"].astype(float)}}
                    env.step(None)
                env.call("render")
                self.assertEqual(env.calls["render"], 1)

    @unittest.skipIf(np is None, "NumPy is unavailable")
    def test_sync_subenvironment_renderers_use_the_same_pixels_and_are_restored(self):
        env = FakeVector(sync=True)
        originals = [child.render for child in env.envs]
        with existing_observations(env, "z_front"):
            env.reset()
            for i, child in enumerate(env.envs):
                np.testing.assert_array_equal(child.render(), env.native_frames()[i])
        self.assertEqual([child.render for child in env.envs], originals)

    def test_native_config_identity_and_custom_hub_are_checked(self):
        class Config:
            camera_name = ["agentview", "hand"]
            camera_name_mapping = {"agentview": "z_front", "hand": "a_hand"}
            hub_path = None
        class Custom(Config):
            pass
        modules = {"lerobot.envs.configs": SimpleNamespace(LiberoEnv=Config),
                   "lerobot.envs.libero": SimpleNamespace(LiberoEnv=ReferenceRenderer)}
        with patch.dict(sys.modules, modules):
            self.assertEqual(standard_libero_camera(Config()), (True, "z_front"))
            self.assertEqual(standard_libero_camera(Custom()), (False, None))
            cfg = Config()
            cfg.hub_path = "owner/custom-env"
            self.assertEqual(standard_libero_camera(cfg), (False, None))

    def test_unknown_or_decorated_renderer_falls_back(self):
        self.assertTrue(observation_render_supported(ReferenceRenderer.render))
        self.assertFalse(observation_render_supported(lambda self: "different renderer"))
        decorated = wraps(ReferenceRenderer.render)(lambda self: None)
        self.assertFalse(observation_render_supported(decorated))

    def test_default_comma_separated_cameras_and_custom_mapping(self):
        class Config:
            camera_name = "agentview_image,robot0_eye_in_hand_image"
            camera_name_mapping = None
        modules = {"lerobot.envs.configs": SimpleNamespace(LiberoEnv=Config),
                   "lerobot.envs.libero": SimpleNamespace(LiberoEnv=ReferenceRenderer)}
        with patch.dict(sys.modules, modules):
            config = Config()
            self.assertEqual(standard_libero_camera(config), (True, "image"))
            config.camera_name_mapping = {"agentview_image": "z_front", "robot0_eye_in_hand_image": "a_hand"}
            self.assertEqual(standard_libero_camera(config), (True, "z_front"))
            config.is_libero_plus = True
            self.assertEqual(standard_libero_camera(config), (False, None))

    def test_unknown_evaluator_api_does_not_block_native_execution(self):
        for evaluator in (SimpleNamespace(), SimpleNamespace(make_env=lambda: {}, rollout=lambda: None)):
            with standard_libero_rollouts(evaluator):
                pass

    def test_factory_and_rollout_hooks_are_scoped_and_threads_have_separate_caches(self):
        envs = [FakeVector("first"), FakeVector("second")]
        def factory(cfg):
            return {"suite": dict(enumerate(envs))}
        def rollout(env):
            env.reset()
            replies = [env.call("task_description") for _ in range(4)]
            return replies
        evaluator = SimpleNamespace(make_env=factory, rollout=rollout)
        with patch("evaluation_efficiency.standard_libero_camera", return_value=(True, None)), \
             standard_libero_rollouts(evaluator):
            evaluator.make_env("native-config")
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(evaluator.rollout, envs))
            self.assertNotEqual(results[0][0], results[1][0])
            self.assertTrue(all(all(reply == values[0] for reply in values) for values in results))
            self.assertEqual([env.calls["task_description"] for env in envs], [1, 1])
        self.assertIs(evaluator.make_env, factory)
        self.assertIs(evaluator.rollout, rollout)

    def test_uncertified_environment_keeps_native_calls(self):
        env = FakeVector()
        evaluator = SimpleNamespace(make_env=lambda cfg: {"task": {0: env}},
                                    rollout=lambda env: [env.call("task_description") for _ in range(3)])
        with patch("evaluation_efficiency.standard_libero_camera", return_value=(False, None)), \
             standard_libero_rollouts(evaluator):
            evaluator.make_env("custom")
            evaluator.rollout(env)
        self.assertEqual(env.calls["task_description"], 3)


if __name__ == "__main__":
    unittest.main()
