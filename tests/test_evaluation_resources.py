"""AI-written checks for preserving concurrency with bounded environment lifetime."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
import time
from types import SimpleNamespace
import unittest

from adapters import NativeAdapter
from parallel_eval import track_parallel_rollouts
from tests.test_parallel_eval import Policy


class EvaluationPlanningTests(unittest.TestCase):
    def stage(self, extra=(), kind="libero", parallel=4):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {"model":"smolvla", "eval":{"policy":None, "env":kind,
                "tasks":["libero_spatial"], "task_ids":[0,1,2,3], "episodes":5,
                "batch":5, "parallel_tasks":parallel, "seed":0, "device":"cuda", "extra":list(extra)}}
            return NativeAdapter(settings, "test", root, root/"externals", {"native":"lerobot"}).evaluate(False)

    def test_default_preserves_all_task_and_episode_slots_without_per_episode_workers(self):
        stage = self.stage()
        for value in ("--env.max_parallel_tasks=4", "--eval.batch_size=5", "--eval.n_episodes=5",
                      "--eval.use_async_envs=false"):
            self.assertIn(value, stage.arguments)
        self.assertIn("eval.use_async_envs", stage.auto_flags)
        self.assertEqual(stage.units, 20)

    def test_explicit_async_choice_is_preserved(self):
        for value in ("true", "false"):
            stage = self.stage([f"--eval.use_async_envs={value}"])
            self.assertEqual([a for a in stage.arguments if a.startswith("--eval.use_async_envs=")],
                             [f"--eval.use_async_envs={value}"])
            self.assertNotIn("eval.use_async_envs", stage.auto_flags)

    def test_single_task_execution_and_custom_or_other_benchmarks_keep_native_defaults(self):
        for options in ({"parallel":1}, {"kind":"libero_plus"},
                        {"extra":["--env.hub_path=owner/custom"]},
                        {"extra":["--env.is_libero_plus=true"]}):
            with self.subTest(options=options):
                self.assertFalse(any(a.startswith("--eval.use_async_envs=") for a in self.stage(**options).arguments))


class EvaluationLifetimeTests(unittest.TestCase):
    def test_next_task_cannot_allocate_before_previous_environment_has_closed(self):
        state = SimpleNamespace(active=0, peak=0, lock=Lock(), barrier=Barrier(2))

        class Env:
            num_envs = 1

            def __init__(self, index):
                self.index, self.started, self.close_calls = index, False, 0

            def start(self):
                with state.lock:
                    self.started = True
                    state.active += 1
                    state.peak = max(state.peak, state.active)
                if self.index < 2:
                    state.barrier.wait(timeout=5)

            def close(self):
                time.sleep(.01)
                with state.lock:
                    self.close_calls += 1
                    if self.started:
                        state.active -= 1

        evaluator = SimpleNamespace(rollout=lambda *a,**k: None, trange=range)

        def run_one(group, task, env, **kwargs):
            env.start()
            time.sleep(.005)
            return task

        def evaluate_all(envs, policy, n_episodes, max_parallel_tasks=1):
            options = {name:{} for name in ("env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor")}
            with ThreadPoolExecutor(max_workers=max_parallel_tasks) as pool:
                pending = {pool.submit(evaluator.run_one, "suite", task, env, policy=policy,
                                       n_episodes=n_episodes, **options):env for task,env in envs["suite"].items()}
                for future in as_completed(pending):
                    future.result()
                    time.sleep(.02)  # A slow result consumer must not retain old environments.
                    pending[future].close()

        evaluator.run_one, evaluator.eval_policy_all = run_one, evaluate_all
        environments = {i:Env(i) for i in range(8)}
        originals = [env.close for env in environments.values()]
        restore = track_parallel_rollouts(evaluator)
        try:
            with redirect_stdout(io.StringIO()):
                evaluator.eval_policy_all({"suite":environments}, Policy(), 1, max_parallel_tasks=2)
        finally:
            restore()
        self.assertEqual(state.peak, 2)
        self.assertEqual(state.active, 0)
        self.assertEqual([env.close_calls for env in environments.values()], [1]*8)
        self.assertEqual([env.close for env in environments.values()], originals)


if __name__ == "__main__":
    unittest.main()
