"""AI-written checks for preserving concurrency with bounded environment lifetime."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
from threading import Event
import signal
import time
from types import SimpleNamespace
import unittest

from adapters import NativeAdapter
from evaluation_efficiency import shared_environment_arguments
from parallel_eval import cancel_on_interrupt, track_parallel_rollouts
from tests.test_parallel_eval import Policy
from tests.test_parallel_eval import evaluator_fixture


class EvaluationPlanningTests(unittest.TestCase):
    def test_saved_commands_get_the_same_default_without_overriding_config_files(self):
        arguments=["--env.type=libero", "--env.max_parallel_tasks=4", "--eval.batch_size=5", "--eval.n_episodes=5"]
        self.assertEqual(shared_environment_arguments(arguments), ["--eval.use_async_envs=false"])
        self.assertEqual(len(arguments), 4)
        for extra in (["--eval.use_async_envs=true"], ["--eval.use_async_envs", "true"],
                      ["--config_path=custom.json"], ["--env.hub_path=owner/custom"]):
            self.assertEqual(shared_environment_arguments([*arguments,*extra]), [])

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
    def test_interrupt_cancels_in_process_task_threads_before_pool_shutdown_waits(self):
        ready = Barrier(3)
        timing = {}
        evaluator = evaluator_fixture()

        class Env:
            num_envs = 1
            closed = False

            def step(self):
                time.sleep(.005)

            def close(self):
                self.closed = True

        def run_one(group, task, env, **options):
            ready.wait(timeout=5)
            for _ in range(200):
                env.step()

        def evaluate_all(envs, policy, n_episodes, max_parallel_tasks=1):
            options = {name:{} for name in ("env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor")}
            with ThreadPoolExecutor(max_workers=2) as pool:
                for task,env in envs["suite"].items():
                    pool.submit(evaluator.run_one,"suite",task,env,policy=policy,n_episodes=1,**options)
                ready.wait(timeout=5)
                timing["interrupted"] = time.monotonic()
                signal.raise_signal(signal.SIGINT)

        evaluator.run_one, evaluator.eval_policy_all = run_one, evaluate_all
        envs = {0:Env(), 1:Env()}
        previous = signal.getsignal(signal.SIGINT)
        if previous is not signal.default_int_handler:
            self.skipTest("Requires Python's normal interrupt handler")
        restore = track_parallel_rollouts(evaluator)
        try:
            with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                evaluator.eval_policy_all({"suite":envs}, Policy(), 1, max_parallel_tasks=2)
        finally:
            restore()
        self.assertLess(time.monotonic()-timing["interrupted"], .5)
        self.assertTrue(all(env.closed for env in envs.values()))
        self.assertIs(signal.getsignal(signal.SIGINT), previous)

    def test_cancelling_other_tasks_does_not_hide_the_original_failure(self):
        ready = Barrier(2)
        evaluator = evaluator_fixture()

        class Env:
            num_envs = 1

            def step(self):
                time.sleep(.005)

            def close(self):
                pass

        def run_one(group, task, env, **options):
            ready.wait(timeout=5)
            if task == 1:
                raise ValueError("original policy failure")
            for _ in range(200):
                env.step()

        def evaluate_all(envs, policy, n_episodes, max_parallel_tasks=1):
            options = {name:{} for name in ("env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor")}
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(evaluator.run_one,"suite",task,env,policy=policy,n_episodes=1,**options)
                         for task,env in envs["suite"].items()]
                # Consume the cancelled task before the original failing task.
                return futures[0].result()

        evaluator.run_one, evaluator.eval_policy_all = run_one, evaluate_all
        restore = track_parallel_rollouts(evaluator)
        try:
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "original policy failure"):
                evaluator.eval_policy_all({"suite":{0:Env(),1:Env()}}, Policy(), 1, max_parallel_tasks=2)
        finally:
            restore()

    def test_custom_interrupt_handler_is_not_replaced(self):
        previous = signal.getsignal(signal.SIGINT)
        custom = lambda *_: None
        try:
            signal.signal(signal.SIGINT, custom)
            with cancel_on_interrupt(Event()):
                self.assertIs(signal.getsignal(signal.SIGINT), custom)
        finally:
            signal.signal(signal.SIGINT, previous)

    def test_failure_still_closes_environments_with_readonly_close_methods(self):
        class Env:
            __slots__ = ("closed",)

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        evaluator = evaluator_fixture()
        def fail(envs, policy, n_episodes, max_parallel_tasks=1):
            raise RuntimeError("task failed")
        evaluator.eval_policy_all = fail
        env = Env()
        restore = track_parallel_rollouts(evaluator)
        try:
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "task failed"):
                evaluator.eval_policy_all({"suite":{0:env}}, Policy(), 1)
        finally:
            restore()
        self.assertTrue(env.closed)

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
