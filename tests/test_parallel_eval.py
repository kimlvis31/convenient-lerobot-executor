"""Concurrent rollout checks: mutable state isolation, shared weights and progress."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from functools import wraps
from importlib.util import find_spec
import io
from threading import Barrier, Lock, RLock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from adapters import Stage
from parallel_eval import task_policy, track_parallel_rollouts
from runner import Progress


class Weight:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.lock = Lock()

    def __deepcopy__(self, memo):
        raise AssertionError("Model parameters must be shared, not copied")


class Cache:
    def __init__(self):
        self.history = []

    def sample(self, task):
        self.history.append(task)
        return task


class Policy:
    def __init__(self, kind="smolvla"):
        self.config = SimpleNamespace(type=kind, use_amp=False)
        self.weight = Weight()
        self.cache = Cache()
        self.reset()

    def parameters(self):
        return iter((self.weight,))

    def modules(self):
        return iter((self, self.cache))

    def reset(self):
        self.queue = deque()

    def select_action(self, task):
        with self.weight.lock:
            self.weight.active += 1
            self.weight.peak = max(self.weight.peak, self.weight.active)
        try:
            if not self.queue:
                self.queue.extend((task, task))
            time.sleep(0.001)
            return self.cache.sample(self.queue.popleft())
        finally:
            with self.weight.lock:
                self.weight.active -= 1


class Environment:
    num_envs = 2

    def __init__(self, task, barrier=None):
        self.task = task
        self.barrier = barrier
        self.closed = False

    def close(self):
        self.closed = True


def evaluator_fixture():
    evaluator = SimpleNamespace()

    def rollout(env, policy, processors):
        action = policy.select_action(env.task)
        for processor in processors:
            processor.setdefault("task", env.task)
            if processor["task"] != env.task:
                raise AssertionError("Processor state leaked across tasks")
        # Both simulations must reach this point concurrently, even though
        # shared-weight inference itself is serialized.
        if env.barrier:
            env.barrier.wait(timeout=5)
        if action != env.task:
            raise AssertionError("Policy action queue leaked across tasks")
        return action

    def run_one(group, task, env, *, policy, n_episodes, **options):
        policy.reset()
        processors = [options[name] for name in
                      ("env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor")]
        for _ in range((n_episodes + env.num_envs - 1) // env.num_envs):
            evaluator.rollout(env, policy, processors)
        return (group, task, list(policy.cache.history))

    def evaluate_all(envs, policy, n_episodes, *, max_parallel_tasks=1):
        options = {name: {} for name in
                   ("env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor")}
        with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            futures = [executor.submit(evaluator.run_one, group, task, env,
                                       policy=policy, n_episodes=n_episodes, **options)
                       for group, tasks in envs.items() for task, env in tasks.items()]
            return [future.result() for future in futures]

    evaluator.run_one = run_one
    evaluator.rollout = rollout
    evaluator.trange = lambda *args, **kwargs: range(*args)
    evaluator.eval_policy_all = evaluate_all
    return evaluator


class ParallelEvaluationTests(unittest.TestCase):
    def instrument(self):
        evaluator = evaluator_fixture()
        restore = track_parallel_rollouts(evaluator)
        self.addCleanup(restore)
        return evaluator

    def test_tasks_overlap_but_keep_queues_nested_caches_and_processors_private(self):
        evaluator = self.instrument()
        policy = Policy()
        barrier = Barrier(2)
        envs = {"suite": {i: Environment(i, barrier) for i in (1, 2)}}
        with redirect_stdout(io.StringIO()) as output:
            result = evaluator.eval_policy_all(envs, policy, 3, max_parallel_tasks=2)
        self.assertEqual(result, [("suite", 1, [1, 1]), ("suite", 2, [2, 2])])
        self.assertEqual(policy.cache.history, [])
        self.assertEqual(list(policy.queue), [])
        self.assertEqual(policy.weight.peak, 1)
        self.assertIn("[Eval progress] Total episodes: 6", output.getvalue())
        counts = [int(line.rsplit(":", 1)[1]) for line in output.getvalue().splitlines()
                  if line.startswith("[Eval progress] Episodes completed:")]
        self.assertEqual(counts[-1], 6)
        self.assertEqual(counts, sorted(set(counts)))

    def test_tasks_reused_on_a_worker_do_not_inherit_previous_task_state(self):
        evaluator = self.instrument()
        envs = {"suite": {i: Environment(i) for i in range(5)}}
        with redirect_stdout(io.StringIO()):
            result = evaluator.eval_policy_all(envs, Policy(), 1, max_parallel_tasks=2)
        self.assertEqual(result, [("suite", i, [i]) for i in range(5)])

    def test_actual_task_count_caps_workers_and_resets_progress_for_each_evaluation(self):
        evaluator = self.instrument()
        envs = {"one": {0: Environment(0)}}
        for _ in range(2):
            with redirect_stdout(io.StringIO()) as output:
                evaluator.eval_policy_all(envs, Policy(), 1, max_parallel_tasks=20)
            self.assertIn("Parallel evaluation tasks: 1 (20 requested, 1 available)", output.getvalue())
            self.assertIn("[Eval progress] Episodes completed: 1", output.getvalue())

    def test_render_limit_excludes_unused_final_batch_slots_and_preserves_lower_limits(self):
        evaluator = evaluator_fixture()
        evaluator.run_one = lambda group, task, env, **options: options["max_episodes_rendered"]
        restore = track_parallel_rollouts(evaluator)
        self.addCleanup(restore)
        for requested, expected in ((10, 3), (2, 2), (0, 0)):
            with self.subTest(requested=requested):
                actual = evaluator.run_one("suite", 0, Environment(0), policy=Policy(), n_episodes=3,
                                           max_episodes_rendered=requested, env_preprocessor={},
                                           env_postprocessor={}, preprocessor={}, postprocessor={})
                self.assertEqual(actual, expected)

    def test_one_available_task_does_not_require_copying_policy_runtime_objects(self):
        evaluator = self.instrument()
        policy = Policy()
        policy.channel = Lock()
        with redirect_stdout(io.StringIO()):
            result = evaluator.eval_policy_all({"one": {0: Environment(0)}}, policy, 1, max_parallel_tasks=20)
        self.assertEqual(result, [("one", 0, [0])])

    def test_finished_task_policy_does_not_wait_for_garbage_collection(self):
        copied = task_policy(Policy(), RLock())
        reference = weakref.ref(copied)
        del copied
        self.assertIsNone(reference())

    def test_accelerator_dispatch_finishes_before_releasing_the_shared_inference_lock(self):
        events = []
        policy = Policy()
        policy.config.device = "cuda:1"
        copied = task_policy(policy, RLock())
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: events.append(device)))
        with patch.dict("sys.modules", {"torch": torch}):
            copied.reset()
            self.assertEqual(events, [])
            copied.select_action(1)
            self.assertEqual(events, ["cuda:1"])

    def test_compiled_bound_method_uses_the_task_cache_instead_of_template(self):
        policy = Policy()

        def compiled(task):
            return policy.cache.sample(task)
        compiled._torchdynamo_orig_callable = policy.cache.sample
        policy.cache.sample = compiled
        copied = task_policy(policy, RLock())
        self.assertEqual(copied.select_action(7), 7)
        self.assertEqual(copied.cache.history, [7])
        self.assertEqual(policy.cache.history, [])
        self.assertIs(copied.weight, policy.weight)
        reference = weakref.ref(copied.cache)
        del copied
        self.assertIsNone(reference())

    def test_compiled_user_decorator_is_not_silently_discarded(self):
        policy = Policy()
        original = policy.cache.sample

        @wraps(original)
        def user_wrapper(task):
            return original(task) + 10

        def compiled(task):
            return user_wrapper(task)
        compiled._torchdynamo_orig_callable = user_wrapper
        policy.cache.sample = compiled
        with self.assertRaisesRegex(RuntimeError, "cannot isolate compiled method sample"):
            task_policy(policy, RLock())

    def test_uncopyable_policy_fails_with_actionable_error_and_closes_all_environments(self):
        evaluator = self.instrument()
        policy = Policy()
        policy.uncopyable = Lock()
        envs = {"suite": {i: Environment(i) for i in range(3)}}
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Use parallel_tasks=1"):
            evaluator.eval_policy_all(envs, policy, 1, max_parallel_tasks=2)
        self.assertTrue(all(env.closed for env in envs["suite"].values()))

    def test_failure_does_not_publish_completed_episodes(self):
        evaluator = evaluator_fixture()
        evaluator.rollout = lambda *a, **kw: (_ for _ in ()).throw(ValueError("physics failed"))
        restore = track_parallel_rollouts(evaluator)
        self.addCleanup(restore)
        with redirect_stdout(io.StringIO()) as output, self.assertRaisesRegex(ValueError, "physics failed"):
            evaluator.eval_policy_all({"suite": {0: Environment(0)}}, Policy(), 1, max_parallel_tasks=2)
        self.assertNotIn("Episodes completed:", output.getvalue())

    def test_tinyvla_keeps_its_thread_local_worker_instead_of_copying_its_channel(self):
        evaluator = evaluator_fixture()
        policy = Policy("tinyvla")
        policy.channel = Lock()
        evaluator.run_one = lambda group, task, env, **opts: opts["policy"]
        restore = track_parallel_rollouts(evaluator)
        self.addCleanup(restore)
        with redirect_stdout(io.StringIO()):
            result = evaluator.eval_policy_all({"suite": {0: Environment(0)}}, policy, 1, max_parallel_tasks=2)
        self.assertIs(result[0], policy)

    def test_restore_reinstates_all_original_evaluator_functions(self):
        evaluator = evaluator_fixture()
        original = vars(evaluator).copy()
        restore = track_parallel_rollouts(evaluator)
        restore()
        self.assertEqual(vars(evaluator), original)

    def test_actual_episode_total_updates_unknown_or_estimated_progress(self):
        for planned in (None, 1):
            stage = Stage("eval", "lerobot", [], (), planned)
            progress = Progress(stage, {"eval": {"episodes": 3, "parallel_tasks": 2}})
            self.assertTrue(progress.update("[Eval progress] Total episodes: 9", 0))
            self.assertEqual((progress.total, stage.units), (9, 9))
            self.assertTrue(progress.update("[Eval progress] Episodes completed: 2", 1))
            self.assertAlmostEqual(progress.fraction(), 2 / 9)
            self.assertFalse(progress.update("[Eval progress] Total episodes: 0", 2))

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_real_module_parameters_share_storage_but_buffers_and_nested_state_do_not(self):
        import torch

        class NativePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(type="native", use_amp=False)
                self.model = torch.nn.Linear(2, 2)
                self.model.register_buffer("counter", torch.zeros(1))
                self.model.cache = []
                frozen = torch.nn.Linear(2, 2).requires_grad_(False)
                frozen.register_buffer("counter", torch.zeros(1))
                frozen.cache = []
                object.__setattr__(self, "frozen", frozen)
                nested = torch.nn.Linear(2, 2).requires_grad_(False)
                nested.register_buffer("counter", torch.zeros(1))
                self.pipeline = SimpleNamespace(components={"text_encoder": nested})
                self.queue = deque()

            def reset(self):
                self.queue.clear()

            def select_action(self, task):
                self.model.counter.add_(1)
                self.model.cache.append(task)
                self.frozen.counter.add_(1)
                self.frozen.cache.append(task)
                self.pipeline.components["text_encoder"].counter.add_(1)
                self.queue.append(task)
                return self.queue.popleft()

        template = NativePolicy()
        self.assertNotIn(id(template.frozen.weight), {id(p) for p in template.parameters()})
        one, two = (task_policy(template, RLock()) for _ in range(2))
        self.assertIs(one.model.weight, template.model.weight)
        self.assertEqual(one.model.weight.data_ptr(), two.model.weight.data_ptr())
        self.assertNotEqual(one.model.counter.data_ptr(), two.model.counter.data_ptr())
        self.assertIs(one.frozen.weight, template.frozen.weight)
        self.assertIs(two.frozen.weight, template.frozen.weight)
        self.assertIs(one.pipeline.components["text_encoder"].weight,
                      template.pipeline.components["text_encoder"].weight)
        self.assertNotEqual(one.frozen.counter.data_ptr(), two.frozen.counter.data_ptr())
        self.assertNotEqual(one.pipeline.components["text_encoder"].counter.data_ptr(),
                            two.pipeline.components["text_encoder"].counter.data_ptr())
        one.select_action(1)
        two.select_action(2)
        self.assertEqual((one.model.cache, two.model.cache, template.model.cache), ([1], [2], []))
        self.assertEqual(template.model.counter.item(), 0)
        self.assertEqual((one.frozen.cache, two.frozen.cache, template.frozen.cache), ([1], [2], []))
        self.assertEqual(template.frozen.counter.item(), 0)
        self.assertEqual(template.pipeline.components["text_encoder"].counter.item(), 0)

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_torch_contexts_are_established_in_each_worker_and_restored_afterward(self):
        import torch
        from parallel_eval import inference_context

        policy = SimpleNamespace(config=SimpleNamespace(device="cpu", use_amp=True))

        def worker():
            self.assertTrue(torch.is_grad_enabled())
            with inference_context(SimpleNamespace(torch=torch), policy):
                self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assertTrue(torch.is_grad_enabled())
            self.assertFalse(torch.is_autocast_enabled("cpu"))
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: worker(), range(2)))

    @unittest.skipUnless(find_spec("torch"), "PyTorch is not installed")
    def test_real_compiled_method_rebinds_nested_tensor_cache(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.zeros(1))

            def sample(self, task):
                self.counter.add_(1)
                return task

        class NativePolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Model()
                self.model.sample = torch.compile(self.model.sample, mode="reduce-overhead")

            def select_action(self, task):
                return self.model.sample(task)

        template = NativePolicy()
        copied = task_policy(template, RLock())
        copied.select_action(torch.ones(1))
        self.assertEqual(copied.model.counter.item(), 1)
        self.assertEqual(template.model.counter.item(), 0)


if __name__ == "__main__":
    unittest.main()
