"""Run native evaluation with isolated task state and shared, serialized weights."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from functools import wraps
from inspect import signature
import logging
from threading import Lock, RLock, local
from types import MethodType, ModuleType, FunctionType, BuiltinFunctionType
from weakref import WeakMethod


def weak_method(method):
    reference = WeakMethod(method)

    @wraps(method.__func__)
    def call(*args, **kwargs):
        return reference()(*args, **kwargs)
    return call


def task_compiled_callable(original, memo, name):
    """Resolve a compiled instance method without unwrapping user decorators."""
    seen = set()
    while original is not None and id(original) not in seen:
        seen.add(id(original))
        if isinstance(original, MethodType) and id(original.__self__) in memo:
            return weak_method(MethodType(original.__func__, memo[id(original.__self__)]))
        if id(original) in memo:
            return memo[id(original)]
        code = getattr(original, "__code__", None)
        # PyTorch's default compiler can wrap bound methods in _wrap_inline's
        # transparent inner(*args, **kwargs). Its __wrapped__ keeps the method's
        # owner; inspect.unwrap would continue past that method and lose it.
        # Restrict this to Torch's wrapper: unwrapping an arbitrary user
        # decorator could silently change the policy's inference behavior.
        if code is None or code.co_name != "inner" or not code.co_filename.replace("\\", "/").endswith(
            "/torch/_dynamo/external_utils.py"
        ):
            break
        original = getattr(original, "__wrapped__", None)
    raise TypeError(f"cannot isolate compiled method {name}")


def parameter_graph(policy):
    """Find declared weights, including frozen modules outside _modules.

    World models can deliberately keep a VAE/text encoder unregistered to
    exclude it from training checkpoints. Sharing only policy.parameters()
    would duplicate those large frozen weights for every evaluation task.
    Tensor buffers remain outside the memo and are still copied per task.
    """
    memo = {id(parameter): parameter for parameter in policy.parameters()}
    modules = list(policy.modules())
    try:
        import torch
    except ImportError:
        return memo, modules
    if not hasattr(torch, "nn") or not hasattr(torch.nn, "Module"):
        return memo, modules
    seen, module_ids = set(), {id(module) for module in modules}
    pending = [policy]
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, torch.nn.Parameter):
            memo[id(value)] = value
            continue
        if isinstance(value, torch.Tensor):
            continue  # Mutable caches/buffers require task-local copies.
        if isinstance(value, (str, bytes, int, float, complex, bool, type(None), type,
                              ModuleType, FunctionType, BuiltinFunctionType, MethodType)):
            continue
        if isinstance(value, torch.nn.Module):
            for parameter in value.parameters(recurse=False):
                memo[id(parameter)] = parameter
            if id(value) not in module_ids:
                modules.append(value)
                module_ids.add(id(value))
        if isinstance(value, dict):
            items = value.values()
        elif isinstance(value, (list, tuple, set, frozenset)):
            items = value
        else:
            try:
                items = vars(value).values()
            except TypeError:
                continue
        pending.extend(items)
    return memo, modules


def task_policy(policy, lock):
    """Copy all mutable policy state, sharing declared model parameters.

    Copy the entire module graph, including buffers and nested streaming caches:
    resetting just the top-level action queue is insufficient for world models.
    Inference is serialized across copies, so only one task needs activation
    memory at a time. The template is never used for a native rollout.
    """
    try:
        memo, modules = parameter_graph(policy)
        auxiliary = len(modules) - len(list(policy.modules()))
        if auxiliary:
            logging.info("Parallel evaluation shares declared weights from %s auxiliary modules too.", auxiliary)
        copied = deepcopy(policy, memo)
        if copied is policy:
            raise TypeError("the policy returned itself from deepcopy")
        for module in modules:
            target = memo.get(id(module))
            if target is None or target is module:
                raise TypeError(f"cannot isolate the {type(module).__name__} module")
            for name, value in vars(module).items():
                original = getattr(value, "_torchdynamo_orig_callable", None)
                if original is None:
                    continue
                # A compiled function's closure can still point at the template
                # after deepcopy. Rebind its eager method to the task's module.
                # Never let a streaming cache escape into the shared template.
                replacement = task_compiled_callable(original, memo, name)
                setattr(target, name, replacement)
                logging.info("Parallel evaluation uses a task-local eager method for %s.%s",
                             type(module).__name__, name)
    except Exception as error:
        raise RuntimeError(
            f"Cannot isolate {type(policy).__name__} for parallel evaluation: {error}. "
            "Use parallel_tasks=1 for this policy/runtime."
        ) from error

    device = str(getattr(getattr(copied, "config", None), "device", "cpu"))

    def guarded(method, synchronize):
        # A wrapper stored on an instance must not retain its bound method;
        # that cycle would keep task caches alive until a later GC pass.
        bound = isinstance(method, MethodType)
        reference = WeakMethod(method) if bound else lambda: method

        @wraps(method.__func__ if bound else method)
        def call(*args, **kwargs):
            with lock:
                result = reference()(*args, **kwargs)
                # Device kernels are asynchronous. Complete them before another
                # task can dispatch inference and allocate its activations.
                if synchronize and device.startswith("cuda"):
                    import torch
                    torch.cuda.synchronize(device)
                elif synchronize and device.startswith("mps"):
                    import torch
                    torch.mps.synchronize()
                return result
        return call

    for name in ("reset", "select_action", "decode_predicted_latents", "_decode_predicted_video"):
        if callable(method := getattr(copied, name, None)):
            setattr(copied, name, guarded(method, name != "reset"))
    return copied


@contextmanager
def inference_context(evaluator, policy):
    """PyTorch inference contexts are thread-local; establish them in each task."""
    with ExitStack() as context:
        torch = getattr(evaluator, "torch", None)
        if torch is not None:
            context.enter_context(torch.no_grad())
            config = policy.config
            if getattr(config, "use_amp", False):
                context.enter_context(torch.autocast(device_type=torch.device(config.device).type))
        yield


def track_parallel_rollouts(evaluator):
    task_state = local()
    lock = Lock()
    policy_lock = RLock()
    completed = 0
    isolate_policy = True
    original_task = evaluator.run_one
    original_rollout = evaluator.rollout
    original_trange = evaluator.trange
    original_all = evaluator.eval_policy_all
    call_signature = signature(original_all)

    @wraps(original_all)
    def evaluate_all(*args, **kwargs):
        nonlocal completed, isolate_policy
        bound = call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        envs = bound.arguments["envs"]
        tasks = [env for group in envs.values() for env in group.values()]
        requested = bound.arguments["max_parallel_tasks"]
        episodes = bound.arguments["n_episodes"]
        if type(requested) is not int or requested < 1 or type(episodes) is not int or episodes < 1:
            raise ValueError("Parallel task and episode counts must be positive integers.")
        if not tasks:
            raise ValueError("The benchmark did not provide any evaluation tasks.")
        bound.arguments["max_parallel_tasks"] = min(requested, len(tasks))
        isolate_policy = bound.arguments["max_parallel_tasks"] > 1
        completed = 0
        print(f"[Eval progress] Total episodes: {len(tasks) * episodes}", flush=True)
        print(f"Parallel evaluation tasks: {min(requested, len(tasks))} "
              f"({requested} requested, {len(tasks)} available)", flush=True)
        try:
            return original_all(*bound.args, **bound.kwargs)
        except BaseException:
            # Upstream closes consumed futures only. Also close environments
            # whose task was waiting/running when another task failed.
            for env in tasks:
                try:
                    env.close()
                except Exception:
                    logging.exception("Could not close an evaluation environment after failure")
            raise

    @wraps(original_task)
    def run_one(task_group, task_id, env, **options):
        task_state.remaining = options["n_episodes"]
        task_state.batch = env.num_envs
        task_state.label = f"{task_group}/{task_id}"
        try:
            if "max_episodes_rendered" in options:
                # The final vector batch can contain unused episode slots.
                # Upstream discards their metrics; do not save extra videos.
                options["max_episodes_rendered"] = min(options["max_episodes_rendered"], options["n_episodes"])
            policy = options["policy"]
            # TinyVLA already owns thread-local queues and a locked shared
            # worker. Copying its subprocess/channel would break that protocol.
            if isolate_policy and getattr(policy.config, "type", None) != "tinyvla":
                with policy_lock:
                    options["policy"] = task_policy(policy, policy_lock)
            for name in ("env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor"):
                options[name] = deepcopy(options[name])
            with inference_context(evaluator, options["policy"]):
                return original_task(task_group, task_id, env, **options)
        finally:
            task_state.remaining = 0
            task_state.label = None

    @wraps(original_rollout)
    def rollout(*args, **kwargs):
        nonlocal completed
        result = original_rollout(*args, **kwargs)
        if getattr(task_state, "remaining", 0):
            count = min(task_state.batch, task_state.remaining)
            task_state.remaining -= count
            with lock:
                completed += count
                print(f"[Eval progress] Rollout ended: {task_state.label}\n"
                      f"[Eval progress] Episodes completed: {completed}", flush=True)
        return result

    @wraps(original_trange)
    def trange(*args, **kwargs):
        description = kwargs.get("desc") or ""
        task = getattr(task_state, "label", None)
        if task and description.startswith("Running rollout with at most "):
            kwargs["desc"] = description.replace("Running rollout", f"Running rollout [{task}]", 1)
        return original_trange(*args, **kwargs)

    evaluator.run_one = run_one
    evaluator.rollout = rollout
    evaluator.trange = trange
    evaluator.eval_policy_all = evaluate_all

    def restore():
        evaluator.run_one = original_task
        evaluator.rollout = original_rollout
        evaluator.trange = original_trange
        evaluator.eval_policy_all = original_all
    return restore


def main():
    import sys
    from evaluation_cameras import prepare_evaluation_runtime, restore_camera_arguments, simulator_directory

    prepare_evaluation_runtime(sys.argv[1:])
    from lerobot.scripts import lerobot_eval
    from robomme_split import simulator_context
    from simulator_worker import simulator_context as source_context

    restore = track_parallel_rollouts(lerobot_eval)
    try:
        restore_camera_arguments()
        with simulator_directory(sys.argv[1:]), simulator_context(), source_context():
            return lerobot_eval.main()
    finally:
        restore()


if __name__ == "__main__":
    main()
