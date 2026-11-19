"""Run LeRobot evaluation with task-local processors and aggregate rollout progress."""

from copy import deepcopy
from functools import wraps
from threading import Lock, local


def track_parallel_rollouts(evaluator):
    task_state = local()
    lock = Lock()
    completed = 0
    original_task = evaluator.run_one
    original_rollout = evaluator.rollout
    original_trange = evaluator.trange

    @wraps(original_task)
    def run_one(task_group, task_id, env, **options):
        task_state.remaining = options["n_episodes"]
        task_state.batch = env.num_envs
        task_state.label = f"{task_group}/{task_id}"
        try:
            for name in ("env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor"):
                options[name] = deepcopy(options[name])
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


def main():
    from lerobot.scripts import lerobot_eval
    from evaluation_cameras import restore_camera_arguments, simulator_directory
    from robomme_split import simulator_context
    from simulator_worker import simulator_context as source_context
    import sys

    track_parallel_rollouts(lerobot_eval)
    restore_camera_arguments()
    with simulator_directory(sys.argv[1:]), simulator_context(), source_context():
        return lerobot_eval.main()


if __name__ == "__main__":
    main()
