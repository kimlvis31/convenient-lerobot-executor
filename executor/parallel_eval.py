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

    @wraps(original_task)
    def run_one(task_group, task_id, env, **options):
        task_state.remaining = options["n_episodes"]
        task_state.batch = env.num_envs
        try:
            for name in ("env_preprocessor", "env_postprocessor", "preprocessor", "postprocessor"):
                options[name] = deepcopy(options[name])
            return original_task(task_group, task_id, env, **options)
        finally:
            task_state.remaining = 0

    @wraps(original_rollout)
    def rollout(*args, **kwargs):
        nonlocal completed
        result = original_rollout(*args, **kwargs)
        if getattr(task_state, "remaining", 0):
            count = min(task_state.batch, task_state.remaining)
            task_state.remaining -= count
            with lock:
                completed += count
                print(f"[Eval progress] Episodes completed: {completed}", flush=True)
        return result

    evaluator.run_one = run_one
    evaluator.rollout = rollout


def main():
    from lerobot.scripts import lerobot_eval

    track_parallel_rollouts(lerobot_eval)
    lerobot_eval.main()


if __name__ == "__main__":
    main()
