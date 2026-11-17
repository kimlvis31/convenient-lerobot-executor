"""Examples for batch evaluation, train-then-evaluate, continuation, and resume."""

# JOBS is an ordered list. Each dictionary describes one executor job.
# To evaluate SmolVLA, use adapter "native", model "smolvla", and set
# eval.policy to a compatible Hugging Face repository ID or the local snapshot
# shown in the inventory. Set eval.device to "cuda" or "cpu". Match the
# checkpoint's environment, camera size, and control mode.
# parallel_tasks and action_steps below are TinyVLA evaluation options.
# Optional TinyVLA training setting: "lora": {"rank": 16, "alpha": 16, "dropout": 0.0}.
# This adds language-attention LoRA to the action head. Exact resume keeps the original settings.
# Native train settings can include policy_options (for example, {"optimizer_lr": 0.0001}).
# Queue evaluations can set eval.environment to select a specific compatible Conda environment.
# A job can contain train, eval, or both. With both present, evaluation uses
# the checkpoint produced by that job's training stage.
# The dataset and checkpoints below are example assets under the convLexecutor root.
JOBS = [
    {
        "label": "Base model: parallel evaluation of tasks 1 and 2",  # Name shown in the queue.
        "adapter": "tinyvla",  # Execution backend. Use "native" for built-in policies.
        "model": "tinyvla",  # Policy name. Native examples include "smolvla" and "act".
        "eval": {  # Evaluation settings. Omitting policy uses the TinyVLA base weights.
            "env": "libero",  # Simulation environment type.
            "tasks": ["libero_object"],  # Benchmark suites to run.
            "task_ids": [1, 2],  # Zero-based task IDs for a single suite. Use [] for all tasks.
            "episodes": 2,  # Episodes to run for each task.
            "batch": 2,  # Episodes evaluated together per task, capped at episodes.
            "parallel_tasks": 2,  # Number of distinct tasks to evaluate concurrently.
            "seed": 0,  # Random seed for the evaluation and the TinyVLA worker.
            "control": "relative",  # Action control mode. TinyVLA requires relative control.
            "size": 256,  # Camera image width and height in pixels.
            "episode_length": 10,  # Maximum control steps per episode. Use 0 for the benchmark default.
            "action_steps": 16,  # Actions executed per prediction, from 1 to 16 for TinyVLA.
        },
    },
    {
        # Evaluate a prepared local checkpoint with one task and one episode.
        "label": "Evaluate the trained checkpoint, task 1",
        "adapter": "tinyvla",
        "model": "tinyvla",
        "eval": {
            "policy": "outputs/example/tinyvla_checkpoint/train/pretrained_model",  # Local model directory.
            "env": "libero",
            "tasks": ["libero_object"],
            "task_ids": [1],
            "episodes": 1,
            "batch": 1,
            "seed": 0,
            "control": "relative",
            "size": 256,
            "episode_length": 10,
        },
    },
    {
        "label": "Reset: train a new action head, then evaluate task 1",
        "adapter": "tinyvla",
        "model": "tinyvla",
        "train": {  # The base weights are loaded with a newly initialized action head and optimizer.
            "task": ["libero_object", 1],  # Benchmark suite and zero-based task ID.
            "dataset_root": "datasets/example/libero",  # Local LeRobot dataset directory.
            "selection": "datasets/example/libero/selection.json",  # Saved training and held-out episode split.
            "updates": 4,  # Optimizer steps per task.
            "batch": 1,  # Samples per optimizer step.
            "lr": 0.0001,  # Learning rate for the action-head optimizer.
            "seed": 0,  # Random seed for model initialization and batch sampling.
            "save_freq": 2,  # Save training state every two optimizer steps.
        },
        "eval": {  # Uses the model saved by this job's training stage.
            "env": "libero",
            "tasks": ["libero_object"],
            "task_ids": [1],
            "episodes": 1,
            "batch": 1,
            "seed": 0,
            "control": "relative",
            "size": 256,
            "episode_length": 10,
        },
    },
    {
        "label": "Continuation: train from saved weights, task 1",
        "adapter": "tinyvla",
        "model": "tinyvla",
        "train": {
            "warm_start": "outputs/example/tinyvla_checkpoint/train/pretrained_model",  # Keep learned weights. Start a new optimizer and step count.
            "task": ["libero_object", 1],
            "dataset_root": "datasets/example/libero",
            "selection": "outputs/example/tinyvla_checkpoint/selection.json",
            "updates": 4,  # Four additional steps in the new run.
            "batch": 1,
            "lr": 0.0001,
            "seed": 0,
            "save_freq": 2,
        },
    },
    {
        "label": "Resume: restore step 2 and finish at step 4, task 1",
        "adapter": "tinyvla",
        "model": "tinyvla",
        "train": {
            "resume": "outputs/example/tinyvla_resume/train/checkpoints/training_state.pt",  # Restore weights, optimizer, random state, and step count.
            "task": ["libero_object", 1],
            "dataset_root": "datasets/example/libero",
            "selection": "outputs/example/tinyvla_resume/selection.json",
            "updates": 4,  # Original target for the interrupted run. This job runs steps 3 and 4.
            "batch": 1,  # Keep the original batch size, learning rate, and seed for an exact resume.
            "lr": 0.0001,
            "seed": 0,
            "save_freq": 2,
        },
    },
]
