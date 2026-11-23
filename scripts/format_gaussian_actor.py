"""Gaussian Actor training and standalone evaluation job formats.

Preview with --queue scripts/format_gaussian_actor.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/rl
"""

# Training block: settings reference only. Gaussian Actor learns through LeRobot's HIL-SERL actor and learner, which need a separate training launcher.
# The manager rejects this training block before launching lerobot-train; it does not return a supervised loss.
# Keep only the evaluation job in JOBS when importing this example. Supply a Gaussian Actor checkpoint with matching observation and action features.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "gaussian_actor",  # Required. LeRobot policy type.
        "label": "Gaussian Actor training settings (separate launcher)",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "your-org/reward-labeled-robot-dataset",  # Required. Replace with your compatible dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": None,  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            # Component settings exposed by this policy (constructor defaults shown).
            #   freeze_vision_encoder = True
            "extra": [],  # Optional. Additional LeRobot options in --name=value form.
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "gaussian_actor",  # Required. Must match the evaluation checkpoint.
        "label": "Gaussian Actor evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "./outputs/training/gaussian_actor-run/pretrained_model",  # Required. Replace with an existing compatible checkpoint path.
            "env": "pusht",  # Required. Evaluation environment.
            "tasks": ["PushT-v0"],  # Required. Task name for this simulator.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "parallel_tasks": 1,  # Optional. Distinct tasks concurrently; use 1 for single-task benchmarks.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 0,  # Optional. Simulator default when 0.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--env.observation_width=96",
                "--env.observation_height=96",
            ],
        },
    },
]
