"""Wall-OSS-0.5 training and standalone evaluation job formats.

Preview with --queue scripts/format_wall_oss_05.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/wall_oss_05
"""

# Replace the dataset and simulator placeholders. Use a fine-tuned checkpoint with matching state, action and sensors.
# For the standard lerobot/libero front/wrist camera pair, the executor supplies
# camera_key_mapping automatically. Other datasets require an explicit verified
# --policy.camera_key_mapping JSON object in train.extra when names differ.
# The published checkpoint uses exactly 10 Euler flow steps; do not reduce them.
# Its released 26-value action space is not the stock LIBERO 7-value action space.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "wall_oss_05",  # Required. LeRobot policy type.
        "label": "Wall-OSS-0.5 training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "your-org/compatible-wall-oss-dataset",  # Required. Replace with your compatible dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": "lerobot/wall-oss-0.5",  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "flow_loss_weight": 1,
                # "text_loss_weight": 0.1,
                # "optimizer_lr": 0.00005,
                # "optimizer_eps": 1e-8,
                # "optimizer_weight_decay": 1e-8,
                # "optimizer_grad_clip_norm": 1,
                # "scheduler_warmup_steps": 1000,
                # "scheduler_decay_steps": 200000,
                # "scheduler_decay_lr": 0.000001,
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.dtype=bfloat16",
            ],
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "wall_oss_05",  # Required. Must match the evaluation checkpoint.
        "label": "Wall-OSS-0.5 evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "./outputs/training/wall_oss_05-run/train/checkpoints/last/pretrained_model",  # Required. Replace with an existing compatible checkpoint path.
            "env": "your_compatible_environment",  # Required. Replace with an installed environment type.
            "tasks": ["your_task"],  # Required. Replace with a task registered by the environment.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "parallel_tasks": 1,  # Optional. Distinct tasks concurrently; use 1 for single-task benchmarks.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 0,  # Optional. Simulator default when 0.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "policy_options": {  # Optional. Omit entries to inherit the checkpoint settings.
                # Constructor defaults below are references, not automatic overrides.
                # "dtype": 'bfloat16',
                # "use_amp": False,
                # "n_action_steps": 32,
                # "num_inference_timesteps": 10,
            },
            "extra": [],  # Optional. Additional LeRobot options in --name=value form.
        },
    },
]
