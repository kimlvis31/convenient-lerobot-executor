"""Long-WAM training and standalone evaluation job formats.

Preview with --queue scripts/format_longwam.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/longwam
"""

# Match observation history, camera order, task text and world-model mode to the checkpoint.
# The released COD checkpoint is a separate co-denoising alternative. Long-WAM is a high-memory policy.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "longwam",  # Required. LeRobot policy type.
        "label": "Long-WAM training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "your-org/compatible-longwam-dataset",  # Required. Replace with your compatible dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": "Efficient-Large-Model/Long-WAM-LIBERO-IDM-LeRobot",  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "use_gradient_checkpointing": True,
                # "optimizer_lr": 0.0001,
                # "optimizer_weight_decay": 0.01,
                # "loss_lambda_video": 1,
                # "loss_lambda_action": 1,
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.dtype=bfloat16",
            ],
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "longwam",  # Required. Must match the evaluation checkpoint.
        "label": "Long-WAM evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "Efficient-Large-Model/Long-WAM-LIBERO-IDM-LeRobot",  # Required. Hub ID or local checkpoint with its processors.
            "env": "libero",  # Required. Evaluation environment.
            "tasks": ["libero_spatial"],  # Required. One or more LIBERO suites.
            "task_ids": [0],  # Optional. Zero-based IDs within one suite. [] evaluates all tasks.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "parallel_tasks": 1,  # Optional. Distinct tasks concurrently; use 1 for single-task benchmarks.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 300,  # Optional. Simulator default when 0.
            "control": "relative",  # Required for LIBERO. Match the checkpoint's action controller.
            "size": 256,  # Required for LIBERO. Square camera input size in pixels.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.dtype=bfloat16",
            ],
        },
    },
]
