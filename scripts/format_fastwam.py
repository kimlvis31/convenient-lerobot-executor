"""FastWAM training and standalone evaluation job formats.

Preview with --queue scripts/format_fastwam.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/fastwam
"""

# Replace the dataset placeholder with matching image, action, state and task features.
# The reference checkpoint uses two 224 px views and a Wan backbone. Confirm the model and backbone fit your device.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "fastwam",  # Required. LeRobot policy type.
        "label": "FastWAM training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "your-org/compatible-fastwam-dataset",  # Required. Replace with your compatible dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": "lerobot/fastwam_libero_uncond_2cam224",  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            # Component settings exposed by this policy (constructor defaults shown).
            #   freeze_video_expert = False
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "use_gradient_checkpointing": False,
                # "optimizer_lr": 0.0001,
                # "optimizer_weight_decay": 0.01,
                # "video_scheduler": {'train_shift': 5.0, 'infer_shift': 5.0, 'num_train_timesteps': 1000},  # Optional. Native constructor default; omit to inherit saved settings. Named numeric settings.
                # "action_scheduler": {'train_shift': 5.0, 'infer_shift': 5.0, 'num_train_timesteps': 1000},  # Optional. Native constructor default; omit to inherit saved settings. Named numeric settings.
                # "loss": {'lambda_video': 1.0, 'lambda_action': 1.0},  # Optional. Native constructor default; omit to inherit saved settings. Named numeric settings.
            },
            "extra": [],  # Optional. Additional LeRobot options in --name=value form.
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "fastwam",  # Required. Must match the evaluation checkpoint.
        "label": "FastWAM evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "lerobot/fastwam_libero_uncond_2cam224",  # Required. Hub ID or local checkpoint with its processors.
            "env": "libero",  # Required. Evaluation environment.
            "tasks": ["libero_10"],  # Required. One or more LIBERO suites.
            "task_ids": [0],  # Optional. Zero-based IDs within one suite. [] evaluates all tasks.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "parallel_tasks": 1,  # Optional. Distinct tasks concurrently; use 1 for single-task benchmarks.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 600,  # Optional. Simulator default when 0.
            "control": "relative",  # Required for LIBERO. Match the checkpoint's action controller.
            "size": 224,  # Required for LIBERO. Square camera input size in pixels.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "policy_options": {  # Optional. Omit entries to inherit the checkpoint settings.
                # Constructor defaults below are references, not automatic overrides.
                # "dtype": 'bfloat16',
                # "use_amp": False,
                # "n_action_steps": 32,
                # "num_inference_steps": 10,
                # "inference_seed": 42,
                # "compile_action_infer": False,
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.dtype=float32",
                "--policy.n_action_steps=10",
            ],
        },
    },
]
