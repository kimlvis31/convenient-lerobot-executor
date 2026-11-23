"""Pi0 training and standalone evaluation job formats.

Preview with --queue scripts/format_pi0.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/pi0
"""

# If switching the model to relative actions, recompute training statistics in relative-action space for that configuration.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "pi0",  # Required. LeRobot policy type.
        "label": "Pi0 training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "lerobot/libero",  # Required. LeRobot dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": "lerobot/pi0_base",  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            # Component settings exposed by this policy (constructor defaults shown).
            #   train_expert_only = False
            #   freeze_vision_encoder = False
            "lora": None,  # Optional. Set {"rank": 16, "alpha": 16} for LoRA.
            # LoRA requires pretrained weights and no explicit policy_training overrides.
            # Optional LoRA keys: target_modules (list or pattern), full_training_modules (list).
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "optimizer_fused": None,  # Optional. None uses the policy's automatic/default behavior. Available: None, True, False.
                # "gradient_checkpointing": False,
                # "optimizer_lr": 0.000025,
                # "optimizer_eps": 1e-8,
                # "optimizer_weight_decay": 0.01,
                # "optimizer_grad_clip_norm": 1,
                # "scheduler_warmup_steps": 1000,
                # "scheduler_decay_steps": 30000,
                # "scheduler_decay_lr": 0.0000025,
                # "optimizer_betas": [0.9, 0.95],  # Optional. Native constructor default; omit to inherit saved settings. Exactly 2 numeric values.
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.dtype=bfloat16",
            ],
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "pi0",  # Required. Must match the evaluation checkpoint.
        "label": "Pi0 evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "lerobot/pi0_libero",  # Required. Hub ID or local checkpoint with its processors.
            "env": "libero",  # Required. Evaluation environment.
            "tasks": ["libero_object"],  # Required. One or more LIBERO suites.
            "task_ids": [0],  # Optional. Zero-based IDs within one suite. [] evaluates all tasks.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "parallel_tasks": 1,  # Optional. Distinct tasks concurrently; use 1 for single-task benchmarks.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 0,  # Optional. Simulator default when 0.
            "control": "relative",  # Required for LIBERO. Match the checkpoint's action controller.
            "size": 224,  # Required for LIBERO. Square camera input size in pixels.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "policy_options": {  # Optional. Omit entries to inherit the checkpoint settings.
                # Constructor defaults below are references, not automatic overrides.
                # "dtype": 'float32',
                # "use_amp": False,
                # "n_action_steps": 50,
                # "num_inference_steps": 10,
                # "compile_model": False,
                # "compile_mode": 'max-autotune',
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.dtype=bfloat16",
            ],
        },
    },
]
