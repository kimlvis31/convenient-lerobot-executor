"""EVO1 training and standalone evaluation job formats.

Preview with --queue scripts/format_evo1.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/evo1
"""

# The action head, language model and vision model are selected for training. The executor adds the companion policy flags.
# The optional native training stage preset is configured through --policy.training_stage when using the checkpoint defaults.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "evo1",  # Required. LeRobot policy type.
        "label": "EVO1 training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "lerobot/libero",  # Required. LeRobot dataset ID.
            "dataset_root": "datasets/example/libero",  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": "zuoxingdong/evo1_libero",  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 8,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 20,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {  # Optional. Explicit trainable components.
                "finetune_language_model": True,
                "finetune_vision_model": True,
                "finetune_action_head": True,
            },
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "enable_gradient_checkpointing": True,  # Optional.
                # "gradient_checkpointing_use_reentrant": False,  # Optional.
                # "dropout": 0,
                # "optimizer_lr": 0.00001,
                # "optimizer_eps": 1e-8,
                # "optimizer_weight_decay": 0.00001,
                # "optimizer_grad_clip_norm": 1,
                # "scheduler_warmup_steps": 300,
            },
            "extra": [],  # Optional. Additional LeRobot options in --name=value form.
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "evo1",  # Required. Must match the evaluation checkpoint.
        "label": "EVO1 evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "zuoxingdong/evo1_libero",  # Required. Hub ID or local checkpoint with its processors.
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
            "size": 448,  # Required for LIBERO. Square camera input size in pixels.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "policy_options": {  # Optional. Omit entries to inherit the checkpoint settings.
                # Constructor defaults below are references, not automatic overrides.
                # "dtype": 'bfloat16',
                # "use_amp": True,
                # "n_action_steps": 50,
                # "use_flash_attn": True,
                # "num_inference_timesteps": 32,
            },
            "extra": [],  # Optional. Additional LeRobot options in --name=value form.
        },
    },
]
