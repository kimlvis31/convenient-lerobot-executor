"""FLUX.3 Action training and standalone evaluation job formats.

Preview with --queue scripts/format_flux3.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/flux3
"""

# Use a 30 Hz SO-101 dataset with matching cameras and action features. The mixed community dataset needs SO-101 selection.
# Supply a compatible environment plugin and its matching checkpoint for simulated evaluation. LeRobot also supports direct SO-101 rollout.
# The required NATTEN wheel must match the installed PyTorch and GPU runtime.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "flux3",  # Required. LeRobot policy type.
        "label": "FLUX.3 Action training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "your-org/so101-only-dataset",  # Required. Replace with your compatible dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": "black-forest-labs/flux-3-action-so101",  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            "lora": None,  # Optional. Set {"rank": 16, "alpha": 16} for LoRA.
            # LoRA requires pretrained weights and no explicit policy_training overrides.
            # Optional LoRA keys: target_modules (list or pattern), full_training_modules (list).
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "loss_reduction": "modalities",
                # "gradient_checkpointing": True,
                # "train_timestep_width": 0.75,
                # "train_timestep_shift": 42,
                # "action_loss_weight": 0.5,
                # "video_loss_weight": 1,
                # "caption_dropout": 0.1,
                # "optimizer_lr": 0.0001,
                # "optimizer_lr_heads_multiplier": 5,
                # "optimizer_eps": 1e-8,
                # "optimizer_weight_decay": 0,
                # "optimizer_grad_clip_norm": 1,
                # "scheduler_freeze_backbone_steps": 0,
                # "scheduler_warmup_steps": 0,
                # "scheduler_warmup_steps_heads": 0,
                # "scheduler_decay_steps": 0,
                # "scheduler_cooldown_steps": 0,
            },
            "extra": [],  # Optional. Additional LeRobot options in --name=value form.
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "flux3",  # Required. Must match the evaluation checkpoint.
        "label": "FLUX.3 Action evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "./outputs/training/flux3-run/train/checkpoints/last/pretrained_model",  # Required. Replace with an existing compatible checkpoint path.
            "env": "your_compatible_environment",  # Required. Replace with an installed environment type.
            "tasks": ["your_task"],  # Required. Replace with a task registered by the environment.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 0,  # Optional. Simulator default when 0.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "extra": [],  # Optional. Additional LeRobot options in --name=value form.
        },
    },
]
