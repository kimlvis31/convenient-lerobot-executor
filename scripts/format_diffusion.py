"""Diffusion training and standalone evaluation job formats.

Preview with --queue scripts/format_diffusion.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/diffusion
"""

# The executor can prepare a temporary processor conversion for the original PushT checkpoint.
# AlohaInsertion-v0 is another supported task with matching weights and training data such as lerobot/aloha_sim_insertion_human.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "diffusion",  # Required. LeRobot policy type.
        "label": "Diffusion training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "lerobot/pusht",  # Required. LeRobot dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": None,  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "num_train_timesteps": 100,  # Optional.
                # "do_mask_loss_for_padding": False,  # Optional.
                # "gradient_checkpointing": False,
                # "optimizer_lr": 0.0001,
                # "optimizer_eps": 1e-8,
                # "optimizer_weight_decay": 0.000001,
                # "scheduler_name": "cosine",
                # "scheduler_warmup_steps": 500,
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.horizon=16",
                "--policy.n_action_steps=8",
                "--policy.crop_shape=[84,84]",
            ],
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "diffusion",  # Required. Must match the evaluation checkpoint.
        "label": "Diffusion evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "lerobot/diffusion_pusht",  # Required. Hub ID or local checkpoint with matching PushT inputs.
            "env": "pusht",  # Required. Evaluation environment.
            "tasks": ["PushT-v0"],  # Required. Task name for this simulator.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "parallel_tasks": 1,  # Optional. Distinct tasks concurrently; use 1 for single-task benchmarks.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 0,  # Optional. Simulator default when 0.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "policy_options": {  # Optional. Omit entries to inherit the checkpoint settings.
                # Constructor defaults below are references, not automatic overrides.
                # "use_amp": False,
                # "n_action_steps": 32,
                # "num_inference_steps": None, None selects the native automatic behavior.
                # "compile_model": False,
                # "compile_mode": 'reduce-overhead',
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.num_inference_steps=10",
            ],
        },
    },
]
