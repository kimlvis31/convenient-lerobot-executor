"""LaWAM training and standalone evaluation job formats.

Preview with --queue scripts/format_lawam.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/lawam
"""

# Released LIBERO weights use 32-D padded state and action features.
# Use the documented MuJoCo version when reproducing published scores.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "lawam",  # Required. LeRobot policy type.
        "label": "LaWAM training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "jialei02/libero_merged_no_noops_20hz",  # Required. LeRobot dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": "jialei02/lawam-pretrain-lerobot",  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            # Component settings exposed by this policy (constructor defaults shown).
            #   freeze_vision_backbone = False
            #   freeze_llm_backbone = False
            #   freeze_embedding = True
            #   unfreeze_vision_merger = True
            #   unfreeze_lam_decoder = True
            #   unfreeze_llm_last_n_layers = -1
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "lam_dropout": 0,
                # "optimizer_lr": 0.0001,
                # "optimizer_eps": 1e-8,
                # "optimizer_weight_decay": 1e-8,
                # "optimizer_grad_clip_norm": 1,
                # "scheduler_warmup_steps": 1500,
                # "scheduler_decay_steps": 25000,
                # "scheduler_decay_lr": 5e-7,
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--dataset.image_transforms.enable=false",
                "--policy.action_horizon=8",
                "--policy.n_action_steps=8",
                "--policy.clip_normalized_actions=true",
                "--policy.pre_snap_gripper_action=true",
                "--policy.binarize_gripper_action=true",
                "--rename_map={\"observation.images.wrist_image\":\"observation.images.image2\"}",
            ],
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "lawam",  # Required. Must match the evaluation checkpoint.
        "label": "LaWAM evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "jialei02/lawam-libero-sft-lerobot",  # Required. Hub ID or local checkpoint with its processors.
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
            "size": 256,  # Required for LIBERO. Square camera input size in pixels.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "extra": [],  # Optional. Additional LeRobot options in --name=value form.
        },
    },
]
