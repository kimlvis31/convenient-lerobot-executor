"""LingBot-VLA 2.0 training and standalone evaluation job formats.

Preview with --queue scripts/format_lingbot_vla_v2.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/lingbot_vla_v2
"""

# The reference RoboTwin checkpoint expects 14-D joint actions.
# The example dataset's six-dimensional state and action need the slot maps.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "lingbot_vla_v2",  # Required. LeRobot policy type.
        "label": "LingBot-VLA 2.0 training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "maximellerbach/omx_multicubes",  # Required. LeRobot dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": "lerobot/lingbot_vla_v2_base",  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            # Component settings exposed by this policy (constructor defaults shown).
            #   train_expert_only = False
            #   freeze_vision_encoder = False
            #   train_state_proj = True
            "lora": None,  # Optional. Set {"rank": 16, "alpha": 16} for LoRA.
            # LoRA requires pretrained weights and no explicit policy_training overrides.
            # Optional LoRA keys: target_modules (list or pattern), full_training_modules (list).
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "gradient_checkpointing": False,
                # "loss_type": "L1_fm",
                # "optimizer_lr": 0.00001,
                # "optimizer_eps": 1e-8,
                # "optimizer_weight_decay": 0,
                # "optimizer_grad_clip_norm": 1,
                # "scheduler_warmup_steps": 1000,
                # "scheduler_decay_steps": 30000,
                # "scheduler_decay_lr": 0.00001,
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                (
                    "--policy.state_slots={\"observation.state.arm.position\":{\"origin_keys\":"
                    "[{\"observation.state\":{\"start\":0,\"end\":5}}]},\"observation.state.effector.position\":"
                    "{\"origin_keys\":[{\"observation.state\":{\"start\":5,\"end\":6}}]}}"
                ),
                (
                    "--policy.action_slots={\"action.arm.position\":{\"origin_keys\":[{\"action\":{\"start\":0,"
                    "\"end\":5}}]},\"action.effector.position\":{\"origin_keys\":[{\"action\":{\"start\":5,\"end\":6}"
                    "}]}}"
                ),
                (
                    "--rename_map={\"observation.images.top\":\"observation.images.camera_top\","
                    "\"observation.images.wrist\":\"observation.images.camera_wrist_left\"}"
                ),
                "--policy.dtype=float32",
                "--accelerator.mixed_precision=bf16",
            ],
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "lingbot_vla_v2",  # Required. Must match the evaluation checkpoint.
        "label": "LingBot-VLA 2.0 evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "lerobot/lingbot_vla_v2_robotwin",  # Required. Hub ID or local checkpoint with its processors.
            "env": "robotwin",  # Required. Evaluation environment.
            "tasks": ["beat_block_hammer"],  # Required. Task name for this simulator.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "parallel_tasks": 1,  # Optional. Distinct tasks concurrently; use 1 for single-task benchmarks.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 0,  # Optional. Simulator default when 0.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                (
                    "--rename_map={\"observation.images.head_camera\":\"observation.images.cam_high\","
                    "\"observation.images.left_camera\":\"observation.images.cam_left_wrist\","
                    "\"observation.images.right_camera\":\"observation.images.cam_right_wrist\"}"
                ),
            ],
        },
    },
]
