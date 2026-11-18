"""MolmoAct2 training and standalone evaluation job formats.

Preview with --queue scripts/format_molmoact2.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/molmoact2
"""

# Training requires compatible LIBERO normalization statistics. Replace the norm_stats_path placeholder.
# The model-specific train_mode_vlm option accepts fft, lora or freeze.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "molmoact2",  # Required. LeRobot policy type.
        "label": "MolmoAct2 training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "lerobot/libero",  # Required. LeRobot dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": None,  # Optional. None uses the raw model initializer in extra.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {"train_mode_vlm": "fft", "freeze_embedding": True},  # Optional. Fine-tuning scope.
            # Component settings exposed by this policy (constructor defaults shown).
            #   freeze_embedding = True
            #   train_mode_vlm = "lora"
            #   lora_rank = 64
            #   lora_alpha = 16
            #   lora_dropout = 0.05
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "gradient_checkpointing": False,
                # "llm_residual_dropout": 0.1,
                # "optimizer_lr": 0.00001,
                # "optimizer_vit_lr": 0.000005,
                # "optimizer_connector_lr": 0.000005,
                # "optimizer_action_expert_lr": 0.00005,
                # "optimizer_eps": 0.000001,
                # "optimizer_weight_decay": 0,
                # "optimizer_grad_clip_norm": 1,
                # "scheduler_warmup_steps": 200,
                # "scheduler_decay_steps": 24000,
                # "scheduler_decay_lr": 0.000001,
            },
            "extra": [
                "--policy.checkpoint_path=allenai/MolmoAct2",  # Required. Raw policy initializer.
                "--policy.norm_tag=libero",  # Required. Normalization profile.
                "--policy.norm_stats_path=datasets/your_libero/norm_stats.json",  # Required. Replace this path.
                "--policy.action_mode=continuous",
                "--policy.chunk_size=10",
                "--policy.n_action_steps=10",
                '--policy.image_keys=["observation.images.image","observation.images.image2"]',
                "--policy.setup_type=single franka robotic arm in libero",
                "--policy.control_mode=delta end-effector pose",
                "--policy.dtype=bfloat16",
            ],
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "molmoact2",  # Required. Must match the evaluation checkpoint.
        "label": "MolmoAct2 evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": None,  # Required field. None loads the native policy type and uses extra below.
            "env": "libero",  # Required. Evaluation environment.
            "tasks": ["libero_object"],  # Required. One or more LIBERO suites.
            "task_ids": [0],  # Optional. Zero-based IDs within one suite. [] evaluates all tasks.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 0,  # Optional. Simulator default when 0.
            "control": "relative",  # Required for LIBERO. Match the checkpoint's action controller.
            "size": 256,  # Required for LIBERO. Square camera input size in pixels.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "extra": [
                "--policy.checkpoint_path=allenai/MolmoAct2-LIBERO",  # Required. Original LIBERO release.
                "--policy.norm_tag=libero",  # Required. Selects the original release's normalization profile.
                "--policy.inference_action_mode=continuous",
                "--policy.dtype=bfloat16",
                "--policy.enable_inference_cuda_graph=false",
                "--env.camera_name_mapping={\"agentview_image\":\"image\",\"robot0_eye_in_hand_image\":\"wrist_image\"}",
            ],
        },
    },
]
