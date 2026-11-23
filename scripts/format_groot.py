"""GR00T N1.7 training and standalone evaluation job formats.

Preview with --queue scripts/format_groot.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/groot
"""

# Use the checkpoint corresponding to the selected LIBERO suite.
# The native base is initialized using base_model_path. Pick the embodiment tag for your data and simulator.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "groot",  # Required. LeRobot policy type.
        "label": "GR00T N1.7 training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot"},  # Optional. Override the preferred Conda environment.
        # Use "resume" with "steps" and "save_freq" to restore a native training checkpoint.
        # For exact resume, remove new-training inputs and scope or optimizer overrides.
        "train": {
            "dataset": "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",  # Required. LeRobot dataset ID.
            "dataset_root": None,  # Optional. Local directory. None lets LeRobot fetch the dataset.
            "episodes": None,  # Optional. Set a list of episode IDs to override the dataset split.
            "pretrained": None,  # Optional. Hub ID or local checkpoint. None uses policy initialization.
            "batch": 1,  # Required. Samples per optimizer batch.
            "device": "cuda",  # Required. Use a device supported by this policy.
            "seed": 0,  # Required. Training seed.
            "steps": 100,  # Required. Target optimizer step count.
            "save_freq": 100,  # Required. Optimizer steps between checkpoints.
            "policy_training": {},  # Optional. Inherit the policy or checkpoint scope.
            # Component settings exposed by this policy (constructor defaults shown).
            #   tune_llm = False
            #   tune_visual = False
            #   tune_projector = True
            #   tune_diffusion_model = True
            #   tune_vlln = True
            #   tune_top_llm_layers = 0
            #   lora_rank = 0 (pass through extra)
            #   lora_alpha = 16 (pass through extra)
            #   lora_dropout = 0.1 (pass through extra)
            "policy_options": {  # Optional. Uncomment or change supported values as needed.
                # "optimizer_fused": None,  # Optional. None uses the policy's automatic/default behavior. Available: None, True, False.
                # "optimizer_lr": 0.0001,
                # "optimizer_eps": 1e-8,
                # "optimizer_weight_decay": 0.00001,
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.base_model_path=nvidia/GR00T-N1.7-3B",
                "--policy.embodiment_tag=libero_sim",
                "--policy.use_relative_actions=false",
            ],
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "groot",  # Required. Must match the evaluation checkpoint.
        "label": "GR00T N1.7 evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "nvidia/gr00t17-lerobot-libero_spatial-640",  # Required. Hub ID or local checkpoint with its processors.
            "env": "libero",  # Required. Evaluation environment.
            "tasks": ["libero_spatial"],  # Required. One or more LIBERO suites.
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
            "policy_options": {  # Optional. Omit entries to inherit the checkpoint settings.
                # Constructor defaults below are references, not automatic overrides.
                # "dtype": 'float32',
                # "use_amp": False,
                # "n_action_steps": 40,
                # "num_inference_timesteps": None, None selects the native automatic behavior.
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.embodiment_tag=libero_sim",
            ],
        },
    },
]
