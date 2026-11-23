"""TD-MPC training and standalone evaluation job formats.

Preview with --queue scripts/format_tdmpc.py --dry-run.
Remove either job from JOBS to run only one mode. Replace any paths or IDs
marked as placeholders before execution. Relative paths start at the project root.

Policy configuration: https://github.com/huggingface/lerobot/tree/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/src/lerobot/policies/tdmpc
"""

# Offline training requires next.reward alongside observations and actions. Use the same embodiment in evaluation.
# Optional planning parameters include cem_iterations, n_gaussian_samples, n_pi_samples and n_elites.
# lerobot/xarm_lift_medium_replay is the upstream TD-MPC benchmark dataset for a separate xArm environment.

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot training and evaluation.
        "model": "tdmpc",  # Required. LeRobot policy type.
        "label": "TD-MPC training",  # Optional. Display name.
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
                # "optimizer_lr": 0.0003,
            },
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--policy.use_mpc=false",
            ],
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "native",  # Required. Built-in LeRobot evaluation.
        "model": "tdmpc",  # Required. Must match the evaluation checkpoint.
        "label": "TD-MPC evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "./outputs/training/tdmpc-run/train/checkpoints/last/pretrained_model",  # Required. Replace with an existing compatible checkpoint path.
            "env": "pusht",  # Required. Evaluation environment.
            "tasks": ["PushT-v0"],  # Required. Task name for this simulator.
            "episodes": 1,  # Required. Episodes per task. Increase this for a benchmark run.
            "batch": 1,  # Required. Parallel episode slots per task, capped by episodes.
            "parallel_tasks": 1,  # Optional. Distinct tasks concurrently; use 1 for single-task benchmarks.
            "seed": 0,  # Required. Evaluation seed.
            "device": "cuda",  # Required. Use a device supported by the policy.
            "episode_length": 0,  # Optional. Simulator default when 0.
            "environment": None,  # Optional. Conda override. None selects a compatible environment.
            "extra": [  # Optional. Additional LeRobot options in --name=value form.
                "--env.observation_width=96",
                "--env.observation_height=96",
                # Keypoint checkpoints with ENV[16] + STATE[2] automatically select
                # environment_state_agent_pos. Image checkpoints keep pixels_agent_pos.
                # Optional override: "--env.obs_type=environment_state_agent_pos".
                "--policy.use_mpc=true",
            ],
        },
    },
]
