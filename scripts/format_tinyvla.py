"""TinyVLA action-head training and standalone evaluation job formats.

Prepare the public example data and checkpoints with executor/prepare_examples.py.
Preview with --queue scripts/format_tinyvla.py --dry-run. Remove either job
from JOBS to run only one mode.
"""

JOBS = [
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "tinyvla",  # Required. External TinyVLA adapter.
        "model": "tinyvla",  # Required. Model identifier for this adapter.
        "label": "TinyVLA action-head training",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        # "environments": {"native": "lerobot", "tinyvla": "lerobot_tinyvla"},
        # Optional. Override the data and model runtimes if their names differ.
        "train": {
            "task": ["libero_object", 1],  # Optional. Omit to train all standard tasks.
            "dataset_root": "datasets/example/libero",  # Optional. Local LeRobot training data.
            "selection": "datasets/example/libero/selection.json",  # Optional. Matching task split.
            "base_model": None,  # Optional. None uses the prepared public base weights.
            "updates": 4,  # Required. Total optimizer steps per task.
            "batch": 1,  # Required. Samples per update.
            "lr": 0.0001,  # Required. Positive learning rate.
            "seed": 0,  # Required. Training seed.
            "save_freq": 2,  # Required. Checkpoint interval in optimizer steps.
            "lora": None,  # Optional. Add language attention LoRA to action-head training.
            # "lora": {"rank": 16, "alpha": 16, "dropout": 0.0},
            # "warm_start": "outputs/example/tinyvla_checkpoint/train/pretrained_model",
            # Optional. Start a new run with a fresh optimizer from the full
            # exported model, including any merged language-adapter changes.
            # Choose new updates or adapter settings here when desired.
            # "resume": "outputs/example/tinyvla_resume/train/checkpoints/training_state.pt",
            # Optional. Restore an interrupted run. Match its data, batch, seed,
            # learning rate, base snapshot, trainer and LoRA settings. Keep its
            # original planned updates target; exact resume does not extend it.
            # Choose one initialization route: base_model, warm_start or resume.
        },
    },
    {
        "schema": 1,  # Optional. Schema 1 is assumed in a queue.
        "adapter": "tinyvla",  # Required. External TinyVLA adapter.
        "model": "tinyvla",  # Required. Model identifier for this adapter.
        "label": "TinyVLA evaluation",  # Optional. Display name.
        "save_outputs": True,  # Optional. False discards outputs after the job.
        "eval": {
            "policy": "outputs/example/tinyvla_checkpoint/train/pretrained_model",  # Optional. None uses the base.
            "env": "libero",  # Required. TinyVLA currently supports LIBERO through this adapter.
            "tasks": ["libero_object"],  # Required. One or more supported suites.
            "task_ids": [1],  # Optional. Use [] for all IDs. Explicit IDs require one suite.
            "episodes": 1,  # Required. Episodes per task.
            "batch": 1,  # Required. Episodes processed together per task.
            "parallel_tasks": 1,  # Optional. Distinct tasks evaluated concurrently.
            "seed": 0,  # Required. Simulator and worker seed.
            "control": "relative",  # Required. TinyVLA uses relative control.
            "size": 256,  # Required. TinyVLA uses 256 px camera input.
            "episode_length": 10,  # Optional. 0 uses the benchmark default.
            "action_steps": 16,  # Optional. Actions per prediction, from 1 to 16.
            "environment": None,  # Optional. Conda override for the LeRobot evaluator.
            "extra": [],  # Optional. Additional supported LeRobot options.
        },
    },
]
