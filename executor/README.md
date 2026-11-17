# LeRobot Executor

On startup, the executor lists local training datasets, model weights, checkpoints with training state, results, and saved job settings. Choose training, evaluation, training followed by evaluation, or checkpoint resume. You can also reuse a previous job's settings.

## Getting started

Follow the [setup guide](../README.md) to install the executor alongside LeRobot. Use a terminal with Python 3.10 or later and Conda.

```bash
python -B ~/convLexecutor/executor/manage.py --inventory
python -B ~/convLexecutor/executor/manage.py --dry-run
python -B ~/convLexecutor/executor/manage.py
```

`--inventory` shows the full list. The executor looks for training datasets in `datasets/` and the local LeRobot cache, checkpoints and results in `outputs/`, and TinyVLA base weights in `externals/TinyVLA/models/`. The inventory checks for the required local files.

`--dry-run` shows the commands and Conda environments selected by the settings. Use `--lerobot-root` to specify a working directory other than `~/lerobot` and `--tinyvla-root` to override the bundled TinyVLA integration.

## Import a job queue

Select **Import a job queue script** to list the `.py` and `.json` files in `executor/scripts/`. In [example_script.py](scripts/example_script.py), `JOBS` is an ordered list of job dictionaries. The interactive prompts fill the same fields. Specify `adapter`, `model`, and at least one of `train` or `eval` for each job. Include both stages in a single dictionary to evaluate the checkpoint produced by training.

The example contains five TinyVLA jobs: base evaluation, checkpoint evaluation, reset training from the base model, continuation from saved weights with a new optimizer, and exact resume from step 2. The preparation script writes the demonstration and its split under `~/lerobot/datasets/example/libero/`, and the checkpoints under `~/lerobot/outputs/example/`. Copy or remove individual dictionaries to select the scenarios you want to run.

```bash
python -B ~/convLexecutor/executor/manage.py --queue ~/convLexecutor/executor/scripts/example_script.py --dry-run
python -B ~/convLexecutor/executor/manage.py --queue ~/convLexecutor/executor/scripts/example_script.py
```

The executor reads literal settings from the file and runs jobs in order. If a job fails, the queue stops and the executor retains the scripts and logs of jobs that started. For one job, use a `SETTINGS` dictionary or `--settings`. Relative dataset and checkpoint paths are resolved from the LeRobot working directory.

## Model and environment selection

Use `"adapter": "native"` for a built-in LeRobot policy. Set `model` to the policy name, such as `"smolvla"`. Native training and evaluation use `lerobot-train` and `lerobot-eval`. TinyVLA trains its action head in a separate Conda environment and uses LeRobot for evaluation.

The default environments are `lerobot` and `tinyvla_infer`. If only `lerobot_tinyvla` is available for the TinyVLA model, the executor selects it. Use `--env-map` to map models to specific environments.

```json
{
  "native": "lerobot",
  "tinyvla": "tinyvla_infer",
  "smolvla": "lerobot_smolvla"
}
```

Use `--name=value` for additional LeRobot options. For example, set a camera mapping with `--env.camera_name_mapping='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}'`. When reproducing an evaluation, check the task IDs, action steps, camera mapping, control mode, resolution, maximum episode length, and random seed.

## Results and resume

After you confirm the settings, the executor generates and runs one script per job.

| Path | Contents |
| --- | --- |
| `executor/log/<run-name>/run.py` | The script that ran, including job settings, commands, and environment details |
| `executor/log/<run-name>/script.py` | The effective settings for that job, saved as a reusable `JOBS` list |
| `executor/log/<run-name>/status.json` | Start and end times, duration, and status for the job and its stages |
| `executor/log/<run-name>/train.log`, `eval.log` | Console output for each stage that ran |
| `outputs/training/<run-name>/` | Checkpoints, training results, and the saved dataset split where applicable |
| `outputs/eval/<run-name>/` | Evaluation results and videos saved by LeRobot |

Overall progress is the average of the stage percentages. Training progress uses optimizer steps. Evaluation progress uses LeRobot's reported batches to count completed episodes. The executor estimates remaining time from measured progress. Estimating the duration of later stages requires a completed run with matching settings and environment. Remaining time is shown as unavailable until there is enough information. Preprocessing, checkpoint saving, and video encoding may take time between progress updates.

- Native resume uses LeRobot's `train_config.json` and `training_state/`. Select a checkpoint or a previous run and set a target step count greater than the saved step.
- TinyVLA resume uses the previous training directory and job settings. It restores the action head, optimizer and random state, and reuses the same training split. To initialize a new run from weights alone, choose **Start training with existing action-head weights**.
- **Reuse settings from a previous run** and `--queue /path/to/script.py` create a new job using saved settings. You can also copy `script.py` into `executor/scripts/` under a new name. Previous `run.py` and `settings.json` files are supported. To evaluate an existing model, select evaluation and specify its `pretrained_model` directory.

## Add another adapter

Extend `NativeAdapter` and give the class a unique `key` and a `label`. Return a `Stage` from `train()` and `evaluate(after_training)` with its environment, command, output directory, and expected files. Keep the module importable and register it with `--plugin module:Class`.

## Verification

Run the local tests with `python -B -m unittest discover -s ~/convLexecutor/executor -v`.
