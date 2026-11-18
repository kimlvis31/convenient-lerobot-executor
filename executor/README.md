# LeRobot Executor

On startup, the executor lists local training datasets, model weights, checkpoints with training state, results, and saved job settings. Choose training, evaluation, training followed by evaluation, or checkpoint resume. You can also reuse a previous job's settings.

## Getting started

Follow the [setup guide](../README.md) to install LeRobot in a Conda environment and run the executor from the project checkout. Use a terminal with Python 3.10 or later and Conda.

```bash
python -B ~/convLexecutor/executor/manage.py --inventory
python -B ~/convLexecutor/executor/manage.py --dry-run
python -B ~/convLexecutor/executor/manage.py
```

`--inventory` shows the full list. The executor looks for training datasets in `datasets/` and its local LeRobot dataset cache, checkpoints and results in `outputs/`, and TinyVLA base weights in `externals/TinyVLA/models/`. The inventory checks for local model files and lists compatible Hugging Face model snapshots already on disk. Snapshots are labeled with their repository and revision. A model-only snapshot can be evaluated or used to initialize training. Resume requires saved training state.

`--dry-run` shows the commands and Conda environments selected by the settings. Use `--lerobot-root` for older relative input paths outside the managed directories. Use `--tinyvla-root` to override the bundled TinyVLA integration.

In a numbered menu, press Enter to select the row marked [Default]. In a text or numeric prompt, press Enter to use the value shown in [Default: ...]. Select a model from one list with [LeRobot Native] or [External] next to its name. Native models come from the selected LeRobot environment and appear before external models. Model size labels show the parameter count of a representative checkpoint in millions or billions. Another checkpoint or policy configuration may have a different count. Policies without a verified count display [N/A]. For plain terminal output, set NO_COLOR=1. The evaluation menu lists LeRobot benchmarks and accepts an environment name manually. Select a LIBERO suite, all four standard suites, or enter a custom suite list. TinyVLA uses LIBERO and accepts an evaluation batch size and a separate number of parallel tasks. The batch size is capped at episodes per task. Parallel tasks have separate action queues and processors, and share one inference worker. For LIBERO, PushT, ALOHA, and Meta-World, set Episode length to the maximum control steps per episode. Enter 0 to use the benchmark default. The display separates status, questions, settings, progress, and completion. Other benchmarks may need their corresponding LeRobot environment dependencies.

Use **Add Another Job?** after configuring a job to build a queue interactively. The Questions section shows job numbers during manual setup and replaces the previous questions on interactive terminals. Enter **B** to go to the previous prompt. Settings shows all jobs before execution. Press Enter to accept the execution confirmation's default of **Y**.

The first menu includes **Delete all evaluation results** and **Delete all training results**. Each action shows its target folder and requires confirmation. Cleanup selects run directories inside `outputs/evaluation/` or `outputs/training/`, including their logs and training checkpoints. It skips active runs and redirected folders. Example assets live in `outputs/example/` and datasets live in `datasets/`.

Choose **N** at **Save Job Outputs?** or set `"save_outputs": False` in a Python job dictionary for temporary runs. Evaluation summaries and failure causes are printed before the temporary files are removed.

Native LeRobot jobs check policy and benchmark dependencies when they run. The executor reuses a ready Conda environment, installs missing dependencies declared by that LeRobot version, and tries a separate clone when pinned versions conflict. Dry runs are read-only. Benchmarks marked **Manual Setup** have their own simulator instructions. An installed LeRobot environment plugin appears in the evaluation menu. The policy menu shows local checkpoint availability, including the Hugging Face cache. The training scope menu lists compatible component combinations. Settings highlights required arguments added automatically in green. Training preflight reads local dataset and checkpoint metadata. Result shows a concise subprocess failure cause and the full log path.

For new native training, **Training Scope** shows the component controls exposed by the selected policy. Supported policies can configure LoRA rank, scale, target modules, and fully trained modules when starting from pretrained weights. MolmoAct2 exposes its own vision-language training mode and LoRA parameters. Resume uses the checkpoint's training configuration. The bundled TinyVLA trainer updates its action head.

## Import a job queue

Select **Import a job queue script** to list job queue and settings files in the project `scripts/` directory. In [example_script.py](../scripts/example_script.py), `JOBS` is an ordered list of job dictionaries. The interactive prompts fill the same fields. Specify `adapter`, `model`, and at least one of `train` or `eval` for each job. Include both stages in a single dictionary to evaluate the checkpoint produced by training.

The example contains five TinyVLA jobs: batched evaluation of the base model across two parallel tasks, checkpoint evaluation, reset training followed by evaluation, continuation from saved weights with a new optimizer, and exact resume from step 2. The preparation script writes the demonstration and its split under `datasets/example/libero/`, and the checkpoints under `outputs/example/`. Copy or remove individual dictionaries to select the scenarios you want to run.

```bash
python -B ~/convLexecutor/executor/manage.py --queue ~/convLexecutor/scripts/example_script.py --dry-run
python -B ~/convLexecutor/executor/manage.py --queue ~/convLexecutor/scripts/example_script.py
```

The executor reads literal settings from the file and runs jobs in order. If a job fails, the queue stops and the executor retains the scripts and logs of jobs that started. For one job, use a `SETTINGS` dictionary or `--settings`. Relative dataset and checkpoint paths are resolved from the convLexecutor checkout.

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

The Execution section shows each numbered job's (Status) and (Result) before moving to the next job. Evaluation results include overall and per-suite success rates, total evaluation time, average time per episode, and the result path.

After you confirm the settings, the executor generates and runs one script per job.

| Path | Contents |
| --- | --- |
| `outputs/{training,evaluation}/<run-name>/executor/run.py` | The script that ran, including job settings, commands, and environment details |
| `outputs/{training,evaluation}/<run-name>/executor/script.py` | The effective settings for that job, saved as a reusable `JOBS` list |
| `outputs/{training,evaluation}/<run-name>/executor/status.json` | Start and end times, duration, and status for the job and its stages |
| `outputs/{training,evaluation}/<run-name>/executor/{train,eval}.log` | Console output for each stage that ran |
| `outputs/training/<run-name>/` | Checkpoints, training results, and the saved dataset split where applicable |
| `outputs/evaluation/<run-name>/` | Evaluation results and videos saved by LeRobot |

With Rich installed, the terminal updates the job gauge in place. Evaluation adds an episode gauge with steps elapsed and the environment's step limit. Episodes can finish before that limit. In parallel TinyVLA evaluation, the episode gauge shows the active batch with the lowest completion percentage and labels it with the suite and task ID. Install it with `python -m pip install rich` in the environment that runs the executor if needed. Redirected output uses plain lines so progress can be read from a file or pipeline.

The status line shows overall completion, the current stage's completed and total units, and the estimated time remaining in hours, minutes, and seconds. Training counts optimizer steps. Evaluation counts completed episodes from LeRobot's reported batches. Seconds Per Episode shows the measured elapsed time divided by the number of completed episodes, including episodes processed in parallel. The executor estimates the whole job when it has measured timings for the remaining stages, including prior compatible TinyVLA evaluation timings. Otherwise it announces the current-stage ETA when measured progress is available. Estimated Time of Completion shows a remaining duration, with a day count for durations of 24 hours or more. It shows N/A when there is no usable estimate.

- Native resume uses LeRobot's `train_config.json` and `training_state/`. Select a checkpoint or a previous run and set a target step count greater than the saved step.
- TinyVLA resume uses the previous training directory and job settings. It restores the action head, optimizer and random state, and reuses the same training split. To initialize a new run from weights alone, choose **Start training with existing action-head weights**.
- **Reuse settings from a previous run** and `--queue /path/to/script.py` create a new job using saved settings. You can also copy `script.py` into `scripts/` under a new name. Previous `run.py` and `settings.json` files are supported. To evaluate an existing model, select evaluation and specify its `pretrained_model` directory.

## Add another adapter

Extend `NativeAdapter` and give the class a unique `key` and a `label`. Return a `Stage` from `train()` and `evaluate(after_training)` with its environment, command, output directory, and expected files. Keep the module importable and register it with `--plugin module:Class`.

## Verification

Run the local tests with `python -B -m unittest discover -s ~/convLexecutor/executor -v`.
