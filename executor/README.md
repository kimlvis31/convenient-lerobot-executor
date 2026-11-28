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

In a numbered menu, press Enter to select the row marked [Default]. In a text or numeric prompt, press Enter to use the value shown in [Default: ...]. Select a model number explicitly from the list labeled [LeRobot Native] or [External]. Native models come from the selected LeRobot environment and appear before external models. Model size labels show the parameter count of a representative checkpoint in millions or billions. Another checkpoint or policy configuration may have a different count. Policies without a verified count display [N/A]. For plain terminal output, set NO_COLOR=1. The evaluation menu lists LeRobot benchmarks and accepts an environment name manually. Select a LIBERO suite, all four standard suites, or enter a custom suite list. Native policies and TinyVLA accept separate evaluation batch and parallel-task counts on multi-task benchmarks. The batch size is capped at episodes per task. Where offered, Episode Length sets the maximum control steps per episode. Meta-World uses its simulator default. Enter 0 to use the benchmark default. The display separates status, questions, settings, progress, and completion. Other benchmarks may need their corresponding LeRobot environment dependencies.

`Parallel Evaluation Tasks` / `eval.parallel_tasks` selects how many distinct tasks run together; the default is 1. `Evaluation Batch Size` / `eval.batch` selects simultaneous episodes within each task. For example, `parallel_tasks=3` and `batch=2` can run six simulation episodes together when at least three tasks and two episodes per task are available. Actual concurrency is capped at the number of tasks returned by the benchmark, and progress uses that actual task inventory. The same setting works for native policies with LIBERO, LIBERO-plus, Meta-World, RoboMME, RoboTwin, RoboCasa, VLABench, and other installed environment plugins that provide multiple tasks. It also works after training and in imported queues. PushT and ALOHA expose one task per job and use `batch` for concurrent episodes.

For standard LIBERO with multiple parallel tasks, the default runs each task's vector batch in the policy process. Task threads still overlap, all requested episode slots remain live, and inference still receives the selected per-task batch. This avoids a separate simulator Python process for every episode slot. An explicit `--eval.use_async_envs=true` in `eval.extra` retains the upstream per-episode processes; custom Hub environments and other benchmarks keep their existing choice. This setting trades within-task simulator process parallelism for lower memory use, while retaining task parallelism. Effective automatic arguments appear in Settings.

Completed task environments close inside their task worker before that worker starts another task. The upstream result consumer can also close the environment without closing it twice. This prevents old and new task resources from overlapping merely because result collection is slower than task scheduling.

For the recognized native evaluator video contract, frames are buffered in anonymous temporary files and read one frame at a time by the unchanged native encoder. The evaluator does not retain an episode-sized RGB list or allocate a second stacked copy. Episode trimming, frame order, encoder settings and output paths are preserved. Encoders release completed batch storage and are joined before cleanup on errors. Unknown evaluator implementations retain their native video storage. Temporary frame data uses the system temporary filesystem; it is not a new persistent model or dataset cache. Predicted-video latent storage is unchanged.

Native parallel evaluation copies each task's policy state, nested caches, buffers and processors while sharing the registered weight parameters. Model inference and predicted-video decoding are serialized to bound activation memory; simulations can advance concurrently. Per-task caches and simulator instances still need additional memory. Compiled instance methods use task-local eager methods so their closures cannot access another task's cache. A policy/runtime whose objects cannot be isolated fails with a diagnostic and can use `parallel_tasks=1`. TinyVLA keeps its thread-local action queues and shared inference worker. Parallel scheduling can change the order of random draws, so use one task at a time when comparing exact seeded traces.

For the verified standard LIBERO configuration, evaluation reuses immutable task descriptions within a rollout and records the primary camera from the latest reset/step observation. The cached frame is copied with the native renderer's orientation, and a reset clears both caches. Unknown renderers, custom Hub environments and other benchmarks keep their native render calls. Hard resets, settling physics steps and model sampling settings are preserved. Batches still wait for their slowest active episode; increasing parallel tasks also creates more simulators and task-local caches.

**Policy Evaluation Options** discovers supported inference controls from the selected runtime's policy configuration. It offers action steps per prediction, sampler/denoising steps, data type, mixed precision and compilation controls when that model declares them. Defaults shown for a local policy come from its saved configuration. Finish without selecting a field to preserve it. Imported dictionaries use `eval.policy_options`, for example `{"n_action_steps": 16}` only for a model that declares that field. Additional native options remain available through `extra`. Choosing a longer action chunk changes prediction frequency and can change task behavior; the manager never makes that choice automatically.

Use **Add Another Job?** after configuring a job to build a queue interactively. The Questions section shows job numbers during manual setup and replaces the previous questions on interactive terminals. Enter **B** to go to the previous prompt. Settings shows all jobs before execution. Press Enter to accept the execution confirmation's default of **Y**.

The first menu includes **Delete all evaluation results** and **Delete all training results**. Each action shows its target folder and requires confirmation. Cleanup selects run directories inside `outputs/evaluation/` or `outputs/training/`, including their logs and training checkpoints. It skips active runs and redirected folders. Example assets live in `outputs/example/` and datasets live in `datasets/`.

Choose **N** at **Save Job Outputs?** or set `"save_outputs": False` in a Python job dictionary for temporary runs. Evaluation summaries and failure causes are printed before the temporary files are removed.

`Ctrl+C` stops the stage and its descendant processes before the manager returns. On Linux, each stage has a dedicated supervisor that adopts orphaned workers, including workers in separate sessions. It allows up to eight seconds for interruption, then requests termination and finally forces remaining workers to exit and reaps them. Windows uses a private kill-on-close Job Object that contains the launcher and its descendants. A launcher exiting early does not leave its workers running; repeated interrupts do not abandon cleanup. Other jobs are outside that ownership boundary. This also covers checkpoint conversion. A normal termination request follows the same cleanup path on POSIX systems, preserving existing custom signal handlers. Other POSIX platforms retain process-group interruption and do not have Linux's orphan adoption.

Native LeRobot jobs check policy and benchmark dependencies when they run. The executor reuses a ready Conda environment, installs missing dependencies declared by that LeRobot version, and tries a separate clone when pinned versions conflict. Dry runs are read-only. On Linux, a confirmed RoboTwin, LIBERO-plus, RoboCasa365, or VLABench job can automatically prepare pinned source, simulator dependencies, and assets when an existing runtime is unavailable. RoboTwin simulation uses a separate interpreter; the other source recipes clone a compatible policy environment before installing their requirements. LIBERO-plus includes its fork, configured task files, complete official perturbation assets, robosuite 1.4.1, and Wand/ImageMagick. The manager checks actual cameras and a physical step before publishing a ready runtime. See the [managed setup guide](../README.md#managed-source-benchmark-setup) for storage, first-use download sizes, and original-environment preservation. Benchmarks that still require **Manual Setup** show the unmet conditions. An installed LeRobot environment plugin appears in the evaluation menu.

The policy menu shows local checkpoint availability, including the Hugging Face cache. The training scope menu lists compatible component combinations. Settings highlights required arguments added automatically in green. Training preflight reads local dataset and checkpoint metadata. Result shows a concise subprocess failure cause and the full log path. Native runs check that local checkpoints include the state files referenced by their saved processors. For legacy Diffusion and VQ-BeT checkpoints that embed normalization tensors, the executor prepares a temporary compatible copy before launching LeRobot and removes it after the stage. The source checkpoint is preserved. For evaluation of recognized legacy MolmoAct2 and Wall-X checkpoints, the executor also preserves the original weights and processor statistics in a temporary compatible view. Explicit Hub policy revisions select the same snapshot for configuration, weights and processors. Settings lists known access pages for gated downloads. If Hugging Face denies access to a policy or an underlying model, the result identifies the affected repository when the service reports it. Open the link with the same Hugging Face account used by the executor, review the terms and request access. If that account already has access, check its token's read permissions. A complete local cache can run offline.

For new native training, **Training Scope** shows the component controls exposed by the selected policy. Supported policies can configure LoRA (Low-Rank Adaptation) rank, scale, target modules, and fully trained modules when starting from pretrained weights. MolmoAct2 exposes its own vision-language training mode and adapter parameters. Resume uses the checkpoint's training configuration. TinyVLA offers action-head training alone or action-head training with language-attention adapters. Set `train.lora` to `{"rank": 16, "alpha": 16, "dropout": 0.0}` to select the latter; omit it or use `None` for action-head training alone. Other TinyVLA backbone weights remain frozen.

## Import a job queue

Select **Import a job queue script** to list job queue and settings files in the project `scripts/` directory. In [example_script.py](../scripts/example_script.py), `JOBS` is an ordered list of job dictionaries. The interactive prompts fill the same fields. Specify `adapter`, `model`, and at least one of `train` or `eval` for each job. Include both stages in a single dictionary to evaluate the checkpoint produced by training.

The example contains five TinyVLA jobs: batched evaluation of the base model across two parallel tasks, checkpoint evaluation, reset training followed by evaluation, continuation from saved weights with a new optimizer, and exact resume from step 2. The preparation script writes the demonstration and its split under `datasets/example/libero/`, and the checkpoints under `outputs/example/`. Copy or remove individual dictionaries to select the scenarios you want to run.

```bash
python -B ~/convLexecutor/executor/manage.py --queue ~/convLexecutor/scripts/example_script.py --dry-run
python -B ~/convLexecutor/executor/manage.py --queue ~/convLexecutor/scripts/example_script.py
```

The executor reads literal settings from the file and runs jobs in order. If a job fails, the queue stops and the executor retains the scripts and logs of jobs that started. For one job, use a `SETTINGS` dictionary or `--settings`. Relative dataset and checkpoint paths are resolved from the convLexecutor checkout. See [the per-model formats](../scripts/) for independent training and evaluation jobs. Review placeholders and select the jobs you want before running the queue.

## Model and environment selection

Use `"adapter": "native"` for a built-in LeRobot policy. Set `model` to the policy name, such as `"smolvla"`. Native training and evaluation use `lerobot-train` and `lerobot-eval`. TinyVLA trains its action head, optionally with language-attention adapters, in a separate Conda environment and uses LeRobot for evaluation.

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

With Rich installed, the terminal updates the job gauge in place. Evaluation adds an episode gauge with steps elapsed and the environment's step limit. Episodes can finish before that limit. In parallel evaluation, the episode gauge shows the active batch with the lowest completion percentage and labels it with the suite and task ID. Install it with `python -m pip install rich` in the environment that runs the executor if needed. Redirected output uses plain lines so progress can be read from a file or pipeline.

The status line shows overall completion, the current stage's completed and total units, and the estimated time remaining in hours, minutes, and seconds. Training counts optimizer steps. Evaluation counts completed episodes from LeRobot's reported batches. Seconds Per Episode shows the measured elapsed time divided by the number of completed episodes, including episodes processed in parallel. The executor estimates the whole job when it has measured timings for the remaining stages, including prior compatible TinyVLA evaluation timings. Otherwise it announces the current-stage ETA when measured progress is available. Estimated Time of Completion shows a remaining duration, with a day count for durations of 24 hours or more. It shows N/A when there is no usable estimate.

- Native resume uses LeRobot's `train_config.json` and `training_state/`. Select a checkpoint or a previous run and set a target step count greater than the saved step.
- TinyVLA exact resume uses the previous training state and job settings. It restores all configured trainable tensors, including any language-attention adapters, the optimizer, random state, and sampler position. Keep the same split, base model snapshot, trainer, batch, learning rate, seed, adapter settings, and planned total updates; exact resume continues an interrupted run toward its original target. Completed training exports a full model with adapter weights merged for evaluation, while `action_head.safetensors` contains only the action head. To initialize a new run from saved weights with a fresh optimizer, choose **Start training with existing action-head weights** under either **Training Continuation** job type. That route loads the exported full model, including previously merged changes, and creates new adapters only if requested.
- **Reuse settings from a previous run** and `--queue /path/to/script.py` create a new job using saved settings. You can also copy `script.py` into `scripts/` under a new name. Previous `run.py` and `settings.json` files are supported. To evaluate an existing model, select evaluation and specify its `pretrained_model` directory.

## Add another adapter

Extend `NativeAdapter` and give the class a unique `key` and a `label`. Return a `Stage` from `train()` and `evaluate(after_training)` with its environment, command, output directory, and expected files. Keep the module importable and register it with `--plugin module:Class`.

Adapters that reuse the base planners must extend `train_fields` and `eval_fields` for their own setting names. Unknown keys are rejected with a spelling hint instead of silently ignored. Native policy dataclass values belong in `policy_options`; other supported LeRobot arguments belong in `extra`. Validate adapter-specific values in the corresponding planner. Gaussian Actor cannot use the ordinary supervised training planner: its actor/learner training requires a separate launcher.

Optional native values such as `seed=None` are serialized as LeRobot's `null` value. TinyVLA's `eval.action_steps` is adapter-specific; for native policies, use `--policy.n_action_steps=<count>` in `eval.extra` when the selected policy supports that field.

The training-options menu includes declared loss controls, knowledge insulation, detach switches, diffusion training timesteps, and gradient checkpointing. Optional Boolean and numeric fields accept `null` for the policy's automatic/default behavior. Leave an option unselected to preserve the checkpoint's existing setting. MolmoAct2 uses its internal vision-language LoRA mode and exposes its bias and discrete-loss weighting choices; the executor does not stack a second outer adapter on that recipe.

Native PushT evaluation inspects checkpoint input features and automatically selects `environment_state_agent_pos` only for the verified ENV[16] + STATE[2] nonvisual contract. Explicit observation, feature, or rename overrides take precedence. An uncached Hub checkpoint is inspected after download, before the evaluator starts. The source checkpoint is unchanged.

`native_runtime.py` wraps only native training/evaluation entrypoints in the child process. It keeps LeRobot's strict checkpoint loading and final device placement, but reads temporary safetensors through host memory when the destination is CUDA or MPS. Other commands and the original LeRobot installation remain unchanged. Parallel task copies share declared parameters in registered and auxiliary frozen modules; mutable tensor buffers, caches and processors remain task-local, and inference is serialized.

Evaluation compares the checkpoint's canonical action shape with the native environment's declared action shape before constructing the policy. A mismatch requires a compatible checkpoint or task adapter; equal dimensions do not verify robot-control semantics. Wall-OSS training recognizes only the standard `lerobot/libero` front/wrist camera pair for automatic `policy.camera_key_mapping`. Explicit camera overrides and saved valid mappings take precedence. For a cold dataset download, the native child resolves the mapping after dataset metadata becomes available and records the added flag in the stage log.

Dependency preparation failures are saved under the job's `executor/` directory as `setup.log`, `status.json`, reusable `script.py`, and a retry `run.py`. The retry script does not imply a model stage ran. Managed Conda clones store a value-free `conda-meta/convlex-source` pointer. During confirmed preparation, older validated `convlex-source.json` pointers are migrated before re-cloning; preview and inventory do not migrate them. A pinned source can reuse an already compatible managed descendant with verified path provenance; other environments remain excluded. Partial final episode batches use the shared native evaluator wrapper to cap video saving at the requested episode count.

## Verification

Automated checks live in `tests/`, separately from the executor's runtime modules. Run them with `python -B -m unittest discover -s ~/convLexecutor/tests -t ~/convLexecutor -v`, or run `python -B -m unittest discover -v` from the project root. Keep machine-specific audit scripts, downloaded weights, and run artifacts outside the source checkout.
