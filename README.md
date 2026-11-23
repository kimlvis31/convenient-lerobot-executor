# convLexecutor

[한국어 README](README.ko.md)

A job executor for LeRobot. Configure jobs interactively or run a queue of training, evaluation, and checkpoint resume jobs. The example queue covers base evaluation, trained-checkpoint evaluation, reset training, continuation with a fresh optimizer, and exact resume.

**AI authorship.** All first-party executable scripts in this project were written by AI, including the executor, example queue, preparation script, and project-specific TinyVLA integration. The bundled upstream TinyVLA code retains its original authorship and notices. See [third-party sources](THIRD_PARTY.md).

## Layout

The executor runs from this checkout and uses LeRobot from the selected Conda environment. An existing editable LeRobot installation works as well. The executor resolves the installed commands automatically.

```text
convLexecutor/
├── executor/           # Interactive executor and example preparation
├── scripts/            # Job queue files
├── tests/              # Automated checks; separate from runtime scripts
├── datasets/           # Local datasets and the LeRobot dataset cache
├── externals/TinyVLA/   # TinyVLA adapter and local base model
└── outputs/
    ├── example/         # Generated example checkpoints
    ├── training/        # Training results and checkpoints
    └── evaluation/      # Evaluation metrics and videos
```

## Manual setup

Job configuration is validated before execution: unknown training/evaluation keys and invalid batch or checkpoint intervals produce an actionable error. Use `policy_options` for native policy fields and `extra` for additional supported LeRobot flags. Gaussian Actor supports evaluation here; training requires LeRobot's separate reinforcement-learning actor/learner launcher.

Imported component and training-option names are checked against the policy fields declared by the selected LeRobot runtime version. An unavailable runtime catalog defers that check until preparation; exact resume keeps the saved configuration.

Model-specific training options include supported loss controls and gradient-checkpointing settings. Nullable controls offer automatic/default behavior through `null`; leaving a field unselected preserves the checkpoint setting. The model templates document these optional values and available enumerated choices.

The training option menu displays valid values from the selected local checkpoint instead of replacing them with constructor defaults. Unselected options remain inherited. For certified stock LIBERO evaluations, the executor reuses unchanged task descriptions and observation camera pixels for video recording. Custom environments and unknown renderer implementations retain native behavior. Simulator resets, settling steps, policy actions, and video orientation are preserved. This removes redundant requests; model inference and simulation initialization can still dominate evaluation time.

For PushT, the executor selects environment-state observations when a checkpoint explicitly requires the supported 16-value environment state and 2-value agent state without image inputs. Image checkpoints retain pixel observations. An explicit observation setting takes precedence. Required arguments added during planning appear in green in Settings.

Native checkpoint tensors are read through host memory before LeRobot places the policy on the requested device, avoiding a second full checkpoint allocation on the accelerator. Parallel evaluation also shares declared weights in auxiliary frozen modules while keeping buffers and rollout state separate. This reduces avoidable copies; the model, activations and simulators must still fit the available memory.

Before loading evaluation weights, the executor checks a checkpoint's declared action shape against the environment. Matching shapes still require matching control semantics. For Wall-OSS training on the standard `lerobot/libero` front/wrist camera pair, it supplies the verified camera-slot mapping unless explicitly overridden. A dataset downloaded during execution receives the same check once its metadata is available.

When dependency preparation fails before a model stage starts, the result includes a concise cause and a link to `executor/setup.log`. Saved runs also include reusable settings, a retry `run.py`, and a failed `status.json` with no completed model stages. Managed Conda clones use a metadata marker that Conda does not treat as a package record; confirmed preparation migrates the older executor marker before re-cloning, while inventory and dry runs remain read-only. A pinned evaluation environment can reuse a verified, already compatible managed descendant of that exact source; unrelated environments are excluded. Partial final episode batches save videos only for the requested episodes.

The following commands use Bash on Linux with Git and Conda available.

The verified configuration uses Linux, Conda, LeRobot 0.6.2 at revision `145cc7ea53e5864bd5efec4e9bb70ee799e47bd8`, and an NVIDIA device that supports bfloat16. Reuse an existing compatible Conda environment when available. On a fresh machine, install the [LIBERO extra](https://huggingface.co/docs/lerobot/libero) in the native `lerobot` environment.

```bash
git clone https://github.com/kimlvis31/convenient-lerobot-executor.git ~/convLexecutor
cd ~/convLexecutor
conda create -n lerobot python=3.12 pip
conda run -n lerobot python -m pip install \
  "lerobot[libero] @ git+https://github.com/huggingface/lerobot.git@145cc7ea53e5864bd5efec4e9bb70ee799e47bd8"
conda run -n lerobot lerobot-train --help
```

TinyVLA runs in a separate model environment. The verified setup used Python 3.10, PyTorch 2.0.1 with CUDA 11.7, torchvision 0.15.2, Transformers 4.37.1, Diffusers 0.11.1, and `huggingface-hub` 0.23.5. The commands below use the [bundled upstream requirements](externals/TinyVLA/sources/TinyVLA/requirements.txt). Check that its PyTorch build supports your graphics driver. Skip the TinyVLA setup and example preparation if you only use native LeRobot policies. The executor recognizes `tinyvla_infer` and `lerobot_tinyvla`. Use `--model-env` during preparation and `--env-map` in the executor for other names.

```bash
conda create -n tinyvla_infer python=3.10 pip
conda run -n tinyvla_infer python -m pip install -r externals/TinyVLA/sources/TinyVLA/requirements.txt
conda run -n tinyvla_infer python -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
conda run -n lerobot python -m pip install --no-deps --no-build-isolation --editable \
  "./externals/TinyVLA/lerobot_policy_tinyvla"
```

Install the bundled evaluation bridge in editable mode so `git pull` updates the application and bridge directly; if an older setup installed a copied package, rerun the bridge installation command above once.

## Prepare and run the examples

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
python -B ~/convLexecutor/executor/prepare_examples.py --dry-run
python -B ~/convLexecutor/executor/prepare_examples.py
python -B ~/convLexecutor/executor/manage.py --inventory
python -B ~/convLexecutor/executor/manage.py \
  --queue ~/convLexecutor/scripts/example_script.py
```

Preparation fetches the pinned public base weights and the files needed for one cream-cheese demonstration, then trains four action-head updates. It saves the full model at step 4 and the optimizer, weights, and random state at step 2 under `outputs/example/`. It verifies and reuses matching local files. The resume fixture refers to the installed base model on the machine where it was generated. Run the preparation command on each machine.

If an update makes a generated example resume state incompatible with its source checks, preserve existing models and user training states, prepare replacement examples in a new `--checkpoint-root` location, and follow the [staging and verification guide](externals/TinyVLA/README.md#regenerate-source-bound-example-resume-states) before replacing a recognized generated example.

For interactive setup, run `manage.py` and choose **Configure a new job**. Select a model from the combined list and choose the evaluation benchmark or LIBERO suite as prompted. Press Enter to use the displayed default. To run a prepared queue, choose **Import a job queue script** or use `--queue`. Choose the policy explicitly when prompted. The [example queue](scripts/example_script.py) and [per-model formats](scripts/) describe the fields inline. Each format includes a training job and a standalone evaluation job, with required settings, optional overrides, and any task-specific prerequisites.

When a job starts, convLexecutor reuses a Conda environment for the selected policy and benchmark where possible. It installs missing dependencies declared by the installed LeRobot version, including policy extras such as the SentencePiece package for the π policies. It keeps installed versions pinned and creates a separate environment if the dependencies conflict. Source benchmarks with a managed setup recipe are prepared when the confirmed job first needs them. Dry runs do not install packages. Benchmarks that still require **Manual Setup** show their missing runtime conditions. Installed LeRobot environment plugins are discovered automatically. Checkpoint selection remains part of job setup. PushT and ALOHA evaluate one task per job. Their default batch evaluation runs the environment steps synchronously and keeps batched policy inference. Meta-World uses its simulator default episode length.

**Parallel Evaluation Tasks** sets the number of distinct tasks evaluated together for native policies and TinyVLA on multi-task benchmarks. **Evaluation Batch Size** sets the simultaneous episodes within each task. For example, 3 tasks with batch 2 can advance six episodes together. The executor isolates policy state and processors per task, shares native weight parameters, and serializes inference. Available task counts cap concurrency; per-task caches and simulators require additional memory. PushT and ALOHA use episode batching for their single task. Imported queues use `eval.parallel_tasks`, defaulting to 1. [Evaluation settings and runtime limits](executor/README.md)

New manager-created dependency clones inherit the source environment's declared `conda-meta/state.env_vars` in child-process memory, including supported model access and cache settings, while preserving the target's controls and the project's dataset location. The clone records only its source prefix; credential values are not copied into that record, settings, or logs. This supports declared variables and does not replay arbitrary source `activate.d` scripts or their computed settings.

For supported Linux builds, confirmed FLUX.3 setup prepares a matching official `natten` binary and verifies its imports without changing the selected Torch installation; unsupported combinations report the required conditions instead of silently compiling from source. [Official extension installation requirements](https://natten.org/install/)

If a policy or one of its dependencies requires Hugging Face access, Settings lists known model pages. For example, GR00T N1.7 can require access to Cosmos-Reason2-2B even when the policy checkpoint is public. Open the reported repository page with the account used on the execution machine, accept the terms or request approval, and retry with a token that can read that repository. The Result section also reports access errors for repositories discovered during the run. [Hugging Face gated model access](https://huggingface.co/docs/hub/models-gated)

Evaluation capability is checked separately from package installation. The inspected stock Unitree G1 MuJoCo module receives motor commands through external DDS (Data Distribution Service); its `step(action)` does not apply policy actions or define task success and episode termination. The executor stops this stock default, `main`, or inspected revision before preparing dependencies and explains that a task adapter is required. Supply a real task environment through `--env.hub_path=owner/task-environment` or a LeRobot environment plugin. Other explicitly selected revisions remain selectable with an unverified-capability notice; their task behavior still needs checking. [Inspected Unitree source](https://huggingface.co/lerobot/unitree-g1-mujoco/blob/b00f8b114c3d5d98b44b7ad0285d631af73f4bc5/env.py)

For the published `lerobot/FineART-VLA-subtask-midtrained` checkpoint with stock ALOHA, Settings explains a known joint/control mismatch: five arm joints and two gripper values per arm versus six arm joints and one normalized gripper. Both have 14 values, but equal shapes and camera renaming do not establish compatible joint order, units, or control. Use a checkpoint fine-tuned for that environment or a matching task adapter. This notice is specific to the known checkpoint and stock environment; it does not prohibit adapted checkpoints or custom environments. [Published FineART checkpoint](https://huggingface.co/lerobot/FineART-VLA-subtask-midtrained)

For a local training dataset, select its directory. If its `selection.json` records a repository and a training split, convLexecutor uses both. TinyVLA automatically uses the local split when it matches the selected tasks. Enter a Hugging Face repository name if you select **Use a dataset repository**.

The policy list marks models without a local checkpoint as <No Local Checkpoint>. The Hugging Face cache counts as local. Choose a checkpoint trained for the selected benchmark. Native training shows supported scope, LoRA (Low-Rank Adaptation), and training options from the selected LeRobot version. TinyVLA can train the action head or combine it with language-attention LoRA. Omitting the LoRA settings selects action-head training. Exact resume restores the prior settings and optimizer state.

Verified saved camera aliases and known virtual camera slots are handled automatically while preserving the original checkpoint configuration and masks; explicit camera or policy overrides take precedence.

Training Scope lists the compatible component combinations together. The executor checks local dataset and checkpoint inputs before training, preserves the original dataset, and shows required arguments it added in green in Settings. Failed subprocesses show a short cause and the full log path in Result.

**Settings** shows the Conda environment for each stage, the reason it was selected, and the model and simulator package requirements with the versions found there. TinyVLA lists its model interpreter and the LeRobot dataset reader or evaluation runner separately. An unverified entry means the executor could not confirm it. `--dry-run` previews this information without installing packages. If dependency preparation changes the selection, the executor shows the resolved environment and saves it in the run script.

Each result has an `executor/` subdirectory with its executed `run.py`, reusable `script.py`, job status, and the stage console log. Evaluation and training results go to `outputs/evaluation/` and `outputs/training/`. Put local datasets in `datasets/`. LeRobot datasets fetched by the executor are cached in `datasets/.hub/`. The repository tracks code, configuration, and source manifests. Downloads, weights, logs, and outputs are local artifacts.

Choose **N** at **Save Job Outputs?** to run in temporary storage and see the result in the terminal. Set `"save_outputs": False` for a job in a Python job queue. Temporary runs discard their outputs and logs after execution. A train-and-evaluate job keeps its checkpoint until evaluation finishes.

See the [executor guide](executor/README.md), [TinyVLA guide](externals/TinyVLA/README.md), and [agent setup guide](AGENTS.md) for more details.

## RoboMME evaluation

Choose **RoboMME** in the evaluation wizard or set `eval.env` to `"robomme"` in a native job. The default `eval.robomme_runtime` is `"auto"`. The executor first looks for a combined environment that can run both the policy and the simulator. If those dependencies conflict, it runs the policy in the native LeRobot environment and starts simulation workers in a separate environment. The official LeRobot evaluator still produces the video, metrics, and execution log.

The first split run searches installed Conda environments before building a simulator under the user cache on Linux. The managed installation needs network access, an NVIDIA driver compatible with CUDA 12.6 or newer, and a working Vulkan renderer. It tests one physical step before reuse. Set `CONVLEX_ROBOMME_CACHE` to choose the installation directory. The execution screen shows the simulator Python path.

Set `eval.robomme_runtime` to `"shared"` to require a combined environment, or `"split"` to request separate processes. Set `eval.simulator_environment` to a Conda environment name or an absolute prefix to select an existing simulator explicitly. `eval.environment` selects the policy environment. [Official RoboMME setup and task IDs](https://huggingface.co/docs/lerobot/main/robomme)

## License

The first-party project is under [MIT](LICENSE). See [third-party sources](THIRD_PARTY.md) for bundled upstream software and public assets.

## RoboTwin runtime

On Linux x86-64, selecting RoboTwin for a confirmed evaluation automatically prepares its pinned source, robot/object assets, and compatible simulator environment when no verified runtime can be reused. The simulator runs in a separate interpreter so its NumPy and native-library requirements do not replace the policy environment's packages. Setup includes the matching CUDA PyTorch runtime, SAPIEN, MPlib, PyTorch3D, CuRobo, and the required upstream compatibility patches. An NVIDIA driver and a working Vulkan renderer are prerequisites. The manager tests rendered cameras and an actual simulation step before marking a runtime ready. The policy still runs through the installed LeRobot evaluator and checkpoint camera mapping.

An existing setup remains usable through `ROBOTWIN_ROOT` or a discoverable `envs` package. The executor checks its required files and MPlib/NumPy compatibility and runs from the simulator asset root. [Pinned upstream integration guide](https://github.com/huggingface/lerobot/blob/145cc7ea53e5864bd5efec4e9bb70ee799e47bd8/docs/source/robotwin.mdx)

## Managed source benchmark setup

The first confirmed evaluation can also prepare LIBERO-plus, RoboCasa365, and VLABench from pinned upstream sources. These recipes clone the selected compatible LeRobot Conda environment into a manager-owned prefix, install the simulator's explicit dependencies, fetch its assets, and check the native LeRobot reset/action/step path. Setup does not download policy weights. Required model checkpoints remain a separate job setting.

| Benchmark | Prepared runtime and assets |
| --- | --- |
| LIBERO-plus | The exact fork and robosuite 1.4.1, private LIBERO configuration, ImageMagick/fontconfig inside the new prefix, and the official 6.40 GB perturbation archive with its published checksum. The clone replaces its own stock `hf-libero`; the original Conda environment keeps its installation. |
| RoboCasa365 | Pinned RoboCasa/robosuite source trees, compatible numerical libraries, kitchen textures and fixtures, and the selected object registries. The default is `lightwheel`. An explicit `--env.obj_registries` request adds its corresponding asset packs. Upstream source metadata stays intact; the runtime imports the pinned source directly to avoid RoboCasa's old `lerobot==0.3.3` installer requirement. |
| VLABench | Pinned VLABench/path-planner source, MuJoCo 3.2.2, dm_control 1.0.22, Open3D, Rtree 1.4.1, and the official scene/object archives. Large Google Drive downloads use bounded HTTP (Hypertext Transfer Protocol) Range requests and resume from preserved partial files. The archives total 6.00 GB and the extracted asset set needs roughly 12 GB, with additional space for the environment and extraction. |

Managed source runtimes live under `externals/.runtimes/` in this checkout. Set `CONVLEX_RUNTIME_CACHE` to an absolute path to use another disk. The cache records source revisions, asset hashes, dependency requirements, and the verified environment. An interrupted preparation never publishes a ready environment. Completed archives remain available for retries until the entire runtime passes package and physical checks; publication then removes the download directory. A later run reuses a compatible runtime and checks its physical simulation path again. Original LeRobot source, user datasets, checkpoints, and existing Conda environments are preserved.

Managed source environments preserve the selected policy environment's Hugging Face and Torch cache paths. Value-free Conda activation hooks inherit supported runtime and model access variables in child-process memory and restore previous values on deactivation. Credential values are not copied into job settings, manifests, or hook files. Managed simulator and compiler paths take precedence.

Fresh setup, physical reset and step, rendered cameras, and environment reuse were verified for all four source recipes with LeRobot 0.6.2 on Linux x86-64. RoboTwin also completed two queued synchronous jobs and an asynchronous job with batch size 2, six short episodes in total, using a temporary initialized policy. Script/log storage and worker cleanup after unexpected parent termination were verified. These checks establish installation and execution behavior; policy performance requires trained checkpoints and a separate evaluation.

For development smoke tests, point `CONVLEX_RUNTIME_CACHE` and job outputs to dedicated temporary folders and remove only those folders when the check finishes. Persistent simulator environments and assets are application dependencies needed for future runs; clearing them causes setup to run again. A missing driver, incompatible hardware, upstream download restriction, or unavailable source version produces a setup diagnosis and its log instead of a false ready status.
