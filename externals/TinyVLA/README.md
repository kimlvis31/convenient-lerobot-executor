# TinyVLA Deployment

The executor finds this integration at `externals/TinyVLA/` in the convLexecutor checkout. Use `--tinyvla-root` to override the location.

| Path | Purpose |
| --- | --- |
| `sources/TinyVLA/` | Bundled TinyVLA upstream code and license |
| `models/Llava-Pythia-700M/` | Model configuration. The preparation script adds the public base weights and tokenizer vocabulary here. |
| `runtime/tinyvla_loader.py` | Loads the pretrained vision-language weights and the action head |
| `lerobot_policy_tinyvla/` | Connects TinyVLA inference to LeRobot evaluation |
| `training/train_head.py` | Trains the action head, optionally with language-attention adapters, on one or all 40 tasks and resumes from checkpoints |
| `training/lerobot_data.py` | Selects training episodes, serves batches, and exports episodes for offline training |

See [third-party sources](../../THIRD_PARTY.md) for upstream code and model assets.

## Setup

Use a LeRobot environment with the LIBERO benchmark. Install the evaluation adapter in the `lerobot` environment. Model inference requires a `tinyvla_infer` or `lerobot_tinyvla` environment and a device that supports bfloat16. The worker records its package versions in `ready.json`.

```bash
conda run -n lerobot python -m pip install --no-deps --no-build-isolation --editable "$HOME/convLexecutor/externals/TinyVLA/lerobot_policy_tinyvla"
```

Editable installation lets `git pull` update the application and bundled evaluation bridge directly; rerun this bridge installation command once to replace an older copied-package installation.

The default model directory, `models/Llava-Pythia-700M/`, needs the original `model.safetensors`, `tokenizer.json`, and model configuration. To use an existing copy elsewhere, give its directory to the executor or pass `--policy.model_path` to LeRobot.

## Training and evaluation

- Training supports one or all 40 tasks. **Training Scope** offers the action head alone or the action head plus language-attention LoRA (Low-Rank Adaptation). The latter adds adapters to the GPT-NeoX language backbone's attention query/key/value projections; other backbone weights remain frozen. In a job dictionary, omit `train.lora` or set it to `None` for the action head alone, or set `train.lora` to `{"rank": 16, "alpha": 16, "dropout": 0.0}` for adapters. Rank and alpha must be positive integers, and dropout must be in `[0, 1)`. The direct trainer equivalent is `--lora-rank 16 --lora-alpha 16 --lora-dropout 0.0`; its default rank of 0 trains the action head alone. See [the job format](../../scripts/format_tinyvla.py).
- The dataset revision and training and held-out episodes are recorded in `selection.json`. Future-action chunks respect episode boundaries and mark padding after the last action.
- `train/pretrained_model/` holds the full exported model, with language-attention adapter weights merged when used. The separate `action_head.safetensors` contains only action-head tensors. When that file is present, the evaluation worker verifies that those saved tensors were loaded. Evaluation uses the full exported model and does not need to recreate the training adapters.
- `train/checkpoints/training_state.pt` stores all trainable tensors, including any language-attention adapters, optimizer state, step count, sampler position, and random state. The trainer replaces this file at each configured save interval. Exact resume requires the same dataset selection, base model snapshot, loader/trainer, batch size, learning rate, seed, adapter settings, and planned total updates, with a compatible runtime. It continues an interrupted run toward its original target; changing the target or training configuration requires a new run.
- An older run's `train/pretrained_model/` and `selection.json` can initialize a new run with a fresh optimizer. This route loads the full exported model, including any previously merged language changes, and creates new adapters only if requested. It does not restore the previous optimizer or unmerged adapter state.
- The evaluation adapter expects two images, eight state values, seven action values, and chunks of up to 16 actions. Use 256-pixel images and relative control. The executor can run several LIBERO tasks concurrently. Each task has its own action queue and processors, and one GPU worker handles requests in sequence. The worker evaluates observation batches on the graphics device. Batch capacity depends on available device memory. The adapter preserves the original camera processing, action bounds, and gripper sign.

The executor passes `--policy.model_path` and `--policy.worker_log_dir`. The original `--policy.step7_script` option remains supported. Each evaluation saves worker logs in `eval/worker_logs/` under its output directory. The adapter removes the worker code from its temporary directory at shutdown. Job script retention is handled by the executor.

## Regenerate source-bound example resume states

Generated example resume states are tied to the trainer, loader, base snapshot, and training configuration used to create them. A source update can make an old example fail verification even when its exported model remains valid. Preserve the existing dataset, base weights, exported models, and all user training states; do not edit saved source hashes or relax the checks.

From the checkout, use an unused staging directory with no existing checkpoint fixtures:

```bash
python -B executor/prepare_examples.py --checkpoint-root outputs/example-refresh --dry-run
python -B executor/prepare_examples.py --checkpoint-root outputs/example-refresh
```

The preparation script verifies the generated fixtures before publishing them in that staging location. Check the staged model and a step-2-to-step-4 resume before changing the normal example paths. You can point your queue at the staged fixtures, or replace only the recognized generated example that needs refreshing after verifying its identity and result. When moving a generated `tinyvla_resume` fixture, update its `settings.json` field `train.selection` to the final `selection.json` path; preserve unrelated examples and user outputs.

This procedure regenerates provided practice fixtures and does not migrate real training history. An older real training state needs its original source version for exact resume, or a new training run initialized from its exported weights with a fresh optimizer.

## Verify the evaluation connection

The following command runs one episode for task 1 in `libero_object`. To verify a trained checkpoint, set `--policy.model_path` to its `pretrained_model/` directory.

```bash
mkdir -p "$HOME/convLexecutor/outputs/evaluation"
run_dir=$(mktemp -d "$HOME/convLexecutor/outputs/evaluation/tinyvla_check_XXXXXX")
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run --no-capture-output -n lerobot lerobot-eval \
  --policy.type=tinyvla --policy.device=cpu --policy.n_action_steps=16 \
  --policy.workspace="$HOME/convLexecutor/externals/TinyVLA" \
  --policy.worker_log_dir="$run_dir/worker_logs" \
  --env.type=libero --env.task=libero_object --env.task_ids='[1]' \
  --env.obs_type=pixels_agent_pos --env.control_mode=relative \
  --env.observation_height=256 --env.observation_width=256 --env.max_parallel_tasks=1 \
  --eval.batch_size=1 --eval.n_episodes=1 --seed=0 --output_dir="$run_dir" \
  > "$run_dir/run.log" 2>&1
```
