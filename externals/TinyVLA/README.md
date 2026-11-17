# TinyVLA Deployment

The executor finds this integration at `externals/TinyVLA/` in the convLexecutor checkout. Use `--tinyvla-root` to override the location.

| Path | Purpose |
| --- | --- |
| `sources/TinyVLA/` | Bundled TinyVLA upstream code and license |
| `models/Llava-Pythia-700M/` | Model configuration. The preparation script adds the public base weights and tokenizer vocabulary here. |
| `runtime/tinyvla_loader.py` | Loads the pretrained vision-language weights and the action head |
| `lerobot_policy_tinyvla/` | Connects TinyVLA inference to LeRobot evaluation |
| `training/train_head.py` | Trains the action head on one or all 40 tasks and resumes from checkpoints |
| `training/lerobot_data.py` | Selects training episodes, serves batches, and exports episodes for offline training |

See [third-party sources](../../THIRD_PARTY.md) for upstream code and model assets.

## Setup

Use a LeRobot environment with the LIBERO benchmark. Install the evaluation adapter in the `lerobot` environment. Model inference requires a `tinyvla_infer` or `lerobot_tinyvla` environment and a device that supports bfloat16. The worker records its package versions in `ready.json`.

```bash
conda run -n lerobot python -m pip install --no-deps --no-build-isolation "$HOME/convLexecutor/externals/TinyVLA/lerobot_policy_tinyvla"
```

The default model directory, `models/Llava-Pythia-700M/`, needs the original `model.safetensors`, `tokenizer.json`, and model configuration. To use an existing copy elsewhere, give its directory to the executor or pass `--policy.model_path` to LeRobot.

## Training and evaluation

- Training updates the action head and supports one or all 40 tasks. The dataset revision and training and held-out episodes are recorded in `selection.json`. Future-action chunks respect episode boundaries and mark padding after the last action.
- `train/pretrained_model/` holds the full model and action-head weights. When `action_head.safetensors` is present, the evaluation worker verifies that the saved action-head tensors were loaded.
- `train/checkpoints/training_state.pt` stores the action head, optimizer state, step count, and random state. The trainer replaces this file at each configured save interval. Resume requires the same training settings, selection, base weights, and runtime environment.
- An older run's `train/pretrained_model/` and `selection.json` can initialize a new run with a fresh optimizer.
- The evaluation adapter expects two images, eight state values, seven action values, and chunks of up to 16 actions. Use one task at a time, 256-pixel images, and relative control. The worker evaluates observation batches on the graphics device. Batch capacity depends on available device memory. The adapter preserves the original camera processing, action bounds, and gripper sign.

The executor passes `--policy.model_path` and `--policy.worker_log_dir`. The original `--policy.step7_script` option remains supported. Each evaluation saves worker logs in `eval/worker_logs/` under its output directory. The adapter removes the worker code from its temporary directory at shutdown. Job script retention is handled by the executor.

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
