# convLexecutor

[한국어 README](README.ko.md)

A job executor for LeRobot. Configure jobs interactively or run a queue of training, evaluation, and checkpoint resume jobs. The example queue covers base evaluation, trained-checkpoint evaluation, reset training, continuation with a fresh optimizer, and exact resume.

**AI authorship.** All first-party executable scripts in this project were written by AI, including the executor, example queue, preparation script, and project-specific TinyVLA integration. The bundled upstream TinyVLA code retains its original authorship and notices. See [third-party sources](THIRD_PARTY.md).

## Layout

Keep the two checkouts side by side. Use the executor and integrations from `convLexecutor/`. LeRobot keeps datasets and model outputs under its own working directory.

```text
~/
├── convLexecutor/
│   ├── executor/                 # CLI, queue definitions, and per-job logs
│   ├── externals/TinyVLA/        # Model integration and generated base weights
│   └── scripts/prepare_examples.py
└── lerobot/
    ├── datasets/example/libero/   # Files fetched and prepared on this machine
    └── outputs/                  # Evaluation, training, and example checkpoints
```

## Setup

The verified configuration uses Linux, Conda, LeRobot 0.6.2 at revision `145cc7ea53e5864bd5efec4e9bb70ee799e47bd8`, and an NVIDIA device that supports bfloat16. Reuse an existing compatible LeRobot checkout and environment when available. On a fresh machine, install the [LIBERO extra](https://huggingface.co/docs/lerobot/libero) in the native `lerobot` environment.

```bash
git clone https://github.com/kimlvis31/convenient-lerobot-executor.git ~/convLexecutor
git clone https://github.com/huggingface/lerobot.git ~/lerobot
git -C ~/lerobot checkout --detach 145cc7ea53e5864bd5efec4e9bb70ee799e47bd8
conda create -n lerobot python=3.12 pip
conda run -n lerobot python -m pip install -e "$HOME/lerobot[libero]"
```

TinyVLA runs in a separate model environment. The verified setup used Python 3.10, PyTorch 2.0.1 with CUDA 11.7, torchvision 0.15.2, Transformers 4.37.1, Diffusers 0.11.1, and `huggingface-hub` 0.23.5. For a new environment, use the [bundled upstream requirements](externals/TinyVLA/sources/TinyVLA/requirements.txt) as the dependency reference and check that the selected PyTorch build supports the machine's NVIDIA graphics driver. The executor recognizes `tinyvla_infer` and `lerobot_tinyvla`. Use `--model-env` during preparation and `--env-map` in the executor for other names.

```bash
conda run -n lerobot python -m pip install --no-deps --no-build-isolation \
  "$HOME/convLexecutor/externals/TinyVLA/lerobot_policy_tinyvla"
```

## Prepare and run the examples

```bash
python -B ~/convLexecutor/scripts/prepare_examples.py --lerobot-root ~/lerobot --dry-run
python -B ~/convLexecutor/scripts/prepare_examples.py --lerobot-root ~/lerobot
python -B ~/convLexecutor/executor/manage.py --lerobot-root ~/lerobot --inventory
python -B ~/convLexecutor/executor/manage.py --lerobot-root ~/lerobot \
  --queue ~/convLexecutor/executor/scripts/example_script.py
```

Preparation fetches the pinned public base weights and the files needed for one cream-cheese demonstration, then trains four action-head updates. It saves the full model at step 4 and the optimizer, weights, and random state at step 2 under `~/lerobot/outputs/example/`. It verifies and reuses matching local files. The resume fixture refers to the installed base model on the machine where it was generated. Run the preparation command on each machine.

For interactive use, run `manage.py` without `--queue` and choose **Import a job queue script**. A queue is a `JOBS` list of job dictionaries. Use `"adapter": "native"` and `"model": "smolvla"` for SmolVLA. The [example](executor/scripts/example_script.py) explains the fields inline. Run with `--dry-run` to preview a plan.

Each job records its executed `run.py`, reusable `script.py`, timings, and console output in `executor/log/<run-name>/`. Evaluation and training results go to `~/lerobot/outputs/eval/` and `~/lerobot/outputs/training/`. The repository tracks code, configuration, and source manifests. Downloads, weights, logs, and outputs are local artifacts.

See the [executor guide](executor/README.md), [TinyVLA guide](externals/TinyVLA/README.md), and [agent setup guide](AGENTS.md) for more details.

## License

The first-party project is under [MIT](LICENSE). See [third-party sources](THIRD_PARTY.md) for bundled upstream software and public assets.
