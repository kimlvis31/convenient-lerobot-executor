# convLexecutor

[한국어 README](README.ko.md)

A job executor for LeRobot. Configure jobs interactively or run a queue of training, evaluation, and checkpoint resume jobs. The example queue covers base evaluation, trained-checkpoint evaluation, reset training, continuation with a fresh optimizer, and exact resume.

**AI authorship.** All first-party executable scripts in this project were written by AI, including the executor, example queue, preparation script, and project-specific TinyVLA integration. The bundled upstream TinyVLA code retains its original authorship and notices. See [third-party sources](THIRD_PARTY.md).

## Layout

The executor runs from this checkout and uses LeRobot from the selected Conda environment. An existing editable LeRobot installation works as well. The executor resolves the installed commands automatically.

```text
convLexecutor/
├── executor/           # Interactive executor and per-job logs
├── scripts/            # Job queues and example preparation
├── datasets/           # Local datasets and the LeRobot dataset cache
├── externals/TinyVLA/   # TinyVLA adapter and local base model
└── outputs/
    ├── example/         # Generated example checkpoints
    ├── training/        # Training results and checkpoints
    └── evaluation/      # Evaluation metrics and videos
```

## Manual setup

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
conda run -n lerobot python -m pip install --no-deps --no-build-isolation \
  "./externals/TinyVLA/lerobot_policy_tinyvla"
```

## Prepare and run the examples

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
python -B ~/convLexecutor/scripts/prepare_examples.py --dry-run
python -B ~/convLexecutor/scripts/prepare_examples.py
python -B ~/convLexecutor/executor/manage.py --inventory
python -B ~/convLexecutor/executor/manage.py \
  --queue ~/convLexecutor/scripts/example_script.py
```

Preparation fetches the pinned public base weights and the files needed for one cream-cheese demonstration, then trains four action-head updates. It saves the full model at step 4 and the optimizer, weights, and random state at step 2 under `outputs/example/`. It verifies and reuses matching local files. The resume fixture refers to the installed base model on the machine where it was generated. Run the preparation command on each machine.

For interactive setup, run `manage.py` and choose **Configure a new job**. Select a model from the combined list and choose the evaluation benchmark or LIBERO suite as prompted. Press Enter to use the displayed default. To run a prepared queue, choose **Import a job queue script** or use `--queue`. The [example](scripts/example_script.py) explains the fields inline.

Each result has an `executor/` subdirectory with its executed `run.py`, reusable `script.py`, job status, and the stage console log. Evaluation and training results go to `outputs/evaluation/` and `outputs/training/`. Put local datasets in `datasets/`. LeRobot datasets fetched by the executor are cached in `datasets/.hub/`. The repository tracks code, configuration, and source manifests. Downloads, weights, logs, and outputs are local artifacts.

See the [executor guide](executor/README.md), [TinyVLA guide](externals/TinyVLA/README.md), and [agent setup guide](AGENTS.md) for more details.

## License

The first-party project is under [MIT](LICENSE). See [third-party sources](THIRD_PARTY.md) for bundled upstream software and public assets.
