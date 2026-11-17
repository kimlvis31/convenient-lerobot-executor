import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from adapters import NativeAdapter, Stage, TinyVLAAdapter, load_adapters, read_json
from inventory import Inventory, scan
from manage import number, plan, run_plans, settings_jobs, wizard
from runner import Progress, execute, load_settings, resolve_arguments


FAKE_BACKEND = r'''
from pathlib import Path
import sys

args = sys.argv[1:]
if args[0] == 'python':
    output = Path(args[args.index('--output') + 1])
    model = output / 'pretrained_model'
    model.mkdir(parents=True)
    (model / 'model.safetensors').write_bytes(b'test')
    (model / 'action_head.safetensors').write_bytes(b'head')
    (output / 'checkpoints').mkdir()
    (output / 'checkpoints/training_state.pt').write_bytes(b'resume')
    (output / 'training_result.json').write_text('{"steps":8}')
    (output.parent / 'selection.json').write_text('{"train_episodes":[0]}')
    print('step:8/8 completed', flush=True)
    sys.exit(0)
output = Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--output_dir=')))
if args[0] == 'lerobot-train':
    steps = int(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--steps=')))
    if '--resume=true' in args:
        previous = Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--config_path=')))
        assert (previous.parent.parent / 'training_state/training_step.json').is_file(), previous
    model = output / 'checkpoints/last/pretrained_model'
    model.mkdir(parents=True)
    (model / 'train_config.json').write_text('{"policy":{"type":"smolvla"}}')
    (model / 'config.json').write_text('{"type":"smolvla"}')
    (model / 'model.safetensors').write_bytes(b'fixture-model')
    state = model.parent / 'training_state'
    state.mkdir()
    (state / 'training_step.json').write_text('{"step":%d}' % steps)
    print('step:4 smoothed loss:0.2', flush=True)
    print(f'Checkpoint policy after step {steps}', flush=True)
elif args[0] == 'lerobot-eval':
    model = Path(next(arg.split('=', 1)[1] for arg in args
                      if arg.startswith(('--policy.path=', '--policy.model_path='))))
    assert model.is_dir(), model
    output.mkdir(parents=True)
    print('Stepping through eval batches: 0%|          | 0/2 [00:00<?, ?it/s]', flush=True)
    print('Stepping through eval batches: 100%|##########| 2/2 [00:00<00:00, 2it/s]', flush=True)
    if '--seed=0' in args:
        (output / 'eval_info.json').write_text('{"overall":{"pc_success":0}}')
        video = output / 'videos/test.mp4'
        video.parent.mkdir()
        video.write_bytes(b'test-video')
'''


def sample_settings():
    return {
        "adapter": "native", "model": "smolvla", "schema": 1,
        "train": {"steps": 8, "dataset": "example/dataset", "device": "cpu", "batch": 1,
                  "seed": 0, "save_freq": 4, "extra": []},
        "eval": {"env": "libero", "tasks": ["libero_object"], "task_ids": [1], "batch": 1,
                 "episodes": 2, "seed": 0, "device": "cpu", "control": "relative", "size": 256},
    }


class ManagerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.logs = self.root / "executor/log"
        self.external = self.root / "externals/TinyVLA"
        self.envs = {"native": "lerobot", "tinyvla": "tinyvla_infer", "smolvla": "smolvla_env"}
        self.backend = self.root / "backend.py"
        self.backend.write_text(FAKE_BACKEND, encoding="utf-8")
        self.environment = {"python": sys.executable, "tinyvla_bridge": str(self.backend), "entrypoints": {
            name: {"source": str(self.backend), "launcher": str(self.backend)}
            for name in ("lerobot-train", "lerobot-eval")
        }}

    def test_train_eval_outputs_and_native_resume(self):
        settings = sample_settings()
        run = self.logs / "smolvla-run"
        adapter = NativeAdapter(settings, run.name, self.root, self.external, self.envs)
        text = io.StringIO()
        with patch("runner.resolve_environment", return_value=self.environment), \
             patch("runner.conda_command", side_effect=lambda c, e, argv: [sys.executable, "-u", str(self.backend), *argv]), \
             contextlib.redirect_stdout(text):
            execute([adapter.train(), adapter.evaluate(True)], settings, run, self.root, "fake", False)
        status = read_json(run / "status.json")
        self.assertEqual(status["status"], "completed")
        self.assertTrue(status["ended_at"] >= status["started_at"])
        self.assertGreater(status["seconds"], 0)
        script = run / "run.py"
        self.assertEqual(load_settings(script), settings)
        self.assertIn("smolvla_env", script.read_text(encoding="utf-8"))
        reusable = run / "script.py"
        self.assertEqual(settings_jobs(load_settings(reusable)), [settings])
        self.assertEqual(set(run.glob("*.py")), {script, reusable})
        self.assertTrue((run / "train.log").is_file())
        self.assertTrue((run / "eval.log").is_file())
        self.assertTrue((adapter.eval_dir / "videos/test.mp4").is_file())
        self.assertFalse(list((self.root / "outputs").rglob("*.py")))
        inventory = scan(self.root, self.external, self.logs)
        self.assertEqual(inventory.settings, [reusable])
        self.assertIn(adapter.eval_dir / "eval_info.json", inventory.results)
        self.assertTrue(inventory.native_resume)
        self.assertIn("Overall 100%", text.getvalue())
        settings["train"] = {"resume": str(adapter.train_dir), "steps": 10, "save_freq": 4}
        retry = NativeAdapter(settings, "resume-run", self.root, self.external, self.envs)
        resumed = retry.train()
        self.assertIn("--resume=true", resumed.arguments)
        self.assertIn(f"--output_dir={retry.train_dir / 'train'}", resumed.arguments)
        self.assertEqual(resumed.start, 8)
        with patch("runner.resolve_environment", return_value=self.environment), \
             patch("runner.conda_command", side_effect=lambda c, e, argv: [sys.executable, "-u", str(self.backend), *argv]), \
             contextlib.redirect_stdout(io.StringIO()):
            execute([resumed], settings, self.logs / retry.run_name, self.root, "fake", False)
        self.assertEqual(read_json(retry.train_dir / "train/checkpoints/last/training_state/training_step.json")["step"], 10)

    def test_zero_exit_without_eval_artifact_is_failed(self):
        settings = sample_settings()
        run = self.logs / "failed"
        settings["eval"].update(seed=1, policy=str(self.root))
        adapter = NativeAdapter(settings, run.name, self.root, self.external, self.envs)
        with patch("runner.resolve_environment", return_value=self.environment), \
             patch("runner.conda_command", side_effect=lambda c, e, argv: [sys.executable, "-u", str(self.backend), *argv]), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                execute([adapter.evaluate(False)], settings, run, self.root, "fake", False)
        self.assertEqual(read_json(run / "status.json")["status"], "failed")

    def test_tinyvla_routes_each_stage_to_its_environment(self):
        for name in ("training/train_head.py", "training/lerobot_data.py", "runtime/tinyvla_loader.py",
                     "models/Llava-Pythia-700M/config.json", "models/Llava-Pythia-700M/model.safetensors"):
            path = self.external / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        settings = sample_settings()
        settings.update(adapter="tinyvla", model="tinyvla", train={
            "task": ["libero_object", 1], "batch": 1, "updates": 8, "seed": 0, "lr": 0.001, "save_freq": 4,
        })
        run = self.logs / "tinyvla-run"
        adapter = TinyVLAAdapter(settings, run.name, self.root, self.external, self.envs)
        train, evaluate = adapter.train(), adapter.evaluate(True)
        self.assertEqual((train.environment, evaluate.environment), ("tinyvla_infer", "lerobot"))
        self.assertIn("@python:lerobot", train.arguments)
        self.assertIn("--policy.inference_python=@python:tinyvla_infer", evaluate.arguments)
        self.assertIn(f"--policy.model_path={adapter.train_dir / 'train/pretrained_model'}", evaluate.arguments)
        self.assertIn(f"--policy.worker_log_dir={adapter.eval_dir / 'worker_logs'}", evaluate.arguments)
        self.assertEqual(train.units, 8)
        actual = resolve_arguments(train.arguments, {"lerobot": self.environment})
        self.assertEqual(Path(actual[2]).parent, self.external / "training")
        self.assertTrue(Path(actual[2]).with_name("lerobot_data.py").is_file())
        self.assertTrue(Path(actual[actual.index("--loader") + 1]).is_file())

        with patch("runner.resolve_environment", return_value=self.environment), \
             patch("runner.conda_command", side_effect=lambda c, e, argv: [sys.executable, "-u", str(self.backend), *argv]), \
             contextlib.redirect_stdout(io.StringIO()):
            execute([train, evaluate], settings, run, self.root, "fake", False)
        self.assertEqual(read_json(run / "status.json")["status"], "completed")
        self.assertTrue((adapter.eval_dir / "videos/test.mp4").is_file())
        self.assertIn(adapter.train_dir, scan(self.root, self.external, self.logs).tiny_resume)
        self.assertEqual(set(run.glob("*.py")), {run / "run.py", run / "script.py"})
        state = adapter.train_dir / "train/checkpoints/training_state.pt"
        settings["train"]["resume"] = str(state)
        resumed = TinyVLAAdapter(settings, "resume", self.root, self.external, self.envs).train()
        self.assertEqual(resumed.arguments[resumed.arguments.index("--resume-state") + 1], str(state))
        self.assertNotIn("--model", resumed.arguments)
        self.assertIsNone(resumed.start)

    def test_inventory_rejects_incomplete_datasets_and_checkpoints(self):
        for name in ("complete", "partial"):
            dataset = self.root / "datasets" / name
            (dataset / "meta").mkdir(parents=True)
            (dataset / "meta/info.json").write_text(json.dumps({"features": {"action": {}}, "total_episodes": 5}))
        (self.root / "datasets/complete/data").mkdir()
        (self.root / "datasets/complete/data/sample.parquet").write_bytes(b"sample")
        model = self.external / "models/Llava-Pythia-700M"
        model.mkdir(parents=True)
        (model / "config.json").write_text('{"model_type":"llava_pythia"}')
        self.assertEqual(scan(self.root, self.external, self.logs).models, [])
        (model / "model.safetensors").write_bytes(b"test")
        inventory = scan(self.root, self.external, self.logs)
        self.assertEqual([path.name for path, _ in inventory.datasets], ["complete"])
        self.assertEqual(inventory.models, [(model, "tinyvla")])

    def test_plan_detects_server_tinyvla_environment(self):
        from argparse import Namespace

        fixture = self.external / "models/Llava-Pythia-700M"
        fixture.mkdir(parents=True)
        (fixture / "config.json").write_text('{"model_type":"llava_pythia"}')
        (fixture / "model.safetensors").write_bytes(b"sample")
        (self.external / "runtime").mkdir()
        (self.external / "runtime/tinyvla_loader.py").write_bytes(b"sample")
        settings = sample_settings()
        del settings["train"]
        settings.update(adapter="tinyvla", model="tinyvla")
        settings_path = self.root / "settings.json"
        settings_path.write_text(json.dumps(settings))
        args = Namespace(settings=settings_path, queue=None, lerobot_root=str(self.root), tinyvla_root=str(self.external),
                         conda="fake", inventory=False, plugin=None, env_map=None)
        with patch("manage.available_environments", return_value={"lerobot", "lerobot_tinyvla"}):
            jobs, _ = plan(args)
        (configured, stages, _, _), = jobs
        self.assertEqual(configured["tinyvla_root"], str(self.external))
        self.assertEqual(configured["environments"]["tinyvla"], "lerobot_tinyvla")
        self.assertIn("--policy.inference_python=@python:lerobot_tinyvla", stages[0].arguments)

    def test_interactive_import_lists_scripts_and_uses_literal_jobs(self):
        directory = self.root / "executor/scripts"
        directory.mkdir(parents=True)
        fixture = directory / "evaluate.py"
        settings = sample_settings()
        fixture.write_text(f"JOBS = {[settings, settings]!r}\nraise RuntimeError('must not be executed')\n", encoding="utf-8")
        (directory / "notes.txt").touch()
        with patch("manage.SCRIPT_ROOT", directory), patch("builtins.input", side_effect=["2", "1"]), \
             contextlib.redirect_stdout(io.StringIO()) as menu:
            imported = wizard(load_adapters([]), Inventory(), self.root)
        self.assertIn("evaluate.py", menu.getvalue())
        self.assertNotIn("notes.txt", menu.getvalue())
        self.assertEqual(settings_jobs(imported), [settings, settings])

    def test_queue_executes_in_order_and_stops_after_a_failed_job(self):
        plans = []
        for index, seed in enumerate((0, 1, 0)):
            settings = sample_settings()
            del settings["train"]
            settings["eval"].update(seed=seed, policy=str(self.root))
            run = self.logs / f"job-{index}"
            adapter = NativeAdapter(settings, run.name, self.root, self.external, self.envs)
            plans.append((settings, [adapter.evaluate(False)], run, self.root))
        with patch("runner.resolve_environment", return_value=self.environment), \
             patch("runner.conda_command", side_effect=lambda c, e, argv: [sys.executable, "-u", str(self.backend), *argv]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(FileNotFoundError):
                run_plans(plans, "fake", False)
        self.assertLess(output.getvalue().index("Job 1/3"), output.getvalue().index("Job 2/3"))
        self.assertEqual(read_json(self.logs / "job-0/status.json")["status"], "completed")
        self.assertEqual(read_json(self.logs / "job-1/status.json")["status"], "failed")
        self.assertFalse((self.logs / "job-2").exists())

    def test_progress_uses_measured_intervals_and_partial_batches(self):
        settings = sample_settings()
        settings["eval"].update(episodes=3, batch=2)
        stage = Stage("eval", "lerobot", [], (), 6, eval_batches=2)
        progress = Progress(stage, settings)
        bar = "Stepping through eval batches: 50%|#####     | {}/2 [00:01<00:01]"
        for count, when in ((0, 0), (1, 2), (2, 3), (0, 4), (1, 6)):
            progress.update(bar.format(count), when)
        self.assertEqual(progress.count, 5)
        self.assertAlmostEqual(progress.eta(), 1.175)
        self.assertTrue(progress.update(bar.format(2), 7))
        self.assertEqual(progress.count, 6)
        self.assertEqual(progress.fraction(), 0.99)
        self.assertIsNone(progress.eta())

    def test_blocks_overriding_output_directory(self):
        settings = sample_settings()
        settings["train"]["extra"] = ["--output_dir=other"]
        adapter = NativeAdapter(settings, "run", self.root, self.external, self.envs)
        with self.assertRaises(ValueError):
            adapter.train()

    def test_accepts_zero_seed_default(self):
        with patch("builtins.input", return_value=""):
            self.assertEqual(number("seed", 0, 0), 0)


if __name__ == "__main__":
    unittest.main()
