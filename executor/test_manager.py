import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from adapters import NativeAdapter, Stage, TinyVLAAdapter, load_adapters, read_json
from inventory import Inventory, scan, show
from manage import choose_policy, eval_settings, finish, model_size, normalize_paths, number, plan, run_plans, settings_jobs, wizard
from runner import Progress, evaluation_timing_key, execute, history, load_settings, resolve_arguments, seconds, status_line


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
    output.mkdir(parents=True, exist_ok=True)
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
        self.assertRegex(text.getvalue(), r"100\.000 % \| 2/2 Episodes \| Seconds Per Episode: \d+\.\d{2} s \| Estimated Time of Completion: 00:00:00")
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
        project = self.root / "convLexecutor"
        with patch("manage.PROJECT_ROOT", project), \
             patch("manage.available_environments", return_value={"lerobot", "lerobot_tinyvla"}):
            jobs, _ = plan(args)
        (configured, stages, _, working_directory), = jobs
        self.assertEqual(working_directory, project)
        self.assertEqual(stages[0].output.parent, project / "outputs/evaluation")
        self.assertEqual(stages[0].env["HF_LEROBOT_HOME"], str(project / "datasets/.hub"))
        self.assertEqual(configured["tinyvla_root"], str(self.external))
        self.assertEqual(configured["environments"]["tinyvla"], "lerobot_tinyvla")
        self.assertIn("--policy.inference_python=@python:lerobot_tinyvla", stages[0].arguments)

    def test_project_paths_are_independent_of_a_lerobot_checkout(self):
        project = self.root / "convLexecutor"
        lerobot_base = self.root / "lerobot"
        data = {"adapter": "tinyvla", "train": {"dataset_root": "datasets/example/libero",
                 "base_model": "externals/TinyVLA/models/base",
                 "resume": "outputs/example/state.pt",
                 "selection": str(lerobot_base / "legacy-split.json")},
                 "eval": {"policy": "outputs/example/model"}}
        with patch("manage.PROJECT_ROOT", project):
            resolved = normalize_paths(data, lerobot_base)
        for name in ("dataset_root", "base_model", "resume"):
            self.assertEqual(Path(resolved["train"][name]), project / data["train"][name])
        self.assertEqual(resolved["train"]["selection"], data["train"]["selection"])
        self.assertEqual(Path(resolved["eval"]["policy"]), project / data["eval"]["policy"])

    def test_interactive_import_lists_scripts_and_uses_literal_jobs(self):
        directory = self.root / "scripts"
        directory.mkdir(parents=True)
        fixture = directory / "evaluate.py"
        settings = sample_settings()
        fixture.write_text(f"JOBS = {[settings, settings]!r}\nraise RuntimeError('must not be executed')\n", encoding="utf-8")
        (directory / "notes.txt").touch()
        (directory / "prepare_examples.py").write_text("SETTINGS = {'steps': 4}\n", encoding="utf-8")
        with patch("manage.SCRIPT_ROOT", directory), patch("sys.stdin", io.StringIO("2\n1\n")), \
             contextlib.redirect_stdout(io.StringIO()) as menu:
            imported = wizard(load_adapters([]), Inventory(), self.root)
        self.assertIn("evaluate.py", menu.getvalue())
        self.assertNotIn("notes.txt", menu.getvalue())
        self.assertNotIn("prepare_examples.py", menu.getvalue())
        self.assertEqual(settings_jobs(imported), [settings, settings])

    def test_combined_run_records_each_stage_in_its_result_folder(self):
        settings = sample_settings()
        adapter = NativeAdapter(settings, "job", self.root, self.external, self.envs)
        training = self.root / "outputs/training/job/executor"
        evaluation = self.root / "outputs/evaluation/job/executor"
        with patch("runner.resolve_environment", return_value=self.environment), \
             patch("runner.conda_command", side_effect=lambda c, e, argv: [sys.executable, "-u", str(self.backend), *argv]), \
             contextlib.redirect_stdout(io.StringIO()):
            execute([adapter.train(), adapter.evaluate(True)], settings, training, self.root, "fake", False)
        self.assertEqual({p.name for p in training.iterdir()}, {"run.py", "script.py", "status.json", "train.log"})
        self.assertEqual({p.name for p in evaluation.iterdir()}, {"run.py", "script.py", "status.json", "eval.log"})
        self.assertEqual(read_json(training / "status.json"), read_json(evaluation / "status.json"))
        self.assertEqual(scan(self.root, self.external, self.logs).settings, [training / "script.py"])
        self.assertFalse(self.logs.exists())

    def test_finish_summarizes_completed_evaluations(self):
        output = self.root / "evaluation"
        output.mkdir()
        report = {"overall": {"n_success": 3, "n_episodes": 4, "eval_s": 8.8, "eval_ep_s": 2.2},
                  "per_group": {"libero_object": {"n_success": 1, "n_episodes": 2},
                                "libero_goal": {"n_success": 2, "n_episodes": 2}}}
        (output / "eval_info.json").write_text(json.dumps(report), encoding="utf-8")
        stage = Stage("eval", "lerobot", [], (), 4, output=output)
        with contextlib.redirect_stdout(io.StringIO()) as text:
            finish([({}, [stage], output / "executor", self.root)])
        self.assertIn("<Finish>", text.getvalue())
        self.assertIn("Overall success rate: 3/4 (75.00%)", text.getvalue())
        self.assertIn("  libero_object: 1/2 (50.00%)", text.getvalue())
        self.assertIn("  libero_goal: 2/2 (100.00%)", text.getvalue())
        self.assertIn("Total evaluation time: 00:00:09", text.getvalue())
        self.assertIn("Average time per episode: 2.20 s", text.getvalue())
        self.assertIn(f"Result path: {output}", text.getvalue())
        self.assertNotIn("Completed 1 job.", text.getvalue())

        report = {"aggregated": {"n_episodes": 2, "pc_success": 50, "eval_s": 3}}
        (output / "eval_info.json").write_text(json.dumps(report), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as text:
            finish([({}, [stage], output / "executor", self.root)])
        self.assertIn("Overall success rate: 50.00%", text.getvalue())
        self.assertIn("Suite success rates: N/A", text.getvalue())
        self.assertIn("Average time per episode: 1.50 s", text.getvalue())

    def test_parallel_rollouts_use_the_aggregate_episode_count(self):
        settings = sample_settings()
        settings["eval"]["parallel_tasks"] = 2
        stage = Stage("eval", "lerobot", [], (), 4, eval_batches=1)
        progress = Progress(stage, settings, started_at=0)
        self.assertFalse(progress.update("Stepping through eval batches: 100%|#| 1/1 [00:01<00:00, 1it/s]", 1))
        self.assertTrue(progress.update("[Eval progress] Episodes completed: 2", 2))
        self.assertTrue(progress.update("[Eval progress] Episodes completed: 4", 3))
        self.assertEqual(progress.count, 4)
        self.assertEqual(progress.average(), 0.75)

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
        self.assertAlmostEqual(progress.average(), 7 / 6)
        self.assertEqual(status_line(99, 6, 6, "eval", None, progress.average()),
                         "[Status] Completion: 99.000 % | 6/6 Episodes | Seconds Per Episode: 1.17 s | Estimated Time of Completion: N/A")
        self.assertIsNone(progress.eta())

    def test_inventory_lists_complete_hugging_face_snapshots_without_assuming_resume_state(self):
        hub = self.root / "hub"
        repo = hub / "models--lerobot--smolvla-test/snapshots"
        snapshot = repo / "abc12345"
        incomplete = repo / "def67890"
        for path in (snapshot, incomplete):
            path.mkdir(parents=True)
            (path / "config.json").write_text('{"type": "smolvla"}', encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"weights")
        with patch.dict("os.environ", {"HF_HUB_CACHE": str(hub)}):
            inventory = scan(self.root, self.external, self.logs)
        self.assertEqual(inventory.models, [(snapshot, "smolvla")])
        self.assertFalse(inventory.native_resume)
        self.assertEqual(inventory.cached_models[snapshot], "lerobot/smolvla-test @ abc12345")

    def test_inventory_lists_numbered_entries_and_item_counts(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            show(Inventory(), {"lerobot", "lerobot_tinyvla"})
        self.assertIn("Conda environments [2 found]\n  1. lerobot\n  2. lerobot_tinyvla", output.getvalue())
        self.assertIn("\n\nTraining datasets [0 found]", output.getvalue())

    def test_estimate_from_first_positive_measurement(self):
        stage = Stage("train", "lerobot", [], (), 8)
        progress = Progress(stage, sample_settings(), started_at=10)
        self.assertTrue(progress.update("step:2/8", 14))
        self.assertEqual(progress.eta(), 12)
        self.assertEqual(status_line(25, 2, 8, "train", progress.eta()),
                         "[Status] Completion: 25.000 % | 2/8 Steps | Estimated Time of Completion: 00:00:12")
        self.assertEqual(status_line(0, 0, None, "eval", None),
                         "[Status] Completion: 0.000 % | 0/N/A Episodes | Seconds Per Episode: N/A | Estimated Time of Completion: N/A")

    def test_reuses_a_legacy_evaluation_timing_with_a_different_checkpoint_path(self):
        settings = {"adapter": "tinyvla", "model": "tinyvla"}
        runtime = {"lerobot": {"python": "python"}}
        stage = Stage("eval", "lerobot", ["--env.type=libero", "--eval.n_episodes=1",
                                         "--policy.model_path=previous"], (), 1)
        log = self.logs / "previous"
        log.mkdir(parents=True)
        (log / "status.json").write_text(json.dumps({"stages": [{
            "name": "eval", "status": "completed", "signature": "legacy", "seconds": 28
        }]}), encoding="utf-8")
        job = {"settings": settings, "runtime": runtime, "stages": [asdict(stage)]}
        (log / "run.py").write_text(f"JOB = {job!r}", encoding="utf-8")
        stage.arguments[-1] = "--policy.model_path=next"
        rates = history(self.logs)
        self.assertEqual(rates[evaluation_timing_key(settings, stage, runtime)], 28)
        stage.arguments[1] = "--eval.n_episodes=2"
        self.assertNotIn(evaluation_timing_key(settings, stage, runtime), rates)

    def test_blocks_overriding_output_directory(self):
        settings = sample_settings()
        settings["train"]["extra"] = ["--output_dir=other"]
        adapter = NativeAdapter(settings, "run", self.root, self.external, self.envs)
        with self.assertRaises(ValueError):
            adapter.train()

    def test_accepts_zero_seed_default(self):
        with patch("builtins.input", return_value="") as prompt:
            self.assertEqual(number("seed", 0, 0), 0)
        prompt.assert_called_once_with("\nseed [Default: 0]: ")

    def test_tinyvla_evaluation_defaults_to_training_suite_and_task(self):
        answers = io.StringIO("\n\n3\n2\n\n\n\n10\n\n\n")
        with patch("sys.stdin", answers), contextlib.redirect_stdout(io.StringIO()) as menu:
            cfg = eval_settings("tinyvla", True, {"task": ["libero_goal", 7]}, Inventory())
        self.assertEqual((cfg["env"], cfg["tasks"], cfg["task_ids"]), ("libero", ["libero_goal"], [7]))
        self.assertEqual(cfg["batch"], 2)
        self.assertIn("3. libero_goal [Default]", menu.getvalue())
        self.assertIn("1. LIBERO (supported by TinyVLA) [Default]", menu.getvalue())
        self.assertNotIn("Select an option", menu.getvalue())

    def test_policy_menu_lists_native_models_before_external_models(self):
        answers = io.StringIO("4\n")
        with patch("sys.stdin", answers), contextlib.redirect_stdout(io.StringIO()) as menu:
            selected = choose_policy(load_adapters([]), ["xvla", "act"])
        self.assertEqual(selected, ("tinyvla", "tinyvla"))
        self.assertIn("1. [LeRobot Native] act [Default]  [51.7 M]", menu.getvalue())
        self.assertIn("2. [LeRobot Native] xvla  [879.7 M]", menu.getvalue())
        self.assertIn("3. [LeRobot Native] Enter a model name manually", menu.getvalue())
        self.assertIn("4. [External] TinyVLA  [740.4 M]", menu.getvalue())

    def test_model_size_uses_parameter_units_or_na(self):
        self.assertEqual(model_size("diffusion"), "[262.7 M]")
        self.assertEqual(model_size("dm05"), "[5.2 B]")
        self.assertEqual(model_size("fastwam"), "[6 B]")
        self.assertEqual(model_size("gaussian_actor"), "[N/A]")

    def test_completion_time_includes_days_when_needed(self):
        self.assertEqual(seconds(None), "N/A")
        self.assertEqual(seconds(86399), "23:59:59")
        self.assertEqual(seconds(86400), "1d 00:00:00")
        self.assertEqual(seconds(2 * 86400 + 3 * 3600 + 4 * 60 + 5), "2d 03:04:05")

    def test_native_evaluation_uses_selected_benchmark(self):
        answers = io.StringIO("2\n\n1\n\n\n\nlerobot/act_pusht\n125\n\n")
        with patch("sys.stdin", answers), contextlib.redirect_stdout(io.StringIO()) as menu:
            cfg = eval_settings("act", False, None, Inventory())
        self.assertEqual((cfg["env"], cfg["tasks"]), ("pusht", ["PushT-v0"]))
        self.assertEqual(cfg["episode_length"], 125)
        job = {"adapter": "native", "model": "act", "eval": cfg}
        adapter = NativeAdapter(job, "run", self.root, self.external, self.envs)
        self.assertIn("--env.episode_length=125", adapter.evaluate(False).arguments)
        cfg["episode_length"] = 0
        self.assertFalse(any(arg.startswith("--env.episode_length=") for arg in adapter.evaluate(False).arguments))
        self.assertIn("1. LIBERO [Default]", menu.getvalue())
        self.assertIn("\nEvaluation benchmark: ", menu.getvalue())
        self.assertIn("[Default: PushT-v0]", menu.getvalue())


if __name__ == "__main__":
    unittest.main()
