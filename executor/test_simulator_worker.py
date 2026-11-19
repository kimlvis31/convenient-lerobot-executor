"""Verify source contracts, safe messages and isolated Gym worker lifecycle."""

import hashlib
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simulator_worker as worker


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_GYM = importlib.util.find_spec("gymnasium") is not None

SOURCE = '''from __future__ import annotations
from lerobot.lerobot_types import RobotObservation
from lerobot.utils.import_utils import _scipy_available
from .utils import _LazyAsyncVectorEnv
RESULT = 41
def native_helper():
    return RESULT + 1
class RoboTwinEnv:
    def reset(self): return native_helper()
    def step(self, action): return action
    def render(self): return RobotObservation()
    def close(self): pass
'''


class SourceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="convlex-worker-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "robotwin.py"
        self.path.write_text(SOURCE, encoding="utf-8")
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.config = {"kind": "robotwin", "source": str(self.path), "source_sha256": self.digest}

    def test_source_version_and_post_selection_drift_are_separate_errors(self):
        with self.assertRaisesRegex(RuntimeError, "not been verified"):
            worker.validate_source(self.config)
        self.path.write_text(SOURCE + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            worker.validate_source(self.config)

    def test_loader_reuses_helpers_and_bodies_without_modifying_source(self):
        before = self.path.read_bytes()
        with patch.object(worker, "ROBOTWIN_SOURCE", self.digest):
            module = worker.load_robotwin(self.config)
        self.assertEqual(module.RoboTwinEnv().reset(), 42)
        self.assertEqual(module.RoboTwinEnv().render(), {})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertNotIn("_LazyAsyncVectorEnv", module.__dict__)

    def test_changed_import_contract_is_rejected(self):
        self.path.write_text(SOURCE.replace("import RobotObservation", "import RobotObservation as RobotObservation"), encoding="utf-8")
        self.config["source_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch.object(worker, "ROBOTWIN_SOURCE", self.config["source_sha256"]):
            with self.assertRaisesRegex(RuntimeError, "import contract"):
                worker.load_robotwin(self.config)

    def test_directory_restores_after_failure(self):
        previous = Path.cwd()
        previous_path = list(sys.path)
        with patch.dict(os.environ, {worker.ROOT_ENV: self.temp.name}):
            with self.assertRaisesRegex(RuntimeError, "probe failure"):
                with worker.source_directory(self.config):
                    self.assertEqual(Path.cwd(), Path(self.temp.name))
                    raise RuntimeError("probe failure")
        self.assertEqual(Path.cwd(), previous)
        self.assertEqual(sys.path, previous_path)

    def test_factory_is_spawn_serializable_and_preserves_episode_stride(self):
        factories = worker.split_factories(self.config, task_name="beat_block_hammer", n_envs=2,
                                           camera_names=["head_camera"], observation_height=240,
                                           observation_width=320, episode_length=10, action_mode="ee")
        restored = pickle.loads(pickle.dumps(factories))
        self.assertEqual([factory.args[1]["episode_index"] for factory in restored], [0, 1])
        self.assertTrue(all(factory.args[1]["n_envs"] == 2 for factory in restored))
        self.assertTrue(all(factory.args[1]["action_mode"] == "ee" for factory in restored))
        with self.assertRaisesRegex(RuntimeError, "factory API changed"):
            worker.split_factories(self.config, task_name="unsupported")


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for simulator messages")
class TransportTests(unittest.TestCase):
    def test_nested_arrays_info_tuples_and_scalars_roundtrip(self):
        import numpy as np

        source = {"obs": {"pixels": {"head": np.arange(24, dtype=np.uint8).reshape(2, 4, 3)},
                          "agent_pos": np.arange(14, dtype=np.float32)},
                  "outcome": (np.float32(0.5), False, True, {"is_success": np.bool_(False),
                                                               "final_info": [{"task": "pick"}]}),
                  "nothing": None}
        stream = io.BytesIO()
        worker.send(stream, source)
        stream.seek(0)
        result = worker.receive(stream)
        np.testing.assert_array_equal(source["obs"]["pixels"]["head"], result["obs"]["pixels"]["head"])
        self.assertEqual(result["obs"]["agent_pos"].dtype, np.dtype("float32"))
        self.assertEqual(result["outcome"], (0.5, False, True, {"is_success": False, "final_info": [{"task": "pick"}]}))
        self.assertIsNone(result["nothing"])

    def test_partial_pipe_reads_and_invalid_message_sizes(self):
        import numpy as np

        class ShortReads(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 7))

        stream = io.BytesIO()
        worker.send(stream, np.arange(100, dtype=np.int32))
        result = worker.receive(ShortReads(stream.getvalue()))
        np.testing.assert_array_equal(result, np.arange(100, dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "length"):
            worker.receive(io.BytesIO(struct.pack("!I", worker.MAX_MESSAGE + 1)))
        with self.assertRaises(EOFError):
            worker.receive(io.BytesIO(stream.getvalue()[:-5]))

    def test_pickle_objects_are_never_accepted(self):
        import numpy as np

        for invalid in [np.array([object()], dtype=object), object(), {4: "integer key"}]:
            with self.assertRaises(TypeError):
                worker.send(io.BytesIO(), invalid)
        data = io.BytesIO()
        np.savez(data, metadata=np.array(object(), dtype=object))
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            worker.receive(io.BytesIO(struct.pack("!I", data.tell()) + data.getvalue()))

    def test_worker_eof_always_closes_environment(self):
        import numpy as np

        env = SimpleNamespace(task_description="first", closed=False)
        def reset(**kwargs):
            env.task_description = "seed " + str(kwargs["seed"])
            return {"agent_pos": np.zeros(14)}, {"is_success": False}
        def close():
            env.closed = True
        env.reset, env.close = reset, close
        reader, writer = io.BytesIO(), io.BytesIO()
        worker.send(reader, {"kind": "robotwin", "kwargs": {}})
        worker.send(reader, {"operation": "reset", "kwargs": {"seed": 17}})
        reader.seek(0)
        with TemporaryDirectory(prefix="convlex-worker-serve-") as root, \
             patch.dict(os.environ, {worker.ROOT_ENV: root}), \
             patch.object(worker, "load_robotwin", return_value=SimpleNamespace(RoboTwinEnv=lambda: env)), \
             patch.object(worker, "environment_details", return_value={"task_description": "first"}):
            worker.serve(reader, writer)
        writer.seek(0)
        self.assertEqual(worker.receive(writer)["result"], {"task_description": "first"})
        response = worker.receive(writer)
        self.assertEqual(response["task_description"], "seed 17")
        self.assertEqual(response["result"][1], {"is_success": False})
        self.assertTrue(env.closed)


@unittest.skipUnless(HAS_GYM and HAS_NUMPY, "Gymnasium and NumPy are required for space schemas")
class GymSpaceTests(unittest.TestCase):
    def test_supported_spaces_reconstruct_shape_bounds_dtype_and_order(self):
        import numpy as np
        from gymnasium import spaces

        source = spaces.Dict([
            ("pixels", spaces.Dict({"head_camera": spaces.Box(0, 255, (2, 4, 3), np.uint8)})),
            ("agent_pos", spaces.Box(-np.inf, np.inf, (14,), np.float32)),
            ("misc", spaces.Tuple((spaces.Discrete(3, start=1), spaces.MultiDiscrete([2, 4], start=[1, 2]),
                                    spaces.MultiBinary((2, 3)), spaces.Text(5, min_length=1, charset="abc")))),
        ])
        stream = io.BytesIO()
        worker.send(stream, worker.space_spec(source))
        stream.seek(0)
        restored = worker.restore_space(worker.receive(stream))
        self.assertEqual(restored, source)
        self.assertEqual(list(restored.spaces), ["pixels", "agent_pos", "misc"])
        self.assertTrue(restored.contains(source.sample()))


class ProcessTests(unittest.TestCase):
    def test_worker_reaped_after_stdin_closes(self):
        process = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   start_new_session=os.name != "nt")
        try:
            worker.close_process(process)
            self.assertEqual(process.returncode, 0)
            self.assertTrue(process.stdout.closed)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


class SourceManagerTests(unittest.TestCase):
    def setUp(self):
        from manage import LeRobotEnvironments

        self.session = LeRobotEnvironments("conda", Path("."), "policy", {"policy"})
        self.session.catalogs = {"policy": {"policies": {"smolvla": True},
                                            "manual_status": {"robotwin": "missing", "robocasa": "missing"}}}
        self.stage = SimpleNamespace(arguments=["lerobot-eval", "--env.type=robotwin"], env={})
        self.origin = {"source": str(Path(__file__).resolve()), "source_sha256": "test", "base_signature": "packages",
                       "base_prefix": sys.prefix,
                       "hub_environment": {"HF_HOME": "/custom/huggingface", "HF_HUB_CACHE": "/custom/models",
                                           "HF_TOKEN_PATH": "/custom/credentials/token", "HF_ASSETS_CACHE": "/custom/assets",
                                           "HF_XET_CACHE": "/custom/xet", "TORCH_HOME": "/custom/torch",
                                           "TRANSFORMERS_CACHE": "/custom/transformers", "HF_MODULES_CACHE": "/custom/modules"}}
        self.report = SimpleNamespace(returncode=0, stderr="",
                                      stdout="CONVLEX_NATIVE_SOURCE " + json.dumps(self.origin))
        self.prepared = {"python": "/managed/bin/python", "prefix": "/managed", "root": "/build",
                         "env": {"PYTHONPATH": "", "ROBOTWIN_ROOT": "/build/sources/robotwin",
                                 worker.ROOT_ENV: "/build/sources/robotwin"}, "recipe": "robotwin-tested",
                         "notes": ["Verified one physics step."]}

    def test_preview_has_no_installation_or_source_probe(self):
        with patch.object(self.session, "prepare", return_value="policy"), \
             patch("manage.subprocess.run", side_effect=AssertionError("Preview started a process")):
            target, notes = self.session.prepare_source(self.stage, "policy", {"env": "robotwin"},
                                                        model="smolvla", peft=False, preview=True)
        self.assertEqual(target, "policy")
        self.assertIn("at launch", " ".join(notes))
        self.assertFalse(self.stage.env)

    def test_split_prepare_keeps_policy_and_records_exact_source(self):
        setup = unittest.mock.Mock(return_value=self.prepared)
        with patch.object(self.session, "lerobot", return_value=self.session.catalogs), \
             patch.object(self.session, "prepare", return_value="policy"), \
             patch("manage.subprocess.run", return_value=self.report), \
             patch.object(worker, "validate_source"), \
             patch.dict(sys.modules, {"runtime_setup": SimpleNamespace(prepare_runtime=setup)}):
            target, notes = self.session.prepare_source(self.stage, "policy", {"env": "robotwin"},
                                                        model="smolvla", peft=False, preview=False)
        self.assertEqual(target, "policy")
        config = json.loads(self.stage.env[worker.SIMULATOR_ENV])
        self.assertEqual(config["source"], self.origin["source"])
        self.assertEqual(config["env"][worker.ROOT_ENV], "/build/sources/robotwin")
        self.assertEqual(Path(self.stage.arguments[2]).name, "evaluation_cameras.py")
        self.assertIn("policy remains in policy", " ".join(notes))
        setup.assert_called_once_with("robotwin", "conda", options={"base_environment": "policy", **self.origin})

    def test_existing_combined_runtime_avoids_provisioning(self):
        self.session.catalogs["policy"]["manual_status"]["robotwin"] = None
        with patch.object(self.session, "lerobot", return_value=self.session.catalogs), \
             patch.object(self.session, "prepare", return_value="policy"), \
             patch("manage.subprocess.run", side_effect=AssertionError("Unnecessary source probe")):
            self.assertEqual(self.session.prepare_source(self.stage, "policy", {"env": "robotwin"},
                                                         model="smolvla", peft=False, preview=False), ("policy", ()))

    def test_combined_source_uses_prefix_and_environment_overrides(self):
        setup = unittest.mock.Mock(return_value=self.prepared)
        self.stage.arguments.append('--env.obj_registries=["lightwheel","objaverse"]')
        with patch.object(self.session, "lerobot", return_value=self.session.catalogs), \
             patch.object(self.session, "prepare", return_value="policy"), \
             patch("manage.subprocess.run", return_value=self.report), \
             patch.dict(sys.modules, {"runtime_setup": SimpleNamespace(prepare_runtime=setup)}):
            target, _ = self.session.prepare_source(self.stage, "policy", {"env": "robocasa"},
                                                    model="smolvla", peft=False, preview=False)
        self.assertEqual(target, "/managed")
        self.assertEqual(self.stage.env[worker.ROOT_ENV], "/build/sources/robotwin")
        self.assertNotIn(worker.SIMULATOR_ENV, self.stage.env)
        self.assertNotIn(target, self.session.names)
        self.assertEqual(setup.call_args.kwargs["options"]["obj_registries"], '["lightwheel","objaverse"]')

    def test_libero_plus_reads_its_shared_native_libero_wrapper(self):
        setup = unittest.mock.Mock(return_value=self.prepared)
        with patch.object(self.session, "lerobot", return_value=self.session.catalogs), \
             patch.object(self.session, "prepare", return_value="policy"), \
             patch("manage.subprocess.run", return_value=self.report) as probe, \
             patch.dict(sys.modules, {"runtime_setup": SimpleNamespace(prepare_runtime=setup)}):
            self.session.prepare_source(self.stage, "policy", {"env": "libero_plus"},
                                        model="smolvla", peft=False, preview=False)
        source_code = probe.call_args.args[0][-1]
        self.assertIn("/'envs'/'libero.py'", source_code)
        self.assertNotIn("/'envs'/'libero_plus.py'", source_code)

    def test_native_probe_preserves_custom_hub_paths_without_reading_credentials(self):
        setup = unittest.mock.Mock(return_value=self.prepared)
        with patch.object(self.session, "lerobot", return_value=self.session.catalogs), \
             patch.object(self.session, "prepare", return_value="policy"), \
             patch("manage.subprocess.run", return_value=self.report) as probe, \
             patch.dict(sys.modules, {"runtime_setup": SimpleNamespace(prepare_runtime=setup)}):
            self.session.prepare_source(self.stage, "policy", {"env": "robocasa"},
                                        model="smolvla", peft=False, preview=False)
        self.assertEqual(setup.call_args.kwargs["options"]["hub_environment"], self.origin["hub_environment"])
        source_code = probe.call_args.args[0][-1]
        self.assertNotIn("import torch", source_code)
        with TemporaryDirectory(prefix="convlex-probe-paths-") as folder:
            package = Path(folder) / "lerobot"
            (package / "envs").mkdir(parents=True)
            wrapper = package / "envs/robocasa.py"
            wrapper.write_text("# native source\n", encoding="utf-8")
            original_read = Path.read_bytes
            def source_only(path):
                if path != wrapper:
                    raise AssertionError("The probe tried to read a non-source file.")
                return original_read(path)
            constants = SimpleNamespace(**self.origin["hub_environment"], HF_TOKEN="secret-never-export")
            output = io.StringIO()
            with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(constants=constants)}), \
                 patch.dict(os.environ, {"TORCH_HOME": "/custom/torch", "TRANSFORMERS_CACHE": "/custom/transformers",
                                         "HF_MODULES_CACHE": "/custom/modules"}), \
                 patch("importlib.util.find_spec", return_value=SimpleNamespace(submodule_search_locations=[str(package)])), \
                 patch("importlib.metadata.distributions", return_value=[]), \
                 patch.object(Path, "read_bytes", source_only), \
                 patch.object(Path, "read_text", side_effect=AssertionError("Credential contents must not be read")), \
                 redirect_stdout(output):
                exec(source_code, {"__name__": "__native_probe__"})
                report = json.loads(output.getvalue().partition(" ")[2])
                custom_output = output.getvalue()
                output.seek(0)
                output.truncate()
                for name in ("TORCH_HOME", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"):
                    os.environ.pop(name, None)
                os.environ["XDG_CACHE_HOME"] = str(Path(folder) / "native-cache")
                exec(source_code, {"__name__": "__native_probe__"})
                fallback = json.loads(output.getvalue().partition(" ")[2])["hub_environment"]
            self.assertEqual(report["hub_environment"], self.origin["hub_environment"])
            self.assertEqual(report["base_prefix"], sys.prefix)
            self.assertEqual(report["hub_environment"]["TORCH_HOME"], "/custom/torch")
            self.assertEqual(fallback["TORCH_HOME"], str(Path(folder) / "native-cache/torch"))
            self.assertNotIn("TRANSFORMERS_CACHE", fallback)
            self.assertNotIn("HF_MODULES_CACHE", fallback)
            self.assertNotIn("secret-never-export", custom_output + output.getvalue())
            self.assertNotIn("HF_TOKEN", report["hub_environment"])

    def test_unexpected_hub_secret_values_are_rejected_before_setup(self):
        invalid = {**self.origin, "hub_environment": {"HF_TOKEN": "must-not-be-reported"}}
        report = SimpleNamespace(returncode=0, stderr="", stdout="CONVLEX_NATIVE_SOURCE " + json.dumps(invalid))
        setup = unittest.mock.Mock(return_value=self.prepared)
        with patch.object(self.session, "lerobot", return_value=self.session.catalogs), \
             patch.object(self.session, "prepare", return_value="policy"), \
             patch("manage.subprocess.run", return_value=report), \
             patch.dict(sys.modules, {"runtime_setup": SimpleNamespace(prepare_runtime=setup)}):
            with self.assertRaisesRegex(RuntimeError, "Invalid robocasa wrapper source report"):
                self.session.prepare_source(self.stage, "policy", {"env": "robocasa"},
                                            model="smolvla", peft=False, preview=False)
        setup.assert_not_called()

    def test_existing_native_helpers_keep_revision_and_parallel_tracking(self):
        from manage import simulator_entrypoint

        for filename, options in (("pinned_policy.py", ["-B", "eval"]),
                                  ("parallel_eval.py", ["-u"]),
                                  ("evaluation_cameras.py", ["-u"])):
            with self.subTest(helper=filename):
                path = str(Path(__file__).with_name(filename).resolve())
                command = ["python", options[0], path, *options[1:], "--policy.pretrained_revision=release",
                           "--env.type=robotwin"]
                stage = SimpleNamespace(arguments=list(command))
                simulator_entrypoint(stage)
                self.assertEqual(stage.arguments, command)

    def test_reusable_job_keeps_base_environment_instead_of_cloning_managed_prefix(self):
        from adapters import Stage
        from manage import prepare_job_environments

        for explicit in (False, True):
            with self.subTest(explicit_base=explicit), TemporaryDirectory(prefix="convlex-runtime-prefix-") as prefix:
                settings = {"adapter": "native", "model": "smolvla", "environments": {"native": "policy"},
                            "eval": {"env": "robocasa", "episodes": 1, "batch": 1}}
                if explicit:
                    settings["eval"]["environment"] = "policy"
                stage = Stage("eval", "policy", ["lerobot-eval"], (), 1)
                sessions = {(Path("."), "policy"): self.session}
                with patch.object(self.session, "prepare_source", return_value=(prefix, ("Managed source runtime.",))), \
                     patch.object(self.session, "packages", return_value={}):
                    prepare_job_environments(settings, [stage], Path("."), "conda", sessions)
                self.assertEqual(stage.environment, prefix)
                self.assertEqual(settings["eval"].get("environment"), "policy" if explicit else None)

    def test_conda_setup_failure_is_reported_in_result(self):
        from manage import main

        plans = [({"model": "smolvla"}, [], Path("run"), Path("."))]
        output = io.StringIO()
        with patch.object(sys, "argv", ["manage.py", "--queue", "unused.py"]), \
             patch("manage.plan", return_value=(plans, "conda")), \
             patch("manage.run_plans", side_effect=subprocess.CalledProcessError(2, ["conda", "create"])), \
             redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(main(), 1)
        self.assertIn("(Result)", output.getvalue())
        self.assertIn("Run failed", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())

    def test_source_setup_failure_reports_short_reason_and_preserved_log(self):
        from manage import main
        from runtime_setup import RuntimeSetupFailure

        plans = [({"model": "smolvla"}, [], Path("run"), Path("."))]
        logfile = Path("managed-runtime/last_failure.log")
        failure = RuntimeSetupFailure("robotwin", "CUDA compiler is unavailable.", logfile)
        output = io.StringIO()
        with patch.object(sys, "argv", ["manage.py", "--queue", "unused.py"]), \
             patch("manage.plan", return_value=(plans, "conda")), \
             patch("manage.run_plans", side_effect=failure), \
             redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(main(), 1)
        self.assertIn("(Result)", output.getvalue())
        self.assertIn("Environment setup failed.", output.getvalue())
        self.assertIn("Cause: CUDA compiler is unavailable.", output.getvalue())
        self.assertIn("Full log: " + str(logfile), output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
