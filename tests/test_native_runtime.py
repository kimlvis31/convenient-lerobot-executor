"""AI-written checks for scoped checkpoint loading without changing LeRobot files."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import os
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from native_runtime import (evaluation_action_contract, host_staged_weights, main,
                            native_invocation, training_dataset_contract)
from tests import EXECUTOR


class TerminationHandlingTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "POSIX signal semantics required")
    def test_termination_unwinds_stage_child_and_temporary_files(self):
        code = '''
from pathlib import Path
from tempfile import TemporaryDirectory
import json,sys
sys.path.insert(0,sys.argv[1])
from runner import Stage,interrupt_on_termination,run_stage
root=Path(sys.argv[2])
try:
    with interrupt_on_termination(), TemporaryDirectory(dir=root,prefix="owned-") as scratch:
        def update(line):
            if line.startswith("CHILD "):
                (root/"ready.json").write_text(json.dumps({"pid":int(line.split()[1]),"scratch":scratch}))
        command=[sys.executable,"-u","-c","import os,time;print('CHILD '+str(os.getpid()),flush=True);time.sleep(60)"]
        run_stage(command,Stage("eval","test",command,(),1),root,root/"stage.log",update)
except KeyboardInterrupt:
    (root/"interrupted").write_text("cleaned")
    raise SystemExit(130)
'''
        with TemporaryDirectory() as directory:
            root = Path(directory)
            process = subprocess.Popen([sys.executable, "-B", "-c", code, str(EXECUTOR), str(root)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for _ in range(160):
                    if (root / "ready.json").is_file() or process.poll() is not None:
                        break
                    time.sleep(0.05)
                self.assertTrue((root / "ready.json").is_file())
                ready = json.loads((root / "ready.json").read_text())
                process.send_signal(signal.SIGTERM)
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 130, output)
                self.assertTrue((root / "interrupted").is_file())
                self.assertFalse(Path(ready["scratch"]).exists())
                with self.assertRaises(ProcessLookupError):
                    os.kill(ready["pid"], 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    @unittest.skipUnless(os.name == "posix", "POSIX signal semantics required")
    def test_custom_handler_is_preserved_and_default_is_restored(self):
        from runner import interrupt_on_termination
        previous = signal.getsignal(signal.SIGTERM)
        custom = lambda *_: None
        try:
            signal.signal(signal.SIGTERM, custom)
            with interrupt_on_termination():
                self.assertIs(signal.getsignal(signal.SIGTERM), custom)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            with interrupt_on_termination():
                self.assertTrue(callable(signal.getsignal(signal.SIGTERM)))
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)
        finally:
            signal.signal(signal.SIGTERM, previous)


class NativeRuntimeTests(unittest.TestCase):
    def test_evaluation_contract_rejects_mismatched_actions_before_factory_and_restores(self):
        calls = []

        def factory(cfg, env_cfg=None):
            calls.append(cfg)
            return "policy"

        evaluator = SimpleNamespace(make_policy=factory)
        modules = {"lerobot.scripts": SimpleNamespace(lerobot_eval=evaluator),
                   "lerobot.policies.factory": SimpleNamespace(env_to_policy_features=lambda env: {
                       "action": SimpleNamespace(shape=(7,))})}
        policy = SimpleNamespace(pretrained_path="team/checkpoint", output_features={"action": {"shape": [26]}})
        with patch.dict(sys.modules, modules), evaluation_action_contract():
            with self.assertRaisesRegex(ValueError, "Checkpoint action shape"):
                evaluator.make_policy(policy, env_cfg=SimpleNamespace(type="libero"))
            self.assertEqual(calls, [])
            policy.output_features["action"]["shape"] = [7]
            self.assertEqual(evaluator.make_policy(policy, env_cfg=SimpleNamespace(type="libero")), "policy")
        self.assertIs(evaluator.make_policy, factory)

    def test_cold_dataset_resolves_camera_defaults_before_native_construction(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "meta").mkdir()
            features = {name: {"dtype": "video", "shape": [256, 256, 3],
                               "names": ["height", "width", "channels"]}
                        for name in ("observation.images.image", "observation.images.image2")}
            (root / "meta/info.json").write_text(json.dumps({"features": features}))
            metadata = SimpleNamespace(repo_id="lerobot/libero", root=root, features=features)
            policy = SimpleNamespace(type="wall_oss_05", camera_key_mapping={"face": "face_view"})
            calls = []

            def factory(cfg, ds_meta=None):
                calls.append(dict(cfg.camera_key_mapping))
                return "trained-policy"

            trainer = SimpleNamespace(make_policy=factory)
            with patch.dict(sys.modules, {"lerobot.scripts": SimpleNamespace(lerobot_train=trainer)}), \
                 patch.object(sys, "argv", ["lerobot-train"]), training_dataset_contract():
                self.assertEqual(trainer.make_policy(policy, ds_meta=metadata), "trained-policy")
            self.assertEqual(calls, [{"observation.images.image": "face_view",
                                      "observation.images.image2": "right_wrist_view"}])
            self.assertIs(trainer.make_policy, factory)

    def test_native_entrypoint_installs_and_restores_the_host_loader(self):
        reads = []

        def loader(model, filename, **options):
            reads.append(options)
            return "loaded-policy"

        pretrained = SimpleNamespace(load_model_as_safetensor=loader)
        policy = SimpleNamespace(pretrained_path="team/checkpoint", output_features={"action": {"shape": [7]}})

        def factory(cfg, env_cfg=None):
            return pretrained.load_model_as_safetensor("model", "file", device="cuda", strict=True)

        evaluator = SimpleNamespace(make_policy=factory, make_env=lambda cfg: {}, rollout=lambda env: None)
        evaluator.main = lambda: evaluator.make_policy(policy, env_cfg=SimpleNamespace(type="libero"))
        modules = {"lerobot.policies": SimpleNamespace(pretrained=pretrained),
                   "lerobot.policies.factory": SimpleNamespace(env_to_policy_features=lambda env: {
                       "action": SimpleNamespace(shape=(7,))}),
                   "lerobot.scripts": SimpleNamespace(lerobot_eval=evaluator),
                   "lerobot.scripts.lerobot_eval": evaluator}
        original_argv = sys.argv
        with patch.dict(sys.modules, modules):
            self.assertEqual(main(["lerobot-eval", "--env.type=libero"]), "loaded-policy")
        self.assertEqual(reads, [{"device": "cpu", "strict": True}])
        self.assertIs(pretrained.load_model_as_safetensor, loader)
        self.assertIs(evaluator.make_policy, factory)
        self.assertIs(sys.argv, original_argv)

    def test_only_exact_native_entrypoints_are_selected(self):
        self.assertEqual(native_invocation(["lerobot-eval", "--policy.type=act"]),
                         ("lerobot-eval", ["--policy.type=act"]))
        helper = (EXECUTOR / "parallel_eval.py").resolve()
        self.assertEqual(native_invocation(["python", "-B", "-u", str(helper), "--env.type=libero"]),
                         (str(helper), ["--env.type=libero"]))
        for command in ([], ["python", "-c", "print('lerobot-eval')"],
                        ["python", str(helper.with_name("benchmarks.py"))],
                        ["python", "/unrelated/parallel_eval.py"],
                        ["python", "-m", "parallel_eval"]):
            self.assertIsNone(native_invocation(command))

    def test_load_device_changes_only_for_accelerated_reads_and_restores_after_failure(self):
        calls = []

        def original(model, filename, **kwargs):
            calls.append((model, filename, kwargs))
            return ["missing"], ["unexpected"]

        module = SimpleNamespace(load_model_as_safetensor=original)
        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            with host_staged_weights(module):
                for device in ("cuda", "cuda:1", "mps", "cpu", "xpu"):
                    self.assertEqual(module.load_model_as_safetensor("model", "weights", device=device, strict=False),
                                     (["missing"], ["unexpected"]))
                raise RuntimeError("stage failed")
        self.assertIs(module.load_model_as_safetensor, original)
        self.assertEqual([x[2]["device"] for x in calls], ["cpu", "cpu", "cpu", "cpu", "xpu"])
        self.assertTrue(all(x[:2] == ("model", "weights") and x[2]["strict"] is False for x in calls))

    def test_versions_without_the_known_loader_are_left_unchanged(self):
        module = SimpleNamespace()
        with host_staged_weights(module):
            self.assertFalse(hasattr(module, "load_model_as_safetensor"))

    def test_real_tensor_loading_preserves_values_dtype_and_shared_parameters(self):
        try:
            import torch
            from safetensors.torch import save_model, load_model
        except ImportError:
            self.skipTest("Torch and safetensors are not available")

        class SharedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left = torch.nn.Linear(3, 2)
                self.right = torch.nn.Linear(3, 2)
                self.right.weight = self.left.weight

        with TemporaryDirectory() as directory:
            source = SharedModel()
            source.left.weight.data.copy_(torch.tensor([[0.5, 2.0, -1.5], [3.0, -4.0, 0.0]]))
            source.left.bias.data.copy_(torch.tensor([1.0, -2.0]))
            filename = str(Path(directory) / "model.safetensors")
            save_model(source, filename)
            target = SharedModel().to(dtype=torch.bfloat16)
            module = SimpleNamespace(load_model_as_safetensor=load_model)
            with host_staged_weights(module):
                missing, unexpected = module.load_model_as_safetensor(target, filename, device="cuda", strict=True)
            self.assertFalse(missing or unexpected)
            self.assertEqual(target.left.weight.dtype, torch.bfloat16)
            self.assertEqual(target.left.weight.device.type, "cpu")
            self.assertIs(target.right.weight, target.left.weight)
            for name, expected in source.state_dict().items():
                self.assertTrue(torch.equal(target.state_dict()[name], expected.to(dtype=torch.bfloat16)))
            if torch.cuda.is_available():
                target.to("cuda")
                self.assertEqual(target.left.weight.device.type, "cuda")
                self.assertIs(target.right.weight, target.left.weight)
                self.assertTrue(torch.equal(target.left.weight.cpu(), source.left.weight.to(dtype=torch.bfloat16)))


if __name__ == "__main__":
    unittest.main()
