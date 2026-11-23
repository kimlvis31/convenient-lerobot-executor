"""AI-written checks for strict Molmo policy loading and adapter ordering."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from molmo_compat import checkpoint_policy_factory, checkpoint_processor_factory


class MolmoRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.policy = self.root / "policy"
        self.metadata = self.root / "metadata"
        self.policy.mkdir()
        self.metadata.mkdir()
        (self.policy / "config.json").write_text('{"type":"molmoact2","train_mode_vlm":"fft"}')
        (self.policy / "model.safetensors").write_bytes(b"full-native-policy")
        self.config = SimpleNamespace(type="molmoact2", pretrained_path=str(self.policy),
                                      checkpoint_path=str(self.metadata), dtype="float32", train_mode_vlm="lora")
        self.events = []
        events = self.events

        class HFModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                events.append("REDUNDANT_BASE_WEIGHTS")
                return SimpleNamespace(weights=None)

            @classmethod
            def _from_config(cls, config, dtype=None):
                events.append(("architecture", config, dtype))
                return SimpleNamespace(weights=None)

        native = SimpleNamespace(MolmoAct2ForConditionalGeneration=HFModel,
                                 _resolve_checkpoint_location=lambda *a, **k: events.append("BASE_DOWNLOAD"),
                                 _strict_load_safetensors_weights=lambda *a: events.append("BASE_RELOAD"))

        class Policy:
            def __init__(self, config):
                self.training, self.outer_loaded, self.lora = True, False, False
                directory = native._resolve_checkpoint_location(config.checkpoint_path, revision=None,
                                                                force_download=False)
                self.model = HFModel.from_pretrained(directory, config={"architecture": "fixed"},
                                                    dtype=config.dtype, low_cpu_mem_usage=True, token=None)
                native._strict_load_safetensors_weights(self.model, directory)
                if config.train_mode_vlm == "lora":
                    self._apply_lora_adapters()
                self._apply_compile()

            def _apply_lora_adapters(self):
                events.append(("lora", self.outer_loaded))
                self.lora = True

            def _apply_compile(self):
                events.append(("compile", self.outer_loaded))

            def train(self, mode):
                self.training = mode
                return self

        native.MolmoAct2Policy = Policy
        self.native, self.Policy, self.HFModel = native, Policy, HFModel
        self.modules = {"lerobot.policies.molmoact2": SimpleNamespace(modeling_molmoact2=native),
                        "transformers.initialization": SimpleNamespace(no_init_weights=nullcontext),
                        "transformers.modeling_utils": SimpleNamespace(no_init_weights=nullcontext)}

    def factory(self, config):
        policy = self.Policy(config)
        self.events.append(("STRICT_NATIVE_LOAD", policy.lora))
        policy.model.weights = [1, 2, 3]
        policy.outer_loaded = True
        return policy.train(False)

    def test_full_dense_checkpoint_loads_before_native_lora_and_compile(self):
        original = (self.policy / "model.safetensors").read_bytes()
        with patch.dict(sys.modules, self.modules):
            policy = checkpoint_policy_factory(self.factory, self.config, (self.config,), {})
        self.assertNotIn("BASE_DOWNLOAD", self.events)
        self.assertNotIn("BASE_RELOAD", self.events)
        self.assertNotIn("REDUNDANT_BASE_WEIGHTS", self.events)
        self.assertLess(self.events.index(("STRICT_NATIVE_LOAD", False)), self.events.index(("lora", True)))
        self.assertLess(self.events.index(("lora", True)), self.events.index(("compile", True)))
        self.assertEqual(policy.model.weights, [1, 2, 3])
        self.assertFalse(policy.training)
        self.assertEqual((self.policy / "model.safetensors").read_bytes(), original)

    def test_saved_lora_checkpoint_keeps_lora_topology_before_strict_load(self):
        (self.policy / "config.json").write_text('{"type":"molmoact2","train_mode_vlm":"lora"}')
        with patch.dict(sys.modules, self.modules):
            checkpoint_policy_factory(self.factory, self.config, (self.config,), {})
        self.assertLess(self.events.index(("lora", False)), self.events.index(("STRICT_NATIVE_LOAD", True)))

    def test_factory_failure_restores_all_native_hooks(self):
        original = (self.native._resolve_checkpoint_location, self.native._strict_load_safetensors_weights,
                    self.HFModel.from_pretrained.__func__, self.Policy._apply_lora_adapters, self.Policy._apply_compile)
        def fail(config):
            self.Policy(config)
            raise RuntimeError("strict native weight mismatch")
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(RuntimeError, "strict native"):
            checkpoint_policy_factory(fail, self.config, (self.config,), {})
        self.assertEqual(original, (self.native._resolve_checkpoint_location,
                         self.native._strict_load_safetensors_weights, self.HFModel.from_pretrained.__func__,
                         self.Policy._apply_lora_adapters, self.Policy._apply_compile))

    def test_fresh_or_incomplete_policy_uses_original_base_initialization(self):
        for path in (None, str(self.policy)):
            self.config.pretrained_path = path
            if path:
                (self.policy / "model.safetensors").unlink()
                (self.policy / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "absent.safetensors"}}))
            calls = []
            result = checkpoint_policy_factory(lambda *a, **k: calls.append((a, k)) or "original",
                                               self.config, (self.config,), {"sentinel": 7})
            self.assertEqual(result, "original")
            self.assertEqual(calls, [((self.config,), {"sentinel": 7})])

    def test_other_models_do_not_import_or_patch_molmo(self):
        self.config.type = "act"
        self.assertEqual(checkpoint_policy_factory(lambda: "native", self.config, (), {}), "native")

    def test_deferred_or_non_strict_factories_keep_original_initialization(self):
        calls = []
        def native_factory(config, *, defer_weight_load=False, strict=True):
            calls.append((config, defer_weight_load, strict))
            return "original"
        for options in ({"defer_weight_load": True}, {"strict": False}):
            result = checkpoint_policy_factory(native_factory, self.config, (self.config,), options)
            self.assertEqual(result, "original")
        self.assertEqual(calls, [(self.config, True, True), (self.config, False, False)])

    def test_resume_override_uses_actual_checkpoint_topology(self):
        checkpoint = self.root / "resumed_lora"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"type":"molmoact2","train_mode_vlm":"lora"}')
        (checkpoint / "model.safetensors").write_bytes(b"resumed-lora-policy")
        def native_factory(config, *, pretrained_path=None):
            return self.factory(config)
        with patch.dict(sys.modules, self.modules):
            checkpoint_policy_factory(native_factory, self.config, (self.config,),
                                      {"pretrained_path": str(checkpoint)})
        self.assertLess(self.events.index(("lora", False)), self.events.index(("STRICT_NATIVE_LOAD", True)))

    def test_older_transformers_initialization_namespace_is_supported(self):
        modules = {**self.modules, "transformers.initialization": None}
        with patch.dict(sys.modules, modules):
            policy = checkpoint_policy_factory(self.factory, self.config, (self.config,), {})
        self.assertEqual(policy.model.weights, [1, 2, 3])


class MolmoProcessorMetadataTests(unittest.TestCase):
    def test_native_processor_gets_only_assets_pinned_to_the_model_metadata(self):
        from fnmatch import fnmatch
        with TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / ('a' * 40)
            snapshot.mkdir()
            (snapshot / 'config.json').write_text('{}')
            calls = []
            native = SimpleNamespace(_resolve_checkpoint_location=lambda *a, **k: 'UNFILTERED_DOWNLOAD')
            def hf_file(repo, filename, **kwargs):
                calls.append(('config', repo, filename, kwargs))
                return str(snapshot / filename)
            def hf_snapshot(repo, **kwargs):
                calls.append(('assets', repo, kwargs))
                return str(snapshot)
            def factory(policy_cfg):
                return native._resolve_checkpoint_location('owner/raw', revision=None, force_download=False)
            config = SimpleNamespace(type='molmoact2', _convlex_molmo_metadata=('owner/raw', None, str(snapshot)))
            modules = {'lerobot.policies.molmoact2': SimpleNamespace(processor_molmoact2=native),
                       'huggingface_hub': SimpleNamespace(hf_hub_download=hf_file, snapshot_download=hf_snapshot)}
            original = native._resolve_checkpoint_location
            with patch.dict(sys.modules, modules):
                self.assertEqual(checkpoint_processor_factory(factory, (config,), {}), str(snapshot))
            self.assertIs(native._resolve_checkpoint_location, original)
            self.assertEqual(calls[0][3]['revision'], snapshot.name)
            assets = calls[1][2]
            self.assertEqual(assets['revision'], snapshot.name)
            self.assertTrue(any(fnmatch('processor_config.json', p) for p in assets['allow_patterns']))
            for filename in ('model.safetensors', 'model-00001-of-00002.safetensors', 'pytorch_model.bin'):
                self.assertFalse(any(fnmatch(filename, p) for p in assets['allow_patterns']))

    def test_other_processors_do_not_import_molmo(self):
        config = SimpleNamespace(type='smolvla')
        self.assertEqual(checkpoint_processor_factory(lambda policy_cfg: 'native', (config,), {}), 'native')


if __name__ == "__main__":
    unittest.main()
