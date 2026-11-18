"""Regression coverage for the legacy checkpoint preparation step."""

from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from checkpoint_compat import PROCESSORS, WEIGHTS, checkpoint_candidate, main, prepare_checkpoint, validate_processor_states


class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.destination = self.root / "migrated"
        self.config = {
            "type": "vqbet",
            "mlp_hidden_dim": 1024,
            "gpt_hidden_dim": 512,
            "crop_shape": [84, 84],
            "normalization_mapping": {"STATE": "MIN_MAX", "ACTION": "MIN_MAX"},
            "input_features": {"observation.state": {"type": "STATE", "shape": [2]}},
            "output_features": {"action": {"type": "ACTION", "shape": [2]}},
        }
        (self.source / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        (self.source / WEIGHTS).write_bytes(b"original")
        self.legacy_key = "normalize_inputs.buffer_observation_state.min"
        self.state = {self.legacy_key: object(), "policy.weight": object()}
        self.stats = {
            "observation.state": {"min": object(), "max": object()},
            "action": {"min": object(), "max": object()},
        }

    def test_candidate_is_limited_to_supported_policy_and_processor_state(self):
        self.assertTrue(checkpoint_candidate(self.source, "vqbet"))
        self.assertFalse(checkpoint_candidate(self.source, "smolvla"))
        self.assertFalse(checkpoint_candidate(self.root / "missing", "diffusion"))
        self.assertTrue(checkpoint_candidate("lerobot/diffusion_pusht", "diffusion"))
        (self.source / PROCESSORS[0]).write_text("{}", encoding="utf-8")
        self.assertFalse(checkpoint_candidate(self.source, "vqbet"))
        self.assertIsNone(prepare_checkpoint(self.source, "vqbet", self.destination))
        self.assertFalse(self.destination.exists())

    def test_modern_or_unrelated_checkpoint_skips_conversion(self):
        header = Mock(return_value=nullcontext({"policy.weight": object()}))
        with patch.dict(sys.modules, {"safetensors": SimpleNamespace(safe_open=header)}):
            self.assertIsNone(prepare_checkpoint(self.source, "diffusion", self.destination))
            header.assert_not_called()
            self.assertIsNone(prepare_checkpoint(self.source, "vqbet", self.destination))
            header.assert_called_once()
        self.assertFalse(self.destination.exists())

    def test_old_vqbet_creates_independent_preprocessors_and_preserves_weights(self):
        decoder = Mock(side_effect=lambda folder: json.loads((folder / "config.json").read_text(encoding="utf-8")))

        def processors(*, policy_cfg, dataset_stats):
            self.assertEqual(policy_cfg["crop_shape"], [84, 84])
            self.assertEqual(policy_cfg["gpt_hidden_dim"], 512)
            self.assertNotIn("mlp_hidden_dim", policy_cfg)
            self.assertEqual(dataset_stats, self.stats)
            return [SimpleNamespace(save_pretrained=lambda folder, name=name: (folder / name).write_text("{}"))
                    for name in PROCESSORS]

        clean = {"policy.weight": self.state["policy.weight"]}
        save_file = Mock()
        migration = SimpleNamespace(
            extract_normalization_stats=Mock(return_value=self.stats),
            remove_normalization_layers=Mock(return_value=clean),
            clean_state_dict=Mock(side_effect=lambda value: value),
        )
        imports = {
            "safetensors": SimpleNamespace(safe_open=lambda *a, **kw: nullcontext(self.state)),
            "safetensors.torch": SimpleNamespace(load_file=lambda *a, **kw: self.state, save_file=save_file),
            "lerobot.configs": SimpleNamespace(PreTrainedConfig=SimpleNamespace(from_pretrained=decoder)),
            "lerobot.policies.factory": SimpleNamespace(make_pre_post_processors=processors),
            "lerobot.processor": SimpleNamespace(migrate_policy_normalization=migration),
        }
        with patch.dict(sys.modules, imports):
            self.assertEqual(prepare_checkpoint(self.source, "vqbet", self.destination), self.destination)
        decoder.assert_called_once_with(self.destination)
        save_file.assert_called_once_with(clean, str(self.destination / WEIGHTS), metadata={"format": "pt"})
        self.assertTrue(all((self.destination / name).is_file() for name in PROCESSORS))
        self.assertEqual(json.loads((self.source / "config.json").read_text()), self.config)
        self.assertEqual((self.source / WEIGHTS).read_bytes(), b"original")

    def test_hub_checks_headers_before_fetching_weights_and_pins_revision(self):
        from checkpoint_compat import _hub_source

        downloads = Mock(side_effect=lambda _, name, **kw: str(self.source / name))
        (self.source / WEIGHTS).unlink()
        header = SimpleNamespace(tensors={})
        api = SimpleNamespace(
            model_info=Mock(return_value=SimpleNamespace(
                siblings=[SimpleNamespace(rfilename=name) for name in ("config.json", WEIGHTS)], sha="pinned"
            )),
            parse_safetensors_file_metadata=Mock(return_value=header),
        )
        hub = SimpleNamespace(
            HfApi=Mock(return_value=api), constants=SimpleNamespace(HF_HUB_OFFLINE=False),
            hf_hub_download=downloads, try_to_load_from_cache=Mock(return_value=None),
        )
        imports = {
            "huggingface_hub": hub,
            "huggingface_hub.errors": SimpleNamespace(
                HfHubHTTPError=type("HubError", (Exception,), {}),
                OfflineModeIsEnabled=type("Offline", (Exception,), {}),
                SafetensorsParsingError=type("HeaderError", (Exception,), {}),
            ),
            "httpx": SimpleNamespace(TransportError=type("TransportError", (Exception,), {})),
        }
        with patch.dict(sys.modules, imports):
            self.assertIsNone(_hub_source("lerobot/vqbet_pusht", "vqbet", "main"))
            downloads.assert_called_once_with("lerobot/vqbet_pusht", "config.json", revision="pinned")
            api.parse_safetensors_file_metadata.assert_called_once_with(
                "lerobot/vqbet_pusht", WEIGHTS, revision="pinned"
            )
            header.tensors[self.legacy_key] = object()
            self.assertIsNotNone(_hub_source("lerobot/vqbet_pusht", "vqbet", "main"))
            downloads.assert_called_with("lerobot/vqbet_pusht", WEIGHTS, revision="pinned")

    def test_processor_validation_checks_only_declared_local_state(self):
        (self.source / PROCESSORS[0]).write_text('{"steps":[{"config":{}},{"state_file":"nested/state.safetensors"}]}')
        (self.source / PROCESSORS[1]).write_text('{"steps":[{"state_file":"state2.safetensors"}]}')
        self.assertIsNone(validate_processor_states("lerobot/vqbet_pusht"))
        with self.assertRaisesRegex(FileNotFoundError, "nested/state.safetensors, state2.safetensors"):
            validate_processor_states(self.source)
        (self.source / "nested").mkdir()
        (self.source / "nested/state.safetensors").write_bytes(b"state")
        with self.assertRaisesRegex(FileNotFoundError, "state2.safetensors"):
            validate_processor_states(self.source)
        (self.source / "state2.safetensors").write_bytes(b"state")
        self.assertIsNone(validate_processor_states(self.source))

    def test_processor_validation_accepts_stateless_or_legacy_sources(self):
        self.assertIsNone(validate_processor_states(self.source))
        (self.source / PROCESSORS[0]).write_text('{"name":"empty","steps":[{}]}')
        self.assertIsNone(validate_processor_states(self.source))

    def test_cli_signals_a_skipped_checkpoint(self):
        output = io.StringIO()
        args = ["compat", "--policy", str(self.root / "missing"), "--output", str(self.destination), "--model", "vqbet"]
        with patch.object(sys, "argv", args), redirect_stdout(output):
            main()
        self.assertEqual(output.getvalue(), "MANAGER_CHECKPOINT null\n")


if __name__ == "__main__":
    unittest.main()
