"""Regression coverage for the legacy checkpoint preparation step."""

from contextlib import nullcontext, redirect_stdout
from dataclasses import make_dataclass
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from checkpoint_compat import PROCESSORS, WEIGHTS, checkpoint_candidate, main, prepare_checkpoint, validate_processor_states
from molmo_compat import LEGACY_FIELDS, LEGACY_FLAGS, evaluation_config
from pinned_policy import pinned_command, pinned_entrypoint, requested_revision


def molmo_runtime_stubs(runtime):
    decoding_error = type("DecodingError", (ValueError,), {})
    decoder = SimpleNamespace(from_pretrained=Mock(side_effect=decoding_error))
    return {"lerobot.policies.molmoact2.configuration_molmoact2": SimpleNamespace(MolmoAct2Config=runtime),
            "lerobot.configs": SimpleNamespace(PreTrainedConfig=decoder),
            "draccus.utils": SimpleNamespace(DecodingError=decoding_error)}


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

    def test_wallx_eval_uses_native_tokenizer_and_original_statistics(self):
        from wallx_compat import LEGACY_STEPS, migrate_processor
        saved = {"steps": [{"registry_name": name, "config": {"old": True}} for name in LEGACY_STEPS]}
        saved["steps"][3]["state_file"] = "pre.safetensors"
        (self.source / "pre.safetensors").write_bytes(b"trained statistics")
        (self.source / PROCESSORS[0]).write_text(json.dumps(saved))
        post = '{"steps":[{"state_file":"post.safetensors"}]}'
        (self.source / PROCESSORS[1]).write_text(post)
        (self.source / "post.safetensors").write_bytes(b"trained action statistics")
        order = ("render_messages", LEGACY_STEPS[0], LEGACY_STEPS[2], LEGACY_STEPS[1], LEGACY_STEPS[3],
                 "wall_x_prompt", "wall_x_tokenizer", LEGACY_STEPS[4])
        current = {"steps": [{"registry_name": name, "config": {"native": True}} for name in order]}
        runtime = SimpleNamespace(WallXTokenizerStep=object, make_wall_x_pre_post_processors=Mock(
            return_value=(SimpleNamespace(get_config=lambda: current), None)))
        types = {"lerobot.policies.wall_x": SimpleNamespace(processor_wall_x=runtime),
                 "lerobot.configs": SimpleNamespace(PreTrainedConfig=SimpleNamespace(
                     from_pretrained=lambda path: SimpleNamespace(type="wall_x")))}
        with patch.dict(sys.modules, types):
            self.assertFalse(checkpoint_candidate(self.source, "wall_x", "train"))
            result = prepare_checkpoint(self.source, "wall_x", self.destination, stage="eval")
        migrated = json.loads((result / PROCESSORS[0]).read_text())
        self.assertEqual([step["registry_name"] for step in migrated["steps"]], list(order))
        self.assertEqual(migrated["steps"][4], saved["steps"][3])
        self.assertEqual((result / "pre.safetensors").read_bytes(), b"trained statistics")
        self.assertEqual((result / PROCESSORS[1]).read_text(), post)
        self.assertEqual((result / WEIGHTS).read_bytes(), b"original")
        self.assertEqual(json.loads((self.source / PROCESSORS[0]).read_text()), saved)
        self.assertIsNone(migrate_processor(migrated, current))
        self.assertIsNone(migrate_processor(saved, saved))
        self.assertIsNone(migrate_processor({"steps": saved["steps"][:-1]}, current))
        with self.assertRaisesRegex(ValueError, "cannot preserve"):
            migrate_processor(saved, {"steps": current["steps"][2:]})

    def test_only_explicit_hub_policy_revisions_require_launcher(self):
        command = ["conda", "run", "-n", "r", "lerobot-eval", "--policy.path=team/policy",
                   "--policy.pretrained_revision=release", "--policy.device=cpu"]
        self.assertIs(pinned_command(command, "eval", "team/policy", None, self.root), command)
        wrapped = pinned_command(command, "eval", "team/policy", "release", self.root)
        self.assertEqual(wrapped[:6], [*command[:4], "python", "-B"])
        self.assertEqual(wrapped[7:], ["eval", *command[5:]])
        for reference in ("./missing", str(self.source), str(self.source / "config.json")):
            with self.subTest(reference=reference):
                self.assertIs(pinned_command(command, "eval", reference, "release", self.root), command)
        with self.assertRaisesRegex(ValueError, "repository ID"):
            pinned_command(command, "eval", "team/policy", "", self.root)
        self.assertEqual(requested_revision([*command, "--policy.pretrained_revision=release"]), "release")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            requested_revision([*command, "--policy.pretrained_revision=main"])

    def test_pinning_with_deferred_camera_detection_uses_one_worker(self):
        camera = str(Path(__file__).with_name("evaluation_cameras.py").resolve())
        command = ["conda", "run", "-n", "test", "python", "-u", camera, "--env.type=vlabench"]
        result = pinned_command(command, "eval", "org/policy", "v1", self.root)
        self.assertEqual(result[:6], ["conda", "run", "-n", "test", "python", "-B"])
        self.assertEqual(result[-2:], ["eval", "--env.type=vlabench"])
        self.assertNotIn(camera, result)

    def test_pinned_policy_preserves_other_sources_and_records_adapter_base(self):
        revision = "c" * 40
        config_dir = self.root / revision
        hub = SimpleNamespace(hf_hub_download=Mock(return_value=str(config_dir / "config.json")))
        calls = []
        class Config:
            @classmethod
            def from_pretrained(cls, pretrained_name_or_path, **kwargs):
                calls.append((pretrained_name_or_path, kwargs))
                return SimpleNamespace(pretrained_path=pretrained_name_or_path, pretrained_revision=kwargs.get("revision"))

        pretrained = SimpleNamespace(get_peft_model=Mock(return_value="adapter"))
        original = pretrained.get_peft_model
        actual = SimpleNamespace(make_pre_post_processors=Mock(return_value="processors"))
        processor = actual.make_pre_post_processors
        def evaluate():
            self.assertIn("--policy.pretrained_revision=" + revision, sys.argv)
            Config.from_pretrained("team/policy", cli_overrides=["--device=cpu"])
            Config.from_pretrained("other/base", revision="other-revision")
            actual.make_pre_post_processors(pretrained_path=Path("team/policy"))
            return 13
        def train():
            config = Config.from_pretrained("team/policy")
            pretrained.get_peft_model(SimpleNamespace(config=SimpleNamespace(pretrained_path="other/base")),
                                      SimpleNamespace(revision="separate-base"))
            pretrained.get_peft_model(SimpleNamespace(config=config), SimpleNamespace(revision=None))
            return 17
        actual.main = evaluate
        modules = {"huggingface_hub": hub, "lerobot.configs": SimpleNamespace(PreTrainedConfig=Config),
                   "lerobot.scripts": SimpleNamespace(lerobot_eval=actual, lerobot_train=SimpleNamespace(main=train)),
                   "lerobot.policies": SimpleNamespace(pretrained=pretrained)}
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["native", "--policy.pretrained_revision=release"]), redirect_stdout(io.StringIO()):
            self.assertEqual(pinned_entrypoint("eval", "team/policy", "release"), 13)
            self.assertEqual(pinned_entrypoint("train", "team/policy", "release"), 17)
        self.assertEqual(calls[0:2], [("team/policy", {"revision": revision, "cli_overrides": ["--device=cpu"]}),
                                      ("other/base", {"revision": "other-revision"})])
        processor.assert_called_once_with(pretrained_path=Path("team/policy"), pretrained_revision=revision)
        self.assertNotIn("revision", original.call_args_list[0].kwargs)
        self.assertEqual(original.call_args.kwargs["revision"], revision)
        self.assertIs(pretrained.get_peft_model, original)
        self.assertIs(actual.make_pre_post_processors, processor)

    def test_molmo_preserves_full_weight_topology_and_training_fields(self):
        config = {"type": "molmoact2", **dict.fromkeys(LEGACY_FLAGS, False), "model_dtype": "bfloat16",
                  "scheduler_decay_steps": None, "action_mode": "both"}
        updated = evaluation_config(config)
        self.assertEqual(updated, {"type": "molmoact2", "scheduler_decay_steps": None, "action_mode": "both",
                                   "dtype": "bfloat16", "train_mode_vlm": "fft"})
        self.assertIsNone(evaluation_config(updated))
        for flag in LEGACY_FLAGS:
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                evaluation_config({**config, flag: True})
        with self.assertRaises(ValueError):
            evaluation_config({**config, "dtype": "float32"})

    def test_molmo_evaluation_writes_only_a_temporary_view_and_skips_training(self):
        config = {"type": "molmoact2", **dict.fromkeys(LEGACY_FLAGS, False), "model_dtype": "bfloat16"}
        config_file = self.source / "config.json"
        original = json.dumps(config)
        config_file.write_text(original)
        (self.source / "stats.safetensors").write_bytes(b"statistics")
        for name in PROCESSORS:
            (self.source / name).write_text(json.dumps({"steps": [{"state_file": "stats.safetensors"}]}))
        runtime = make_dataclass("Config", ["dtype", "train_mode_vlm"])
        imports = molmo_runtime_stubs(runtime)
        with patch.dict(sys.modules, imports):
            self.assertIsNone(prepare_checkpoint(self.source, "molmoact2", self.destination, stage="train"))
            self.assertEqual(prepare_checkpoint(self.source, "molmoact2", self.destination, stage="eval"), self.destination)
        self.assertEqual(config_file.read_text(), original)
        self.assertEqual((self.destination / "stats.safetensors").read_bytes(), b"statistics")
        self.assertEqual((self.destination / WEIGHTS).read_bytes(), b"original")
        self.assertEqual(json.loads((self.destination / "config.json").read_text())["train_mode_vlm"], "fft")

    def test_molmo_hub_probe_pins_weights_and_processors_to_config_revision(self):
        snapshot = self.source / "immutable-revision"
        snapshot.mkdir()
        config = {"type": "molmoact2", **dict.fromkeys(LEGACY_FLAGS, False), "model_dtype": "float32"}
        (snapshot / "config.json").write_text(json.dumps(config))
        for name in PROCESSORS:
            (snapshot / name).write_text('{"steps":[]}')
        (snapshot / WEIGHTS).write_bytes(b"weights")
        downloads = Mock(side_effect=lambda repo, name, revision: str(snapshot / name))
        runtime = make_dataclass("Config", ["dtype", "train_mode_vlm"])
        imports = {**molmo_runtime_stubs(runtime), "huggingface_hub": SimpleNamespace(hf_hub_download=downloads)}
        with patch.dict(sys.modules, imports):
            destination = prepare_checkpoint("example/molmo", "molmoact2", self.destination, revision="tag", stage="eval")
        self.assertEqual(destination, self.destination)
        self.assertEqual(downloads.call_args_list[0].kwargs["revision"], "tag")
        self.assertEqual({call.kwargs["revision"] for call in downloads.call_args_list[1:]}, {snapshot.name})
        downloads.reset_mock()
        imports["lerobot.policies.molmoact2.configuration_molmoact2"] = SimpleNamespace(
            MolmoAct2Config=make_dataclass("LegacyAware", sorted(LEGACY_FIELDS | {"dtype", "train_mode_vlm"})))
        with patch.dict(sys.modules, imports):
            self.assertIsNone(prepare_checkpoint("example/molmo", "molmoact2", self.destination, stage="eval"))
        downloads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
