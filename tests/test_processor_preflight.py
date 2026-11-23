"""AI-written checks for early processor-file checks and native extension fallbacks."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from checkpoint_compat import PROCESSORS, validate_evaluation_processor_files
from runner import stage_failure_reason


class ProcessorPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = SimpleNamespace(type='eo1', pretrained_path=str(self.root))
        self.module = SimpleNamespace()
        self.factory = SimpleNamespace(_import_sibling_policy_module=Mock(return_value=self.module))

    def test_missing_default_processors_fail_before_model_construction(self):
        with self.assertRaisesRegex(FileNotFoundError, 'training initializer'):
            validate_evaluation_processor_files(self.config, self.factory)
        self.factory._import_sibling_policy_module.assert_called_once_with(type(self.config), 'processor')

    def test_complete_and_partial_exports_are_distinguished(self):
        (self.root / PROCESSORS[0]).write_text('{}')
        with self.assertRaisesRegex(FileNotFoundError, PROCESSORS[1]):
            validate_evaluation_processor_files(self.config, self.factory)
        (self.root / PROCESSORS[1]).write_text('{}')
        self.factory._import_sibling_policy_module.reset_mock()
        validate_evaluation_processor_files(self.config, self.factory)
        self.factory._import_sibling_policy_module.assert_not_called()

    def test_native_custom_factories_and_unknown_apis_keep_their_own_behavior(self):
        self.module.make_eo1_pre_post_processors_from_pretrained = lambda **kwargs: None
        validate_evaluation_processor_files(self.config, self.factory)
        validate_evaluation_processor_files(self.config, SimpleNamespace())
        self.factory._import_sibling_policy_module.return_value = None
        validate_evaluation_processor_files(self.config, self.factory)

    def test_fresh_initializers_and_unknown_hub_extension_apis_are_not_probed(self):
        self.config.pretrained_path = None
        validate_evaluation_processor_files(self.config, self.factory)
        self.config.pretrained_path = 'example/public-policy'
        validate_evaluation_processor_files(self.config, SimpleNamespace())
        self.factory._import_sibling_policy_module.assert_not_called()

    def hub(self):
        class Unavailable(Exception):
            pass
        api = SimpleNamespace(model_info=Mock(return_value=SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=name) for name in PROCESSORS])))
        constants = SimpleNamespace(HF_HUB_OFFLINE=False)
        modules = {'huggingface_hub': SimpleNamespace(HfApi=lambda: api, constants=constants),
                   'huggingface_hub.errors': SimpleNamespace(HfHubHTTPError=Unavailable),
                   'httpx': SimpleNamespace(TransportError=Unavailable)}
        self.config.pretrained_path = 'example/public-policy'
        self.config.pretrained_revision = 'immutable-revision'
        return api, constants, modules, Unavailable

    def test_hub_metadata_checks_pinned_processor_names_without_downloading_weights(self):
        api, constants, modules, _ = self.hub()
        with patch.dict(sys.modules, modules):
            validate_evaluation_processor_files(self.config, self.factory)
            api.model_info.assert_called_once_with('example/public-policy', revision='immutable-revision', timeout=10)
            api.model_info.return_value.siblings = []
            with self.assertRaisesRegex(FileNotFoundError, 'training initializer'):
                validate_evaluation_processor_files(self.config, self.factory)

    def test_offline_and_failed_metadata_lookups_preserve_native_cached_loading(self):
        api, constants, modules, Unavailable = self.hub()
        with patch.dict(sys.modules, modules):
            constants.HF_HUB_OFFLINE = True
            validate_evaluation_processor_files(self.config, self.factory)
            api.model_info.assert_not_called()
            constants.HF_HUB_OFFLINE = False
            api.model_info.side_effect = Unavailable('Network or authentication unavailable')
            validate_evaluation_processor_files(self.config, self.factory)

    def test_native_migration_error_has_a_short_actionable_reason(self):
        log = self.root / 'eval.log'
        log.write_text('lerobot.processor.pipeline.ProcessorMigrationError: Model requires migration to processor format.\n')
        reason = stage_failure_reason(log)
        self.assertIn('Evaluation processor files are missing', reason)
        self.assertIn('saved normalization statistics', reason)

    def test_cudnn_diagnostic_explains_a_retry_without_asserting_memory_exhaustion(self):
        log = self.root / 'eval.log'
        log.write_text('RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR\n')
        reason = stage_failure_reason(log)
        self.assertIn('smaller batch', reason)
        self.assertIn('parallel evaluation tasks', reason)
        self.assertNotIn('out of memory', reason)

    def test_mixed_dtype_diagnostic_identifies_precision_settings(self):
        log = self.root / 'eval.log'
        for detail in ('mat1 and mat2 must have the same dtype, but got Float and BFloat16',
                       'expected m1 and m2 to have the same dtype, but got: c10::BFloat16 != float'):
            with self.subTest(detail=detail):
                log.write_text('RuntimeError: ' + detail + '\n')
                self.assertIn('tensor data types differ', stage_failure_reason(log))


if __name__ == '__main__':
    unittest.main()
