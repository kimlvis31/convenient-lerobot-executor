"""AI-written coverage for native adapter scopes and structured optimizer/loss inputs."""

from contextlib import redirect_stdout
from dataclasses import dataclass, field
import io
import math
import unittest
from unittest.mock import patch

from adapters import structured_policy_value, training_flags
from benchmarks import policy_training_options
from manage import configure_policy_options, validate_model_training_fields


class StructuredOptionTests(unittest.TestCase):
    def test_inspection_exposes_default_factory_losses_and_optimizer_tuples(self):
        @dataclass
        class Config:
            loss: dict = field(default_factory=lambda: {'lambda_video': 1.0, 'lambda_action': 1.0})
            optimizer_betas: tuple = (0.9, 0.95)
        options = policy_training_options(Config)
        self.assertEqual(options['loss']['value_type'], 'numeric_object')
        self.assertEqual(options['optimizer_betas']['default'], [0.9, 0.95])
        train = {}
        with patch('sys.stdin', io.StringIO('2\n{"lambda_action":0.5}\n1\n')), redirect_stdout(io.StringIO()):
            configure_policy_options(train, options)
        self.assertEqual(train['policy_options']['loss'], {'lambda_video': 1.0, 'lambda_action': 0.5})
        self.assertIn('--policy.loss={"lambda_video":1.0,"lambda_action":0.5}', training_flags(train, 'fastwam'))

    def test_native_lora_routes_to_scope_and_keeps_component_choices(self):
        @dataclass
        class Config:
            lora_rank: int = 0
            lora_alpha: int = 16
            lora_dropout: float = 0.1
            lora_full_model: bool = False
        options = policy_training_options(Config)
        train = {'policy_training': {'tune_diffusion_model': True}}
        with patch('sys.stdin', io.StringIO('5\n2\n1\n')), redirect_stdout(io.StringIO()):
            configure_policy_options(train, options)
        self.assertEqual(train['policy_training'], {'tune_diffusion_model': True, 'lora_rank': 2})
        self.assertNotIn('policy_options', train)
        self.assertIn('--policy.lora_rank=2', training_flags(train, 'groot'))

    def test_native_lora_invalid_scope_values_fail_before_launch(self):
        for name, value in [('lora_rank', -1), ('lora_alpha', 0), ('lora_dropout', math.nan),
                            ('lora_dropout', 2), ('lora_full_model', 1)]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                training_flags({'policy_training': {name: value}}, 'groot')
        self.assertIn('--policy.lora_rank=0', training_flags({'policy_training': {'lora_rank': 0}}, 'groot'))

    def test_unknown_nonfinite_or_wrong_sequence_inputs_are_rejected(self):
        spec = {'default': [0.9, 0.95], 'value_type': 'numeric_sequence', 'minimum': 0, 'maximum_exclusive': 1}
        for value in ([0.9], [0.9, 1.0], [-0.1, 0.9], [0.9, math.nan], [True, 0.9], [10**500, 0.9]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                structured_policy_value(value, spec)
        object_spec = {'default': {'lambda_action': 1.0}, 'value_type': 'numeric_object'}
        with self.assertRaises(ValueError):
            structured_policy_value({'misspelled': 1.0}, object_spec)

    def test_invalid_json_retries_the_same_option_then_preserves_default_dimensions(self):
        options = {'optimizer_betas': {'default': [0.9, 0.95], 'value_type': 'numeric_sequence',
                                       'minimum': 0, 'maximum_exclusive': 1}}
        train, output = {}, io.StringIO()
        with patch('sys.stdin', io.StringIO('2\n[0.9]\n[0.9,1.0]\n[0.8,0.99]\n1\n')), redirect_stdout(output):
            configure_policy_options(train, options)
        self.assertEqual(train['policy_options']['optimizer_betas'], [0.8, 0.99])
        self.assertEqual(output.getvalue().count('Enter a valid JSON value'), 2)

    def test_imported_structures_follow_selected_runtime_members(self):
        spec = {'default': {'lambda_video': 1.0, 'lambda_action': 1.0}, 'value_type': 'numeric_object'}
        catalog = {'policy_fields': {'fastwam': ['loss']}, 'training_options': {'fastwam': {'loss': spec}}}
        validate_model_training_fields('fastwam', {'policy_options': {'loss': {'lambda_action': 0.5}}}, catalog)
        with self.assertRaisesRegex(ValueError, 'Invalid fastwam'):
            validate_model_training_fields('fastwam', {'policy_options': {'loss': {'lambda_wrong': 0.5}}}, catalog)


if __name__ == '__main__':
    unittest.main()
