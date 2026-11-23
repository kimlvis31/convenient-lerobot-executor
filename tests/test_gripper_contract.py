"""AI-written checks for documented dataset names without changing normalization."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training_preflight import molmo_feature_defaults, native_training_preflight
from native_runtime import dataset_preflight, training_dataset_contract


class GripperContractTests(unittest.TestCase):
    def setUp(self):
        self.train = {"dataset": "lerobot/libero", "extra": []}
        self.dataset = {"robot_type": "panda", "features": {
            "action": {"shape": [7], "dtype": "float32", "names": ["actions"]},
            "observation.state": {"shape": [8], "dtype": "float32", "names": ["state"]}}}

    def test_standard_coarse_labels_receive_exact_gripper_positions(self):
        before = deepcopy(self.dataset)
        result = molmo_feature_defaults(self.train, self.dataset)
        names = result["policy.dataset_feature_names"]
        self.assertEqual(["gripper" in name for name in names["action"]], [False] * 6 + [True])
        self.assertEqual(["gripper" in name for name in names["observation.state"]], [False] * 6 + [True, True])
        self.assertEqual(self.dataset, before)
        self.assertNotIn("policy.normalize_gripper", result)

    def test_unknown_layouts_and_explicit_overrides_are_preserved(self):
        cases = [(dict(self.train, dataset="user/robot"), self.dataset, {}),
                 (self.train, dict(self.dataset, robot_type="other"), {}),
                 (dict(self.train, extra=['--policy.dataset_feature_names={"action":["custom"]}']), self.dataset, {}),
                 (dict(self.train, extra=['--policy.normalize_gripper=true']), self.dataset, {}),
                 (self.train, self.dataset, {"normalize_gripper": True})]
        changed = deepcopy(self.dataset)
        changed['features']['action']['shape'] = [14]
        cases.append((self.train, changed, {}))
        for train, dataset, policy in cases:
            with self.subTest(train=train, dataset=dataset, policy=policy):
                self.assertEqual(molmo_feature_defaults(train, dataset, policy), {})

    def test_target_labels_take_precedence_over_saved_source_dataset_labels(self):
        saved = {"action": ["saved_gripper"] * 7}
        result = molmo_feature_defaults(self.train, self.dataset, {"dataset_feature_names": saved})
        self.assertNotEqual(result['policy.dataset_feature_names']['action'], saved['action'])
        self.assertEqual(saved, {"action": ["saved_gripper"] * 7})
        dataset = deepcopy(self.dataset)
        for key in dataset['features']:
            dataset['features'][key]['names'] = ['dimension'] * dataset['features'][key]['shape'][0]
        self.assertEqual(molmo_feature_defaults(self.train, dataset), {})

    def test_preflight_surfaces_names_and_exact_resume_keeps_saved_configuration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'meta').mkdir()
            (root / 'meta/info.json').write_text(json.dumps(self.dataset))
            config = {'model': 'molmoact2', 'train': {**self.train, 'dataset_root': str(root)}}
            self.assertIn('policy.dataset_feature_names', native_training_preflight(config))
            config['train']['resume'] = 'saved-checkpoint'
            self.assertEqual(native_training_preflight(config), {})

    def test_cold_hub_policy_uses_resolved_runtime_settings(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'meta').mkdir()
            (root / 'meta/info.json').write_text(json.dumps(self.dataset))
            config = {'model': 'molmoact2', 'train': {**self.train, 'dataset_root': str(root),
                                                   'pretrained': 'public/repository'}}
            self.assertEqual(native_training_preflight(config), {})
            self.assertIn('policy.dataset_feature_names', native_training_preflight(config, policy_config={
                'type': 'molmoact2', 'normalize_gripper': False}))
            self.assertEqual(native_training_preflight(config, policy_config={'type': 'molmoact2',
                                                                             'normalize_gripper': True}), {})

    def test_native_metadata_setter_cannot_erase_required_or_explicit_names(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'meta').mkdir()
            (root / 'meta/info.json').write_text(json.dumps(self.dataset))
            metadata = SimpleNamespace(root=root, repo_id='lerobot/libero', features=self.dataset['features'])
            def native_factory(cfg, ds_meta=None):
                cfg.dataset_feature_names = {'action': ['actions'], 'observation.state': ['state']}
                return SimpleNamespace(config=cfg)
            trainer = SimpleNamespace(make_policy=native_factory)
            modules = {'lerobot.scripts': SimpleNamespace(lerobot_train=trainer),
                       'draccus': SimpleNamespace(encode=lambda cfg: {'type': cfg.type,
                            'normalize_gripper': False, 'dataset_feature_names': cfg.dataset_feature_names})}
            for arguments in ([], ['--resume=true'], ['--policy.dataset_feature_names={"action":["explicit"]}']):
                saved_names = {'action': ['explicit'] * 7, 'observation.state': ['saved'] * 8}
                policy = SimpleNamespace(type='molmoact2', pretrained_path=None,
                                         dataset_feature_names=deepcopy(saved_names))
                with self.subTest(arguments=arguments), patch.dict(sys.modules, modules), \
                     patch.object(sys, 'argv', ['lerobot-train', *arguments]), training_dataset_contract():
                    result = trainer.make_policy(policy, ds_meta=metadata)
                if arguments:
                    self.assertEqual(result.config.dataset_feature_names, saved_names)
                else:
                    self.assertEqual(result.config.dataset_feature_names['action'][-1], 'gripper')
                    self.assertEqual(result.config.dataset_feature_names['observation.state'][-2:],
                                     ['gripper_left', 'gripper_right'])

    def test_cold_camera_mapping_applies_before_dataset_timestamp_resolution(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'meta').mkdir()
            data = deepcopy(self.dataset)
            for camera in ('image', 'image2'):
                data['features']['observation.images.' + camera] = {
                    'dtype': 'video', 'shape': [256, 256, 3], 'names': ['height', 'width', 'channels']}
            (root / 'meta/info.json').write_text(json.dumps(data))
            policy = SimpleNamespace(type='molmoact2', pretrained_path='public/repository')
            cfg = SimpleNamespace(policy=policy, resume=False, rename_map={},
                                  dataset=SimpleNamespace(repo_id='lerobot/libero'))
            metadata = SimpleNamespace(root=root, repo_id=cfg.dataset.repo_id)
            original_loader = lambda *args, **kwargs: metadata
            factory = SimpleNamespace(load_dataset_metadata=original_loader)
            def prepare(config):
                factory.load_dataset_metadata(config.dataset.repo_id)
                self.assertEqual(config.rename_map, {'observation.images.image2': 'observation.images.wrist_image'})
                self.assertEqual(config.policy.dataset_feature_names['action'][-1], 'gripper')
                return 'dataset', 'eval-dataset'
            trainer = SimpleNamespace(make_train_eval_datasets=prepare)
            encoded = {'type': 'molmoact2', 'input_features': {
                'observation.images.' + camera: {'type': 'VISUAL', 'shape': [3, 256, 256]}
                for camera in ('image', 'wrist_image')}}
            modules = {'lerobot.datasets': SimpleNamespace(factory=factory),
                       'draccus': SimpleNamespace(encode=lambda policy: encoded)}
            with patch.dict(sys.modules, modules), patch.object(sys, 'argv', ['lerobot-train']), dataset_preflight(trainer):
                self.assertEqual(trainer.make_train_eval_datasets(cfg), ('dataset', 'eval-dataset'))
            self.assertIs(factory.load_dataset_metadata, original_loader)
            self.assertIs(trainer.make_train_eval_datasets, prepare)


if __name__ == '__main__':
    unittest.main()
