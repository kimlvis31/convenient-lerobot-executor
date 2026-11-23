"""AI-written guards for native WALL-X host precision and unchanged initialization."""

from contextlib import redirect_stdout
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from checkpoint_initialization import checkpoint_model_placement


class CheckpointInitializationTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events

        class Model:
            def __init__(self):
                self.weight = SimpleNamespace(device=SimpleNamespace(type="cpu"))

            def parameters(self):
                return iter([self.weight])

            def to(self, *args, **kwargs):
                events.append(("move", args, kwargs))
                target = args[0] if args else kwargs.get("device")
                if isinstance(target, str) and target.split(":")[0] in ("cpu", "cuda"):
                    self.weight.device.type = target.split(":")[0]
                return self

            def to_bfloat16_for_selected_params(self):
                events.append(("native_cast", self.weight.device.type))
                self.to(dtype="bfloat16")

        self.Model = Model
        self.config = SimpleNamespace(type="wall_x", device="cuda")
        self.modules = {"lerobot.policies.wall_x.modeling_wall_x":
                        SimpleNamespace(Qwen2_5_VLMoEForAction=Model)}
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, self.modules).start()

    def context(self):
        return checkpoint_model_placement(None, self.config, (), {})

    def test_host_cast_precedes_cuda_copy_without_recursing(self):
        model = self.Model()
        with redirect_stdout(io.StringIO()), self.context():
            self.assertIs(model.to("cuda:0"), model)
        self.assertEqual(self.events[0], ("native_cast", "cpu"))
        self.assertEqual(self.events[-1], ("move", ("cuda:0",), {}))
        self.assertEqual(len(self.events), 3)

    def test_cpu_and_explicit_dtype_placement_keep_native_calls(self):
        model = self.Model()
        with self.context():
            model.to("cpu")
            model.to("cuda", dtype="float32")
        self.assertEqual([event[0] for event in self.events], ["move", "move"])

    def test_existing_cuda_or_meta_parameters_are_not_recast(self):
        for device in ("cuda", "meta"):
            model = self.Model()
            model.weight.device.type = device
            with self.context():
                model.to("cuda")
        self.assertEqual([event[0] for event in self.events], ["move", "move"])

    def test_other_policy_or_missing_native_cast_api_is_unchanged(self):
        original = self.Model.to
        self.config.type = "another_policy"
        with self.context():
            self.assertIs(self.Model.to, original)
        self.config.type = "wall_x"
        with patch.object(self.Model, "to_bfloat16_for_selected_params", None), self.context():
            self.assertIs(self.Model.to, original)

    def test_native_method_is_restored_after_an_exception(self):
        original = self.Model.to
        with self.assertRaisesRegex(RuntimeError, "interrupted"), self.context():
            raise RuntimeError("interrupted")
        self.assertIs(self.Model.to, original)

    def test_keyword_device_works_and_does_not_replace_native_initialization_or_loaders(self):
        resize = lambda: self.events.append(("native_vocabulary_initialization",))
        load = lambda: self.events.append(("native_weight_load",))
        model = self.Model()
        with redirect_stdout(io.StringIO()), self.context():
            resize()
            model.to(device="cuda")
            model.to_bfloat16_for_selected_params()
            load()
        self.assertEqual(self.events[0], ("native_vocabulary_initialization",))
        self.assertEqual(self.events[-1], ("native_weight_load",))
        self.assertEqual(sum(event[0] == "native_vocabulary_initialization" for event in self.events), 1)


if __name__ == "__main__":
    unittest.main()
