"""Check publication, reuse and failure cleanup of a managed simulator prefix."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robomme_setup import prepare_simulator, torch_index, RECIPE


class SimulatorSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / RECIPE
        self.calls = []
        self.addCleanup(patch.stopall)
        patch("robomme_setup.sys.platform", "linux").start()
        patch.dict(os.environ, {"CONVLEX_ROBOMME_CACHE": str(self.root)}).start()
        patch.dict(sys.modules, {"fcntl": SimpleNamespace(LOCK_EX=2, flock=lambda *args: None)}).start()
        patch("robomme_setup.torch_index", return_value="https://download.pytorch.org/whl/cu128").start()

    def invoke(self, args, **kwargs):
        self.calls.append(args)
        if "create" in args:
            prefix = Path(args[args.index("--prefix") + 1])
            (prefix / "bin").mkdir()
            (prefix / "bin/python").touch()
        return SimpleNamespace(returncode=0)

    def test_new_prefix_is_published_only_after_a_real_probe(self):
        with patch("robomme_setup.subprocess.run", side_effect=self.invoke):
            python, environment = prepare_simulator("conda")
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.calls[-1][-1], "--probe")
        self.assertEqual((self.folder / "ready").read_text().strip(), Path(python).parent.parent.name)
        self.assertEqual(environment["PYTHONPATH"], "")
        self.assertEqual(environment["CONDA_PREFIX"], str(Path(python).parent.parent))

    def test_verified_prefix_is_reused_without_installs(self):
        prefix = self.folder / "conda-existing"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin/python").touch()
        (self.folder / "ready").write_text(prefix.name)
        with patch("robomme_setup.subprocess.run", side_effect=self.invoke):
            python, _ = prepare_simulator("conda")
        self.assertEqual(Path(python), prefix / "bin/python")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][-1], "--check")

    def test_failed_probe_cleans_only_its_new_prefix_and_keeps_old_marker(self):
        self.folder.mkdir()
        (self.folder / "ready").write_text("conda-previous")
        def failing_probe(args, **kwargs):
            if args[-1] == "--probe":
                raise subprocess.CalledProcessError(1, args)
            return self.invoke(args, **kwargs)
        with patch("robomme_setup.subprocess.run", side_effect=failing_probe):
            with self.assertRaises(subprocess.CalledProcessError):
                prepare_simulator("conda")
        self.assertEqual((self.folder / "ready").read_text(), "conda-previous")
        self.assertFalse(list(self.folder.glob("conda-*")))

    def test_invalid_saved_prefix_cannot_escape_the_managed_folder(self):
        self.folder.mkdir()
        (self.folder / "ready").write_text("../../other")
        with patch("robomme_setup.subprocess.run", side_effect=self.invoke):
            python, _ = prepare_simulator("conda")
        self.assertEqual(self.calls[0][1], "create")
        self.assertEqual(Path(python).parent.parent.parent, self.folder)


class TorchWheelSelectionTests(unittest.TestCase):
    def test_driver_compatibility_selects_an_available_official_wheel(self):
        for driver, wheel in (("12.6", "cu126"), ("12.8", "cu128"), ("13.1", "cu128")):
            with self.subTest(driver=driver), patch("robomme_setup.subprocess.run", return_value=SimpleNamespace(
                stdout=f"<nvidia_smi_log><cuda_version>{driver}</cuda_version></nvidia_smi_log>"
            )):
                self.assertTrue(torch_index().endswith(wheel))
        with patch("robomme_setup.subprocess.run", return_value=SimpleNamespace(
            stdout="<nvidia_smi_log><cuda_version>12.4</cuda_version></nvidia_smi_log>"
        )):
            with self.assertRaisesRegex(RuntimeError, "12.6-compatible"):
                torch_index()


if __name__ == "__main__":
    unittest.main()
