"""AI-written checks for managed RoboTwin's guarded compatibility boundaries."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import robotwin_setup


class RoboTwinSetupTests(unittest.TestCase):
    def test_unsupported_host_is_rejected_before_driver_or_package_work(self):
        with patch.object(robotwin_setup.sys, "platform", "linux"), \
             patch.object(robotwin_setup.platform, "machine", return_value="aarch64"):
            with self.assertRaisesRegex(RuntimeError, "Linux x86-64"):
                robotwin_setup.check_graphics()

    @staticmethod
    def cuda_fixture(folder):
        prefix = Path(folder).resolve()
        toolkit = prefix / "targets/x86_64-linux/include"
        toolkit.mkdir(parents=True)
        (toolkit / "cuda_runtime.h").touch()
        (toolkit / "cuda_runtime_api.h").touch()
        site = prefix / "lib/python3.12/site-packages"
        for vendor, headers in (("cusparse", ["cusparse.h"]),
                                ("cublas", ["cublas_v2.h", "cublasLt.h"]),
                                ("cusolver", ["cusolverDn.h"]),
                                ("curand", ["curand_kernel.h"])):
            directory = site / "nvidia" / vendor / "include"
            directory.mkdir(parents=True)
            for header in headers:
                (directory / header).touch()
        return prefix, site

    def test_cuda_headers_use_toolkit_and_only_owned_nvidia_wheels(self):
        with TemporaryDirectory() as folder:
            prefix, site = self.cuda_fixture(folder)
            paths = robotwin_setup.cuda_header_paths(prefix, [site])
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_relative_to(prefix) for path in paths))
            with self.assertRaisesRegex(RuntimeError, "outside the simulator prefix"):
                robotwin_setup.cuda_header_paths(prefix, [prefix.parent / "other-policy/site-packages"])
            (site / "nvidia/cusparse/include/cusparse.h").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing development headers: cusparse.h"):
                robotwin_setup.cuda_header_paths(prefix, [site])

    def test_build_environment_propagates_headers_to_gcc_and_nvcc(self):
        with TemporaryDirectory(prefix="cuda prefix ") as folder:
            prefix, site = self.cuda_fixture(folder)
            fake_torch = SimpleNamespace(__version__="2.9.1+cu128", version=SimpleNamespace(cuda="12.8"),
                                         cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                                              get_device_capability=lambda _: (8, 6)))
            cc = str(prefix / "bin/x86_64-conda-linux-gnu-cc")
            cxx = str(prefix / "bin/x86_64-conda-linux-gnu-c++")
            output = prefix / "build.json"
            with patch.dict(sys.modules, {"torch": fake_torch}), \
                    patch.dict(os.environ, {"CONDA_PREFIX": str(prefix), "CC": cc, "CXX": cxx}), \
                    patch.object(robotwin_setup.sysconfig, "get_path", return_value=str(site)), \
                    patch.object(robotwin_setup.shutil, "which", side_effect=lambda value: value):
                robotwin_setup.write_build_environment(output)
            env = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(env["CPATH"], env["CPLUS_INCLUDE_PATH"])
            flags = shlex.split(env["NVCC_PREPEND_FLAGS"])
            for directory in robotwin_setup.cuda_header_paths(prefix, [site]):
                self.assertIn("-I" + str(directory), flags)
            self.assertEqual(env["TORCH_CUDA_ARCH_LIST"], "8.6")

    def test_cuda_header_smoke_uses_managed_nvcc_and_cleans_scratch(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []

            def run(arguments):
                source = Path(arguments[arguments.index("-c") + 1])
                calls.append((arguments, source.read_text(encoding="utf-8")))

            build = SimpleNamespace(root=root, prefix=root / "env", run=run)
            robotwin_setup.check_cuda_headers(build)
            self.assertEqual(calls[0][0][0], str(root / "env/bin/nvcc"))
            for header in robotwin_setup.CUDA_HEADERS:
                self.assertIn(f"#include <{header}>", calls[0][1])
            self.assertEqual(list(root.glob("cuda-headers-*")), [])

    def test_sapien_cp312_wheel_contract_and_already_patched_loader(self):
        # Exact statements inspected in the public 3.0.0b1 cp312 wheel. Its
        # docstrings already contain ".srdf", so matching that word is unsafe.
        original = ('def parse(self, urdf_file, srdf_file=None):\n'
                    '    """Defaults to the ".srdf" file."""\n'
                    '    with open(urdf_file, "r") as f:\n'
                    '        urdf_string = f.read()\n'
                    '    if srdf_file is None:\n'
                    '        srdf_file = urdf_file[:-4] + "srdf"\n'
                    '    if os.path.isfile(srdf_file):\n'
                    '        with open(srdf_file, "r") as f:\n'
                    '            self.ignore_pairs = self.parse_srdf(f.read())\n')
        with TemporaryDirectory() as folder:
            path = Path(folder) / "urdf_loader.py"
            path.write_text(original, encoding="utf-8")
            robotwin_setup.patch_sapien_loader(path)
            first = path.read_text(encoding="utf-8")
            self.assertEqual(first.count('encoding="utf-8"'), 2)
            self.assertIn('srdf_file = urdf_file[:-4] + ".srdf"', first)
            robotwin_setup.patch_sapien_loader(path)
            self.assertEqual(first, path.read_text(encoding="utf-8"))

    def test_sapien_patch_refuses_unrecognized_loader(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "urdf_loader.py"
            path.write_text("def changed_loader(): pass\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pinned version"):
                robotwin_setup.patch_sapien_loader(path)

    def test_source_patch_is_idempotent_and_refuses_changed_upstream(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "source.py"
            path.write_text("value = wp.torch.device_from_torch(device)\n", encoding="utf-8")
            robotwin_setup.replace_checked(path, "wp.torch.device_from_torch(", "wp.device_from_torch(")
            robotwin_setup.replace_checked(path, "wp.torch.device_from_torch(", "wp.device_from_torch(")
            self.assertEqual(path.read_text(), "value = wp.device_from_torch(device)\n")
            path.write_text("changed upstream API\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pinned version"):
                robotwin_setup.replace_checked(path, "wp.torch.device_from_torch(", "wp.device_from_torch(")

    def test_ambiguous_patch_does_not_modify_source(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "source.py"
            original = "old_call()\nold_call()\n"
            path.write_text(original, encoding="utf-8")
            with self.assertRaises(RuntimeError):
                robotwin_setup.replace_checked(path, "old_call", "new_call")
            self.assertEqual(path.read_text(), original)

    def test_runtime_patches_refuse_other_package_versions(self):
        with patch.object(robotwin_setup.importlib.metadata, "version", return_value="9.0"):
            with self.assertRaisesRegex(RuntimeError, "unrecognized mplib"):
                robotwin_setup.patch_runtime_packages()

    def test_probe_keeps_official_source_hash_and_argument_boundaries(self):
        source = "/path with spaces/lerobot/envs/robotwin.py"
        build = SimpleNamespace(python="/managed prefix/bin/python", options={
            "source": source, "source_sha256": "a" * 64,
        })
        command = robotwin_setup._worker_command(build, "--probe")
        self.assertEqual(command[0], build.python)
        self.assertEqual(command[-4:], ["--source", source, "--source-sha256", "a" * 64])
        self.assertIn("--probe", command)

    def test_probe_can_hash_a_source_without_importing_lerobot(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "robotwin.py"
            content = b"raise RuntimeError('must not execute while hashing')\n"
            source.write_bytes(content)
            build = SimpleNamespace(python="python", options={"source": str(source)})
            command = robotwin_setup._worker_command(build, "--inspect")
            self.assertEqual(command[-1], hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
