"""Policy-kernel selection must preserve the runtime and reject mismatched binaries."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from packaging.requirements import Requirement
from packaging.tags import Tag

from policy_extensions import (NATTEN, WheelRuntime, extension_problems, extension_runtime_issue,
                               extension_specs, import_issue, install_extensions, resolve_wheel,
                               select_wheel, wheel_runtime)


class PolicyExtensionTests(unittest.TestCase):
    runtime = WheelRuntime("2110", "130", frozenset({Tag("cp312", "cp312", "linux_x86_64")}))

    @staticmethod
    def link(version="0.21.6", build="torch2110cu130", abi="cp312-cp312-linux_x86_64", fragment=""):
        return (f"https://github.com/SHI-Labs/NATTEN/releases/download/v{version}/"
                f"natten-{version}%2B{build}-{abi}.whl{fragment}")

    def html(self, *links):
        return "\n".join(f'<a href="{url}">binary</a>' for url in links)

    def test_selection_matches_torch_cuda_python_and_platform_without_upgrade(self):
        wanted = self.link()
        page = self.html(self.link("0.21.7", "torch2130cu130"), self.link(build="torch2110cu128"),
                         self.link(abi="cp311-cp311-linux_x86_64"),
                         self.link(abi="cp312-cp312-linux_aarch64"), wanted)
        selected = select_wheel(NATTEN, page, self.runtime)
        self.assertEqual(selected["url"], wanted)
        self.assertEqual(selected["version"], "0.21.6+torch2110cu130")

    def test_unavailable_build_has_no_compiler_or_torch_upgrade_fallback(self):
        with self.assertRaisesRegex(ValueError, "Torch will not be changed.*compilation is disabled"):
            select_wheel(NATTEN, self.html(self.link(build="torch2100cu128")), self.runtime)

    def test_index_links_cannot_select_an_untrusted_origin_or_invalid_digest(self):
        valid = self.link(fragment="#sha256=" + "a" * 64)
        bad = (self.link().replace("https://", "http://"),
               self.link().replace("github.com", "github.com.evil.invalid"),
               self.link().replace("/SHI-Labs/", "/untrusted/"),
               self.link(fragment="#sha256=not-a-digest"))
        selected = select_wheel(NATTEN, self.html(*bad, valid), self.runtime)
        self.assertEqual(selected["url"], valid)
        self.assertEqual(selected["sha256"], "a" * 64)
        with self.assertRaises(ValueError):
            select_wheel(NATTEN, self.html(*bad), self.runtime)

    def test_upstream_declared_version_constraints_are_preserved(self):
        page = self.html(self.link("0.21.6"), self.link("0.21.7"))
        selected = select_wheel(NATTEN, page, self.runtime, [Requirement("natten<0.21.7")])
        self.assertEqual(selected["version"], "0.21.6+torch2110cu130")

    def test_prerelease_wheels_do_not_replace_a_compatible_stable_binary(self):
        selected = select_wheel(NATTEN, self.html(self.link("0.21.6"), self.link("0.22.0rc1")), self.runtime)
        self.assertEqual(selected["version"], "0.21.6+torch2110cu130")

    def test_catalog_checks_are_local_and_an_installed_broken_extension_is_not_ready(self):
        with patch("policy_extensions.urlopen") as network:
            self.assertIn("natten", extension_problems("flux3", {}))
            failure = subprocess.CompletedProcess([], 1, stdout="", stderr="ImportError: undefined symbol")
            with patch("policy_extensions.subprocess.run", return_value=failure):
                issues = extension_problems("flux3", {"natten": "0.21.6+torch2110cu130"})
            self.assertIn("undefined symbol", issues["natten"])
            self.assertFalse(extension_problems("act", {}))
            network.assert_not_called()

    def test_native_runtime_identity_does_not_query_a_gpu(self):
        torch = SimpleNamespace(__version__="2.11.0+cu130", version=SimpleNamespace(cuda="13.0"))
        with patch.dict(sys.modules, {"torch": torch}), patch("policy_extensions.sys.platform", "linux"), \
             patch("policy_extensions.platform.python_implementation", return_value="CPython"):
            result = wheel_runtime()
        self.assertEqual((result.torch, result.cuda), ("2110", "130"))
        self.assertFalse(extension_runtime_issue("act"))

    def test_unknown_nightly_cpu_or_inconsistent_build_is_not_silently_replaced(self):
        for version, cuda in (("2.12.0.dev20261119+cu130", "13.0"), ("2.11.0+cpu", None),
                              ("2.11.0+cu128", "13.0")):
            with self.subTest(version=version), \
                 patch.dict(sys.modules, {"torch": SimpleNamespace(__version__=version,
                                                                   version=SimpleNamespace(cuda=cuda))}), \
                 patch("policy_extensions.sys.platform", "linux"), \
                 patch("policy_extensions.platform.python_implementation", return_value="CPython"):
                with self.assertRaises(ValueError):
                    wheel_runtime()

    def test_official_index_is_bounded_and_redirect_origin_is_checked(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *exc): pass
            def geturl(self): return "https://untrusted.invalid/"
            def read(self, _): return b"unused"
        with patch("policy_extensions.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "unexpected origin"):
                resolve_wheel(NATTEN, self.runtime)

    def test_extension_installer_uses_only_the_exact_wheel_without_dependencies(self):
        selected = {"distribution": "natten", "version": "0.21.6+torch2110cu130",
                    "url": self.link(), "sha256": None}
        commands = []

        def install(command, **kwargs):
            commands.append(command)
            report = Path(command[command.index("--report") + 1])
            report.write_text(json.dumps({"install": [{"metadata": {"name": "natten"},
                "download_info": {"archive_info": {"hashes": {"sha256": "b" * 64}}}}]}))
            return subprocess.CompletedProcess(command, 0, stdout="installed\n")

        with tempfile.TemporaryDirectory() as target, \
             patch("policy_extensions.extension_problems", return_value={"natten": "missing"}), \
             patch("policy_extensions.wheel_runtime", return_value=self.runtime), \
             patch("policy_extensions.resolve_wheel", return_value=selected), \
             patch("policy_extensions.subprocess.run", side_effect=install), \
             patch("policy_extensions.import_issue", return_value=None) as check, \
             contextlib.redirect_stdout(io.StringIO()):
            results = install_extensions("flux3", {}, target=target)
        self.assertEqual(results[0]["sha256"], "b" * 64)
        command = commands[0]
        self.assertIn("--no-deps", command)
        self.assertIn("--only-binary=:all:", command)
        self.assertIn("--target", command)
        self.assertEqual(command[-1], self.link())
        self.assertFalse(any(item.startswith("torch==") for item in command))
        check.assert_called_once_with(NATTEN, target=target)

    def test_overlay_import_verification_uses_a_fresh_child_and_the_private_target(self):
        with patch("policy_extensions.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertIsNone(import_issue(NATTEN, target="/tmp/owned/natten"))
        command = run.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-B", "-c"])
        self.assertTrue(run.call_args.kwargs["env"]["PYTHONPATH"].startswith("/tmp/owned/natten"))
        self.assertNotIn("cuda()", command[-1])
        self.assertEqual(extension_specs("flux3"), ("natten>=0.21.6",))


if __name__ == "__main__":
    unittest.main()
