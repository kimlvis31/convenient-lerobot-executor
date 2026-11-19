"""AI-written tests using fake Conda states and dummy credentials only."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import runtime_activation as activation


SCOPE = "0123456789abcdef"


def apply_commands(commands, environment):
    """Decode the exact two-word export/unset statements without executing them."""
    result = dict(environment)
    tokens = shlex.split(commands)
    for index in range(0, len(tokens), 2):
        command, argument = tokens[index:index + 2]
        if command == "export":
            name, value = argument.split("=", 1)
            result[name] = value
        elif command == "unset":
            result.pop(argument, None)
        else:
            raise AssertionError("Unexpected shell command")
    return result


class RuntimeActivationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="convlex-activation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "source prefix"
        (self.base / "conda-meta").mkdir(parents=True)
        self.state = self.base / "conda-meta/state"
        self.prefix = self.root / "build/env"
        (self.prefix / "bin").mkdir(parents=True)

    def write_state(self, variables):
        self.state.write_text(json.dumps({"env_vars": variables}), encoding="utf-8")

    def test_inherits_restores_and_keeps_all_values_out_of_hook_files(self):
        secret = "dummy-token-for-unit-test-only"
        self.write_state({"HF_TOKEN": secret, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "2"})
        original = self.state.read_bytes()
        with patch.object(activation, "source_variables", side_effect=AssertionError("Setup must not read env_vars")):
            installed = activation.install_activation(self.root / "build", self.prefix, self.base)
        for path in (self.root / "build").rglob("*"):
            if path.is_file():
                self.assertNotIn(secret, path.read_text(encoding="utf-8"))
        self.assertEqual(installed["base_prefix"], str(self.base))
        before = {"HF_TOKEN": "dummy-parent", "CUDA_VISIBLE_DEVICES": "0"}
        live = apply_commands(activation.activation_commands(self.base, SCOPE, before), before)
        self.assertEqual(live["HF_TOKEN"], secret)
        self.assertEqual(live["HF_HUB_OFFLINE"], "1")
        self.assertEqual(live["CUDA_VISIBLE_DEVICES"], "2")
        restored = apply_commands(activation.deactivation_commands(SCOPE, live), live)
        self.assertEqual(restored, before)
        self.assertEqual(self.state.read_bytes(), original)

    def test_managed_paths_cache_and_compilers_are_never_overridden(self):
        protected = {name: "source-must-not-win" for name in (
            "PATH", "PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "LD_LIBRARY_PATH", "XDG_CACHE_HOME",
            "MPLCONFIGDIR", "HF_HOME", "HF_HUB_CACHE", "HF_TOKEN_PATH", "TORCH_HOME", "HF_MODULES_CACHE",
            "HF_LEROBOT_HOME", "CUDA_HOME", "CXX", "CONVLEX_SIMULATOR_ROOT", "ROBOTWIN_ROOT",
        )}
        self.write_state({**protected, "HF_TOKEN": "dummy-inherited"})
        before = {name: "managed-value" for name in protected}
        live = apply_commands(activation.activation_commands(self.base, SCOPE, before), before)
        self.assertTrue(all(live[name] == "managed-value" for name in protected))
        self.assertEqual(live["HF_TOKEN"], "dummy-inherited")

    def test_conda_unset_sentinel_and_duplicate_activation_match_conda_semantics(self):
        self.write_state({"SKIPPED": activation.UNSET_SENTINEL, "NEW_VALUE": "dummy"})
        before = {"SKIPPED": "leave-parent-value"}
        live = apply_commands(activation.activation_commands(self.base, SCOPE, before), before)
        self.assertEqual(live["SKIPPED"], before["SKIPPED"])
        self.assertEqual(activation.activation_commands(self.base, SCOPE, live), "")
        self.assertEqual(apply_commands(activation.deactivation_commands(SCOPE, live), live), before)

    def test_source_library_paths_append_after_managed_libraries_and_restore_exact_prior(self):
        before = {"LD_LIBRARY_PATH": os.pathsep.join(("/target/current", "/shared/lib"))}
        self.write_state({"LD_LIBRARY_PATH": os.pathsep.join(("/source/extensions", "/shared/lib")),
                          "LD_PRELOAD": "must-not-inherit"})
        code = activation.activation_commands(self.base, SCOPE, before, prefix=self.prefix)
        live = apply_commands(code, before)
        self.assertEqual(live["LD_LIBRARY_PATH"].split(os.pathsep),
                         [str(self.prefix / "lib"), "/target/current", "/shared/lib", "/source/extensions"])
        self.assertNotIn("LD_PRELOAD", live)
        self.assertEqual(apply_commands(activation.deactivation_commands(SCOPE, live), live), before)
        empty = apply_commands(activation.activation_commands(self.base, SCOPE, {}, prefix=self.prefix), {})
        restored = apply_commands(activation.deactivation_commands(SCOPE, empty), empty)
        self.assertNotIn("LD_LIBRARY_PATH", restored)

    def test_missing_env_vars_need_no_marker_and_bad_json_is_generic(self):
        self.assertEqual(activation.activation_commands(self.base, SCOPE, {}), "")
        self.assertEqual(activation.deactivation_commands(SCOPE, {}), "")
        self.state.write_text('{"env_vars":{"HF_TOKEN":"dummy-secret-not-in-error"', encoding="utf-8")
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = activation.main(["activate", "--scope", SCOPE, "--base-prefix", str(self.base)])
        self.assertEqual(code, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Cannot read the source Conda", errors.getvalue())
        self.assertNotIn("dummy-secret", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())

    def test_invalid_names_and_corrupt_backups_never_emit_partial_commands(self):
        self.write_state({"VALID": "dummy", "BAD;touch nope": "dummy-secret"})
        with self.assertRaises(activation.ActivationError):
            activation.activation_commands(self.base, SCOPE, {})
        with self.assertRaisesRegex(activation.ActivationError, "Cannot restore"):
            activation.deactivation_commands(SCOPE, {activation.marker(SCOPE): "bad dummy-secret json"})

    def test_unsafe_scope_is_rejected(self):
        with self.assertRaises(activation.ActivationError):
            activation.activation_commands(self.base, "$(touch wrong)", {})

    @unittest.skipUnless(os.name == "posix", "POSIX activation hooks")
    def test_shell_quotes_injection_and_restores_values_in_a_real_shell(self):
        dummy = "dummy'quoted\n$(touch " + str(self.root / "injected") + ");`false`"
        self.write_state({"HF_TOKEN": dummy, "NEW_VALUE": "dummy-new", "LD_LIBRARY_PATH": "/source/extensions"})
        (self.prefix / "bin/python").symlink_to(sys.executable)
        activation.install_activation(self.root / "build", self.prefix, self.base)
        verify = ("import os; assert os.environ['HF_TOKEN']==" + repr(dummy) + "; assert os.environ['NEW_VALUE']=='dummy-new'; "
                  "assert os.environ['LD_LIBRARY_PATH'].split(':')==" + repr([str(self.prefix / "lib"), "/target/current", "/source/extensions"]))
        restore = "import os; assert os.environ['HF_TOKEN']=='dummy-parent'; assert os.environ['LD_LIBRARY_PATH']=='/target/current'; assert 'NEW_VALUE' not in os.environ; assert not any(k.startswith('__CONVLEX_SOURCE_VARS_') for k in os.environ)"
        script = "\n".join(("set -e", ". " + shlex.quote(str(self.prefix / "etc/conda/activate.d/convlex_source_vars.sh")),
                            shlex.join([sys.executable, "-B", "-c", verify]),
                            ". " + shlex.quote(str(self.prefix / "etc/conda/deactivate.d/convlex_source_vars.sh")),
                            shlex.join([sys.executable, "-B", "-c", restore])))
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                                env={**os.environ, "HF_TOKEN": "dummy-parent", "LD_LIBRARY_PATH": "/target/current"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "injected").exists())
        self.assertEqual(result.stdout, "")
        for path in (self.root / "build").rglob("*"):
            if path.is_file() and not path.is_symlink():
                self.assertNotIn(dummy, path.read_text(encoding="utf-8"))

    @unittest.skipUnless(os.name == "posix", "Symlink ownership check")
    def test_hook_directory_symlink_cannot_modify_source_environment(self):
        (self.prefix / "etc/conda").mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        (self.prefix / "etc/conda/activate.d").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside the managed"):
            activation.install_activation(self.root / "build", self.prefix, self.base)
        self.assertFalse(list(outside.iterdir()))

    def test_common_runtime_uses_reported_base_prefix_without_reading_source_state(self):
        import runtime_setup

        self.write_state({"HF_TOKEN": "dummy-token-never-read-during-setup"})
        build = SimpleNamespace(root=self.root / "build", prefix=self.prefix,
                                options={"base_environment": "source", "base_prefix": str(self.base)}, conda="conda")
        with patch.object(runtime_setup.subprocess, "run", side_effect=AssertionError("No prefix probe needed")), \
             patch.object(activation, "source_variables", side_effect=AssertionError("No state read during setup")):
            result = runtime_setup.configure_runtime_activation(build)
        self.assertEqual(result["base_prefix"], str(self.base))
        self.assertEqual(build.options["base_prefix"], str(self.base))

    def test_common_runtime_fallback_queries_only_prefix_and_does_not_expose_probe_output(self):
        import runtime_setup

        build = SimpleNamespace(root=self.root / "build", prefix=self.prefix,
                                options={"base_environment": "source"}, conda="conda")
        report = SimpleNamespace(returncode=0, stdout="CONVLEX_BASE_PREFIX " + str(self.base) + "\n")
        with patch.object(runtime_setup.subprocess, "run", return_value=report) as probe:
            runtime_setup.configure_runtime_activation(build)
        self.assertIn("sys.prefix", probe.call_args.args[0][-1])
        self.assertNotIn("env_vars", probe.call_args.args[0][-1])
        self.assertNotIn("HF_TOKEN", probe.call_args.args[0][-1])


if __name__ == "__main__":
    unittest.main()
