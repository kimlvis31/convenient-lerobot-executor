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
    def test_clone_pointer_is_not_in_condas_package_record_json_glob(self):
        target = self.clone_prefix("target")
        activation.install_clone_inheritance(self.base, target)
        self.assertEqual(list((target / "conda-meta").glob("*.json")), [])
        self.assertTrue((target / "conda-meta" / activation.CLONE_MARKER).is_file())

    def test_legacy_clone_reading_is_readonly_and_confirmed_migration_preserves_inheritance(self):
        target = self.clone_prefix("target")
        self.write_state({"HF_TOKEN": "dummy-source", "CUSTOM_SOURCE": "yes"})
        original_state = self.state.read_bytes()
        legacy = target / "conda-meta" / activation.LEGACY_CLONE_MARKER
        legacy.write_text(json.dumps({"schema": 1, "source_prefix": str(self.base)}))
        before = legacy.read_bytes()
        self.assertEqual(activation.clone_environment(target, {})["CUSTOM_SOURCE"], "yes")
        self.assertEqual(legacy.read_bytes(), before)
        self.assertFalse((target / "conda-meta" / activation.CLONE_MARKER).exists())
        self.assertTrue(activation.migrate_clone_inheritance(target))
        self.assertFalse(legacy.exists())
        self.assertEqual(list((target / "conda-meta").glob("*.json")), [])
        self.assertEqual(activation.clone_environment(target, {})["HF_TOKEN"], "dummy-source")
        self.assertEqual(self.state.read_bytes(), original_state)
        self.assertFalse(activation.migrate_clone_inheritance(target))

    def test_clone_migration_does_not_touch_unmarked_environments(self):
        self.write_state({"CUSTOM_SOURCE": "unchanged"})
        before = self.state.read_bytes()
        self.assertFalse(activation.migrate_clone_inheritance(self.base))
        self.assertEqual(self.state.read_bytes(), before)
        self.assertEqual([p.name for p in (self.base / "conda-meta").iterdir()], ["state"])

    def test_conflicting_legacy_clone_metadata_is_not_silently_replaced(self):
        target, other = self.clone_prefix("target"), self.clone_prefix("other")
        activation.install_clone_inheritance(self.base, target)
        current = target / "conda-meta" / activation.CLONE_MARKER
        before = current.read_bytes()
        legacy = target / "conda-meta" / activation.LEGACY_CLONE_MARKER
        legacy.write_text(json.dumps({"schema": 1, "source_prefix": str(other)}))
        with self.assertRaises(activation.ActivationError):
            activation.migrate_clone_inheritance(target)
        self.assertEqual(current.read_bytes(), before)
        self.assertTrue(legacy.exists())

    def setUp(self):
        self.temp = TemporaryDirectory(prefix="convlex-activation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
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

    def test_clone_lineage_reads_paths_only_for_environment_selection(self):
        target = self.clone_prefix("lineage target")
        activation.install_clone_inheritance(self.base, target)
        with patch.object(activation, "_environment_variables", side_effect=AssertionError("Do not read credentials")):
            lineage = activation.clone_lineage(target)
        self.assertEqual(lineage, {"prefix": str(target.resolve()), "clone_sources": [str(self.base.resolve())]})
        self.assertEqual(activation.clone_lineage(self.base)["clone_sources"], [])

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

    def clone_prefix(self, name):
        prefix = self.root / name
        (prefix / "conda-meta").mkdir(parents=True, exist_ok=True)
        return prefix

    def run_clone_child(self, prefix, command, environment=None):
        bootstrap = ("import sys; sys.path.insert(0, " + repr(str(Path(activation.__file__).parent)) + "); "
                     "import runtime_activation; sys.prefix=" + repr(str(prefix)) + "; "
                     "sys.exit(runtime_activation.main(['run', '--', *sys.argv[1:]]))")
        return subprocess.run([sys.executable, "-B", "-c", bootstrap, *command],
                              env=dict(os.environ if environment is None else environment),
                              capture_output=True, text=True, timeout=20)

    def test_named_clone_records_only_a_pointer_without_reading_credentials(self):
        target = self.clone_prefix("named target")
        secret = "dummy-token-not-for-marker"
        self.write_state({"HF_TOKEN": secret, "HF_HUB_CACHE": "dummy-cache-path"})
        original = self.state.read_bytes()
        with patch.object(activation, "_environment_variables", side_effect=AssertionError("No state values at setup")):
            record = activation.install_clone_inheritance(self.base, target)
        self.assertEqual(record, {"schema": 1, "source_prefix": str(self.base)})
        files = list(target.rglob("*"))
        self.assertEqual([file.name for file in files if file.is_file()], [activation.CLONE_MARKER])
        self.assertNotIn(secret, (target / "conda-meta" / activation.CLONE_MARKER).read_text())
        self.assertEqual(self.state.read_bytes(), original)

    def test_named_clone_inherits_caches_and_credentials_but_keeps_target_controls(self):
        target = self.clone_prefix("named target")
        self.write_state({"HF_TOKEN": "dummy-source", "HF_HUB_CACHE": "source-hub", "TORCH_HOME": "source-torch",
                          "CUSTOM_SOURCE": "yes", "PATH": "unsafe-source-path", "PYTHONPATH": "unsafe-python",
                          "CUDA_HOME": "unsafe-toolkit", "CONDA_PREFIX": "unsafe-conda"})
        (target / "conda-meta/state").write_text(json.dumps({"env_vars": {"HF_TOKEN": "dummy-target",
                                                                         "HF_HOME": "target-home"}}))
        activation.install_clone_inheritance(self.base, target)
        before = {"PATH": "target-path", "PYTHONPATH": "target-python", "CUDA_HOME": "target-toolkit",
                  "CONDA_PREFIX": str(target), "HF_HOME": "parent-home", "HF_HUB_CACHE": "parent-cache"}
        actual = activation.clone_environment(target, before)
        self.assertEqual(actual["HF_TOKEN"], "dummy-target")
        self.assertEqual(actual["HF_HUB_CACHE"], "source-hub")
        self.assertEqual(actual["TORCH_HOME"], "source-torch")
        self.assertEqual(actual["HF_HOME"], "target-home")
        self.assertEqual(actual["CUSTOM_SOURCE"], "yes")
        for name in ("PATH", "PYTHONPATH", "CUDA_HOME", "CONDA_PREFIX"):
            self.assertEqual(actual[name], before[name])
        self.assertEqual(before["HF_HUB_CACHE"], "parent-cache")

    def test_clone_chains_apply_nearer_overrides_and_conda_unsets(self):
        middle = self.clone_prefix("middle")
        target = self.clone_prefix("target")
        self.write_state({"HF_TOKEN": "dummy-base", "CUSTOM_BASE": "yes", "DROP_ME": "base"})
        (middle / "conda-meta/state").write_text(json.dumps({"env_vars": {"HF_TOKEN": "dummy-middle"}}))
        (target / "conda-meta/state").write_text(json.dumps({"env_vars": {"DROP_ME": activation.UNSET_SENTINEL}}))
        activation.install_clone_inheritance(self.base, middle)
        activation.install_clone_inheritance(middle, target)
        actual = activation.clone_environment(target, {"DROP_ME": "parent"})
        self.assertEqual(actual["HF_TOKEN"], "dummy-middle")
        self.assertEqual(actual["CUSTOM_BASE"], "yes")
        self.assertNotIn("DROP_ME", actual)
        with self.assertRaisesRegex(activation.ActivationError, "cyclic"):
            activation.install_clone_inheritance(target, self.base)
        (middle / "conda-meta" / activation.CLONE_MARKER).write_text(
            json.dumps({"schema": 1, "source_prefix": str(target)}))
        with self.assertRaisesRegex(activation.ActivationError, "cyclic"):
            activation.clone_environment(target, {})

    def test_clone_preserves_managed_dataset_and_system_cache_paths(self):
        target = self.clone_prefix("target")
        self.write_state({"HF_LEROBOT_HOME": "source-datasets-outside-project",
                          "XDG_CACHE_HOME": "source-system-cache", "HF_HUB_CACHE": "source-model-cache"})
        (target / "conda-meta/state").write_text(json.dumps({"env_vars": {
            "HF_LEROBOT_HOME": "target-declared-datasets-outside-project"}}))
        activation.install_clone_inheritance(self.base, target)
        before = {"HF_LEROBOT_HOME": "project/datasets/.hub", "XDG_CACHE_HOME": "managed-system-cache"}
        actual = activation.clone_environment(target, before)
        self.assertEqual(actual["HF_LEROBOT_HOME"], before["HF_LEROBOT_HOME"])
        self.assertEqual(actual["XDG_CACHE_HOME"], before["XDG_CACHE_HOME"])
        self.assertEqual(actual["HF_HUB_CACHE"], "source-model-cache")
        self.assertNotIn("HF_LEROBOT_HOME", activation.clone_environment(target, {}))

    def test_clone_chain_depth_is_bounded_before_new_marker_is_written(self):
        middle, target = self.clone_prefix("middle"), self.clone_prefix("target")
        activation.install_clone_inheritance(self.base, middle)
        with patch.object(activation, "MAX_CLONE_DEPTH", 1):
            with self.assertRaisesRegex(activation.ActivationError, "too deep"):
                activation.install_clone_inheritance(middle, target)
        self.assertFalse((target / "conda-meta" / activation.CLONE_MARKER).exists())

    def test_unmarked_environment_passes_through_without_reading_state_or_writing_files(self):
        target = self.clone_prefix("unmarked")
        before = {"HF_TOKEN": "dummy-parent", "PATH": "parent-path"}
        with patch.object(activation, "_environment_variables", side_effect=AssertionError("No inheritance")):
            self.assertEqual(activation.clone_environment(target, before), before)
        self.assertEqual(list((target / "conda-meta").iterdir()), [])
        result = self.run_clone_child(target, [sys.executable, "-c", "import sys; sys.exit(7)"])
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    @unittest.skipUnless(os.name == "posix", "POSIX exec signal propagation")
    def test_unmarked_command_replaces_wrapper_and_preserves_termination_signal(self):
        import signal

        target = self.clone_prefix("unmarked")
        result = self.run_clone_child(target, [sys.executable, "-c",
                                              "import os,signal; os.kill(os.getpid(), signal.SIGTERM)"])
        self.assertEqual(result.returncode, -signal.SIGTERM)
        self.assertEqual(result.stderr, "")

    def test_conda_commands_use_the_same_portable_wrapper_and_keep_original_argument_tail(self):
        from runner import conda_command

        for name, selector in (("named", "-n"), (str(self.prefix), "--prefix")):
            arguments = ["lerobot-train", "--job_name=literal ; $(false)"]
            command = conda_command("conda", name, arguments)
            self.assertEqual(command[:5], ["conda", "run", "--no-capture-output", selector, name])
            self.assertEqual(command[5:10], ["python", "-B", str(Path(activation.__file__)), "run", "--"])
            self.assertEqual(command[10:], arguments)

    def test_real_clone_child_preserves_argv_and_literal_environment_values_without_shell(self):
        target = self.clone_prefix("quoted target")
        literal = "dummy'\"\n$(false); & %PATH% ! `false`"
        self.write_state({"HF_TOKEN": literal, "HF_HUB_CACHE": "source cache with spaces"})
        activation.install_clone_inheritance(self.base, target)
        argument = "literal' \" ; $(false) & %PATH%"
        child = ("import os,sys; assert os.environ['HF_TOKEN']==" + repr(literal) + "; "
                 "assert os.environ['HF_HUB_CACHE']=='source cache with spaces'; "
                 "assert sys.argv[1]==" + repr(argument) + "; print('child-ok')")
        result = self.run_clone_child(target, [sys.executable, "-B", "-c", child, argument])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "child-ok\n")
        self.assertEqual(result.stderr, "")
        self.assertNotIn(literal, (target / "conda-meta" / activation.CLONE_MARKER).read_text())

    def test_bad_clone_source_state_fails_without_leaking_its_contents(self):
        target = self.clone_prefix("target")
        activation.install_clone_inheritance(self.base, target)
        self.state.write_text('{"env_vars":{"HF_TOKEN":"dummy-secret-malformed"')
        result = self.run_clone_child(target, [sys.executable, "-c", "raise SystemExit(99)"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("Cannot read the source Conda", result.stderr)
        self.assertNotIn("dummy-secret", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_clone_prefix_probe_reports_only_prefixes_and_uses_no_wrapper(self):
        target = self.clone_prefix("target")
        replies = [SimpleNamespace(returncode=0, stdout="CONVLEX_CLONE_PREFIX " + str(prefix) + "\n")
                   for prefix in (self.base, target)]
        with patch.object(activation.subprocess, "run", side_effect=replies) as probe:
            activation.configure_clone_inheritance("conda", "source", "target", self.root)
        self.assertEqual(len(probe.call_args_list), 2)
        for call in probe.call_args_list:
            command = call.args[0]
            self.assertIn("sys.prefix", command[-1])
            self.assertNotIn("env_vars", command[-1])
            self.assertNotIn("runtime_activation.py", command)

    @unittest.skipUnless(os.name == "posix", "Symlink marker ownership")
    def test_clone_marker_cannot_redirect_writes_outside_new_target(self):
        target = self.clone_prefix("target")
        outside = self.root / "outside-marker"
        outside.write_text("keep")
        (target / "conda-meta" / activation.CLONE_MARKER).symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "inside the managed"):
            activation.install_clone_inheritance(self.base, target)
        self.assertEqual(outside.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
