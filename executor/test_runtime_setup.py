"""AI-written offline tests for runtime publication, recovery and ownership."""

from contextlib import ExitStack, redirect_stdout
from hashlib import sha256
from http.client import IncompleteRead
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import runtime_setup


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


class InterruptedResponse(Response):
    def read(self, size=-1):
        if self.tell():
            raise IncompleteRead(b"", 3)
        return super().read(size)


class RuntimeSetupTests(unittest.TestCase):
    @staticmethod
    def ranged_response(request, content):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        start, end = map(int, match.groups())
        packet = content[start:end + 1]
        return Response(packet, 206, {"Content-Range": f"bytes {start}-{end}/{len(content)}",
                                      "Content-Length": str(len(packet))})

    def test_segmented_download_has_a_fresh_retry_budget_for_each_packet(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "asset.bin"
            content = b"abcdefghijklmnopqrst"
            attempts, requests = {}, []
            checksum = runtime_setup.sha256

            def open_packet(request, **kwargs):
                byte_range = request.get_header("Range")
                requests.append(byte_range)
                attempts[byte_range] = attempts.get(byte_range, 0) + 1
                self.assertEqual(request.get_header("Connection"), "close")
                if attempts[byte_range] < 4:
                    raise OSError("temporary network error")
                return self.ranged_response(request, content)

            def full_checksum(path):
                self.assertEqual(Path(path).stat().st_size, len(content))
                return checksum(path)

            with patch.object(runtime_setup, "urlopen", side_effect=open_packet), \
                    patch.object(runtime_setup.time, "sleep"), \
                    patch.object(runtime_setup, "sha256", side_effect=full_checksum) as verify:
                runtime_setup.download("https://example.invalid/asset", target, size=len(content),
                                       digest=sha256(content).hexdigest(), range_size=4)
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(len(requests), 20)
            self.assertEqual(set(attempts.values()), {4})
            self.assertEqual(verify.call_count, 1)

    def test_segmented_truncated_packet_retries_without_duplicating_its_tail(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "asset.bin"
            partial = target.with_name("asset.bin.part")
            content = b"abcdefghijkl"
            partial.write_bytes(content[:3])
            requests = []

            def open_packet(request, **kwargs):
                requests.append(request.get_header("Range"))
                if len(requests) == 1:
                    return Response(b"de", 206, {"Content-Range": "bytes 3-6/12", "Content-Length": "4"})
                if len(requests) == 2:
                    self.assertEqual(partial.read_bytes(), b"abc")
                return self.ranged_response(request, content)

            with patch.object(runtime_setup, "urlopen", side_effect=open_packet), patch.object(runtime_setup.time, "sleep"):
                runtime_setup.download("https://example.invalid/asset", target, size=12, range_size=4)
            self.assertEqual(requests, ["bytes=3-6", "bytes=3-6", "bytes=7-10", "bytes=11-11"])
            self.assertEqual(target.read_bytes(), content)

    def test_segmented_download_rejects_inconsistent_range_before_appending(self):
        for header in ("bytes 3-6/8", "bytes 2-5/9", "bytes 2-4/8", "bytes 2-5/*"):
            with self.subTest(header=header), TemporaryDirectory() as folder:
                target = Path(folder) / "asset.bin"
                partial = target.with_name("asset.bin.part")
                partial.write_bytes(b"ab")
                response = Response(b"cdef", 206, {"Content-Range": header, "Content-Length": "4"})
                with patch.object(runtime_setup, "urlopen", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, "inconsistent Content-Range"):
                        runtime_setup.download("https://example.invalid/asset", target, size=8, range_size=4)
                self.assertEqual(partial.read_bytes(), b"ab")
                self.assertFalse(target.exists())

    def test_segmented_download_checks_content_length_and_ignored_ranges(self):
        cases = [(206, {"Content-Range": "bytes 2-5/8", "Content-Length": "3"}, "Content-Length"),
                 (206, {"Content-Range": "bytes 2-5/8"}, "Content-Length"),
                 (200, {"Content-Length": "8"}, "ignored the requested byte range")]
        for status, headers, reason in cases:
            with self.subTest(status=status, headers=headers), TemporaryDirectory() as folder:
                target = Path(folder) / "asset.bin"
                partial = target.with_name("asset.bin.part")
                partial.write_bytes(b"ab")
                with patch.object(runtime_setup, "urlopen", return_value=Response(b"cdef", status, headers)):
                    with self.assertRaisesRegex(RuntimeError, reason):
                        runtime_setup.download("https://example.invalid/asset", target, size=8, range_size=4)
                self.assertEqual(partial.read_bytes(), b"ab")
                self.assertFalse(target.exists())

    def test_segmented_download_without_a_digest_resumes_cached_partial(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "asset.bin"
            target.with_name("asset.bin.part").write_bytes(b"abcd")
            content = b"abcdefghij"
            ranges = []

            def open_packet(request, **kwargs):
                ranges.append(request.get_header("Range"))
                return self.ranged_response(request, content)

            with patch.object(runtime_setup, "urlopen", side_effect=open_packet):
                runtime_setup.download("https://example.invalid/asset", target, size=10, range_size=3)
            self.assertEqual(ranges, ["bytes=4-6", "bytes=7-9"])
            self.assertEqual(target.read_bytes(), content)
            other = Path(folder) / "complete.bin"
            other.with_name("complete.bin.part").write_bytes(content)
            with patch.object(runtime_setup, "urlopen", side_effect=AssertionError("unnecessary download")):
                runtime_setup.download("https://example.invalid/asset", other, size=10, range_size=3)
            self.assertEqual(other.read_bytes(), content)

    def test_segmented_download_exhaustion_keeps_only_committed_packets(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "asset.bin"
            requests = []

            def open_packet(request, **kwargs):
                requests.append(request.get_header("Range"))
                if len(requests) == 1:
                    return self.ranged_response(request, b"abcdefgh")
                raise OSError("network unavailable")

            with patch.object(runtime_setup, "urlopen", side_effect=open_packet), patch.object(runtime_setup.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "at byte 4"):
                    runtime_setup.download("https://example.invalid/asset", target, size=8, range_size=4)
            self.assertEqual(len(requests), 5)
            self.assertEqual(target.with_name("asset.bin.part").read_bytes(), b"abcd")
            self.assertFalse(target.exists())

    def test_segmented_disk_guard_reserves_space_for_the_whole_remaining_file(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "asset.bin"
            target.with_name("asset.bin.part").write_bytes(b"a" * 20)
            with patch.object(runtime_setup.shutil, "disk_usage", return_value=SimpleNamespace(free=1024**3 + 10)), \
                    patch.object(runtime_setup, "urlopen", side_effect=AssertionError("must not start downloading")):
                with self.assertRaisesRegex(RuntimeError, "Not enough disk space"):
                    runtime_setup.download("https://example.invalid/asset", target, size=100, range_size=4)

    def test_segmented_download_requires_known_size_and_forwards_build_option(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "known complete file size"):
                runtime_setup.download("https://example.invalid/asset", root / "asset.bin", range_size=4)
            build = runtime_setup.RuntimeBuild(root, root / "env", root / "env/bin/python", "conda", root / "downloads")
            with patch.object(runtime_setup, "download", return_value=root / "asset.bin") as transfer:
                build.download("https://example.invalid/asset", "asset.bin", size=10, range_size=4)
            self.assertEqual(transfer.call_args.kwargs["range_size"], 4)

    def test_package_mutations_reject_an_external_prefix_before_running_anything(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / "owned"
            external = Path(folder) / "preserved-original-environment"
            build = runtime_setup.RuntimeBuild(root, external, external / "bin/python", "conda", root / "downloads")
            with patch.object(build, "run", side_effect=AssertionError("Original environment was changed")), \
                 patch.object(build, "execute", side_effect=AssertionError("Original environment was changed")):
                with self.assertRaisesRegex(RuntimeError, "outside the managed runtime"):
                    build.pip_install("numpy")
                with self.assertRaisesRegex(RuntimeError, "outside the managed runtime"):
                    build.conda_install("python")

    def fixture(self, folder, install=None, probe=None):
        cache = (Path(folder) / "managed").resolve()
        recipe = {"revision": "fixture-1", "mode": "split", "python": "3.12",
                  "constraints": (), "description": "offline fixture",
                  "install": install or Mock(), "probe": probe or Mock()}

        def execute(build, command, **kwargs):
            if "create" in command:
                build.python.parent.mkdir(parents=True)
                build.python.write_text("offline Python placeholder", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "")

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(runtime_setup, "supported", return_value=True))
        stack.enter_context(patch.object(runtime_setup, "recipes", return_value={"fixture": recipe}))
        stack.enter_context(patch.object(runtime_setup, "cache_directory", return_value=cache))
        stack.enter_context(patch.object(runtime_setup.RuntimeBuild, "execute", execute))
        stack.enter_context(patch.object(runtime_setup.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12)))
        stack.enter_context(patch.dict(sys.modules, {"fcntl": SimpleNamespace(LOCK_EX=2, flock=lambda *_: None)}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return cache, recipe

    def test_ready_is_published_only_once_and_reuse_does_not_install(self):
        with TemporaryDirectory() as folder:
            cache, recipe = self.fixture(folder)
            first = runtime_setup.prepare_runtime("fixture", "conda")
            second = runtime_setup.prepare_runtime("fixture", "conda")
            self.assertEqual(first, second)
            self.assertEqual(recipe["install"].call_count, 1)
            self.assertEqual(recipe["probe"].call_count, 2)
            self.assertTrue(Path(first["python"]).is_file())
            self.assertEqual(first["env"]["CONDA_PREFIX"], first["prefix"])
            self.assertEqual(first["env"]["CONDA_DEFAULT_ENV"], first["prefix"])
            self.assertTrue(first["env"]["PATH"].startswith(str(Path(first["prefix"]) / "bin") + os.pathsep))

    def test_interrupt_immediately_after_publication_does_not_delete_verified_root(self):
        with TemporaryDirectory() as folder:
            _, recipe = self.fixture(folder)
            atomic = runtime_setup._atomic_json
            published = []

            def publish_then_interrupt(path, value):
                atomic(path, value)
                if path.name == "ready.json":
                    published.append(value)
                    raise KeyboardInterrupt()

            with patch.object(runtime_setup, "_atomic_json", side_effect=publish_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runtime_setup.prepare_runtime("fixture", "conda")
            self.assertTrue(Path(published[0]["python"]).is_file())
            self.assertEqual(runtime_setup.prepare_runtime("fixture", "conda"), published[0])
            self.assertEqual(recipe["install"].call_count, 1)

    def test_cleanup_failure_keeps_the_published_runtime_reusable(self):
        with TemporaryDirectory() as folder:
            def install(build):
                build.downloads.mkdir(parents=True, exist_ok=True)
                (build.downloads / "completed.zip").write_text("owned download")

            self.fixture(folder, install=Mock(side_effect=install))
            with patch.object(runtime_setup, "discard", side_effect=PermissionError("busy download")):
                ready = runtime_setup.prepare_runtime("fixture", "conda")
            self.assertTrue(Path(ready["python"]).is_file())
            self.assertEqual(runtime_setup.prepare_runtime("fixture", "conda"), ready)
            self.assertFalse((Path(ready["root"]).parent / "downloads").exists())

    def test_stale_external_pointer_cannot_delete_user_files(self):
        with TemporaryDirectory() as folder:
            cache, recipe = self.fixture(folder)
            outside = Path(folder) / "preserved-user-source"
            outside.mkdir()
            preserved = outside / "model.safetensors"
            preserved.write_text("preserved")
            identity = runtime_setup.identity("fixture", recipe, {})
            target = cache / ("fixture-" + identity)
            target.mkdir(parents=True)
            (target / "ready.json").write_text(json.dumps({"root": str(outside)}))
            ready = runtime_setup.prepare_runtime("fixture", "conda")
            self.assertEqual(preserved.read_text(), "preserved")
            self.assertTrue(Path(ready["root"]).is_relative_to(cache))

    def test_failed_rebuild_keeps_previous_ready_and_primary_cause(self):
        with TemporaryDirectory() as folder:
            _, recipe = self.fixture(folder)
            previous = runtime_setup.prepare_runtime("fixture", "conda")
            recipe["probe"].side_effect = RuntimeError("real camera failure")
            original_atomic = runtime_setup._atomic_json

            def write(path, data):
                if path.name == "last_failure.json":
                    raise PermissionError("cannot write failure marker")
                original_atomic(path, data)

            with patch.object(runtime_setup, "_atomic_json", side_effect=write):
                with self.assertRaisesRegex(RuntimeError, "real camera failure"):
                    runtime_setup.prepare_runtime("fixture", "conda")
            self.assertTrue(Path(previous["python"]).is_file())
            pointer = Path(previous["root"]).parent / "ready.json"
            self.assertEqual(json.loads(pointer.read_text())["root"], previous["root"])
            self.assertEqual(len(list(pointer.parent.glob("build-*"))), 1)

    def test_split_cache_ignores_policy_environment_but_tracks_wrapper(self):
        recipe = {"revision": "recipe-1", "constraints": (), "mode": "split"}
        one = {"source_sha256": "a", "base_environment": "model-one", "base_signature": "1"}
        two = {**one, "base_environment": "model-two", "base_signature": "2"}
        self.assertEqual(runtime_setup.identity("robotwin", recipe, one),
                         runtime_setup.identity("robotwin", recipe, two))
        self.assertNotEqual(runtime_setup.identity("robotwin", recipe, one),
                            runtime_setup.identity("robotwin", recipe, {**one, "source_sha256": "b"}))
        previous_cli_details = {"kind": "robotwin", "recipe": "recipe-1", "constraints": (),
                                "base": None, "base_signature": None, "source_sha256": "a", "assets": ()}
        previous_cli_key = sha256(json.dumps(previous_cli_details, sort_keys=True).encode()).hexdigest()[:16]
        self.assertEqual(runtime_setup.identity("robotwin", recipe, one), previous_cli_key)

    def test_power_loss_remnants_are_reclaimed_without_touching_an_active_build(self):
        with TemporaryDirectory() as folder:
            cache, recipe = self.fixture(folder)
            profile = cache / ("fixture-" + runtime_setup.identity("fixture", recipe, {}))
            abandoned = profile / ("build-" + "a" * 12)
            active = profile / ("build-" + "b" * 12)
            for root in (abandoned, active):
                root.mkdir(parents=True)
                (root / "partial-install").write_text("owned temporary files")
            with patch.object(runtime_setup, "runtime_in_use", side_effect=lambda root: root == active):
                ready = runtime_setup.prepare_runtime("fixture", "conda")
            self.assertFalse(abandoned.exists())
            self.assertTrue(active.is_dir())
            self.assertTrue(Path(ready["python"]).is_file())

    def test_combined_cache_tracks_base_and_canonical_asset_packs(self):
        recipe = {"revision": "recipe-1", "constraints": (), "mode": "combined"}
        one = {"base_environment": "env", "base_signature": "1", "obj_registries": '["lightwheel","aigen"]'}
        two = {**one, "obj_registries": ["aigen", "lightwheel", "aigen"]}
        self.assertEqual(runtime_setup.identity("robocasa", recipe, one),
                         runtime_setup.identity("robocasa", recipe, two))
        self.assertNotEqual(runtime_setup.identity("robocasa", recipe, one),
                            runtime_setup.identity("robocasa", recipe, {**one, "base_signature": "2"}))
        self.assertNotEqual(runtime_setup.identity("robocasa", recipe, one),
                            runtime_setup.identity("robocasa", recipe, {**one, "obj_registries": ["lightwheel"]}))

    def test_interrupted_http_read_resumes_before_digest_publication(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "asset.zip"
            responses = [InterruptedResponse(b"abc"), Response(b"def", 206, {"Content-Range": "bytes 3-5/6"})]
            requests = []

            def open_url(request, **kwargs):
                requests.append(dict(request.header_items()))
                return responses.pop(0)

            with patch.object(runtime_setup, "urlopen", side_effect=open_url), patch.object(runtime_setup.time, "sleep"):
                runtime_setup.download("https://example.invalid/asset", target,
                                       digest=sha256(b"abcdef").hexdigest(), size=6)
            self.assertEqual(target.read_bytes(), b"abcdef")
            self.assertEqual(requests[1]["Range"], "bytes=3-")
            self.assertFalse(target.with_name("asset.zip.part").exists())

    def test_zip_traversal_is_rejected_before_extracting_any_member(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "asset.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("good/file.txt", "safe")
                bundle.writestr("../user-file.txt", "escape")
            with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                runtime_setup.extract(archive, root / "destination")
            self.assertFalse((root / "user-file.txt").exists())
            self.assertFalse((root / "destination/good/file.txt").exists())

    def test_complete_verified_partial_is_published_without_redownloading(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "asset.zip"
            target.with_name("asset.zip.part").write_bytes(b"complete asset")
            with patch.object(runtime_setup, "urlopen", side_effect=AssertionError("Unnecessary download")):
                runtime_setup.download("https://example.invalid/asset", target,
                                       digest=sha256(b"complete asset").hexdigest(), size=14)
            self.assertEqual(target.read_bytes(), b"complete asset")

    def test_unknown_digest_still_requires_the_complete_http_body(self):
        with TemporaryDirectory() as folder:
            target = Path(folder) / "source.tar.gz"
            responses = [Response(b"abc", headers={"Content-Length": "6"}),
                         Response(b"def", 206, {"Content-Length": "3", "Content-Range": "bytes 3-5/6"})]
            with patch.object(runtime_setup, "urlopen", side_effect=responses), patch.object(runtime_setup.time, "sleep"):
                runtime_setup.download("https://example.invalid/source", target)
            self.assertEqual(target.read_bytes(), b"abcdef")

    def test_cancel_kills_grandchildren_even_after_conda_exits(self):
        process = Mock(pid=4321)
        process.wait.return_value = 0
        fake_os = SimpleNamespace(name="posix", killpg=Mock())
        with patch.object(runtime_setup, "os", fake_os), \
                patch.object(runtime_setup, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9)):
            runtime_setup.stop_process_tree(process)
        self.assertEqual(fake_os.killpg.call_args_list[0].args, (4321, 15))
        self.assertEqual(fake_os.killpg.call_args_list[1].args, (4321, 9))

    def test_replaced_runtime_is_preserved_while_referenced_by_a_worker(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            process = root / "proc/123"
            process.mkdir(parents=True)
            runtime = root / "build-old"
            (process / "cmdline").write_bytes(b"python\0worker.py\0")
            (process / "environ").write_bytes(b"CONDA_PREFIX=" + os.fsencode(str(runtime)) + b"/env\0")
            fake_os = SimpleNamespace(name="posix", fsencode=os.fsencode,
                                      getuid=lambda: process.stat().st_uid)
            with patch.object(runtime_setup, "os", fake_os):
                self.assertTrue(runtime_setup.runtime_in_use(runtime, root / "proc"))
                (process / "environ").write_bytes(b"CONDA_PREFIX=/unrelated\0")
                self.assertFalse(runtime_setup.runtime_in_use(runtime, root / "proc"))


if __name__ == "__main__":
    unittest.main()
