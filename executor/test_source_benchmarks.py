"""Small offline checks for managed source recipe ownership and asset lifecycle."""

import contextlib
from importlib.util import find_spec
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from source_benchmarks import (CASA_PACKS, DRIVE_RANGE_BYTES, LIBERO_ASSETS, PROBE, RECIPES,
                               _adopt_drive_partial, _already_have, _asset_archive,
                               _patch_once, _prepend_paths, drive_download_url,
                               install_libero_plus, install_robocasa, install_vlabench,
                               robocasa_asset_packs)


class FakeBuild:
    def __init__(self, root, options=None):
        self.root = Path(root)
        self.prefix = self.root / "conda-owned"
        self.prefix.mkdir()
        self.python = self.prefix / "bin/python"
        self.conda = "conda"
        self.downloads = self.root / "downloads"
        self.downloads.mkdir()
        self.options = options or {}
        self.env = {"PYTHONPATH": "/existing/native/source"}
        self.actions = []

    def source(self, name, repository, revision):
        self.actions.append(("source", name, repository, revision))
        root = self.root / "sources" / name
        root.mkdir(parents=True, exist_ok=True)
        if name == "libero-plus":
            for rel in ("libero/libero/benchmark/task_classification.json",
                        "libero/libero/bddl_files/libero_spatial/task.bddl",
                        "libero/libero/init_files/libero_newobj/libero_spatial/task.pruned_init"):
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("fixture")
        elif name == "robocasa":
            package = root / "robocasa/models/assets/box_links"
            package.mkdir(parents=True)
            links = {key: "https://example.invalid/s/" + key for key, *_ in CASA_PACKS.values()}
            (package / "box_links_assets.json").write_text(json.dumps(links))
            (root / "setup.py").write_text('install_requires=["lerobot==0.3.3"]')
        elif name == "vlabench":
            target = root / "VLABench/configs/constant.py"
            target.parent.mkdir(parents=True)
            target.write_text("def get_object_list(xml_dir):\n    subdirs = os.listdir(xml_dir)\n")
        return root

    @staticmethod
    def archive(target, members):
        with zipfile.ZipFile(target, "w") as package:
            for member in members:
                package.writestr(member, "fixture")
        return target

    def download(self, url, filename, **kwargs):
        self.actions.append(("download", url, filename, kwargs))
        target = self.downloads / filename
        if filename == "libero-plus-assets.zip":
            return self.archive(target, ["author/deep/assets/scenes/libero_tabletop_base_style.xml"])
        if filename.startswith("vlabench-"):
            return self.vla_archive(target)
        name = filename.removeprefix("robocasa-").removesuffix(".zip")
        _, suffix, patterns = CASA_PACKS[name]
        folder = Path(suffix).name
        member = f"{folder}/category/object/model.xml" if "objs" in name or "fixtures" in name else f"{folder}/wall/tex.png"
        return self.archive(target, [member])

    def extract(self, archive, destination):
        self.actions.append(("extract", str(archive), str(destination)))
        with zipfile.ZipFile(archive) as package:
            package.extractall(destination)

    def pip_install(self, *requirements, **kwargs):
        self.actions.append(("pip", requirements, kwargs))

    def conda_install(self, *packages, **kwargs):
        self.actions.append(("conda", packages, kwargs))

    def run(self, args, **kwargs):
        self.actions.append(("run", args, kwargs))
        if "gdown" in args:
            target = Path(args[args.index("--output") + 1])
            self.vla_archive(target)

    def vla_archive(self, target):
        if target.name == "vlabench-obj.zip":
            members = ["obj/meshes/poker/poker_asset.xml", "obj/meshes/containers/basket/one/model.xml",
                       "obj/meshes/tablewares/plates/one/model.xml", "obj/meshes/fruit/one/model.xml"]
        else:
            members = ["scenes/default/empty.xml"]
        return self.archive(target, members)


class SourceRecipeTests(unittest.TestCase):
    def setUp(self):
        resolver = patch("source_benchmarks.drive_download_url",
                         side_effect=lambda file_id: "https://drive.usercontent.google.com/download?id=" + file_id)
        self.drive_resolver = resolver.start()
        self.addCleanup(resolver.stop)

    def test_explicit_renderer_selection_survives_source_preparation(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            with patch.dict(os.environ, {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}):
                _prepend_paths(build, Path(root) / "source")
            self.assertEqual("osmesa", build.env["MUJOCO_GL"])
            self.assertEqual("osmesa", build.env["PYOPENGL_PLATFORM"])

    def test_libero_fork_is_private_and_replaces_stock_distribution_only_inside_clone(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            install_libero_plus(build)
            config = json.loads((Path(build.env["LIBERO_CONFIG_PATH"]) / "config.yaml").read_text())
            self.assertTrue(Path(config["assets"]).is_relative_to(root))
            self.assertTrue((Path(config["assets"]) / "scenes/libero_tabletop_base_style.xml").is_file())
            self.assertIn("/existing/native/source", build.env["PYTHONPATH"])
            self.assertTrue(any(a[0] == "download" and LIBERO_ASSETS in a[1] for a in build.actions))
            self.assertIn(("run", ["python", "-m", "pip", "uninstall", "--yes", "hf-libero"], {}), build.actions)
            self.assertEqual(1, len(list(build.downloads.glob("*.zip"))))
            self.assertFalse(any(a[0] == "run" and "apt" in a[1] for a in build.actions))

    def test_uninstall_is_rejected_for_an_original_environment_outside_the_recipe(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as original:
            build = FakeBuild(root)
            build.prefix = Path(original)
            with self.assertRaisesRegex(RuntimeError, "outside the managed"):
                install_libero_plus(build)
            self.assertEqual([], build.actions)

    def test_libero_source_layout_is_checked_before_package_or_asset_downloads(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            with patch.object(build, "source", return_value=Path(root) / "incomplete-source"):
                with self.assertRaisesRegex(RuntimeError, "Incomplete simulator assets"):
                    install_libero_plus(build)
            self.assertEqual([], build.actions)

    def test_dependency_failure_is_reported_before_any_large_asset_download(self):
        for installer in (install_libero_plus, install_robocasa, install_vlabench):
            with self.subTest(recipe=installer.__name__), tempfile.TemporaryDirectory() as root:
                build = FakeBuild(root)
                original_run = build.run
                def run(args, **kwargs):
                    if list(args[-3:]) == ["-m", "pip", "check"]:
                        raise RuntimeError("incompatible installed packages")
                    return original_run(args, **kwargs)
                with patch.object(build, "run", side_effect=run), self.assertRaisesRegex(RuntimeError, "incompatible"):
                    installer(build)
                self.assertFalse(any(action[0] == "download" for action in build.actions))

    def test_robocasa_downloads_requested_registries_and_keeps_upstream_metadata_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root, {"obj_registries": ["lightwheel", "objaverse"]})
            install_robocasa(build)
            names = [a[2] for a in build.actions if a[0] == "download"]
            self.assertEqual(["robocasa-tex.zip", "robocasa-tex_generative.zip", "robocasa-fixtures_lw.zip",
                              "robocasa-objs_lw.zip", "robocasa-objs_objaverse.zip"], names)
            self.assertEqual('install_requires=["lerobot==0.3.3"]',
                             (Path(root) / "sources/robocasa/setup.py").read_text())
            self.assertFalse(any(a[0] == "pip" and a[2].get("editable") for a in build.actions))
            self.assertEqual(5, len(list(build.downloads.glob("*.zip"))))
            self.assertIn("robosuite", json.loads(build.env["CONVLEX_SOURCE_BENCHMARK"]))

    def test_bad_registry_fails_before_any_source_fetch(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root, {"obj_registries": ["invented"]})
            with self.assertRaisesRegex(ValueError, "Unknown RoboCasa"):
                install_robocasa(build)
            self.assertEqual([], build.actions)

    def test_robocasa_default_is_minimal_but_never_drops_explicit_choices(self):
        self.assertEqual(["tex", "tex_generative", "fixtures_lw", "objs_lw"], robocasa_asset_packs({}))
        self.assertEqual(["tex", "tex_generative", "fixtures_lw", "objs_objaverse", "objs_aigen"],
                         robocasa_asset_packs({"obj_registries": '["objaverse","aigen"]'}))

    def test_robocasa_registry_parser_accepts_the_valid_lerobot_cli_list_encodings(self):
        expected = ["tex", "tex_generative", "fixtures_lw", "objs_lw", "objs_objaverse"]
        for value in ('["lightwheel","objaverse"]', "['lightwheel', 'objaverse']", "[lightwheel,objaverse]",
                      "lightwheel,objaverse"):
            with self.subTest(value=value):
                self.assertEqual(expected, robocasa_asset_packs({"obj_registries": value}))
        for value in ("[]", '[1]', '[["lightwheel"]]', '{"lightwheel": true}'):
            with self.subTest(invalid=value), self.assertRaisesRegex(ValueError, "nonempty list of registry names"):
                robocasa_asset_packs({"obj_registries": value})

    def test_vlabench_uses_official_resumable_assets_and_checks_native_packages(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            with patch.dict("source_benchmarks.VLA_ASSET_BYTES", {"obj": None, "scenes": None}):
                install_vlabench(build)
            downloads = [a for a in build.actions if a[0] == "download"]
            self.assertEqual(2, len(downloads))
            self.assertTrue(all(action[3]["range_size"] == DRIVE_RANGE_BYTES for action in downloads))
            self.assertTrue(all(action[1].startswith("https://drive.usercontent.google.com/download?id=")
                                for action in downloads))
            package = Path(build.env["VLABENCH_ROOT"])
            self.assertIn("if not os.path.isdir(xml_dir):", (package / "configs/constant.py").read_text())
            self.assertTrue((package / "assets/scenes/default/empty.xml").is_file())
            self.assertTrue((package / "assets/obj/meshes/poker/poker_asset.xml").is_file())
            self.assertEqual(2, len(list(build.downloads.glob("*.zip"))))

    def test_vlabench_rejects_and_removes_a_truncated_completed_archive(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            with self.assertRaisesRegex(RuntimeError, "archive has .* bytes; expected"):
                install_vlabench(build)
            self.assertFalse((build.downloads / "vlabench-obj.zip").exists())

    def test_vlabench_repairs_from_complete_archives_without_contacting_google_drive(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            sizes = {}
            for name in ("obj", "scenes"):
                archive = build.downloads / f"vlabench-{name}.zip"
                build.run(["python", "-m", "gdown", "--output", str(archive), "fixture"])
                sizes[name] = archive.stat().st_size
            build.actions.clear()
            with patch.dict("source_benchmarks.VLA_ASSET_BYTES", sizes):
                install_vlabench(build)
            self.assertFalse(any(a[0] == "run" and "gdown" in a[1] for a in build.actions))
            self.assertFalse(any(a[0] == "download" for a in build.actions))
            self.drive_resolver.assert_not_called()
            self.assertTrue((Path(build.env["VLABENCH_ROOT"]) / "assets/scenes/default/empty.xml").is_file())

    def test_official_drive_confirmation_is_resolved_without_cookie_files(self):
        class Response(io.BytesIO):
            headers = {"Content-Type": "text/html"}

        page = b'''<form id="download-form" action="https://drive.usercontent.google.com/download?export=download&amp;confirm=t">
                   <input type="hidden" name="id" value="official_file"><input type="hidden" name="uuid" value="token"></form>'''
        with patch("source_benchmarks.urlopen", return_value=Response(page)):
            url = drive_download_url("official_file")
        self.assertIn("id=official_file", url)
        self.assertIn("confirm=t", url)
        self.assertIn("uuid=token", url)

    def test_drive_confirmation_rejects_wrong_file_ids_and_unexpected_hosts(self):
        class Response(io.BytesIO):
            headers = {"Content-Type": "text/html"}

        for host, file_id in (("drive.usercontent.google.com", "wrong_file"), ("example.invalid", "official_file")):
            page = f'<form id="download-form" action="https://{host}/download"><input type="hidden" name="id" value="{file_id}"></form>'
            with self.subTest(host=host), patch("source_benchmarks.urlopen", return_value=Response(page.encode())):
                with self.assertRaises(RuntimeError):
                    drive_download_url("official_file")

    def test_legacy_drive_partial_is_adopted_without_copying_or_losing_received_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            archive = build.downloads / "vlabench-scenes.zip"
            legacy = build.downloads / "vlabench-scenes.zipold.part"
            legacy.write_bytes(b"received data")
            original_inode = legacy.stat().st_ino
            _adopt_drive_partial(build, archive, 100)
            partial = archive.with_name(archive.name + ".part")
            self.assertEqual(b"received data", partial.read_bytes())
            self.assertEqual(original_inode, partial.stat().st_ino)
            self.assertFalse(legacy.exists())

    def test_corrupt_final_archive_is_removed_so_a_future_download_can_repair_it(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            archive = build.downloads / "corrupt.zip"
            archive.write_text("truncated network body")
            with self.assertRaisesRegex(RuntimeError, "archive is corrupt"):
                _asset_archive(build, archive, Path(root) / "assets", "assets", ("required.xml",), source="fixed-url")
            self.assertFalse(archive.exists())

    def test_invalid_assets_keep_the_download_and_do_not_publish_a_success_marker(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            archive = build.archive(build.downloads / "bad.zip", ["assets/unrelated.txt"])
            with self.assertRaisesRegex(RuntimeError, "Expected one assets"):
                _asset_archive(build, archive, Path(root) / "assets", "assets", ("required.xml",), source="fixed-url")
            self.assertTrue(archive.is_file())
            self.assertFalse((Path(root) / "asset-manifests/assets.json").exists())
            self.assertEqual([], list(Path(root).glob("extract-*")))

    def test_corrupted_previously_completed_assets_are_repaired_with_archive_kept_until_runtime_ready(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            destination = Path(root) / "assets"
            archive = build.archive(build.downloads / "first.zip", ["assets/required.xml"])
            _asset_archive(build, archive, destination, "assets", ("required.xml",), source="fixed-url")
            self.assertTrue(_already_have(build, destination, "assets", ("required.xml",), "fixed-url"))
            (destination / "required.xml").unlink()
            self.assertFalse(_already_have(build, destination, "assets", ("required.xml",), "fixed-url"))
            archive = build.archive(build.downloads / "second.zip", ["assets/required.xml"])
            _asset_archive(build, archive, destination, "assets", ("required.xml",), source="fixed-url")
            self.assertTrue((destination / "required.xml").is_file())
            self.assertTrue(archive.exists())

    def test_owned_source_patch_is_idempotent_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as root:
            build = FakeBuild(root)
            target = Path(root) / "source.py"
            target.write_text("old line\n")
            _patch_once(build, target, "old line", "new line")
            _patch_once(build, target, "old line", "new line")
            self.assertEqual("new line\n", target.read_text())
            target.write_text("different upstream\n")
            with self.assertRaisesRegex(RuntimeError, "expected patch"):
                _patch_once(build, target, "old line", "new line")

    def run_probe(self, root, *, black_camera=False, wrong_source=False, nonfinite_state=False,
                  kind="vlabench", evaluator_failure=False):
        import numpy as np

        calls = []
        class Image:
            def __init__(self, **kwargs):
                calls.append(("wand", kwargs))

            def __enter__(self):
                return self

            def __exit__(self, *error):
                calls.append(("wand-close",))

        class VectorEnv:
            action_space = types.SimpleNamespace(shape=(1, 7))

            def reset(self, seed=None):
                calls.append(("reset", seed))
                images = {name: np.ones((1, 8, 8, 3), dtype=np.uint8)
                          for name in ("image", "second_image", "wrist_image")}
                if black_camera:
                    images["wrist_image"][:] = 0
                state = np.array([[np.nan if nonfinite_state else 0.0] * 7])
                return {"pixels": images, "agent_pos": state}, {}

            def step(self, action):
                calls.append(("step", action.shape))
                return {"pixels": {}}, np.array([0.0]), [False], [True], {}

            def close(self):
                calls.append(("close",))

        class Config:
            def __init__(self, **kwargs):
                calls.append(("config", kwargs))

            def create_envs(self, count, use_async):
                calls.append(("create", count, use_async))
                return {"task": {0: VectorEnv()}}

        configs = types.ModuleType("lerobot.envs.configs")
        configs.VLABenchEnv = Config
        configs.LiberoPlusEnv = Config
        source = types.ModuleType("pinned_fixture")
        source.__file__ = str(Path(root) / ("wrong/source.py" if wrong_source else "source/source.py"))
        records = {"pinned_fixture": {"directory": str(Path(root) / "source"), "revision": "a" * 40}}
        image_module = types.ModuleType("wand.image")
        image_module.Image = Image
        evaluator = types.ModuleType("lerobot.scripts.lerobot_eval")
        modules = {"lerobot.envs.configs": configs, "pinned_fixture": source,
                   "wand.image": image_module, evaluator.__name__: evaluator}
        import importlib
        original_import = importlib.import_module

        def guarded_import(name, *args, **kwargs):
            if name == evaluator.__name__:
                calls.append(("evaluator",))
                if evaluator_failure:
                    raise RuntimeError("evaluator native extension cannot load")
                if kind == "libero_plus" and ("wand-close",) not in calls:
                    raise RuntimeError("ImageMagick must initialize before the evaluator")
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"CONVLEX_SOURCE_BENCHMARK": json.dumps(records)}), \
                patch.object(sys, "argv", ["probe", kind]), patch("importlib.import_module", side_effect=guarded_import), \
                patch("benchmarks.libero_plus_issue", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(PROBE, "source-runtime-probe", "exec"), {})
            except RuntimeError as error:
                return calls, error
        return calls, None

    @unittest.skipUnless(find_spec("numpy"), "NumPy is required for simulator readiness probes")
    def test_ready_probe_executes_reset_then_action_step_and_closes(self):
        with tempfile.TemporaryDirectory() as root:
            calls, error = self.run_probe(root)
            self.assertIsNone(error)
            self.assertEqual([("reset", 0), ("step", (1, 7)), ("close",)], calls[-3:])

    @unittest.skipUnless(find_spec("numpy"), "NumPy is required for simulator readiness probes")
    def test_failed_camera_probe_closes_the_environment_without_publishing_readiness(self):
        with tempfile.TemporaryDirectory() as root:
            calls, error = self.run_probe(root, black_camera=True)
            self.assertIn("Empty rendered camera", str(error))
            self.assertEqual(("close",), calls[-1])
            self.assertFalse(any(call[0] == "step" for call in calls))

    @unittest.skipUnless(find_spec("numpy"), "NumPy is required for simulator readiness probes")
    def test_shadowed_upstream_package_is_rejected_before_creating_the_simulator(self):
        with tempfile.TemporaryDirectory() as root:
            calls, error = self.run_probe(root, wrong_source=True)
            self.assertIn("outside the managed pinned source", str(error))
            self.assertEqual([("evaluator",)], calls)

    @unittest.skipUnless(find_spec("numpy"), "NumPy is required for simulator readiness probes")
    def test_nonfinite_joint_state_is_rejected_and_the_environment_is_closed(self):
        with tempfile.TemporaryDirectory() as root:
            calls, error = self.run_probe(root, nonfinite_state=True)
            self.assertIn("Nonfinite robot state", str(error))
            self.assertEqual(("close",), calls[-1])
            self.assertFalse(any(call[0] == "step" for call in calls))

    @unittest.skipUnless(find_spec("numpy"), "NumPy is required for simulator readiness probes")
    def test_libero_plus_probe_loads_image_libraries_then_the_real_evaluator_before_reset(self):
        with tempfile.TemporaryDirectory() as root:
            calls, error = self.run_probe(root, kind="libero_plus")
            self.assertIsNone(error)
            self.assertEqual(calls[:3], [("wand", {"width": 1, "height": 1}),
                                        ("wand-close",), ("evaluator",)])
            self.assertEqual(calls[-3:], [("reset", 0), ("step", (1, 7)), ("close",)])

    @unittest.skipUnless(find_spec("numpy"), "NumPy is required for simulator readiness probes")
    def test_evaluator_import_failure_cannot_publish_readiness_even_if_simulator_alone_would_work(self):
        with tempfile.TemporaryDirectory() as root:
            calls, error = self.run_probe(root, evaluator_failure=True)
            self.assertIn("native extension cannot load", str(error))
            self.assertEqual(calls, [("evaluator",)])


if __name__ == "__main__":
    unittest.main()
