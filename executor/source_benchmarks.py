"""Pinned source-benchmark recipes for manager-owned combined LeRobot runtimes.

Executable code in this module was written by AI. Upstream source checkouts and
assets retain their own licenses. See THIRD_PARTY.md for attribution.
"""

import ast
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import zipfile
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


LIBERO_PLUS = "4976dc30028e805ff8094b55501d532c48fec182"
LIBERO_ASSETS = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
LIBERO_ASSETS_SHA256 = "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
LIBERO_ASSETS_BYTES = 6_395_849_578
ROBOCASA = "56e355ccc64389dfc1b8a61a33b9127b975ba681"
ROBOSUITE = "aaa8b9b214ce8e77e82926d677b4d61d55e577ab"
VLABENCH = "cf588fe60c0c7282174fe979f5913170cfe69017"
RRT = "e51d95ee489a225220d6ae2a764c4111f6ba7d85"

CASA_PACKS = {
    "tex": ("textures", "textures", ("*/*",)),
    "tex_generative": ("generative_textures", "generative_textures", ("*/*",)),
    "fixtures_lw": ("fixtures_lightwheel", "fixtures", ("*/*/model.xml",)),
    "objs_lw": ("objects_lightwheel", "objects/lightwheel", ("*/*/model.xml",)),
    "objs_objaverse": ("objaverse", "objects/objaverse", ("**/*.xml",)),
    "objs_aigen": ("aigen_objs", "objects/aigen_objs", ("**/*.xml",)),
}
CASA_ASSET_BYTES = {"tex": 543_180_421, "tex_generative": 1_184_675_536,
                    "fixtures_lw": 529_745_465, "objs_lw": 791_939_656}
VLA_DRIVE = {
    "obj": "1ldEMZua2OzXHJTYTCP0IGVU1aFYBCMu-",
    "scenes": "1KdReRkibJClBHHD32jz_wTkaBzhEJ9Kw",
}
VLA_ASSET_BYTES = {"obj": 4_987_960_728, "scenes": 1_015_309_482}
DRIVE_RANGE_BYTES = 32 * 1024 * 1024


class _DriveConfirmation(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action, self.active = attrs.get("action"), True
        if self.active and tag == "input" and attrs.get("type") == "hidden" and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def drive_download_url(file_id):
    """Resolve only the official public file confirmation, without cookie files."""
    if not isinstance(file_id, str) or not re.fullmatch(r"[-_a-zA-Z0-9]+", file_id):
        raise ValueError("Invalid Google Drive file identifier.")
    origin = "https://drive.google.com/uc?" + urlencode({"export": "download", "id": file_id})
    request = Request(origin, headers={"User-Agent": "convLexecutor-runtime/1"})
    with urlopen(request, timeout=30) as response:
        if response.headers.get("Content-Disposition"):
            direct = response.geturl()
        else:
            page = response.read(128 * 1024 + 1)
            if len(page) > 128 * 1024:
                raise RuntimeError(f"Unexpected Google Drive confirmation size. Official source: {origin}")
            form = _DriveConfirmation()
            form.feed(page.decode("utf-8", errors="replace"))
            if not form.action or form.fields.get("id") != file_id:
                raise RuntimeError(f"Cannot resolve the public Google Drive archive; check access or quota: {origin}")
            parts = urlsplit(form.action)
            direct = urlunsplit(parts._replace(query=urlencode({**dict(parse_qsl(parts.query)), **form.fields})))
    target = urlsplit(direct)
    if target.scheme != "https" or target.hostname not in {
        "drive.google.com", "drive.usercontent.google.com", "docs.google.com",
    }:
        raise RuntimeError("Google Drive returned an unexpected archive download location.")
    return direct


def _adopt_drive_partial(build, archive, size):
    """Preserve an interrupted legacy gdown download when switching transports."""
    partial = archive.with_name(archive.name + ".part")
    legacy = [path for path in archive.parent.glob(archive.name + "*.part") if path != partial]
    if partial.exists() or not legacy:
        return
    if len(legacy) != 1:
        raise RuntimeError(f"Multiple partial downloads exist for {archive.name}; preserve one before retrying.")
    candidate = _owned(build, legacy[0])
    if not candidate.is_file() or candidate.stat().st_size > size:
        raise RuntimeError(f"Invalid partial simulator archive: {candidate}")
    candidate.replace(partial)


def _owned(build, path):
    """Restrict cleanup to a recipe's own build and download directories."""
    resolved = Path(path).resolve()
    parents = (Path(build.root).resolve(), Path(build.downloads).resolve())
    if not any(resolved != parent and resolved.is_relative_to(parent) for parent in parents):
        raise RuntimeError(f"Refusing to modify a path outside the managed recipe: {resolved}")
    return resolved


def _discard(build, path):
    path = _owned(build, path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _prepend_paths(build, *paths):
    previous = build.env.get("PYTHONPATH", "")
    values = [str(Path(path)) for path in paths]
    values.extend(part for part in previous.split(os.pathsep) if part and part not in values)
    build.env["PYTHONPATH"] = os.pathsep.join(values)
    build.env["PYTHONNOUSERSITE"] = "1"
    renderer = build.env.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    if os.environ.get("PYOPENGL_PLATFORM"):
        build.env.setdefault("PYOPENGL_PLATFORM", os.environ["PYOPENGL_PLATFORM"])
    elif renderer in {"egl", "osmesa"}:
        build.env.setdefault("PYOPENGL_PLATFORM", renderer)


def _require_tree(root, patterns):
    root = Path(root)
    missing = [pattern for pattern in patterns if not any(p.is_file() for p in root.glob(pattern))]
    if missing:
        raise RuntimeError(f"Incomplete simulator assets under {root}: {', '.join(missing)}")


def _merge_move(source, destination):
    """Merge extracted assets without retaining a second full-sized copy."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir() and target.is_dir():
            _merge_move(item, target)
            item.rmdir()
        elif target.exists():
            if target.is_dir():
                raise RuntimeError(f"Asset file conflicts with an existing directory: {target}")
            item.replace(target)
        else:
            shutil.move(str(item), str(target))


def _asset_root(staging, suffix, patterns):
    candidates = []
    for path in Path(staging).rglob(Path(suffix).name):
        if not path.is_dir():
            continue
        try:
            _require_tree(path, patterns)
        except RuntimeError:
            continue
        candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {suffix} directory in downloaded assets, found {len(candidates)}.")
    return candidates[0]


def _asset_archive(build, archive, destination, suffix, patterns, *, source):
    """Record verified content; keep archives until the whole runtime is ready."""
    destination = _owned(build, destination)
    marker = Path(build.root) / "asset-manifests" / (str(suffix).replace("/", "-") + ".json")
    staging = Path(tempfile.mkdtemp(prefix="extract-", dir=build.root))
    try:
        try:
            build.extract(archive, staging)
        except (zipfile.BadZipFile, tarfile.ReadError, EOFError) as error:
            # gdown skips an existing completed filename when --continue is
            # used. Drop a corrupt final archive so the next launch can repair
            # it instead of repeatedly accepting that same filename.
            _discard(build, archive)
            raise RuntimeError(f"Downloaded simulator archive is corrupt; retry setup: {archive}") from error
        unpacked = _asset_root(staging, suffix, patterns)
        _require_tree(unpacked, patterns)
        digest = hashlib.sha256()
        with Path(archive).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        _merge_move(unpacked, destination)
        _require_tree(destination, patterns)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"source": source, "sha256": digest.hexdigest(),
                                      "bytes": Path(archive).stat().st_size}, indent=2) + "\n",
                          encoding="utf-8")
        # Common runtime publication removes the downloads directory after
        # package and physical probes succeed. Retaining the archive until
        # then allows dependency repairs without repeating a large download.
    finally:
        _discard(build, staging)


def _already_have(build, destination, suffix, patterns, source):
    marker = Path(build.root) / "asset-manifests" / (suffix.replace("/", "-") + ".json")
    try:
        if json.loads(marker.read_text(encoding="utf-8")).get("source") != source:
            return False
        _require_tree(destination, patterns)
    except (OSError, ValueError, RuntimeError):
        return False
    return True


def _patch_once(build, filename, original, replacement):
    """Apply an exact recipe patch only inside the manager-owned checkout."""
    filename = _owned(build, filename)
    text = filename.read_text(encoding="utf-8")
    if replacement in text:
        return
    if text.count(original) != 1:
        raise RuntimeError(f"Pinned source does not match the expected patch: {filename}")
    filename.write_text(text.replace(original, replacement, 1), encoding="utf-8")


def _record_sources(build, records):
    """Keep effective source identities separate from installed wheel metadata."""
    build.env["CONVLEX_SOURCE_BENCHMARK"] = json.dumps(records, sort_keys=True)


def robocasa_registries(options):
    """Validate explicit registry choices without silently changing the job."""
    registries = options.get("obj_registries", ["lightwheel"])
    if isinstance(registries, str):
        try:
            registries = json.loads(registries)
        except ValueError:
            try:
                registries = ast.literal_eval(registries)
            except (ValueError, SyntaxError):
                # Draccus also accepts YAML-style [lightwheel,objaverse].
                # Shell parsing may have removed the quotes around names.
                value = registries.strip()
                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1]
                registries = [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(registries, str):
        registries = [registries]
    if not isinstance(registries, (list, tuple)) or not registries or not all(isinstance(v, str) for v in registries):
        raise ValueError("RoboCasa obj_registries must be a nonempty list of registry names.")
    mapping = {"lightwheel": "objs_lw", "objaverse": "objs_objaverse", "aigen": "objs_aigen"}
    unknown = [value for value in registries if value not in mapping]
    if unknown:
        raise ValueError(f"Unknown RoboCasa object registries: {unknown}. Known registries: {sorted(mapping)}")
    return list(dict.fromkeys(registries))


def robocasa_asset_packs(options):
    """Map each selected registry to its official archive."""
    mapping = {"lightwheel": "objs_lw", "objaverse": "objs_objaverse", "aigen": "objs_aigen"}
    return ["tex", "tex_generative", "fixtures_lw",
            *dict.fromkeys(mapping[value] for value in robocasa_registries(options))]


def install_libero_plus(build):
    _owned(build, build.prefix)
    source = build.source("libero-plus", "sylvestf/LIBERO-plus", LIBERO_PLUS)
    package = source / "libero/libero"
    # Initial states are nested by suite, so validate their real layout before
    # changing the clone or fetching the multi-gigabyte perturbation archive.
    _require_tree(package, ("benchmark/task_classification.json", "bddl_files/**/*.bddl",
                            "init_files/libero_newobj/**/*.pruned_init"))
    config = Path(build.root) / "libero-config"
    config.mkdir(exist_ok=True)
    # JSON is a YAML subset and quotes paths containing colons or spaces safely.
    values = {"benchmark_root": str(package), "assets": str(package / "assets"),
              "bddl_files": str(package / "bddl_files"), "init_states": str(package / "init_files"),
              "datasets": str(Path(build.root) / "datasets")}
    (config / "config.yaml").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    build.env["LIBERO_CONFIG_PATH"] = str(config)
    build.env["LIBERO_PLUS_ROOT"] = str(package)
    build.env["MAGICK_HOME"] = str(build.prefix)
    _prepend_paths(build, source)
    _record_sources(build, {"libero": {"directory": str(source / "libero"), "revision": LIBERO_PLUS}})
    # Native libraries stay in the isolated prefix, with no sudo/apt changes.
    build.conda_install("imagemagick", "fontconfig")
    # Stock hf-libero pins robosuite 1.4.0. Remove it only in this new isolated
    # clone, replacing it with the real fork distribution and its 1.4.1 runtime.
    build.run(["python", "-m", "pip", "uninstall", "--yes", "hf-libero"])
    build.pip_install("robosuite==1.4.1", "bddl==1.0.1", "easydict==1.13", "mujoco==3.7.0",
                      "matplotlib==3.10.8", "Wand==0.6.13", "scikit-image==0.25.2", "gym==0.26.2")
    build.pip_install(str(source), no_deps=True, editable=True)
    build.run([str(build.python), "-m", "pip", "check"])
    url = f"https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/{LIBERO_ASSETS}/assets.zip"
    patterns = ("scenes/libero_tabletop_base_style.xml",)
    if not _already_have(build, package / "assets", "assets", patterns, url):
        archive = build.download(url, "libero-plus-assets.zip", sha256=LIBERO_ASSETS_SHA256,
                                 size=LIBERO_ASSETS_BYTES)
        _asset_archive(build, archive, package / "assets", "assets", patterns, source=url)


def install_robocasa(build):
    _owned(build, build.prefix)
    packs = robocasa_asset_packs(build.options)
    source = build.source("robocasa", "robocasa/robocasa", ROBOCASA)
    suite = build.source("robosuite", "ARISE-Initiative/robosuite", ROBOSUITE)
    _prepend_paths(build, source, suite)
    _record_sources(build, {"robocasa": {"directory": str(source / "robocasa"), "revision": ROBOCASA},
                            "robosuite": {"directory": str(suite / "robosuite"), "revision": ROBOSUITE,
                                          "version": "1.5.2"}})
    # Import both immutable source checkouts directly. Installing RoboCasa's
    # distribution would incorrectly request LeRobot 0.3.3. We keep upstream
    # metadata intact and install its actual runtime dependencies explicitly.
    build.pip_install("numpy==2.2.5", "numba==0.61.2", "scipy==1.15.3", "mujoco==3.3.1",
                      "pygame==2.6.1", "Pillow==12.2.0", "opencv-python==4.13.0.92", "pyyaml==6.0.3",
                      "pynput==1.8.1", "tqdm==4.67.3", "termcolor==3.3.0", "imageio==2.37.3",
                      "h5py==3.16.0", "lxml==6.0.4", "hidapi==0.14.0.post4",
                      "gymnasium==1.2.3", "qpsolvers[quadprog]>=4.3.1", "pytest")
    # Tianshou appears only in the optional upstream bench_speed.py utility.
    # Its protobuf~=3.19.0 requirement conflicts with current policy tools and
    # is unnecessary for the native LeRobot evaluator used here.
    build.run(["python", "-m", "robocasa.scripts.setup_macros"])
    build.run([str(build.python), "-m", "pip", "check"])
    assets = source / "robocasa/models/assets"
    links = json.loads((assets / "box_links/box_links_assets.json").read_text(encoding="utf-8"))
    for name in packs:
        key, suffix, patterns = CASA_PACKS[name]
        shared = links[key]
        host, share = shared.rstrip("/").rsplit("/s/", 1)
        url = f"{host}/shared/static/{share}.zip"
        destination = assets / suffix
        if not _already_have(build, destination, suffix, patterns, url):
            archive = build.download(url, f"robocasa-{name}.zip", size=CASA_ASSET_BYTES.get(name))
            _asset_archive(build, archive, destination, suffix, patterns, source=url)


def install_vlabench(build):
    _owned(build, build.prefix)
    source = build.source("vlabench", "OpenMOSS/VLABench", VLABENCH)
    rrt = build.source("rrt-algorithms", "motion-planning/rrt-algorithms", RRT)
    package = source / "VLABench"
    _prepend_paths(build, source, rrt)
    build.env["VLABENCH_ROOT"] = str(package)
    _record_sources(build, {"VLABench": {"directory": str(package), "revision": VLABENCH},
                            "rrt_algorithms.rrt.rrt_star": {"directory": str(rrt / "rrt_algorithms"),
                                                           "revision": RRT}})
    # Upstream evaluates every object category during import. The official
    # LeRobot benchmark recipe applies this exact missing-directory guard.
    _patch_once(build, package / "configs/constant.py", "    subdirs = os.listdir(xml_dir)",
                "    if not os.path.isdir(xml_dir):\n        return []\n    subdirs = os.listdir(xml_dir)")
    build.pip_install("mujoco==3.2.2", "dm-control==1.0.22", "open3d==0.19.0", "colorlog",
                      "scikit-learn", "openai", "gym==0.26.2", "ipdb",
                      "imageio", "rtree==1.4.1", "plotly")
    # RRT's source pins Rtree 1.2.0, whose published Linux wheel separates its
    # Tag headers with a blank line and fails modern pip platform validation.
    # Use the compatible 1.x API with well-formed wheel metadata, and verify
    # the pinned planner's real spatial queries before fetching large assets.
    build.run([str(build.python), "-m", "pip", "check"])
    build.run(["python", "-B", "-c",
               "import numpy as np; from rrt_algorithms.search_space.search_space import SearchSpace; "
               "space=SearchSpace(np.array([[0.,1.],[0.,1.]]),np.array([[.2,.2,.8,.8]])); "
               "assert not space.obstacle_free((.5,.5)); assert space.obstacle_free((.1,.1)); "
               "assert not space.collision_free((.1,.1),(.9,.9),.1); print('RRT spatial queries verified')"])
    assets = package / "assets"
    assets.mkdir(exist_ok=True)
    for name, drive_id in VLA_DRIVE.items():
        patterns = (("meshes/poker/poker_asset.xml", "meshes/containers/basket/**/*.xml",
                     "meshes/tablewares/plates/**/*.xml", "meshes/fruit/**/*.xml")
                    if name == "obj" else ("default/empty.xml",))
        url = f"https://drive.google.com/uc?id={drive_id}"
        if _already_have(build, assets / name, name, patterns, url):
            continue
        archive = Path(build.downloads) / f"vlabench-{name}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        expected = VLA_ASSET_BYTES[name]
        cached = (archive.is_file() and (expected is None or archive.stat().st_size == expected)
                  and zipfile.is_zipfile(archive))
        if not cached:
            # Resolve the public confirmation only when data is missing. Each
            # bounded Range request then uses a fresh connection so a throttled
            # long-lived stream does not stall the entire first-use setup.
            if archive.exists():
                _discard(build, archive)
            _adopt_drive_partial(build, archive, expected)
            archive = build.download(drive_download_url(drive_id), archive.name,
                                     size=expected, range_size=DRIVE_RANGE_BYTES)
        if expected is not None and archive.stat().st_size != expected:
            observed = archive.stat().st_size
            _discard(build, archive)
            raise RuntimeError(f"VLABench {name} archive has {observed} bytes; expected {expected}. Retry setup.")
        _asset_archive(build, archive, assets / name, name, patterns, source=url)


PROBE = r'''
import importlib,json,os,sys
from pathlib import Path
kind=sys.argv[1]
options=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
from evaluation_cameras import prepare_evaluation_runtime
prepare_evaluation_runtime(['--env.type='+kind])
# Match the real entry point's native-extension import order before declaring
# this runtime ready. Importing the simulator alone misses evaluator conflicts.
importlib.import_module('lerobot.scripts.lerobot_eval')
import numpy as np
sources=json.loads(os.environ['CONVLEX_SOURCE_BENCHMARK'])
for name,record in sources.items():
    module=importlib.import_module(name)
    observed=([Path(module.__file__).resolve()] if getattr(module,'__file__',None)
              else [Path(path).resolve() for path in module.__path__])
    expected=Path(record['directory']).resolve()
    if not observed or not all(path==expected or path.is_relative_to(expected) for path in observed):
        raise RuntimeError(f'{name} resolves outside the managed pinned source: {observed}')
    if record.get('version') and module.__version__!=record['version']:
        raise RuntimeError(f'{name} source version differs: {module.__version__}')
if kind=='libero_plus':
    from benchmarks import libero_plus_issue
    issue=libero_plus_issue()
    if issue: raise RuntimeError(issue)
    from lerobot.envs.configs import LiberoPlusEnv
    config=LiberoPlusEnv(task='libero_spatial',task_ids=[0],episode_length=1)
elif kind=='robocasa':
    from lerobot.envs.configs import RoboCasaEnv
    config=RoboCasaEnv(task='CloseFridge',episode_length=1,obj_registries=options.get('obj_registries',['lightwheel']))
else:
    from lerobot.envs.configs import VLABenchEnv
    config=VLABenchEnv(task='select_poker',episode_length=1)
envs=config.create_envs(1,False)
try:
    env=next(iter(next(iter(envs.values())).values()))
    obs,_=env.reset(seed=0)
    def check_state(observation):
        if 'agent_pos' in observation:
            state=np.asarray(observation['agent_pos'])
            if not np.all(np.isfinite(state)): raise RuntimeError('Nonfinite robot state in simulator observation')
            return list(state.shape)
        return None
    state_shape=check_state(obs)
    cameras=obs.get('pixels',{})
    expected=2 if kind=='libero_plus' else 3
    if len(cameras)<expected: raise RuntimeError(f'Missing raw cameras: {list(cameras)}')
    for name,image in cameras.items():
        image=np.asarray(image)
        if image.dtype!=np.uint8 or image.ndim!=4 or image.shape[0]!=1 or image.shape[-1]!=3:
            raise RuntimeError(f'Unexpected {name} camera shape/dtype: {image.shape}/{image.dtype}')
        if not np.any(image): raise RuntimeError(f'Empty rendered camera: {name}')
    action=np.zeros(env.action_space.shape,dtype=np.float32)
    obs,reward,terminated,truncated,info=env.step(action)
    if not isinstance(obs,dict) or not np.all(np.isfinite(np.asarray(reward))):
        raise RuntimeError('Invalid physical-step observation/reward')
    check_state(obs)
    print('SOURCE_BENCHMARK_READY '+json.dumps({'benchmark':kind,'cameras':list(cameras),'step':1,
          'camera_shapes':{name:list(np.asarray(image).shape) for name,image in cameras.items()},
          'state_shape':state_shape,'action_shape':list(action.shape),'sources':sources}))
finally:
    for tasks in envs.values():
        for env in tasks.values(): env.close()
'''


def _probe(build, kind):
    # The probe imports this checkout's existing readiness checks and uses the
    # installed LeRobot adapter, exercising its actual reset/action/step path.
    _prepend_paths(build, Path(__file__).resolve().parent)
    options = {"obj_registries": robocasa_registries(build.options)} if kind == "robocasa" else {}
    build.run(["python", "-B", "-c", PROBE, kind, json.dumps(options)])


def probe_libero_plus(build):
    _probe(build, "libero_plus")


def probe_robocasa(build):
    _probe(build, "robocasa")


def probe_vlabench(build):
    _probe(build, "vlabench")


RECIPES = {
    "libero_plus": {"revision": "libero-plus-" + LIBERO_PLUS[:8] + "-1", "mode": "combined",
                    "python": "3.12", "constraints": ("robosuite==1.4.1", "mujoco==3.7.0"),
                    "install": install_libero_plus, "probe": probe_libero_plus,
                    "asset_bytes": LIBERO_ASSETS_BYTES, "description": "LIBERO-plus fork and perturbation assets"},
    "robocasa": {"revision": "robocasa-" + ROBOCASA[:8] + "-" + ROBOSUITE[:8] + "-1", "mode": "combined",
                 "python": "3.12", "constraints": ("numpy==2.2.5", "numba==0.61.2", "scipy==1.15.3",
                                                      "mujoco==3.3.1", "gymnasium==1.2.3"),
                 "install": install_robocasa, "probe": probe_robocasa,
                 "asset_bytes": 10_000_000_000, "description": "RoboCasa365 kitchen simulator and selected object registries"},
    "vlabench": {"revision": "vlabench-" + VLABENCH[:8] + "-" + RRT[:8] + "-1", "mode": "combined",
                 "python": "3.12", "constraints": ("mujoco==3.2.2", "dm-control==1.0.22", "open3d==0.19.0"),
                 "install": install_vlabench, "probe": probe_vlabench,
                 "asset_bytes": 12_100_000_000, "description": "VLABench simulator, path planner and scene/object assets"},
}
