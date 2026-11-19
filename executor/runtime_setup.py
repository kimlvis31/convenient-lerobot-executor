"""Prepare verified simulator runtimes in owned folders. Executable code is AI-written.

Recipes share locking, atomic readiness, safe downloads and failure cleanup.
Conda prefixes stay at their final absolute path; user environments are never
changed by a source recipe. No packages or assets are prepared during previews.
"""

from dataclasses import dataclass, field
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from urllib.request import Request, urlopen
import uuid
import zipfile

from robomme_setup import torch_index


class RuntimeSetupFailure(RuntimeError):
    def __init__(self, kind, reason, logfile=None):
        self.reason, self.logfile = reason, logfile
        super().__init__(f"{kind} environment setup failed: {reason}")


def recipes():
    import robotwin_setup
    from source_benchmarks import RECIPES

    return {**RECIPES, "robotwin": {
        "revision": robotwin_setup.REVISION, "mode": "split", "python": "3.12",
        "constraints": robotwin_setup.CONSTRAINTS, "install": robotwin_setup.install,
        "probe": robotwin_setup.probe, "asset_bytes": 4_000_000_000,
        "description": "RoboTwin, NumPy 1, SAPIEN, MPlib, CUDA PyTorch, CuRobo and robot/object assets",
    }}


def supported(kind):
    return sys.platform == "linux" and kind in recipes()


def cache_directory():
    configured = os.environ.get("CONVLEX_RUNTIME_CACHE")
    directory = Path(configured or Path(__file__).resolve().parents[1] / "externals/.runtimes").expanduser()
    if configured and not directory.is_absolute():
        raise ValueError("CONVLEX_RUNTIME_CACHE must name an absolute directory.")
    return directory.resolve()


def owned(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise RuntimeError(f"Refusing to modify a path outside the managed runtime: {path}")
    return path


def discard(path, root):
    path = owned(path, root)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, *, digest=None, size=None, range_size=None):
    """Resume only into an owned partial file, then verify before publication."""
    if range_size is not None:
        if not isinstance(range_size, int) or isinstance(range_size, bool) or range_size <= 0:
            raise ValueError("range_size must be a positive integer number of bytes.")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("Segmented downloads require the known complete file size.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def valid(path):
        return path.is_file() and (size is None or path.stat().st_size == size) and (
            digest is None or sha256(path) == digest)

    if valid(destination):
        return destination
    destination.unlink(missing_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if (size is not None or digest is not None) and valid(partial):
        partial.replace(destination)
        return destination
    if range_size is not None:
        return _download_ranges(url, destination, partial, digest=digest, size=size, range_size=range_size)
    for attempt in range(4):
        offset = partial.stat().st_size if partial.is_file() else 0
        if size is not None and offset >= size:
            partial.unlink()
            offset = 0
        request = Request(url, headers={"User-Agent": "convLexecutor-runtime/1"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urlopen(request, timeout=120) as response:
                resumed = offset and response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Download server returned an inconsistent byte range.")
                length = response.headers.get("Content-Length", "")
                length = int(length) if length.isdigit() else None
                total = re.fullmatch(r"bytes \d+-\d+/(\d+)", response.headers.get("Content-Range", ""))
                remaining = (size - offset) if size is not None and resumed else size
                if remaining is None:
                    remaining = length
                if remaining is not None and shutil.disk_usage(destination.parent).free < remaining + 1024**3:
                    raise RuntimeError(f"Not enough disk space to download {destination.name} "
                                       f"({remaining / 1e9:.2f} GB still needed, plus 1 GiB reserve).")
                with partial.open("ab" if resumed else "wb") as stream:
                    before = stream.tell()
                    shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
                    received = stream.tell() - before
                if length is not None and received != length:
                    raise OSError("Incomplete HTTP response body")
                if total and partial.stat().st_size != int(total[1]):
                    raise OSError("Incomplete HTTP byte range")
            if not valid(partial):
                if size is not None and partial.stat().st_size < size:
                    raise OSError("Incomplete asset download")
                partial.unlink(missing_ok=True)
                raise OSError(f"Asset size or SHA-256 verification failed: {destination.name}")
            partial.replace(destination)
            return destination
        except (OSError, TimeoutError, IncompleteRead) as error:
            if attempt == 3:
                raise RuntimeError(f"Could not download {url}: {error}") from error
            time.sleep(min(2 ** attempt, 5))
    raise AssertionError("Unreachable")


def _download_ranges(url, destination, partial, *, digest, size, range_size):
    """Commit complete HTTP packets to a resumable file, using fresh connections."""
    if partial.is_file() and partial.stat().st_size > size:
        partial.unlink()
    partial.touch(exist_ok=True)
    while partial.stat().st_size < size:
        offset = partial.stat().st_size
        end = min(offset + range_size, size) - 1
        expected = end - offset + 1
        # Reserve space for the entire unfinished file, not merely one packet.
        remaining = size - offset
        if shutil.disk_usage(destination.parent).free < remaining + 1024**3:
            raise RuntimeError(f"Not enough disk space to download {destination.name} "
                               f"({remaining / 1e9:.2f} GB still needed, plus 1 GiB reserve).")
        for attempt in range(4):
            request = Request(url, headers={"User-Agent": "convLexecutor-runtime/1",
                                           "Accept-Encoding": "identity", "Connection": "close",
                                           "Range": f"bytes={offset}-{end}"})
            try:
                with urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError("Download server ignored the requested byte range "
                                           f"(HTTP {response.status}); retry without range_size.")
                    content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                                 response.headers.get("Content-Range", ""))
                    if not content_range or tuple(map(int, content_range.groups())) != (offset, end, size):
                        raise RuntimeError("Download server returned an inconsistent Content-Range.")
                    length = response.headers.get("Content-Length", "")
                    if not length.isdigit() or int(length) != expected:
                        raise RuntimeError("Download server returned an inconsistent packet Content-Length.")
                    try:
                        with partial.open("ab") as stream:
                            received = 0
                            while received < expected:
                                block = response.read(min(4 * 1024 * 1024, expected - received))
                                if not block:
                                    raise OSError("Incomplete HTTP range packet")
                                stream.write(block)
                                received += len(block)
                            if received != expected or response.read(1):
                                raise OSError("HTTP range packet exceeded its declared length")
                    except BaseException:
                        # Never append a retry to the uncommitted tail of a
                        # truncated/oversized packet, including on Ctrl-C.
                        try:
                            with partial.open("r+b") as stream:
                                stream.truncate(offset)
                        except OSError:
                            pass
                        raise
                break
            except (OSError, TimeoutError, IncompleteRead) as error:
                # Some HTTP wrappers fail while closing an otherwise complete
                # response. Roll it back as well before retrying the same range.
                if partial.stat().st_size > offset:
                    with partial.open("r+b") as stream:
                        stream.truncate(offset)
                if attempt == 3:
                    raise RuntimeError(f"Could not download {url} at byte {offset}: {error}") from error
                time.sleep(min(2 ** attempt, 5))
        # A successful packet starts a new independent retry budget.
    if partial.stat().st_size != size or (digest is not None and sha256(partial) != digest):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Asset size or SHA-256 verification failed: {destination.name}")
    partial.replace(destination)
    return destination


def stop_process_tree(process):
    """Terminate the owned command group, including compiler grandchildren."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "posix":
            process.kill()
    finally:
        # Conda can exit before a child compiler which ignored SIGTERM. Reap
        # the remaining process group even when its original leader is gone.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def extract(archive, destination):
    """Extract source/assets without allowing traversal or escaping links."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            required = sum(member.file_size for member in bundle.infolist())
            if shutil.disk_usage(destination).free < required + 1024**3:
                raise RuntimeError(f"Not enough disk space to extract {Path(archive).name} "
                                   f"({required / 1e9:.2f} GB plus 1 GiB reserve required).")
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination) or member.isdev() or member.isfifo():
                    raise ValueError(f"Unsafe archive member: {member.name}")
                if member.issym() or member.islnk():
                    link = (target.parent if member.issym() else destination) / member.linkname
                    if not link.resolve().is_relative_to(destination):
                        raise ValueError(f"Unsafe archive link: {member.name}")
            required = sum(member.size for member in bundle.getmembers() if member.isfile())
            if shutil.disk_usage(destination).free < required + 1024**3:
                raise RuntimeError(f"Not enough disk space to extract {Path(archive).name} "
                                   f"({required / 1e9:.2f} GB plus 1 GiB reserve required).")
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(destination, filter="data")
            else:
                if any(member.issym() or member.islnk() for member in bundle.getmembers()):
                    raise RuntimeError("Update the manager's Python to extract source archive links safely.")
                bundle.extractall(destination)


@dataclass
class RuntimeBuild:
    root: Path
    prefix: Path
    python: Path
    conda: str
    downloads: Path
    options: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    constraints: tuple = ()
    sources: dict = field(default_factory=dict)

    def run(self, args, cwd=None, env=None):
        command = [self.conda, "run", "--no-capture-output", "--prefix", str(self.prefix), *map(str, args)]
        return self.execute(command, cwd=cwd, env=env)

    def execute(self, command, cwd=None, env=None):
        """Run a setup command with an owned lifetime and a bounded failure tail."""
        overrides = {**os.environ, **self.env, **(env or {}), "PYTHONNOUSERSITE": "1",
                     "PIP_CACHE_DIR": str(self.downloads / "pip-cache"),
                     "CONDA_PKGS_DIRS": str(self.downloads / "conda-pkgs"),
                     "CONDA_ALWAYS_COPY": "true"}
        print("  Runtime setup: " + " ".join(map(str, command[:4])), flush=True)
        log = self.root / "setup.log"
        tail = []
        with log.open("a", encoding="utf-8") as stream:
            with subprocess.Popen(command, cwd=cwd or self.root, env=overrides, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                  start_new_session=os.name == "posix") as process:
                try:
                    for line in process.stdout:
                        stream.write(line)
                        tail.append(line.rstrip())
                        tail = tail[-30:]
                        print(line, end="", flush=True)
                    code = process.wait()
                except BaseException:
                    stop_process_tree(process)
                    raise
                if code:
                    stop_process_tree(process)
        if code:
            raise RuntimeError(f"Runtime setup command failed ({code}): {' '.join(map(str, command[:4]))}\n" + "\n".join(tail))
        return subprocess.CompletedProcess(command, code, "\n".join(tail))

    def pip_install(self, *requirements, no_deps=False, index_url=None, no_build_isolation=False,
                    editable=False, cwd=None, wheelhouse=False):
        owned(self.prefix, self.root)
        command = [str(self.python), "-m", "pip", "install", "--no-input", "--no-user",
                   "--disable-pip-version-check", "--progress-bar", "off", "--cache-dir",
                   str(self.downloads / "pip-cache")]
        if self.constraints:
            constraint = self.root / "constraints.txt"
            constraint.write_text("\n".join(self.constraints) + "\n", encoding="utf-8")
            command += ["--constraint", str(constraint)]
        if wheelhouse:
            if no_deps or no_build_isolation or editable:
                raise ValueError("A wheelhouse install requires binary packages with their dependencies.")
            wheels = self.downloads / "wheels"
            wheels.mkdir(parents=True, exist_ok=True)
            fetch = [str(self.python), "-m", "pip", "download", "--dest", str(wheels),
                     "--only-binary=:all:", "--disable-pip-version-check", "--progress-bar", "off"]
            if self.constraints:
                fetch += ["--constraint", str(constraint)]
            if index_url:
                fetch += ["--index-url", index_url, "--extra-index-url", "https://pypi.org/simple"]
            self.run([*fetch, *requirements], cwd=cwd)
            command += ["--no-index", "--find-links", str(wheels)]
        if no_deps:
            command.append("--no-deps")
        if index_url and not wheelhouse:
            command += ["--index-url", index_url]
        if no_build_isolation:
            command.append("--no-build-isolation")
        if editable:
            if len(requirements) != 1:
                raise ValueError("Editable installation requires exactly one source directory.")
            command.append("--editable")
        return self.run([*command, *requirements], cwd=cwd)

    def conda_install(self, *packages, channels=("conda-forge",)):
        owned(self.prefix, self.root)
        command = [self.conda, "install", "--yes", "--prefix", str(self.prefix), "--override-channels"]
        for channel in channels:
            command += ["--channel", channel]
        return self.execute([*command, *packages], env={"CONDA_ALWAYS_YES": "true"})

    def source(self, name, github_repo, revision):
        if name in {".", ".."} or not re.fullmatch(r"[a-zA-Z0-9_.-]+", name) or not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", github_repo):
            raise ValueError("Invalid source repository or directory name.")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Source recipes must pin a complete Git commit.")
        source = self.root / "sources" / name
        archive = self.download(f"https://codeload.github.com/{github_repo}/tar.gz/{revision}",
                                f"{name}-{revision}.tar.gz")
        unpack = self.root / f"unpack-{name}"
        self.extract(archive, unpack)
        entries = list(unpack.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise RuntimeError(f"Unexpected source archive layout: {github_repo}")
        source.parent.mkdir(parents=True, exist_ok=True)
        entries[0].rename(source)
        unpack.rmdir()
        self.sources[name] = {"repository": github_repo, "revision": revision, "archive_sha256": sha256(archive)}
        archive.unlink()
        return source

    def download(self, url, filename, sha256=None, size=None, range_size=None):
        if filename in {"", ".", ".."} or Path(filename).name != filename:
            raise ValueError("Downloads require a basename.")
        return download(url, self.downloads / filename, digest=sha256, size=size, range_size=range_size)

    @staticmethod
    def extract(archive, destination):
        return extract(archive, destination)


def identity(kind, recipe, options):
    # Cloned policy distributions and source wrappers form part of the ABI.
    details = {"kind": kind, "recipe": recipe["revision"], "constraints": recipe["constraints"],
               "base": None, "base_signature": None,
               "source_sha256": options.get("source_sha256"), "assets": options.get("asset_groups", ())}
    if recipe["mode"] == "combined":
        details.update(base=options.get("base_environment"), base_signature=options.get("base_signature"))
    if kind == "robocasa":
        from source_benchmarks import robocasa_registries

        details["obj_registries"] = sorted(robocasa_registries(options))
    return hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()[:16]


def managed_build(path, folder):
    root = owned(path, folder)
    if root.parent != Path(folder).resolve() or not re.fullmatch(r"build-[0-9a-f]{12}", root.name):
        raise ValueError(f"Not a managed build directory: {root}")
    return root


def published_root(pointer, folder):
    try:
        return managed_build(json.loads(pointer.read_text(encoding="utf-8"))["root"], folder)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return None


def runtime_environment(build):
    """Workers launched directly must not inherit the policy's active prefix."""
    original_cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    hub_home = str(Path(os.environ.get("HF_HOME") or original_cache / "huggingface").expanduser())
    hub_environment = {"HF_HOME": hub_home,
                       "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or str(Path(hub_home) / "hub"),
                       "HF_TOKEN_PATH": os.environ.get("HF_TOKEN_PATH") or str(Path(hub_home) / "token"),
                       "HF_ASSETS_CACHE": os.environ.get("HF_ASSETS_CACHE") or str(Path(hub_home) / "assets"),
                       "HF_XET_CACHE": os.environ.get("HF_XET_CACHE") or str(Path(hub_home) / "xet"),
                       "TORCH_HOME": os.environ.get("TORCH_HOME") or str(original_cache / "torch")}
    for key in ("TRANSFORMERS_CACHE", "HF_MODULES_CACHE"):
        if os.environ.get(key):
            hub_environment[key] = os.environ[key]
    selected = build.options.get("hub_environment", {})
    allowed = {*hub_environment, "TRANSFORMERS_CACHE", "HF_MODULES_CACHE"}
    if not isinstance(selected, dict) or any(key not in allowed or not isinstance(value, str)
                                              for key, value in selected.items()):
        raise ValueError("Invalid policy cache path report.")
    hub_environment.update(selected)
    return {"CONDA_PREFIX": str(build.prefix), "CONDA_DEFAULT_ENV": str(build.prefix),
            "PATH": str(build.prefix / "bin") + os.pathsep + os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", **hub_environment}


def configure_runtime_activation(build):
    """Install value-free source-env hooks without reading source env_vars here."""
    base = build.options.get("base_prefix")
    if base is None:
        selected = build.options.get("base_environment")
        if not isinstance(selected, str) or not selected:
            raise ValueError("A source environment is required for managed Conda activation.")
        selector = "--prefix" if Path(selected).is_absolute() else "--name"
        command = [str(build.conda), "run", "--no-capture-output", selector, selected,
                   "python", "-B", "-c", "import sys; print('CONVLEX_BASE_PREFIX '+sys.prefix)"]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=60)
        base = next((line.removeprefix("CONVLEX_BASE_PREFIX ") for line in reversed(result.stdout.splitlines())
                     if line.startswith("CONVLEX_BASE_PREFIX ")), None)
        if result.returncode or not base:
            raise RuntimeError("Cannot determine the source Conda environment prefix.")
    if not isinstance(base, str) or not Path(base).is_absolute():
        raise ValueError("The source Conda environment prefix must be absolute.")
    from runtime_activation import install_activation

    result = install_activation(build.root, build.prefix, base)
    build.options["base_prefix"] = result["base_prefix"]
    return result


def runtime_in_use(root, process_root="/proc"):
    """Preserve a replaced prefix still referenced by one of this user's jobs."""
    if os.name != "posix":
        return True
    needle = os.fsencode(str(Path(root).resolve()))
    try:
        processes = list(Path(process_root).iterdir())
    except OSError:
        return True
    for process in processes:
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
        except FileNotFoundError:
            continue
        except OSError:
            return True
        for name in ("cmdline", "environ"):
            try:
                if needle in (process / name).read_bytes():
                    return True
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError:
                return True
    return False


def _atomic_json(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def finish_cleanup(folder, current, old=None):
    """Retry harmless housekeeping without invalidating a published runtime."""
    for candidate in (folder / "downloads", old, folder / "last_failure.json", folder / "last_failure.log"):
        if not candidate or not Path(candidate).exists() or Path(candidate).resolve() == current:
            continue
        if old is not None and Path(candidate).resolve() == old and runtime_in_use(old):
            print(f"  Runtime ready; preserving a previous runtime still in use: {old}", flush=True)
            continue
        try:
            discard(candidate, folder)
        except (OSError, RuntimeError) as error:
            print(f"  Runtime ready; cleanup deferred for {candidate}: {error}", flush=True)


def prepare_runtime(kind, conda, *, options=None):
    """Return an actual verified runtime; only called after a job is confirmed."""
    if not supported(kind):
        raise RuntimeError(f"Automatic source setup for {kind} requires Linux.")
    import fcntl

    options = dict(options or {})
    recipe = recipes()[kind]
    if recipe["mode"] == "combined" and not options.get("base_environment"):
        raise ValueError("A combined simulator needs a selected LeRobot base environment.")
    if kind == "robotwin":
        from robotwin_setup import check_graphics
        from simulator_worker import validate_source
        if not options.get("source"):
            raise ValueError("RoboTwin setup requires the selected LeRobot wrapper source (--source PATH).")
        validate_source({"kind": kind, **options})
        check_graphics()
        torch_index()  # Fail on an unsupported driver before any large download.
    folder = cache_directory() / (kind + "-" + identity(kind, recipe, options))
    folder.mkdir(parents=True, exist_ok=True)
    pointer = folder / "ready.json"
    with (folder / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        published = published_root(pointer, folder)
        # A power loss cannot run Python's finally block. Reclaim only our
        # unpublished builds once their installers/workers are no longer alive.
        for candidate in folder.glob("build-*"):
            try:
                candidate = managed_build(candidate, folder)
                if candidate != published and not runtime_in_use(candidate):
                    discard(candidate, folder)
            except (OSError, RuntimeError, ValueError) as error:
                print(f"  Incomplete runtime cleanup deferred: {error}", flush=True)
        if pointer.is_file():
            try:
                ready = json.loads(pointer.read_text(encoding="utf-8"))
                root = managed_build(ready["root"], folder)
                if (ready["recipe"] == recipe["revision"] and ready.get("kind") == kind
                        and ready.get("python") == str(root / "env/bin/python")
                        and ready.get("prefix") == str(root / "env")
                        and isinstance(ready.get("env"), dict)):
                    build = RuntimeBuild(root, root / "env", root / "env/bin/python", conda,
                                         folder / "downloads", options, ready["env"], tuple(recipe["constraints"]))
                    if build.python.is_file():
                        build.env.update(runtime_environment(build))
                        if recipe["mode"] == "combined":
                            ready["base_prefix"] = configure_runtime_activation(build)["base_prefix"]
                        # Asset structure and an actual physical reset/step are checked on
                        # reuse, not just a stale marker left by an interrupted setup.
                        recipe["probe"](build)
                        print(f"  Reusing verified {kind} runtime: {build.prefix}", flush=True)
                        finish_cleanup(folder, root)
                        return ready
            except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                print(f"  Rebuilding incomplete {kind} runtime: {error}", flush=True)

        # Never let an untrusted/stale pointer nominate an external deletion.
        old = published_root(pointer, folder)
        root = folder / ("build-" + uuid.uuid4().hex[:12])
        root.mkdir()
        build = RuntimeBuild(root, root / "env", root / "env/bin/python", conda,
                             folder / "downloads", options, {}, tuple(recipe["constraints"]))
        try:
            needed = max(16_000_000_000, recipe.get("asset_bytes", 0) * 3 + 16_000_000_000)
            if shutil.disk_usage(folder).free < needed:
                raise RuntimeError(f"Not enough disk space for {kind} setup (at least {needed / 1e9:.1f} GB free required).")
            print(f"  Preparing {kind}: {recipe['description']}.\n  Managed runtime: {root}", flush=True)
            command = [conda, "create", "--yes", "--prefix", str(build.prefix)]
            if recipe["mode"] == "combined":
                command += ["--clone", options["base_environment"]]
            else:
                command += ["--override-channels", "--channel", "conda-forge", "python=" + recipe["python"], "pip"]
            build.execute(command)
            build.env.update({**runtime_environment(build), "PYTHONPATH": "",
                              "XDG_CACHE_HOME": str(root / "cache"),
                              "MPLCONFIGDIR": str(root / "matplotlib")})
            if recipe["mode"] == "combined":
                configure_runtime_activation(build)
            recipe["install"](build)
            build.run([str(build.python), "-m", "pip", "check"])
            recipe["probe"](build)
            source_overrides = json.loads(build.env.get("CONVLEX_SOURCE_BENCHMARK") or "{}")
            source_notes = [f"Effective source: {' '.join(filter(None, (name, entry.get('version'))))} "
                            f"@ {entry['revision'][:12]} ({entry['directory']})."
                            for name, entry in source_overrides.items()]
            ready = {"kind": kind, "recipe": recipe["revision"], "root": str(root), "prefix": str(build.prefix),
                     "python": str(build.python), "env": build.env, "sources": build.sources,
                     "source_overrides": source_overrides,
                     "base_environment": options.get("base_environment"),
                     "base_prefix": options.get("base_prefix"),
                     "base_signature": options.get("base_signature"),
                     "source_sha256": options.get("source_sha256"),
                     "requirements": list(recipe["constraints"]),
                     "notes": [recipe["description"],
                               "Required simulator packages: " + ", ".join(recipe["constraints"]),
                               *source_notes,
                               "Physical reset, camera observation and one simulator step verified."]}
            _atomic_json(root / "manifest.json", ready)
            _atomic_json(pointer, ready)
        except BaseException as error:
            # A signal can arrive immediately after ready.json was replaced.
            # Preserve that verified build instead of leaving a dangling pointer.
            if published_root(pointer, folder) != root:
                details = None
                if (root / "setup.log").is_file():
                    try:
                        details = folder / "last_failure.log"
                        shutil.copyfile(root / "setup.log", details)
                    except OSError:
                        details = None
                try:
                    _atomic_json(folder / "last_failure.json", {"kind": kind, "recipe": recipe["revision"],
                                                               "error": str(error)[-12000:]})
                except OSError as cleanup_error:
                    print(f"  Could not record runtime failure: {cleanup_error}", flush=True)
                try:
                    discard(root, folder)
                except OSError as cleanup_error:
                    print(f"  Could not remove incomplete runtime: {cleanup_error}", flush=True)
                if isinstance(error, Exception):
                    from runner import stage_failure_reason

                    reason = stage_failure_reason(details) if details else None
                    reason = reason or str(error).splitlines()[0] or type(error).__name__
                    raise RuntimeSetupFailure(kind, reason, details) from error
            raise

        # Publication is the commit point. Housekeeping must never delete a
        # successfully published runtime or replace its real outcome with an error.
        finish_cleanup(folder, root, old)
        return ready


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=tuple(recipes()))
    parser.add_argument("--conda", default=os.environ.get("CONDA_EXE") or shutil.which("conda"))
    parser.add_argument("--source")
    parser.add_argument("--base-environment")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        recipe = recipes()[args.kind]
        print(json.dumps({key: recipe[key] for key in ("revision", "mode", "python", "constraints", "description")}, indent=2))
        return
    if not args.conda:
        parser.error("Conda was not found.")
    options = {"base_environment": args.base_environment}
    if args.source:
        options.update(source=str(Path(args.source).resolve()), source_sha256=sha256(args.source))
    ready = prepare_runtime(args.kind, args.conda, options=options)
    print("CONVLEX_RUNTIME_READY " + json.dumps(ready))


if __name__ == "__main__":
    main()
