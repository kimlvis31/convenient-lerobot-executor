"""Prepare optional policy kernels from exact official binaries, preserving Torch."""

from dataclasses import dataclass
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ExtensionRecipe:
    distribution: str
    minimum: str
    index: str
    documentation: str
    release_prefix: str
    imports: tuple[str, ...]


NATTEN = ExtensionRecipe(
    "natten", "0.21.6", "https://whl.natten.org/", "https://natten.org/install/",
    "https://github.com/SHI-Labs/NATTEN/releases/download/",
    ("natten.functional:na2d", "natten.functional:na3d",
     "natten.backends:can_run_cutlass_fna", "natten.backends:can_run_cutlass_hopper_fna",
     "natten.backends:can_run_cutlass_blackwell_fna"),
)
POLICY_EXTENSIONS = {"flux3": (NATTEN,)}


@dataclass(frozen=True)
class WheelRuntime:
    torch: str
    cuda: str
    tags: frozenset


def extension_specs(policy):
    return tuple(f"{item.distribution}>={item.minimum}" for item in POLICY_EXTENSIONS.get(policy, ()))


def wheel_runtime():
    """Inspect a local runtime without querying a driver or allocating CUDA memory."""
    from packaging.tags import sys_tags
    import torch

    release = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\+cu(\d+))?", str(torch.__version__))
    cuda = str(torch.version.cuda or "")
    if sys.platform != "linux" or platform.python_implementation() != "CPython":
        raise ValueError("Automatic policy-kernel binaries require Linux and CPython; no compiler fallback is used.")
    if not release or not re.fullmatch(r"\d+\.\d+", cuda):
        raise ValueError("Automatic policy-kernel binaries require an official stable CUDA-enabled Torch build.")
    cuda_tag = cuda.replace(".", "")
    if release[4] and release[4] != cuda_tag:
        raise ValueError("Torch's CUDA version tag differs from its runtime; select a consistent environment.")
    return WheelRuntime("".join(release.group(i) for i in (1, 2, 3)), cuda_tag, frozenset(sys_tags()))


def extension_runtime_issue(policy):
    if not POLICY_EXTENSIONS.get(policy):
        return None
    try:
        wheel_runtime()
    except (ImportError, OSError, ValueError) as error:
        return str(error)
    return None


def import_issue(recipe, *, target=None):
    """Use a fresh interpreter so replaced or broken extension modules cannot linger."""
    code = (
        "import importlib,importlib.metadata as m\n"
        "from packaging.version import Version\n"
        f"assert Version(m.version({recipe.distribution!r})) >= Version({recipe.minimum!r}), 'extension is too old'\n"
        f"checks={recipe.imports!r}\n"
        "for value in checks:\n"
        "    module,_,name=value.partition(':')\n"
        "    assert callable(getattr(importlib.import_module(module),name)), 'extension API missing: '+value\n"
    )
    env = None
    if target is not None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(target), env.get("PYTHONPATH"))))
    try:
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=45, env=env)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"{recipe.distribution} runtime import could not be verified: {error}"
    if result.returncode:
        lines = (result.stderr or result.stdout).strip().splitlines()
        detail = lines[-1] if lines else f"exit code {result.returncode}"
        return f"{recipe.distribution} runtime import failed: {detail}"
    return None


def extension_problems(policy, versions):
    """Only installed files and imports are read; catalog calls never access the network."""
    from packaging.version import InvalidVersion, Version

    problems = {}
    for recipe in POLICY_EXTENSIONS.get(policy, ()):
        version = versions.get(recipe.distribution)
        try:
            sufficient = version is not None and Version(version) >= Version(recipe.minimum)
        except InvalidVersion:
            sufficient = False
        if not sufficient:
            problems[recipe.distribution] = f"Missing or outdated: {recipe.distribution}>={recipe.minimum}"
        elif issue := import_issue(recipe):
            problems[recipe.distribution] = issue
    return problems


class WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        if tag == "a" and (href := dict(attributes).get("href")):
            self.links.append(href)


def select_wheel(recipe, html, runtime, requirements=()):
    """Select only a published compatible binary from the recipe's trusted release origin."""
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
    from packaging.version import Version

    parser = WheelLinks()
    parser.feed(html)
    candidates = []
    trusted = urlsplit(recipe.release_prefix)
    for href in parser.links:
        url = urljoin(recipe.index, href)
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.netloc != trusted.netloc or parts.query
                or not parts.path.startswith(trusted.path)):
            continue
        try:
            name, version, _, tags = parse_wheel_filename(unquote(Path(parts.path).name))
        except InvalidWheelFilename:
            continue
        if (name != recipe.distribution or version < Version(recipe.minimum) or version.is_prerelease
                or version.local != f"torch{runtime.torch}cu{runtime.cuda}" or not (tags & runtime.tags)
                or any(not requirement.specifier.contains(str(version), prereleases=False) for requirement in requirements)):
            continue
        digest = None
        if parts.fragment:
            if not re.fullmatch(r"sha256=[a-fA-F0-9]{64}", parts.fragment):
                continue
            digest = parts.fragment.partition("=")[2].lower()
        candidates.append((version, urlunsplit(parts), digest))
    if not candidates:
        raise ValueError(
            f"No exact official {recipe.distribution} wheel matches Torch {runtime.torch}, CUDA {runtime.cuda}, "
            f"Python and platform. Torch will not be changed and source compilation is disabled. {recipe.documentation}"
        )
    version, url, digest = max(candidates)
    return {"distribution": recipe.distribution, "version": str(version), "url": url, "sha256": digest}


def resolve_wheel(recipe, runtime, requirements=()):
    request = Request(recipe.index, headers={"User-Agent": "convLexecutor-policy-extensions"})
    with urlopen(request, timeout=30) as response:
        final = urlsplit(response.geturl())
        if final.scheme != "https" or final.netloc != urlsplit(recipe.index).netloc:
            raise ValueError("The official policy-kernel index redirected to an unexpected origin.")
        page = response.read(2 * 1024 * 1024 + 1)
    if len(page) > 2 * 1024 * 1024:
        raise ValueError("The official policy-kernel index exceeds the supported size.")
    return select_wheel(recipe, page.decode("utf-8"), runtime, requirements)


def install_extensions(policy, versions, requirements=(), *, target=None):
    """Confirmed-run path only; exact wheels never install dependencies or replace Torch."""
    from packaging.utils import canonicalize_name

    problems = extension_problems(policy, versions)
    if not problems:
        return []
    runtime = wheel_runtime()
    installed = []
    for recipe in POLICY_EXTENSIONS.get(policy, ()):
        if recipe.distribution not in problems:
            continue
        constraints = [item for item in requirements if canonicalize_name(item.name) == recipe.distribution]
        selected = resolve_wheel(recipe, runtime, constraints)
        with tempfile.TemporaryDirectory(prefix="convlexecutor-kernel-") as directory:
            report = Path(directory) / "pip-report.json"
            command = [sys.executable, "-m", "pip", "install", "--no-input", "--no-user",
                       "--disable-pip-version-check", "--progress-bar", "off", "--no-deps",
                       "--only-binary=:all:", "--force-reinstall", "--report", str(report)]
            if target is not None:
                command.extend(("--target", str(target)))
            command.append(selected["url"])
            print(f"Installing official policy kernel: {selected['url']}", flush=True)
            result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.stdout:
                print(result.stdout, end="")
            if result.returncode:
                raise ValueError(f"Installing the exact {recipe.distribution} binary failed; no source build was attempted.")
            if report.is_file():
                payload = json.loads(report.read_text(encoding="utf-8"))
                for item in payload.get("install", ()):
                    if canonicalize_name(item.get("metadata", {}).get("name", "")) == recipe.distribution:
                        selected["sha256"] = item.get("download_info", {}).get("archive_info", {}).get("hashes", {}).get("sha256") or selected["sha256"]
        if issue := import_issue(recipe, target=target):
            raise ValueError(issue)
        print("POLICY_EXTENSION_READY " + json.dumps(selected, sort_keys=True), flush=True)
        installed.append(selected)
    return installed
