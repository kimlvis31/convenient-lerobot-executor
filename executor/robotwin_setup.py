"""AI-written setup for an isolated RoboTwin simulator; preserve upstream sources."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import sysconfig
from tempfile import TemporaryDirectory


KEY = "robotwin"
REVISION = "robotwin-0aeea2d6-curobo-d64c4b00-torch291-1"
ROBOTWIN = "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4"
CUROBO = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
PYTORCH3D = "75ebeeaea0908c5527e7b1e305fbc7681382db47"
ASSET_REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
ASSETS = (
    ("embodiments.zip", 219847741,
     "85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789"),
    ("objects.zip", 3737778549,
     "6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27"),
)
# MPlib's compiled ABI requires NumPy 1. Keep this environment separate from
# native LeRobot; installing a LeRobot wheel with ignored dependencies is unsafe.
CONSTRAINTS = (
    "torch==2.9.1", "torchvision==0.24.1", "numpy==1.26.4",
    "gymnasium==0.29.1", "scipy==1.15.3", "sapien==3.0.0b1",
    "mplib==0.2.1", "transforms3d==0.4.2", "trimesh==4.4.3",
    "open3d==0.19.0", "imageio==2.34.2", "setuptools==80.9.0",
    "opencv-python==4.11.0.86", "zarr==2.18.7", "warp-lang==1.19.0",
)


def replace_checked(path, old, new, *, count=1):
    """Patch a known upstream contract, accepting an already patched file."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    if source.count(old) == count:
        path.write_text(source.replace(old, new), encoding="utf-8")
    elif old not in source and source.count(new) == count:
        return
    else:
        raise RuntimeError(f"RoboTwin source patch no longer matches its pinned version: {path}")


def patch_sapien_loader(path):
    """Patch the exact Python 3.12 SAPIEN 3.0.0b1 wheel's parse method."""
    replace_checked(path, 'with open(urdf_file, "r") as f:',
                    'with open(urdf_file, "r", encoding="utf-8") as f:')
    replace_checked(path, 'with open(srdf_file, "r") as f:',
                    'with open(srdf_file, "r", encoding="utf-8") as f:')
    replace_checked(path, 'srdf_file = urdf_file[:-4] + "srdf"',
                    'srdf_file = urdf_file[:-4] + ".srdf"')


def patch_runtime_packages():
    """Apply only the upstream's version-specific MPlib and SAPIEN fixes."""
    expected = {"mplib": "0.2.1", "sapien": "3.0.0b1"}
    for name, version in expected.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Refusing to patch an unrecognized {name} version.")
    mplib = Path(importlib.metadata.distribution("mplib").locate_file("mplib/planner.py"))
    # RoboTwin/script/_install.sh removes this exact broken collision check.
    replace_checked(mplib,
                    "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
                    "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:")
    sapien = Path(importlib.metadata.distribution("sapien").locate_file("sapien/wrapper/urdf_loader.py"))
    patch_sapien_loader(sapien)


CUDA_HEADERS = ("cuda_runtime_api.h", "cusparse.h", "cublas_v2.h", "cublasLt.h",
                "cusolverDn.h", "curand_kernel.h")


def cuda_header_paths(prefix, sites=None):
    """Find development headers in the toolkit and this prefix's NVIDIA wheels."""
    prefix = Path(prefix).resolve()
    toolkit = prefix / "targets/x86_64-linux/include"
    if not (toolkit / "cuda_runtime.h").is_file():
        toolkit = prefix / "include"
    if not (toolkit / "cuda_runtime.h").is_file():
        raise RuntimeError("The managed CUDA toolkit does not contain cuda_runtime.h.")
    include = [toolkit.resolve()]
    if sites is None:
        sites = {sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}
    for site in sorted(map(str, sites)):
        site = Path(site).resolve()
        if not site.is_relative_to(prefix):
            raise RuntimeError(f"Refusing CUDA headers from outside the simulator prefix: {site}")
        for directory in sorted((site / "nvidia").glob("*/include")):
            directory = directory.resolve()
            if not directory.is_relative_to(prefix):
                raise RuntimeError(f"Refusing an external NVIDIA include link: {directory}")
            if directory.is_dir() and directory not in include:
                include.append(directory)
    missing = [name for name in CUDA_HEADERS if not any((folder / name).is_file() for folder in include)]
    if missing:
        raise RuntimeError("The managed PyTorch CUDA packages are missing development headers: " +
                           ", ".join(missing))
    return include


def write_build_environment(destination):
    """Read the activated Conda toolchain and restrict compilation to this GPU."""
    import torch

    prefix = Path(os.environ["CONDA_PREFIX"]).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("RoboTwin setup needs an NVIDIA GPU visible to the simulator environment.")
    if torch.__version__.split("+")[0] != "2.9.1" or torch.version.cuda not in {"12.6", "12.8"}:
        raise RuntimeError("The RoboTwin recipe requires its pinned PyTorch 2.9.1 CUDA 12.x wheel.")
    include = cuda_header_paths(prefix)
    header_path = os.pathsep.join(map(str, include))
    result = {
        "CUDA_HOME": str(prefix), "CUDACXX": str(prefix / "bin/nvcc"),
        "CUB_HOME": str(include[0]), "CPATH": header_path, "CPLUS_INCLUDE_PATH": header_path,
        # NVCC does not consistently consume GCC's CPATH for its device-side
        # preprocessor. Its official prepend variable applies to both projects.
        "NVCC_PREPEND_FLAGS": shlex.join("-I" + str(folder) for folder in include),
        "LIBRARY_PATH": os.pathsep.join((str(prefix / "lib"),
                                          str(prefix / "targets/x86_64-linux/lib"))),
        "TORCH_CUDA_ARCH_LIST": ";".join(sorted({
            ".".join(map(str, torch.cuda.get_device_capability(index)))
            for index in range(torch.cuda.device_count())
        })),
        "MAX_JOBS": "2", "FORCE_CUDA": "1",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO": "0.7.8",
    }
    for variable, fallback in (("CC", "x86_64-conda-linux-gnu-cc"),
                               ("CXX", "x86_64-conda-linux-gnu-c++")):
        executable = os.environ.get(variable) or str(prefix / "bin" / fallback)
        resolved = shutil.which(executable)
        if not resolved or not Path(resolved).resolve().is_relative_to(prefix):
            raise RuntimeError(f"The managed Conda {variable} compiler is missing: {executable}")
        result[variable] = resolved
    Path(destination).write_text(json.dumps(result), encoding="utf-8")


def check_cuda_headers(build):
    """Compile the required public headers before either expensive extension."""
    with TemporaryDirectory(prefix="cuda-headers-", dir=build.root) as folder:
        source = Path(folder) / "headers.cu"
        output = Path(folder) / "headers.o"
        source.write_text("".join(f"#include <{name}>\n" for name in CUDA_HEADERS) +
                          "int main() { return 0; }\n", encoding="utf-8")
        build.run([str(Path(build.prefix) / "bin/nvcc"), "-std=c++17", "-c", str(source),
                   "-o", str(output)])


def check_graphics():
    """Fail before downloading assets when host NVIDIA graphics are unavailable."""
    import ctypes

    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("Automatic RoboTwin setup currently requires Linux x86-64 and an NVIDIA GPU.")
    try:
        ctypes.CDLL("libcuda.so.1")
        ctypes.CDLL("libGLX_nvidia.so.0")
    except OSError as error:
        raise RuntimeError("RoboTwin needs the host NVIDIA CUDA and Vulkan/GLX driver libraries. "
                           "Install the NVIDIA graphics driver before retrying.") from error


def _worker_command(build, mode):
    source = str(build.options["source"])
    digest = build.options.get("source_sha256")
    if not digest:
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return [str(build.python), "-B", str(Path(__file__).with_name("simulator_worker.py")),
            mode, "--kind", KEY, "--source", source, "--source-sha256", digest]


def install(build):
    """Build the pinned simulator and return its runtime environment overrides."""
    from runtime_setup import torch_index

    check_graphics()
    index = torch_index()
    if index.endswith("cu128"):
        cuda = ("cuda-nvcc=12.8.93", "cuda-cudart-dev=12.8.90")
    elif index.endswith("cu126"):
        cuda = ("cuda-nvcc=12.6.85", "cuda-cudart-dev=12.6.77")
    else:
        raise RuntimeError(f"Unsupported RoboTwin CUDA wheel index: {index}")
    build.conda_install(*cuda, "gcc_linux-64=13", "gxx_linux-64=13", "libvulkan-loader",
                        channels=("nvidia", "conda-forge"))
    # Bootstrap PyPI dependencies before selecting the CUDA-only wheel index.
    # Fresh Conda prefixes may contain a newer setuptools; the Torch index does
    # not carry the pinned legacy version required by SAPIEN's pkg_resources.
    build.pip_install(*CONSTRAINTS[2:], "wheel", "ninja", "setuptools-scm>=6.2", "pybind11",
                      "pyyaml", "pydantic", "h5py", "termcolor", "av", "matplotlib", "huggingface-hub",
                      "iopath", "numpy-quaternion", "yourdfpy>=0.0.53", "scikit-image")
    cuda_tag = index.rsplit("/", 1)[-1]
    build.pip_install(f"torch==2.9.1+{cuda_tag}", f"torchvision==0.24.1+{cuda_tag}",
                      index_url=index, wheelhouse=True)
    script = str(Path(__file__).resolve())
    build.run([str(build.python), "-B", script, "--patch-runtime"])
    settings = Path(build.root) / "build-environment.json"
    try:
        build.run([str(build.python), "-B", script, "--configure", str(settings)])
        build.env.update(json.loads(settings.read_text(encoding="utf-8")))
    finally:
        settings.unlink(missing_ok=True)
    check_cuda_headers(build)

    source = build.source("robotwin", "RoboTwin-Platform/RoboTwin", ROBOTWIN)
    pytorch3d = build.source("pytorch3d", "facebookresearch/pytorch3d", PYTORCH3D)
    curobo = build.source("curobo", "NVlabs/curobo", CUROBO)
    replace_checked(curobo / "src/curobo/geom/sdf/world_mesh.py",
                    "wp.torch.device_from_torch(", "wp.device_from_torch(")
    # PyTorch 2.9.1 uses C++17. Keep both sources' original standard; newer
    # mutable image versions require different source and compiler contracts.
    build.pip_install(str(pytorch3d), no_build_isolation=True, cwd=pytorch3d)
    build.pip_install(str(curobo), no_build_isolation=True, cwd=curobo)

    assets = source / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for filename, size, sha256 in ASSETS:
        url = ("https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/"
               f"{ASSET_REVISION}/{filename}")
        archive = build.download(url, filename, sha256=sha256, size=size)
        build.extract(archive, assets)
    build.run([str(build.python), "script/update_embodiment_config_path.py"], cwd=source)
    overrides = {
        "ROBOTWIN_ROOT": str(source), "CONVLEX_SIMULATOR_ROOT": str(source),
        "PYTHONPATH": str(source), "PYTHONNOUSERSITE": "1",
        "CUDA_HOME": str(build.prefix),
        "WARP_CACHE_PATH": str(Path(build.root) / "warp-cache"),
        "TORCH_EXTENSIONS_DIR": str(Path(build.root) / "torch-extensions"),
        "CUDA_CACHE_PATH": str(Path(build.root) / "cuda-cache"),
        "LD_LIBRARY_PATH": os.pathsep.join((str(Path(build.prefix) / "lib"),
                                              str(Path(build.prefix) / "targets/x86_64-linux/lib"),
                                              os.environ.get("LD_LIBRARY_PATH", ""))),
    }
    # A per-runtime ICD avoids altering /usr/share/vulkan when the host only
    # supplied the NVIDIA driver library. Existing explicit ICD settings win.
    if not os.environ.get("VK_ICD_FILENAMES"):
        icd = Path(build.root) / "nvidia_icd.json"
        icd.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {
            "library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0",
        }}), encoding="utf-8")
        overrides["VK_ICD_FILENAMES"] = str(icd)
    build.env.update(overrides)
    # Source directories are owned by this build. Installed non-editable
    # extensions no longer reference their large compiler intermediates.
    for directory in (pytorch3d, curobo):
        directory = Path(directory).resolve()
        if not directory.is_relative_to(Path(build.root).resolve()):
            raise RuntimeError(f"Refusing to clean an unowned RoboTwin build source: {directory}")
        shutil.rmtree(directory)
    return overrides


def probe(build):
    """Verify official wrapper, camera observations, reset and one physical step."""
    build.run(_worker_command(build, "--probe"), cwd=build.env.get("ROBOTWIN_ROOT"))


if __name__ == "__main__":
    if sys.argv[1:] == ["--patch-runtime"]:
        patch_runtime_packages()
    elif len(sys.argv) == 3 and sys.argv[1] == "--configure":
        write_build_environment(sys.argv[2])
    else:
        raise SystemExit("Expected --patch-runtime or --configure PATH.")
