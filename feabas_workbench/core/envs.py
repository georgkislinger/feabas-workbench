"""
Python environments, GPU detection and installation.

The workbench needs up to three interpreters:

* the GUI's own (this process);
* a *FEABAS* environment (feabas + tensorstore, numpy<2);
* a *deep-learning* environment (torch + segmentation-models-pytorch +
  ultralytics + careamics) for the fold U-Net, YOLO-seg and N2V workers.

They are discovered by probing conda/micromamba environments, remembered in
the global settings file, and can be created by the installer with conda,
micromamba (downloaded on demand) or plain venv.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .. import ORG_NAME

PROBE_SCRIPT = r"""
import json, sys, importlib, importlib.metadata as md
out = {"python": sys.executable, "version": sys.version.split()[0], "packages": {}}
try:
    dists = md.packages_distributions()   # import name -> distribution name(s), e.g. cv2 -> opencv-python-headless
except Exception:
    dists = {}
for name in ["feabas", "tensorstore", "numpy", "torch", "segmentation_models_pytorch", "ultralytics",
             "careamics", "pytorch_lightning", "lightning", "tifffile", "cv2", "PySide6", "h5py", "scipy", "skimage"]:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", None) or getattr(m, "VERSION", None)
        if not v:
            # feabas and tensorstore set no __version__; the installed distribution still knows it
            for dist in dists.get(name, [name]):
                try:
                    v = md.version(dist)
                    break
                except md.PackageNotFoundError:
                    pass
        out["packages"][name] = str(v or "?")
    except Exception as e:
        out["packages"][name] = None
try:
    import torch
    out["cuda"] = bool(torch.cuda.is_available())
    out["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        x = torch.ones(8, device="cuda").sum().item()
        out["cuda_alloc_ok"] = (x == 8.0)
except Exception as e:
    out["cuda"] = False
    out["cuda_error"] = str(e)[:200]
print("##PROBE " + json.dumps(out))
"""


def settings_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    d = base / ORG_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


SETTINGS_FILE = settings_dir() / "settings.json"


@dataclass
class Settings:
    feabas_python: str = ""
    dl_python: str = ""
    fiji_path: str = ""
    vast_path: str = ""
    recent_projects: list[str] = field(default_factory=list)
    conda_exe: str = ""
    torch_index: str = "https://download.pytorch.org/whl/cu126"
    last_project: str = ""
    theme: str = "dark"

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        if SETTINGS_FILE.is_file():
            try:
                raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    if k in cls.__dataclass_fields__:
                        setattr(s, k, v)
            except (OSError, ValueError):
                pass
        return s

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def remember_project(self, path: str) -> None:
        path = str(Path(path))
        self.recent_projects = [path] + [p for p in self.recent_projects if p != path]
        self.recent_projects = self.recent_projects[:12]
        self.last_project = path


# ----------------------------------------------------------------------
# conda / micromamba discovery
# ----------------------------------------------------------------------

def find_conda() -> Path | None:
    for name in ("conda", "mamba", "micromamba"):
        exe = shutil.which(name)
        if exe:
            return Path(exe)
    cands = []
    home = Path.home()
    if os.name == "nt":
        for base in (home / "AppData" / "Local", home, Path("C:/ProgramData"), Path("C:/")):
            for dist in ("miniforge3", "mambaforge", "miniconda3", "anaconda3", "Miniconda3", "Anaconda3"):
                cands.append(base / dist / "Scripts" / "conda.exe")
                cands.append(base / dist / "condabin" / "conda.bat")
        cands.append(settings_dir() / "micromamba" / "micromamba.exe")
    else:
        for dist in ("miniforge3", "mambaforge", "miniconda3", "anaconda3"):
            cands.append(home / dist / "bin" / "conda")
            cands.append(Path("/opt") / dist / "bin" / "conda")
        cands.append(settings_dir() / "micromamba" / "bin" / "micromamba")
    for c in cands:
        if c.is_file():
            return c
    return None


def conda_envs(conda: Path | None = None) -> list[Path]:
    """Environment folders known to conda (plus the environments.txt registry)."""
    out: list[Path] = []
    reg = Path.home() / ".conda" / "environments.txt"
    if reg.is_file():
        for ln in reg.read_text(encoding="utf-8", errors="replace").splitlines():
            p = Path(ln.strip())
            if ln.strip() and p.is_dir():
                out.append(p)
    conda = conda or find_conda()
    if conda is not None:
        try:
            r = subprocess.run([str(conda), "env", "list", "--json"], capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                for e in json.loads(r.stdout).get("envs", []):
                    p = Path(e)
                    if p.is_dir():
                        out.append(p)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    uniq = []
    seen = set()
    for p in out:
        k = str(p).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def env_python(env_dir: Path) -> Path | None:
    for c in (env_dir / "python.exe", env_dir / "bin" / "python", env_dir / "Scripts" / "python.exe"):
        if c.is_file():
            return c
    return None


# ----------------------------------------------------------------------
# probing
# ----------------------------------------------------------------------

@dataclass
class ProbeResult:
    python: str
    ok: bool
    version: str = ""
    packages: dict = field(default_factory=dict)
    cuda: bool = False
    cuda_version: str | None = None
    gpu: str = ""
    gpu_mem_gb: float = 0.0
    cuda_alloc_ok: bool = False
    error: str = ""

    def has(self, *names: str) -> bool:
        return all(self.packages.get(n) for n in names)

    @property
    def is_feabas_env(self) -> bool:
        return self.has("feabas")

    @property
    def is_dl_env(self) -> bool:
        return self.has("torch")

    def describe(self) -> str:
        if not self.ok:
            return f"unusable: {self.error}"
        bits = [f"Python {self.version}"]
        for n, label in (("feabas", "feabas"), ("torch", "torch"), ("careamics", "careamics"),
                         ("ultralytics", "ultralytics"), ("segmentation_models_pytorch", "smp"),
                         ("tensorstore", "tensorstore")):
            v = self.packages.get(n)
            if v:
                bits.append(f"{label} {v}")
        if self.packages.get("torch"):
            bits.append(f"CUDA {'ok: ' + self.gpu if self.cuda else 'not available'}")
        return ", ".join(bits)


def probe_python(python: str | Path, timeout: int = 240) -> ProbeResult:
    python = str(python)
    if not Path(python).is_file():
        return ProbeResult(python, False, error="interpreter not found")
    try:
        r = subprocess.run([python, "-c", PROBE_SCRIPT], capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return ProbeResult(python, False, error="probe timed out")
    except OSError as e:
        return ProbeResult(python, False, error=str(e))
    for ln in r.stdout.splitlines():
        if ln.startswith("##PROBE "):
            d = json.loads(ln[8:])
            return ProbeResult(python, True, d.get("version", ""), d.get("packages", {}), d.get("cuda", False),
                               d.get("cuda_version"), d.get("gpu", ""), d.get("gpu_mem_gb", 0.0),
                               d.get("cuda_alloc_ok", False), d.get("cuda_error", ""))
    return ProbeResult(python, False, error=(r.stderr or r.stdout)[-400:])


def check_imports(python: str | Path, modules: Iterable[str], timeout: int = 90) -> str:
    """
    Empty string if *python* imports every module in *modules*, else a one-line reason.

    A cheap pre-flight before starting a step in another environment: a wrong
    interpreter should be a readable message, not a traceback from a subprocess.
    """
    python = str(python)
    mods = list(modules)
    if not python:
        return "no interpreter configured"
    if not Path(python).is_file():
        return f"interpreter not found: {python}"
    code = (
        "import json, sys\n"
        f"mods = {mods!r}\n"
        "bad = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        __import__(m)\n"
        "    except Exception as e:\n"
        "        bad.append('%s (%s: %s)' % (m, type(e).__name__, e))\n"
        "print('##CHECK ' + json.dumps(bad))\n"
    )
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return f"the interpreter did not answer within {timeout}s"
    except OSError as e:
        return str(e)
    for ln in r.stdout.splitlines():
        if ln.startswith("##CHECK "):
            bad = json.loads(ln[8:])
            if not bad:
                return ""
            return "cannot import " + ", ".join(bad)
    return (r.stderr or r.stdout or "the interpreter produced no answer").strip()[-300:]


def discover_environments() -> list[ProbeResult]:
    """Probe every conda env (and the current interpreter) for feabas / torch."""
    pys: list[Path] = [Path(sys.executable)]
    for e in conda_envs():
        p = env_python(e)
        if p:
            pys.append(p)
    seen = set()
    out = []
    for p in pys:
        k = str(p).lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(probe_python(p))
    return out


# ----------------------------------------------------------------------
# GPU / driver
# ----------------------------------------------------------------------

@dataclass
class GpuInfo:
    found: bool = False
    name: str = ""
    driver: str = ""
    memory_mb: int = 0

    def torch_index(self) -> str:
        """Pick a PyTorch wheel index from the driver version (coarse, conservative)."""
        try:
            major = float(self.driver.split(".")[0]) if self.driver else 0
        except ValueError:
            major = 0
        if not self.found or major == 0:
            return "https://download.pytorch.org/whl/cpu"
        if major >= 528:
            return "https://download.pytorch.org/whl/cu126"
        if major >= 452:
            return "https://download.pytorch.org/whl/cu118"
        return "https://download.pytorch.org/whl/cpu"


def detect_gpu() -> GpuInfo:
    exe = shutil.which("nvidia-smi")
    if exe is None and os.name == "nt":
        for c in (Path("C:/Windows/System32/nvidia-smi.exe"),
                  Path("C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe")):
            if c.is_file():
                exe = str(c)
                break
    if exe is None:
        return GpuInfo()
    try:
        r = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        line = r.stdout.strip().splitlines()[0]
        name, drv, mem = [s.strip() for s in line.split(",")]
        return GpuInfo(True, name, drv, int(float(mem)))
    except Exception:
        return GpuInfo()


# ----------------------------------------------------------------------
# installation plans
# ----------------------------------------------------------------------

FEABAS_PIP = ["feabas==3.0.5", "tifffile", "imagecodecs"]
# careamics 0.3.2 requires torch<2.10 and torchvision<=0.25. Install torch within those bounds
# up front, from the chosen index: otherwise the careamics step downgrades torch again, from
# PyPI, which throws away the CPU/CUDA build the user selected (and the download).
TORCH_PIP = ["torch<2.10", "torchvision<=0.25"]
DL_PIP = ["segmentation-models-pytorch", "ultralytics", "careamics==0.3.2", "tifffile", "imagecodecs",
          "scikit-image", "opencv-python-headless", "h5py", "psutil", "pyyaml", "scipy"]


@dataclass
class InstallPlan:
    kind: str                       # feabas | dl
    env_name: str
    python_version: str
    conda: Path | None
    torch_index: str = ""

    def commands(self) -> list[list[str]]:
        """Shell commands to run in order. The env's python is resolved after creation."""
        cmds: list[list[str]] = []
        if self.conda is None:
            raise RuntimeError("No conda/mamba/micromamba found. Install Miniforge or let the workbench download micromamba.")
        cmds.append([str(self.conda), "create", "-y", "-n", self.env_name, "--override-channels", "-c", "conda-forge",
                     f"python={self.python_version}", "pip"])
        return cmds

    def pip_commands(self, python: str) -> list[list[str]]:
        base = [python, "-m", "pip", "install", "--index-url", "https://pypi.org/simple"]
        if self.kind == "feabas":
            return [base + FEABAS_PIP]
        cmds = []
        if self.torch_index:
            cmds.append([python, "-m", "pip", "install", "--index-url", self.torch_index] + TORCH_PIP)
        else:
            cmds.append(base + TORCH_PIP)
        cmds.append(base + DL_PIP)
        return cmds


def env_dir_for(conda: Path, name: str) -> Path | None:
    for e in conda_envs(conda):
        if e.name == name:
            return e
    return None


MICROMAMBA_URLS = {
    ("Windows", "AMD64"): "https://micro.mamba.pm/api/micromamba/win-64/latest",
    ("Linux", "x86_64"): "https://micro.mamba.pm/api/micromamba/linux-64/latest",
    ("Linux", "aarch64"): "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest",
    ("Darwin", "arm64"): "https://micro.mamba.pm/api/micromamba/osx-arm64/latest",
    ("Darwin", "x86_64"): "https://micro.mamba.pm/api/micromamba/osx-64/latest",
}


def download_micromamba(log=None) -> Path:
    """Fetch a standalone micromamba into the settings folder (tar.bz2 archives)."""
    import tarfile
    key = (platform.system(), platform.machine())
    url = MICROMAMBA_URLS.get(key)
    if url is None:
        raise RuntimeError(f"No micromamba build known for {key}")
    dest_dir = settings_dir() / "micromamba"
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "micromamba.tar.bz2"
    if log:
        log(f"downloading micromamba from {url}")
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(dest_dir)
    exe = dest_dir / ("Library/bin/micromamba.exe" if os.name == "nt" else "bin/micromamba")
    if not exe.is_file():
        # some archives put it at the top level
        for c in dest_dir.rglob("micromamba*"):
            if c.is_file() and c.suffix in ("", ".exe"):
                exe = c
                break
    if not exe.is_file():
        raise RuntimeError("micromamba archive did not contain an executable")
    if os.name != "nt":
        exe.chmod(0o755)
    return exe


def default_feabas_root_for(python: str) -> Path | None:
    """Location of the feabas package in an environment (for reference/diagnostics)."""
    try:
        r = subprocess.run([python, "-c", "import feabas, os; print(os.path.dirname(feabas.__file__))"],
                           capture_output=True, text=True, timeout=60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0:
            return Path(r.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def find_fiji() -> str:
    cands = []
    if os.name == "nt":
        for base in (Path("D:/"), Path("C:/"), Path.home(), Path.home() / "Desktop", Path("C:/Program Files")):
            cands += [base / "Fiji.app" / "ImageJ-win64.exe", base / "Fiji" / "ImageJ-win64.exe"]
    else:
        for base in (Path.home(), Path("/opt"), Path("/Applications")):
            cands += [base / "Fiji.app" / "ImageJ-linux64", base / "Fiji.app" / "Contents" / "MacOS" / "ImageJ-macosx"]
    for c in cands:
        if c.is_file():
            return str(c)
    return ""


def find_vast() -> str:
    if os.name != "nt":
        return ""
    for base in (Path("C:/Program Files"), Path("C:/Program Files (x86)"), Path("D:/"), Path.home() / "Desktop"):
        if not base.is_dir():
            continue
        for d in sorted(base.glob("VAST*"), reverse=True):
            for exe in sorted(d.glob("VAST*.exe"), reverse=True):
                return str(exe)
    return ""


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])
