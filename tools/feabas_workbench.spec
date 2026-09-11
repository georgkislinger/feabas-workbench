# PyInstaller spec for the GUI only (FEABAS and deep-learning environments stay external).
#   pip install pyinstaller
#   pyinstaller tools/feabas_workbench.spec
# Produces dist/FEABAS-Workbench/FEABAS-Workbench.exe. The vendored FEABAS scripts,
# default configs and worker modules are bundled as data so the app can run them
# in the configured environments.

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

root = Path(SPECPATH).parent
pkg = root / "feabas_workbench"

# Plain-file copies of the parts workers need (core.jobs.worker_package_root stages them into a
# clean folder at run time, since the bundle folder itself is full of compiled packages) plus the
# bundled fold U-Net weights.
datas = [
    (str(pkg / "vendor"), "feabas_workbench/vendor"),
    (str(pkg / "workers"), "feabas_workbench/workers"),
    (str(pkg / "core"), "feabas_workbench/core"),
    (str(pkg / "resources"), "feabas_workbench/resources"),
    (str(pkg / "__init__.py"), "feabas_workbench"),
]
datas += collect_data_files("cv2", include_py_files=False)

a = Analysis(
    [str(root / "feabas_workbench" / "__main__.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=collect_submodules("feabas_workbench") + ["tifffile", "imagecodecs", "h5py", "psutil", "yaml", "cv2", "scipy", "PIL"],
    hookspath=[],
    excludes=["torch", "torchvision", "careamics", "ultralytics", "tensorstore", "matplotlib", "IPython", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="FEABAS-Workbench",
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="FEABAS-Workbench")
