# Third-party notices

The workbench itself is Copyright 2026 Georg Kislinger and licensed under the Apache License 2.0
(`LICENSE`); its author attribution is in `NOTICE`. What follows are the notices of bundled third-party
software.

## FEABAS

This repository vendors the driver scripts, tools and default configuration files of
**FEABAS 3.0.5**, unmodified, under `feabas_workbench/vendor/feabas_3_0_5/`. FEABAS does the actual
stitching and alignment; the workbench is a front end for it.

* Project: <https://github.com/YuelongWu/feabas>
* Paper: Wu, Y. & Lichtman, J. W. (2026). *FEABAS: A Stitching and Alignment Tool for Serial EM Data.*
  bioRxiv, <https://doi.org/10.64898/2026.06.07.730510>
* Copyright (c) 2022 Yuelong Wu, Center for Brain Science, Harvard University
* License: MIT – see [`feabas_workbench/vendor/feabas_3_0_5/LICENSE`](feabas_workbench/vendor/feabas_3_0_5/LICENSE)

FEABAS itself is installed separately, as a Python package, into an environment of your choice; the
workbench never modifies that installation. `feabas_workbench/vendor/winfix/sitecustomize.py` applies two
run-time fixes to FEABAS 3.0.5 from the outside (see the comments in that file).

## The Windows build

The wheel and the source installs contain only the workbench: pip installs each dependency on its
own, with its own license files. The frozen Windows build (`FEABAS-Workbench.exe`, made with
PyInstaller from `tools/feabas_workbench.spec`) is different: it carries a Python interpreter and the
GUI's dependencies. The ones that matter for redistribution:

| Component | License |
| --- | --- |
| Python | PSF License 2.0 |
| Qt 6 (QtCore, QtGui, QtWidgets), through PySide6 and shiboken6 | LGPL-3.0 (see below) |
| paramiko | LGPL-2.1 (see below) |
| OpenCV (`opencv-python-headless`) | Apache-2.0; contains FFmpeg (LGPL-2.1) and other libraries listed in `cv2/LICENSE-3RD-PARTY.txt` |
| Globus CLI 3.43, Globus SDK 4.9, requests, cryptography, bcrypt, PyNaCl, packaging | Apache-2.0 (cryptography and packaging: Apache-2.0 or BSD) |
| NumPy, SciPy, h5py, tifffile, imagecodecs, psutil, click, idna, pycparser, invoke | BSD |
| Pillow | MIT-CMU |
| PyYAML, urllib3, charset-normalizer, PyJWT, jmespath, cffi | MIT |
| certifi | MPL-2.0 |

Several of these contain compiled libraries of their own (OpenBLAS in NumPy and SciPy, HDF5 in
h5py, image codecs in Pillow and imagecodecs, OpenSSL in cryptography, libsodium in PyNaCl); each
package's license files list them. The build keeps every package's metadata folder
(`_internal/<name>-<version>.dist-info`), which holds its license files where the package ships
them, and `cv2/LICENSE*.txt`; the `_internal/LICENSES` folder holds the GNU license texts below.

**Qt and PySide6 (LGPL-3.0).** The workbench uses Qt only through PySide6's dynamically loaded
libraries (`_internal/PySide6/*.dll`, `*.pyd`, `_internal/shiboken6/`); you may replace them with
other, interface-compatible builds of the same Qt/PySide6 version. The texts of the LGPL-3.0 and of
the GPL-3.0 it builds on are in `LICENSES/LGPL-3.0.txt` and `LICENSES/GPL-3.0.txt`. The version in a
build is the one its PySide6 `*.dist-info` folder names; the corresponding sources are at
<https://download.qt.io/official_releases/QtForPython/> (PySide6, shiboken6) and
<https://download.qt.io/official_releases/qt/> (Qt).

**paramiko (LGPL-2.1).** paramiko is included as plain Python source files
(`_internal/paramiko/*.py`), not inside the build's compiled archive, so it can be read, changed or
replaced in place. License text: `LICENSES/LGPL-2.1.txt`; project: <https://github.com/paramiko/paramiko>.

**Globus CLI and Globus SDK** provide the browser authentication and the verified transfers of
cluster mode. Projects: <https://github.com/globus/globus-cli> and
<https://github.com/globus/globus-sdk-python>.
