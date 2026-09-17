# Changelog

All notable changes to FEABAS Workbench. The format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.3.3] – 2026-09-17

### Added
- Log dock: a three-level detail switch instead of the *only warnings/errors* box — *messages only*
  (the workbench's own lines plus every process error), *messages + warnings*, *full log*. The choice
  is remembered in the settings.
- *Help → Workbench manual* (F1) opens the user guide; the HTML guide is now bundled in the package
  (`feabas_workbench/resources/user_guide.html`, written by `tools/build_guide_html.py` alongside
  `docs/user_guide.html`), so it opens offline from a wheel or the frozen build; the online guide is
  the fallback. *Help* is now: Workbench manual, Workbench on GitHub · FEABAS on GitHub, FEABAS paper
  · About.
- The first log line and the About box name the folder the app runs from, so an old checkout
  shadowing a newer install (`python -m feabas_workbench` from inside it) is visible at once.

### Changed
- Every process the workbench starts gets `OPENCV_LOG_LEVEL=ERROR` unless the environment sets it:
  OpenCV's WARN lines about unknown private TIFF tags (34682/34683 in Thermo Maps tiles) were printed
  once per tile per step and drowned the log.

### Fixed
- `tools\install.bat`: the micromamba download on a PC without any package manager now calls Windows'
  own `curl.exe` and `tar.exe` in `System32` by full path. With a GNU `tar` earlier on `PATH` (Git Bash,
  MSYS2, Cygwin) the extraction failed with "Cannot connect to C: resolve failed".

## [0.3.2] – 2026-09-17

### Added
- Published on PyPI: `pip install feabas-workbench`. The release workflow uploads the wheel and sdist
  through PyPI trusted publishing after the GitHub Release; README links are made absolute for the
  PyPI page (`tools/absolutize_readme.py`).
- The FEABAS paper (Wu & Lichtman, 2026, bioRxiv, doi:10.64898/2026.06.07.730510) is cited in the
  README, the user guide, `CITATION.cff`, the third-party notices and the app's About box; *Help* has
  links to the FEABAS repository, the paper and the workbench repository.

### Changed
- README and user guide brought in line: how to obtain the workbench (Windows exe, wheel from the
  Releases page, source; not on PyPI), the fresh-PC routes in both, the manual environment commands in
  the guide instead of a dangling reference to the README, `--selfcheck` in the flag table, current CI
  description, and where the fine compare distance / tissue settings / interface switch are stored.

## [0.3.1] – 2026-09-16

### Fixed
- Setup → *Install fw-feabas* / *Install fw-dl* failed at the pip step when run from the frozen
  Windows exe (it tried to use the exe as a Python interpreter). The pip installs now run inside the new
  environment through `micromamba run -n …` / `conda run --no-capture-output -n …`, with streamed output,
  and the environment's interpreter is looked up afterwards.
- `tools\install.bat` on a PC without any conda/mamba/micromamba now downloads a standalone micromamba
  (with Windows' own `curl` and `tar`) instead of giving up.
- User guide: what to do on a fresh Windows PC without Python (exe route and installer route).

## [0.3.0] – 2026-09-16

### Added
- Synthetic demo dataset and project (`python -m feabas_workbench.core.synthetic DIR`) and an
  end-to-end runner (`tools/run_demo_pipeline.py`) that drives every FEABAS step on it; CI now runs
  the whole pipeline on Linux on every commit and renders the page screenshots on a real project.
  The Project-page smoke test runs on the synthetic tiles instead of being skipped.
- `core.pipeline` with the ordered standard step list, shared by the runner and the GUI.
- Frozen Windows build (PyInstaller): built and self-checked in CI on every push, attached to each
  GitHub Release as `FEABAS-Workbench-<tag>-windows-x64.zip`; `feabas-workbench --selfcheck report.json`
  verifies any installation without opening a window.
- `ruff check` (pyflakes level) in CI; all README screenshots regenerated consistently.
- *Pipeline → Run the standard pipeline…* (Ctrl+R): queues every not-yet-done step of a plain run
  (stitch, thumbnails, coarse and fine alignment, optionally the PNG render) in order and stops at the
  first failure. The dialog shows the mip levels the run will use and can stop after the thumbnails so
  masks (folds, tissue) can be made by hand before the alignment. CI runs it through the real job
  queue on the synthetic dataset (`tools/run_demo_pipeline_gui.py`).
- Every CI run offers the zipped frozen Windows build as an artifact (kept for a week).

### Changed
- Alignment page: *Quality check* is step 3; *Test on subset* is an optional, unnumbered tab (it comes
  before a real run, not after it).

### Fixed
- Thumbnail mip level 0 (small sections) with the default high-pass filter made FEABAS's thumbnail
  step die with an AssertionError; the workbench now switches the high-pass off at mip 0 wherever
  it sets the mip (Project page, Masks page, synthetic projects) and says so.
- Expected pair counts of the fine and coarse matching steps follow the compare distance everywhere.

## [0.2.0] – 2026-09-16

### Added
- Masks: two new tissue methods — *detect a uniform frame (black / white) from the outside in* (peels
  black padding, a saturated white rim or any flat grey layer by layer; thin streaks of the frame colour
  stay tissue so they can be labelled as folds) and *fixed margins from the image edge* with a live
  preview. White regions inside the section can be excluded like black ones.
- Fine alignment has its own compare distance (written to `align/match_name.txt`, FEABAS's pair list,
  refreshed before every fine step).
- Setup → Interface switch for the experimental structure-guided alignment tab (hidden by default).
- `tools/install.bat` / `install.sh` record the interpreter they installed into in
  `start_gui.local.bat` / `.sh`; the launchers read it first and fall back to discovery.
- `CITATION.cff`, `NOTICE`, `CHANGELOG.md`; a release workflow that builds the wheel and attaches it to
  the GitHub Release when a `v*` tag is pushed; Dependabot for the GitHub Actions.

### Changed
- Masks page: one method selector that shows only the chosen method's settings, tooltips on every
  control, clean-up settings renamed to what they do (*drop tissue pieces smaller than px*, *fill holes
  up to px*), folds card hides irrelevant rows, tabs numbered, viewer buttons in two rows.
- Alignment page: tabs numbered in workflow order; tooltips on all coarse and fine settings.
- Preprocessing: the denoising card shows the everyday controls, the rest under *Advanced settings*.
- Launchers and installers search conda / mamba / micromamba in many more places (`PATH`,
  `CONDA_EXE` / `MAMBA_EXE`, install roots on `C:`, `D:`, `E:`), `--envs` and `FW_DEBUG=1` help.
- Progress of *Make thumbnails* counts the stitched-section mip levels (the slow part) as well as the
  thumbnails; full FEABAS runs report absolute output counts, so a re-run of a half-done step starts
  the bar half full.
- Projects store the bundled fold checkpoint as `"bundled"` instead of an absolute path, so they
  survive a move to another machine or a different install location.
- License changed from MIT to Apache-2.0 with a NOTICE file (author attribution travels with every
  redistribution). FEABAS stays MIT.

### Fixed
- Log panel: *autoscroll* off now really keeps the view where it is.
- Thumbnail progress bar stuck at 0 during mip-mapping.
- `start_gui.bat` from the repository did not find environments under `%USERPROFILE%\micromamba\envs`.
- `install.bat` re-created an existing environment with micromamba (its `env list` format differs
  from conda's) and did not find a conda on another drive.
- Training progress lines glued onto tqdm output were not recognised (N2V training showed no progress).
- Process abort at exit ("QThread: Destroyed while thread is still running") when a window was closed
  within 200 ms of start.

## [0.1.0] – 2026-09-14

First public version: project setup, preprocessing (histogram matching, N2V denoising), stitching,
masks with the bundled fold U-Net, coarse and fine alignment with sandbox test runs, structure-guided
alignment, export to VAST / Neuroglancer precomputed.
