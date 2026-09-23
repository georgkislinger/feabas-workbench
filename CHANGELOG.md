# Changelog

All notable changes to FEABAS Workbench. The format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.3.3+cluster.5] – 2026-09-23 (local build)

### Added
- The home folder is offered as project storage next to assigned DSS containers; its quota
  is read from `dssusrinfo`. Setup estimates the project's storage need (about 5× the raw
  images, plus one copy per preprocessing variant and the FEABAS environment) and warns
  before a folder that is too small is used.
- Images, job inputs, previews, exports and rendered stacks can travel directly over SSH:
  resumable, size-checked, never deleting at either end, no Globus Connect Personal needed.
  Default for home-folder projects and datasets below ~200 GB; Globus stays available.
- **Leave cluster mode** in the top bar, the Pipeline menu and the setup window. Submitted
  jobs keep running at LRZ; monitoring pauses until cluster mode is chosen again.
- The Sync button becomes **Stop transfer** while an SSH copy runs.

### Fixed
- **Prepare FEABAS at LRZ** creates a Miniforge Python 3.11 environment (LRZ's python modules
  stop at 3.8, its default python3 is 3.6) with headless OpenCV for compute nodes.
- Leaving cluster mode no longer refuses while a job or Globus transfer is active, including
  a transfer left unconfirmed; only an operation in progress has to finish.
- Closing the LRZ sign-in dialog, declining the host key or stopping a copy is a status line,
  not a failure; error dialogs show the message, the traceback goes to the log only.
- After signing in, the first storage entry is selected; before, the list stayed unselected
  and "Use this storage folder" received an empty path.
- "Sync project & images" and "Save cluster settings & use cluster" showed an underlined
  letter instead of "&".
- Bulk preview downloads compared local files with the new remote manifest instead of the
  previous one, so previews updated by a later run were kept as if they were local edits.

### Removed
- The cluster.1 "Run on cluster" window (`ui/cluster_dialog.py`), unreachable since cluster.2
  replaced it with the setup window and per-project cluster mode.

## [0.3.3+cluster.4] – 2026-09-22 (local build)

### Fixed
- Query supported SSH authentication methods after verifying the host, before
  asking for credentials. Interactive password/MFA login no longer asks for an
  unused extra password before the server's own challenges.

### Validation
- Added a real local SSH server test for interactive-only password/MFA; password
  plus MFA, encrypted-key plus MFA, and changed-host-key rejection still pass.
- 134 regression tests passed. Live LRZ evidence is recorded separately in
  LRZ_TEST_RESULTS.json; local tests do not establish account access.

## [0.3.3+cluster.3] – 2026-09-19 (local build)

### Fixed
- DSS discovery only offers assigned container paths, excluding mounted filesystem
  roots and home directories. An empty container list is distinguished from an
  unrecognized or failed discovery report.
- The guided storage selector rejects filesystem roots, home and scratch paths.

### Validation
- All 133 automated tests passed, including 17 storage regression cases based on
  the reported empty-container output. See the separate live LRZ access report
  for account-specific results; this test does not run FEABAS or upload images.

## [0.3.3+cluster.2] – 2026-09-19 (local build)

### Added
- Persistent per-project cluster mode: the normal Run buttons use LRZ; green accents
  and an execution banner identify the selected backend.
- Guided SSH/MFA, DSS discovery, private FEABAS environment setup, official Globus
  browser login, verified synchronization, transfer recovery and return downloads.
- Bounded section concurrency combined with workers inside each section, Teramem
  resource support and observed memory reporting.
- Remote preprocessing, mask workers, shared settings, previews, models, logs and
  versioned exports; active jobs survive closing and reopening the interface.

### Fixed
- Missing transfer dependency in source launch environments and the packaged Windows app.
- Portable paths for mirrored Windows results, stale downstream results after settings
  changes, large input staging, and preservation of local edits during preview refresh.
- Empty preprocessing inputs now fail clearly; remote preprocessing selection does
  not fall back to raw data because its outputs are absent from the PC.

### Validation
- 116 tests passed; the full 13-command synthetic FEABAS pipeline passed with two
  simultaneous sections and two workers per section. Live LRZ verification is pending.

## [0.3.3+cluster.1] – 2026-09-18 (local build)

### Added
- Separate **Pipeline → Run on cluster…** window: LRZ SSH/MFA, portable input snapshots,
  single-node Slurm resource presets, preview, submission receipts, job history, status,
  cancellation, logs and explicit small-file result downloads. Workstation controls remain local.
- Cluster setup/data-transfer guide. Raw image data stays out of small job uploads.
- Offline tests including a local SSH server exercising password/key plus MFA.

### Fixed
- Source launchers now report missing cluster dependencies in their selected environment.
  Opening the cluster window without Paramiko shows a repair command for that exact Python
  instead of an unhandled import error; workstation pages remain usable.
- A missing executable could deadlock the local job queue. Log-file creation failures now also
  finish the job and let queue error handling run.
- Headless pipeline timeouts now stop silent subprocess trees as well as processes that print output.
- Windows packaging isolates DLL discovery from unrelated programs on PATH; an incompatible
  Poppler ICU library could otherwise make the bundled Qt interface fail to start.
- The module/frozen entry point now returns failures to the caller, so a failed self-check
  no longer reports process exit code zero.

This is a local modification of upstream 0.3.3, not an upstream release. Live LRZ execution
requires the user's account, DSS allocation and FEABAS environment and has not been verified.

## [0.3.3] – 2026-09-17

### Added
- CI tests Python 3.13 and 3.14 as well (lint, tests, offscreen render, wheel), on Ubuntu and Windows.
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
