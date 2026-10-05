# Changelog

All notable changes to FEABAS Workbench. The format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.3.8] – 2026-10-05

### Added – segmentation masks follow the alignment
- For masks drawn on an image stack whose alignment you improve: import the stack as one image per
  section, align it, and **Segmentation masks (labels)** (Export & view → Render) renders the masks
  through exactly the transforms the images went through. Each mask is placed with its section's
  stitching transform and moved by its alignment mesh onto the aligned stack's canvas, with
  nearest-neighbour sampling throughout and none of the images' intensity processing. Every output
  value is one of the input labels.
  - **Input:** 8- and 16-bit greyscale PNG or TIFF, one per input image, matched by file name or in
    section order. Masks that don't fit (size, bit depth, count, format) are refused with the reason
    before anything is rendered.
  - **Output:** the aligned PNG stack's layout (same tiles, names and section folders at every mip
    level), with mipmaps that keep each 2×2 block's majority label.
  - **Export and checking:** export it like the images (VAST tiles, OME-Zarr, or the new one image per
    section), and check it as a colour overlay under View aligned sections. The status line says when
    the alignment or the mask files changed after rendering.
- Export: *one image per section* writes whole sections at a chosen mip level (PNG or TIFF, 8- or 16-bit)
  for tools that import image sequences; the export tab can export an aligned mask stack as well as the
  images.

### Fixed
- The VAST export always wrote `SourceBytesPerPixel: 1`; a 16-bit stack now gets 2.
- The end-to-end check reported "0 matched tile pairs" for every section of an image stack (one image
  per section), which has no tile pairs to match. FEABAS itself handles such sections fine.

### Packaging
- Public version 0.3.8 for the Windows executable, wheel, source archive and PyPI package.

### Validation
- With FEABAS 3.0.5, masks made from the input images themselves line up with the aligned images.
  The tile layout is identical at every mip level, and the remaining offsets are within the
  nearest-neighbour ±0.5 px: ≤ 0.31 px for one image per section, ≤ 0.7 px (in y) with overlapping
  tiles, ≤ 0.14 px at mip 2 thanks to the majority mipmaps.
- 16-bit labels come out exactly; no new values at any level.
- CI carries both 8-bit images and 16-bit labels through the alignment and checks them, for 2×2 tiles
  per section and for an image stack with masks matched in order.
- The Windows and Linux test matrix (Python 3.10–3.14) and both FEABAS end-to-end runs passed; the
  release wheel was installed and started in a clean environment.

## [0.3.7] – 2026-10-02

### Changed – mipmaps use the whole PC
- *Make thumbnails* and *Mipmaps for PNG stack* left a workstation mostly idle (~1% CPU on 64
  threads, one section a minute). FEABAS's own way mipmaps one section at a time, and within it every
  mip level is a new pool of worker processes with only as many jobs as the level has tiles (25, 9,
  4, 4, … for 64 tiles; ~40 short-lived processes per section). Their new Local parallelism default,
  **Automatic**, hands whole sections to the workers when there are enough sections (FEABAS's own
  `parallel_within_section: false`), one per core of the CPU budget as far as RAM allows. It also
  raises the tile read cache from 4 to 16 tiles, which stops the re-reading of each source tile. The
  output is the same file for file. On 4 cores and four 64-tile sections the step took 170 s instead
  of 325 s; a 32-core PC gains far more. Settings chosen under *downsample* are kept, the log says
  what each run used, and *Existing FEABAS settings* runs FEABAS exactly as configured. The values
  reach FEABAS through the job environment only; the project's YAML is not touched.

### Fixed
- *Cancel* could leave a half-written PNG/JPEG tile in a montage, the aligned stack or a mipmap
  level, and FEABAS keeps every tile file that exists when it renders a level again. Tiles cut short
  by the kill are now removed with the other incomplete outputs, so the next run makes them again.

### Packaging
- Public version 0.3.7 for the Windows executable, wheel, source archive and PyPI package.

### Validation
- The Windows and Linux test matrix (Python 3.10–3.14) and the FEABAS end-to-end pipeline passed, the
  pipeline now with both mipmap steps on Automatic; the parallelism tests passed against the FEABAS
  runtime. With FEABAS 3.0.5, four sections of 64 PNG tiles gave byte-identical mipmaps on every path
  (FEABAS's default, sections side by side, with and without the larger cache). The release wheel was
  installed and started in a clean environment.

## [0.3.6] – 2026-09-28

### Fixed
- Local parallelism never used hyper-threads: the CPU budget was capped at the physical cores even
  when a larger total CPU budget was set, so on a 32-core / 64-thread PC "4 sections × 12 workers"
  ran 2 sections at once. Without a set budget the physical cores remain the default (FEABAS's
  own); a budget you set may now go up to the logical CPUs. The plan says which limit reduced
  the sections at once (CPU budget, RAM or section count) and what would lift it, and the dialog
  shows physical cores and logical CPUs.
- Steps went *stale* right after running. Staleness compared whole config files, and the coarse
  alignment's settings share the thumbnail config: saving any of them (even the worker count) made
  *Make thumbnails* stale, and since 0.3.5 that passed on to the masks, *Match thumbnails* and
  *Optimize coarse stack*, although those had just been run with the new settings. The workbench
  now records which setting changed when; each step reacts only to the settings that decide its
  results, worker counts and cache sizes never count, and the reason names the changed settings.
  Files edited outside the workbench still count as a change for every step that reads them;
  changes made before 0.3.6 are not held against existing outputs. Cluster mode ignores the same
  run-only settings when it compares configurations.

### Packaging
- Public version 0.3.6 for the Windows executable, wheel, source archive and PyPI package.

### Validation
- The Windows and Linux test matrix (Python 3.10–3.14) passed, with regression tests for the
  32-core / 64-thread case and for settings changes after *Make thumbnails*; the parallelism tests passed against the FEABAS runtime; the release
  wheel was installed and started in a clean environment.

## [0.3.5] – 2026-09-28

### Fixed – what "done", "stale" and "Clear" mean
- *Clear…* on a test run's step card cleared the **project's** outputs instead of the test run's.
  It now clears the test run's own folder, and clearing never deletes through a link or junction
  (alignment test runs link the project's rendered montages; the dialog lists what it left alone).
- Clearing *Mipmaps for PNG stack* deleted the whole aligned stack including the full-resolution
  render, and clearing *Mipmaps for volume* deleted the whole volume. Both now remove only the
  downsampled levels (the volume's `info` and `align/ts_spec.json` are cut back to match).
- Output folders set in the configs (`rendering.out_dir` for montages and the aligned stack,
  `tensorstore_rendering.out_dir`) were ignored: those steps never showed as done, and *Clear*
  removed nothing, so FEABAS skipped the old outputs on the next run. Steps now look where FEABAS
  writes (relative paths are relative to the project), at the configured render mip, and clear only
  FEABAS's own folders in a folder you chose.
- *Mipmaps for volume* showed *done* as soon as the volume render started, and *Mipmaps for PNG
  stack* as soon as the first section had one level. They now count the finished mip levels
  (`align/ts_spec.json`) and the sections whose coarsest level is written.
- Pressing *Apply* with nothing changed rewrote the config file and made every output of that stage
  *stale*. Config files are now written only when their content changes; an override file whose
  last key is reset stays as `{}` (FEABAS fails on an empty one).
- Staleness is judged section by section (pair by pair for matches): running an earlier step for a
  few more sections no longer marks the finished ones stale, and a section whose inputs changed is
  no longer missed. Staleness is passed on downstream, rewritten coordinate files make *Match tiles*
  stale, and a new coarse solution makes the meshes stale (FEABAS starts them from it).
- *Run* on a stale step did nothing (FEABAS skips outputs that exist). It now offers *Clear, then
  run* or *Run without clearing*.
- Restoring a snapshot rewrote every file, which hid what it changed. Only differing files are
  written now, so outputs computed later from other inputs show as stale; snapshot paths are checked
  against the list of snapshotted folders.

### Fixed – projects, settings and test runs
- A copied, moved or renamed project (and any test run in it) still pointed FEABAS's
  `working_directory` at the original folder, so FEABAS read and wrote the original project. The
  path is repaired when a project is opened and before every FEABAS step or tool.
- A corrected pixel size had no effect: FEABAS caches the first run's resolution in
  `configs/resolutions.yaml`. The cache is removed when new coordinate files carry another value
  (in cluster mode too).
- Writing the coordinate files overwrote thumbnail and working mip levels chosen on the Masks and
  Alignment pages, and every thumbnail mip change also rewrote `meshing.mask_mip_level`, so FEABAS
  read existing high-resolution masks at the wrong scale. Suggested mips now replace only values the
  workbench set itself; the mask mip belongs to `align/material_masks` alone.
- The compute settings on the Setup page (CPU budget, parallel framework, log level) were not
  written by *Apply*.
- Creating a montage test run wrote the quick settings into the project's configs (making project
  outputs stale); they now go into the test run's copy only. Test runs no longer inherit the
  project's configured output folders, and alignment test runs link the montages from wherever the
  project renders them.
- *Open project* turned any folder into a project without asking, and a missing last or recent
  project was silently recreated empty (e.g. a disconnected drive). Both now ask or refuse; *New
  project* warns about a non-empty folder.
- Duplicate names in `section_order.txt` listed a section twice.

### Fixed – masks
- *Split section* painted the line as *exclude* (255), leaving a blank strip in the output; it now
  uses FEABAS's split material (200) as documented.
- Masks made for earlier thumbnails (thumbnails made again at another mip) crashed composing or were
  resized silently: the footprint backup is rebuilt, a tissue mask older than its thumbnail is
  computed again (imported ones resampled), and stale fold masks are left out after asking.
  *Clear…* on *Make thumbnails* removes the masks drawn on them.
- Composing read widget values from the worker thread; hole filling and border bands are
  vectorized (they looped over every connected component); mask temp files no longer look like
  masks to the section listing.

### Fixed – tiles, jobs and robustness
- TIFFs with a print resolution (72 or 300 dpi) were read as pixel sizes of hundreds of
  micrometres; whole-number dpi values are ignored, ImageJ units are honoured and implausible sizes
  rejected.
- A project folder inside the tile folder made tile scans count its preprocessed copies as more raw
  tiles.
- Switching between raw and preprocessed tiles ignored coordinate files written with absolute paths,
  and switched to incomplete or missing folders without a word.
- A cancelled FEABAS step could leave a truncated `.h5` that FEABAS then skipped as finished;
  unreadable outputs written by the cancelled run are removed. Histogram matching and denoising
  write each tile under a temporary name first.
- Two jobs queued in the same second shared one spec file.
- The staged worker package was replaced in place, under workers still importing it (a second
  installation or version); each version now gets its own folder. `settings.json` is written
  atomically.
- The FEABAS run-time patch on `PYTHONPATH` shadowed an environment's own `sitecustomize`; it now
  runs it too.
- Closing the window while a background task was just finishing could crash the application:
  stopping a task released its worker before the worker's thread had ended.

### Fixed – setup and export
- RTX 50-series (Blackwell) GPUs were given CUDA 12.6 PyTorch builds, which cannot run on them;
  *auto* now reads the compute capability and picks the CUDA 12.8 build (driver ≥ 570), which is
  also selectable.
- Installs forced `--index-url https://pypi.org/simple`, overriding pip mirrors and proxies (also in
  `tools/install.sh` / `install.bat`); only PyTorch keeps its dedicated index.
- *Download micromamba* and *Check selected* froze the window; both run in the background. The
  download takes the official release binary and checks it against its published SHA-256.
- A failed environment install is reported; the install handler is connected once.
- Export and the aligned-stack viewer assumed the stack was rendered at mip 0: a stack rendered at
  a coarser `rendering.mip_level` now exports with that level as full resolution (voxel size scaled).
- Structure detection progress assumed a `structures.json` existed; structure sources and the
  quality-check viewer use the configured montage folder.

### Tests, CI and packaging
- `tests/test_safety.py`: 42 regression tests for the fixes above, core and offscreen GUI.
- The end-to-end run now checks the matches themselves (every montage connected, several coarse and
  fine match points per pair). FEABAS's default mesh and matching grids left one fine match point
  per section pair on the synthetic sections, so the demo project scales them to its section size.
- CI also tests Python 3.10, the oldest version `pyproject.toml` accepts.
- `THIRD_PARTY_NOTICES.md` covers what the Windows build carries (Qt/PySide6 under LGPL-3.0,
  paramiko under LGPL-2.1, and the rest); the build includes the GNU license texts (`LICENSES/`)
  and each package's license files, and keeps paramiko as replaceable source files.

### Packaging
- Public version 0.3.5 for the Windows executable, wheel, source archive and PyPI package.

### Validation
- The Windows and Linux test matrix (Python 3.10–3.14) passed: 208 tests on Linux, 3 skipped
  there (a Windows-only test and the two that need the separate FEABAS runtime, which run in the
  end-to-end job).
- The synthetic FEABAS pipeline passed run directly and through the GUI's job queue, now with the
  match checks (14–17 fine match points per section pair); the frozen Windows application passed
  its self-check and rendered every page; the release wheel was installed and started in a clean
  environment.
- Full FEABAS image processing at LRZ remains unvalidated.

## [0.3.4] – 2026-09-25

### Added
- Bring the cluster-mode development builds into the public release: guided LRZ setup,
  SSH or Globus data transfer, persistent remote job monitoring, previews and export downloads
  through the usual workbench controls. See the cluster.1–cluster.6 entries for details.
- Local section/worker parallelism with CPU and RAM planning; pixel-based N2V training
  selection; live training/validation loss and best-patch previews with graceful stopping.

### Fixed
- Run the FEABAS-dependent parallelism tests in the dedicated runtime CI job; GUI-only
  installations skip those two integration checks because FEABAS lives in a separate environment.

### Packaging
- Use public version 0.3.4 for the Windows executable, wheel, source archive and PyPI package.
- Full FEABAS image processing at LRZ remains unvalidated; live training previews and graceful
  N2V stopping currently apply to local execution.

## [0.3.3+cluster.6] – 2026-09-24 (development build)

### Added
- Local parallelism controls for tile matching, montage optimization/rendering, stitched
  mipmaps/thumbnails, aligned PNG section rendering and aligned PNG mipmaps. Choose workers
  within sections, across sections, or both; CPU/RAM estimates limit simultaneous sections.
  Project settings and section logs remain separate from cluster resource settings.
- N2V training selection based on usable pixels instead of a minimum number of files, with
  pixel-based auto-selection. A sufficiently large single image is accepted; training and
  validation use non-overlapping spatial regions with recorded patch locations.
- Live training and validation loss curves and a fixed held-out original/best-denoised patch
  comparison for local N2V-family training. Stop after the current epoch while retaining the
  checkpoint with the lowest validation loss.

### Fixed
- Clearing the denoising training selection now persists; model selection prefers the saved
  best checkpoint. Existing training runs are protected by requiring a new run name.
- Parallel thumbnail phases stay on the same named section when other mipmaps are incomplete.
  Aligned rendering initializes the shared canvas once before processing sections in parallel.
- A failed parallel section stops sibling and pending work and reports the original failure.
- Disable CAREamics' console progress bar at construction so epoch progress messages remain
  readable by the GUI. Large training images are sampled one at a time with bounded patch arrays.

### Validation
- 168 regression tests passed; one Linux-only test skipped on Windows.
- A complete synthetic parallel pipeline, including a partial-section thumbnail run, passed.
- Short CPU N2V and N2V2 runs verified both loss curves, best-model previews, graceful stopping,
  checkpoint reload and prediction. RAM planning remains approximate; live denoising feedback
  and graceful stopping are currently local-mode features.

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
- **Optional: prepare deep-learning tools at LRZ**: the counterpart of the local deep-learning
  environment (CPU PyTorch, segmentation-models-pytorch, ultralytics, careamics), used by fold
  detection, YOLO and Noise2Void at LRZ.
- **Free this project's LRZ storage…**: shows the folder's size and which exports are on this PC,
  then deletes only the folder the storage step created (optionally the environments too).
- The setup window's log keeps a history and streams environment installs line by line; it
  shares a draggable splitter with the tabs.

### Fixed
- **Prepare FEABAS at LRZ** creates a Miniforge Python 3.11 environment (LRZ's python modules
  stop at 3.8, its default python3 is 3.6) with headless OpenCV for compute nodes.
- Leaving cluster mode no longer refuses while a job or Globus transfer is active, including
  a transfer left unconfirmed; only an operation in progress has to finish.
- Jobs failed at once on CoolMUC-4 with "module: command not found": batch shells do not
  define `module`. job.sh no longer loads modules unless some are chosen under Advanced, and then
  initialises the module system itself.
- The setup window fits the screen it opens on and its tabs scroll; on a 1280x800 display at
  200 % its bottom buttons were off-screen and it could not be made smaller.
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
