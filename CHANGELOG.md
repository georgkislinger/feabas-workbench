# Changelog

All notable changes to FEABAS Workbench. The format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

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
