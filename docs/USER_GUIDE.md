# FEABAS Workbench – user guide

Every window, every setting: what it does, when to touch it, and what it breaks if you change it later.
If you only want the short version, read [Installing and starting the workbench](#1-installing-and-starting-the-workbench) and
[A first dataset, end to end](#10-a-first-dataset-end-to-end).

**Contents**

1. [Installing and starting the workbench](#1-installing-and-starting-the-workbench)
2. [How the workbench thinks](#2-how-the-workbench-thinks) – project folder, step states, where settings live
3. [Window 0 – Setup](#3-window-0--setup)
4. [Window 1 – Project & data](#4-window-1--project--data)
5. [Window 2 – Preprocessing](#5-window-2--preprocessing-optional)
6. [Window 3 – Stitching](#6-window-3--stitching)
7. [Window 4 – Masks](#7-window-4--masks)
8. [Window 5 – Alignment](#8-window-5--alignment)
9. [Window 6 – Export & view](#9-window-6--export--view)
10. [A first dataset, end to end](#10-a-first-dataset-end-to-end)
11. [Changing your mind: stale, clear, snapshots](#11-changing-your-mind-stale-clear-snapshots)
12. [Troubleshooting](#12-troubleshooting)
13. [Where every setting is stored](#13-where-every-setting-is-stored)

---

## 1. Installing and starting the workbench

### 1.1 Get the code

The workbench is distributed from GitHub for now (a PyPI package will follow). Either

* **Download a ZIP**: open the repository page, click the green **Code** button → **Download ZIP**, and
  unpack it somewhere permanent – the folder you unpack is the one you will start the app from. Avoid
  paths with spaces or non-ASCII characters; they still work, but they make every later command harder
  to type.
* **Clone it** if you have git: `git clone https://github.com/<owner>/<repo>.git`. Cloning makes later
  updates a single `git pull`.

The download is small (about 50 MB, most of it the bundled fold-detection network). The example dataset
is **not** included.

### 1.2 Install – pick the row that matches your machine

**A fresh Windows PC with nothing on it** (no Python, no conda) works either way:

* **Without installing anything:** download `FEABAS-Workbench-<version>-windows-x64.zip` from the
  [Releases page](https://github.com/georgkislinger/feabas-workbench/releases), unpack it anywhere and
  start `FEABAS-Workbench\FEABAS-Workbench.exe`. It contains its own Python, Qt and image libraries.
  Windows SmartScreen will warn once about an unsigned program from the internet — *More info → Run
  anyway*. Then, on the **Setup** page: *Download micromamba* → *Install fw-feabas* → *Install fw-dl*
  (internet needed; the deep-learning one is several GB). Those two environments are where FEABAS and
  the models run; the exe is only the workbench.
* **From the source zip:** double-click `tools\install.bat`. If the PC has no conda, mamba or
  micromamba at all, it downloads a standalone micromamba (10 MB, using the `curl` and `tar` that are
  part of Windows 10/11) into `%APPDATA%\FeabasWorkbench\micromamba`, creates the `feabas-workbench`
  environment, installs the workbench and starts it. FEABAS and deep-learning environments as above.


The GUI itself needs Python 3.10 or newer with PySide6 and a few scientific packages. The heavy parts
(FEABAS, PyTorch) live in *separate* environments that the app creates for you afterwards (§1.3), so
the choice here only concerns the GUI.

| You already have | Do this |
|---|---|
| **Nothing** – no Python, no conda | Install **Miniforge** from <https://conda-forge.org/download/> (accept the defaults; on Windows tick *Add to PATH* if offered, it is not required). Then follow the *conda* row. Miniforge is the recommended starting point because the FEABAS and deep-learning environments are conda environments too. |
| **micromamba, Miniforge, Miniconda or Anaconda** | Windows: double-click **`tools\install.bat`**. Linux/macOS: `bash tools/install.sh`. The script finds a package manager (micromamba and mamba first, then conda: on `PATH`, in `CONDA_EXE` / `MAMBA_EXE`, and in the usual install folders on `C:`, `D:` and `E:`; `set FW_CONDA=…` forces one, `set FW_DEBUG=1` shows the search), creates an environment called `feabas-workbench` unless it exists, installs the app into it, **records that interpreter in `start_gui.local.bat`** so the launcher is hard-wired to this installation, puts a shortcut on the Windows Desktop and starts the GUI. Later, start it with `start_gui.bat` / `./start_gui.sh`. |
| **Plain Python 3.10+** (python.org, Microsoft Store, your distribution) but no conda | In the unpacked folder: <br>Windows: `python -m venv .venv` then `.venv\Scripts\python.exe -m pip install -e .` <br>Linux/macOS: `python3 -m venv .venv` then `.venv/bin/python -m pip install -e .` <br>Afterwards `start_gui.bat` / `./start_gui.sh` finds the `.venv` automatically. The Setup page will offer to **download micromamba** – a single-file conda – when it needs to create the FEABAS and deep-learning environments, so you never have to install conda yourself. |

Linux only: PySide6 needs a handful of system libraries that minimal server installs lack. On
Debian/Ubuntu: `sudo apt install libegl1 libopengl0 libxkbcommon0 libdbus-1-3 libxcb-cursor0 libfontconfig1`.
Desktop installs already have them.

To check the install worked, the test suite runs without any project or data:

```bash
python -m pip install pytest
python -m pytest tests -q
```

### 1.3 The two pipeline environments

On first start the app opens **Setup** (§3). Click **Detect environments on this machine** – if you
already have FEABAS or PyTorch installed somewhere, it finds them. Otherwise:

* **Install fw-feabas** creates the FEABAS environment (feabas 3.0.5 + TensorStore; a few hundred MB).
  Needed for stitching, alignment and export – i.e. for everything.
* **Install fw-dl** creates the deep-learning environment (PyTorch, CAREamics, ultralytics,
  segmentation-models-pytorch; several GB). Needed only for the optional model-based steps: fold
  U-Net, N2V denoising, YOLO structure detection. Pick the PyTorch build for your GPU driver in the
  drop-down, or *cpu* if you have no NVIDIA card – everything still works, just slower.

Both need internet and 10–30 minutes. They can also be created by hand; the commands are in the README.

### 1.4 Starting

Double-click **`start_gui.bat`** (Windows) or run **`./start_gui.sh`** (Linux/macOS) in the unpacked
folder. The script first reads `start_gui.local.bat` / `start_gui.local.sh` next to it — the interpreter
`tools\install.bat` / `install.sh` installed into — and uses that if it still exists (delete the file to
go back to pure discovery). Otherwise it looks, in this order, for `FW_PYTHON`; the environment that is active in the shell;
a `.venv` (or `venv`, `env`) next to the script; a conda / mamba / micromamba environment whose name
contains `feabas` (Miniforge, Miniconda, Anaconda and micromamba roots, `~/.conda/envs`,
`MAMBA_ROOT_PREFIX`, `CONDA_ENVS_PATH`, plus any folders in `FW_ENV_DIRS`); any other environment there
that has PySide6; and finally what `conda env list` / `micromamba env list` and `PATH` report — then it
runs the app. `start_gui.bat --envs` (or `./start_gui.sh --envs`) lists every interpreter that would
work, and `FW_DEBUG=1` shows each candidate as it is tried.

Both scripts forward their arguments, so from a terminal you can also do:

```bat
start_gui.bat --project D:\data\my_stack
```

To force a particular interpreter (e.g. a second environment you keep for testing):

```bat
set FW_PYTHON=C:\Users\me\AppData\Local\miniforge3\envs\other-env\python.exe
start_gui.bat
```

(`FW_PYTHON=/path/to/env/bin/python ./start_gui.sh` on Linux.)

A console window stays open behind the GUI when started this way. That is deliberate: unhandled errors
and the raw FEABAS subprocess output are printed there as well as in the log dock.

Because the installer pip-installs the package, the environment's own launchers work from any folder
too: `feabas-workbench` (no console window) and `feabas-workbench-cli` (with one, for `--screenshot`
and tracebacks), or `python -m feabas_workbench`.

### Command-line flags

| Flag | Use |
|---|---|
| `--project DIR` | open that project folder immediately |
| `--page N` | jump to window N (0 = Setup … 6 = Export) |
| `--screenshot DIR` | render every window to PNG offscreen and exit (for checking layouts) |
| `--offscreen` | use the offscreen Qt platform (no window) |

On Windows set `QT_QPA_FONTDIR=C:/Windows/Fonts` when using `--screenshot`, otherwise the offscreen
platform has no fonts.

### What happens on the first start

* No FEABAS interpreter configured yet → the app opens **Setup** and logs a warning.
* Otherwise it reopens the last project.

Machine-level settings live in `%APPDATA%\FeabasWorkbench\settings.json` (Linux/macOS:
`~/.config/FeabasWorkbench/settings.json`): the two interpreters, the Fiji and VASTlite paths, the conda
executable, the PyTorch wheel index, the recent-project list and the last project. Delete that file to
start clean; no project folder depends on it.

---

## 2. How the workbench thinks

Four ideas explain almost every button.

### 2.1 The project folder *is* the FEABAS working directory

FEABAS reads `configs/general_configs.yaml` from its current working directory, and the workbench runs
every FEABAS step as a subprocess with `cwd = <project folder>`. A project folder therefore holds the
plain FEABAS layout plus a little workbench state:

```
project/
  workbench_project.json          everything the GUI remembers about this project
  configs/                        general_configs.yaml + default_*.yaml + your overrides
  stitch/stitch_coord/            one TSV per section – what FEABAS starts from
  stitch/, stitched_sections/     stitching outputs
  thumbnail_align/, align/        coarse and fine alignment
  aligned_stack/, aligned_tensorstore/   rendered output
  masks/{tissue,folds,structures} workbench intermediates
  models/{n2v,folds,yolo}/        models you train here
  tests/<name>/                   sandbox working directories for subset test runs
  snapshots/<stamp>/              rollback copies
  exports/                        VAST / OME-Zarr exports
  logs/                           FEABAS log files
```

Your raw tiles stay where they are; only the coordinate files point at them. The FEABAS installation is
never modified.

### 2.2 State is read from disk, every time

There is no hidden database. A step is "done" when its output files exist. Each step card shows one of:

| Badge | Meaning |
|---|---|
| **not started** | no outputs |
| **partly done** | some sections have outputs (crash, cancel, or a deliberate subset run) |
| **done** | as many outputs as expected |
| **stale** | outputs exist, but the config file or an upstream step is newer than they are |
| **errors** | FEABAS left `*_err` markers – those sections failed |
| **blocked** | a required earlier step has no outputs yet |

"Stale" is a warning, not a lock: nothing stops you from running a stale step, but the result can mix
old and new settings. The honest fix is **Clear…** on the earliest stale step and re-run.

Press **F5** (Pipeline → Re-read pipeline state) after changing files outside the GUI.

### 2.3 Every step card works the same way

* **Run** – run this step for all sections.
* **subset** – reveals `from` / `to` / `every`, which become FEABAS's `--start / --stop / --step`. Use it
  to process sections 0–99 here and 100–199 on another machine, or to redo one broken section.
* **Inspect** – jump to the quality-check tab for that step.
* **Remove error files** – appears only when `*_err` markers exist; deleting them makes the next run
  retry exactly those sections.
* **Clear…** – deletes this step's outputs **and every later step's outputs**, after showing the list.
  Snapshot first if the old state was expensive.
* **Run all steps below in order** – queues the steps of that panel that are not already *done*.

Only one job runs at a time; further submissions queue. The status bar shows progress and a **Cancel
job** button. The log dock opens on its own while a job runs (or when an error is logged); **View →
Log** (Ctrl+L) shows or hides it, and it carries FEABAS's own messages. Untick **autoscroll** to read
while a job keeps writing: the view then stays on the lines you are looking at (it keeps the last 5000
lines; *only warnings/errors* and the filter box narrow it down).

**One click for the plain case.** *Pipeline → Run the standard pipeline…* (Ctrl+R) lists every step
of a plain run — stitching, thumbnails, coarse and fine alignment, optionally the PNG render and its
mipmaps — with its current state, and queues the ones that are not done yet, in order; the queue stops at
the first failure. Use it once the coordinate files exist and you are happy with the defaults, or after a
*Test on subset* told you which settings to change; the step cards on the pages stay the way to run,
inspect and clear individual steps. Three things to know:

* **Mip levels** are taken from the project's settings, which were chosen for the data when the
  coordinate files were written (thumbnails of about 1000 px; fine matching at the mip whose pixel is
  just below the section thickness). The dialog shows them; change them on the Masks / Alignment pages
  first if they do not suit.
* **Masks:** the pipeline uses FEABAS's default — everything the tiles cover is tissue, **no fold
  detection**. That is right for intact sections. For sections with folds, tears or empty areas tick
  *stop after 'Make thumbnails'*: the run ends after the thumbnails, you make the masks on the Masks page
  (tissue method, fold detection, *Compose*), then run the pipeline again — it skips what is done and
  continues with your masks.
* **Stale steps** (a config or an input changed after their outputs) are left alone: FEABAS never
  recomputes outputs that exist, so queuing them again would do nothing. To redo a step, *Clear…* it on
  its step card (that removes everything after it too), then run the pipeline again.

### 2.4 Settings come in three flavours

| Kind | Stored in | Examples |
|---|---|---|
| Machine settings | `%APPDATA%\FeabasWorkbench\settings.json` | interpreters, Fiji/VAST paths, conda |
| Project settings | `<project>/workbench_project.json` | tile folder, naming rule, voxel size, mask and structure settings |
| FEABAS settings | `<project>/configs/*.yaml` | everything the FEABAS scripts read |

FEABAS merges `configs/default_<name>.yaml` with your `configs/<name>.yaml`. The workbench shows the
merged view and **saves only the differences**, so your override file stays short and every overridden
key is visible as such in the *All settings* tree. Resetting a key deletes the override.

Buttons labelled **Apply** write the config file. Nothing is written while you are only typing.

---

## 3. Window 0 – Setup

Machine-level configuration, done once per workstation. Only *Compute settings* is per project.

### Environments

| Control | What it does | When to touch it |
|---|---|---|
| **FEABAS Python** | `python.exe` of an environment where `import feabas` works (plus tensorstore, numpy < 2). Runs every stitching/alignment/rendering step. | Once, or after rebuilding the environment. |
| **Deep-learning Python** | `python.exe` with torch + careamics + ultralytics + segmentation-models-pytorch. Runs fold detection, YOLO-seg and N2V denoising. | Only needed for the optional deep-learning features. |
| **Detect environments on this machine** | Probes every conda environment it can find and reports what each imports. Fills empty fields automatically, preferring a CUDA-capable DL environment. | First run, or after creating environments by hand. |
| **Check selected** | Re-probes the two chosen interpreters; warns if `feabas` or `torch` is missing. | After editing a path by hand. |
| **Use detected → as FEABAS / as deep-learning** | Copies the environment picked in the drop-down into the field above. | |
| **Save** | Writes the paths (plus Fiji/VAST/conda) to the global settings file. | **Always press this** – the fields are not saved automatically. |

### Install environments

Only needed if you do not have the environments yet.

| Control | Notes |
|---|---|
| **conda** | Path to `conda`, `mamba` or `micromamba`; auto-detected if Miniforge/Miniconda is installed. |
| **Download micromamba** | Fetches a standalone micromamba if you have no conda at all. |
| **PyTorch build** | `auto from driver` picks the wheel index from your NVIDIA driver version; pin CUDA 12.6 / 11.8 / CPU-only if you know better. CUDA 12.6 wheels need driver ≥ 528. |
| **Install fw-feabas** | Creates `fw-feabas` (Python 3.12) with feabas 3.0.5 + tensorstore. |
| **Install fw-dl** | Creates `fw-dl` (Python 3.11) with torch, segmentation-models-pytorch, ultralytics, careamics 0.3.2. Several GB, 10–30 minutes. |

When an install finishes, the corresponding interpreter field is filled in and saved for you.

### Hardware

Read-only: physical cores, threads, RAM, and the GPU with its driver version and the recommended PyTorch
build. Treat the core count as the ceiling for the per-step `workers` settings — FEABAS runs one process
per worker and each caches images, so RAM is usually the real limit. FEABAS itself never uses the GPU;
only the deep-learning workers do.

### External viewers

`Fiji` (`ImageJ-win64.exe`) and `VASTlite` (`VAST_Lite.exe`). Needed for *Edit mask in Fiji*, *Open
section in Fiji* and *Open in VASTlite*. Auto-detected in the usual install locations.

### Interface

**Show the experimental 'Structure-guided' tab on the Alignment page** – off by default. The tab lets a
YOLO-seg model decide where the alignment is driven from (see §8); it is not part of the normal FEABAS
workflow, so it stays out of the way until you switch it on here. The change is immediate and remembered
across sessions.

### Compute settings (this project)

**Apply to project** writes these into `configs/general_configs.yaml`, so a project must be open.

| Setting | Meaning | Recommendation |
|---|---|---|
| **CPU budget** | FEABAS's global hint for how many cores it may use; `0` = all physical cores. | Leave at 0 unless you share the machine. The per-step `num_workers` values are what actually control load. |
| **Parallel framework** | `process` (default), `thread`, `dask`. | Keep `process`. `thread` is for debugging; `dask` is for cluster setups. |
| **Log file level** | `INFO` (default), `DEBUG`, `WARNING`. | Switch to `DEBUG` when a step fails for reasons the log does not explain, then switch back – DEBUG logs get large. |

---

## 4. Window 1 – Project & data

This window turns a folder of tiles into FEABAS's `stitch/stitch_coord/*.txt`. Everything downstream
depends on it, and it is cheap to redo — as long as you have not stitched yet.

### Project

* **New project…** / **Open…** – both pick a folder; an empty folder becomes a new project (configs are
  copied in), an existing one is reopened. Use a fast disk with room for the rendered stack (roughly the
  size of the raw data; twice that if you render both PNG tiles and a precomputed volume).
* **Project name** – free text; shown in the title bar and used as the Neuroglancer layer name.

### Raw tiles

| Control | What it does | When to change |
|---|---|---|
| **Tile folder** | Root of the raw data; subfolders are fine. | |
| **Extension** | `tif` by default. | For `png`, `bmp`, … |
| **include subfolders** | Recurse into subdirectories. | Off only if a subfolder holds images you must ignore. |
| **Naming rule** | How row/column/section are encoded: *Thermo (Maps)* `Tile_row-col-id_0-000.sZZZZ_e00`, *Zeiss (Atlas)* `Tile_rY-cX_S_ZZZZ`, *Sequential numbers*, *Custom pattern*. | Start with **Guess rule from filenames**. |
| **Number order** (sequential only) | One letter per number in the name, in order: `y` row, `x` column, `z` section, `r` = ignore. E.g. `yxz`. | When filenames are just numbers. |
| **Pattern** (custom only) | The filename with `#x#`, `#y#`, `#z#`, `#r#` placeholders, e.g. `Tile_#y#-#x#-#r#_0-000.s#z#_e00`. | Anything the presets do not cover. |
| **Guess rule from filenames** | Reads up to 400 filenames and offers ranked candidates; **Use** applies the selected one. | Always try this first. |

If every section is a folder of identically named tiles, run **Tools → Prefix filenames with folder
name…** first so the section number lands in the filename.

### Voxel size & tile layout

| Control | What it does | Notes |
|---|---|---|
| **Pixel size** | xy pixel size at mip0, in nm. | Drives the mip suggestions and the exported voxel size. A wrong value gives wrong scale bars everywhere downstream. |
| **Section thickness** | z step in nm. | Used to pick the fine-alignment working mip: the mip whose xy resolution is closest below the thickness. |
| **Read from image metadata** | Reads the first tile: width, height, dtype, pixel size, presence of stage coordinates. | Press it before scanning; it fills tile width/height for you. |
| **Tile width / height** | Tile size in pixels. | Set by hand only if the metadata is unreadable. |
| **Placement** | *Regular grid from indices*: tile (col,row) goes to `(col·(width−overlap), row·(height−overlap))`. *Microscope stage positions*: reads stage coordinates from the TIFF metadata (Thermo FEI tags, or the Zeiss Atlas Fibics XML block where the pixel size is field of view ÷ image size and tile offsets come from the mosaic entry when the stage position is per mosaic) and rotates them into image space. | Grid is fine for almost everything — FEABAS refines positions during matching, so ±10 % is harmless. Use stage placement for irregular or drifting acquisitions. |
| **Overlap x / y** + **unit** | Nominal overlap between neighbours, in % of tile or in pixels. | ~10 % is typical. Too small an overlap combined with a small search margin is the classic cause of failed matches. |
| **Estimate overlap from stage** | Derives the real overlap from one section's stage coordinates and switches the unit to pixels. | Use whenever stage positions exist. |
| **flip columns / flip rows** | Mirrors the index → position mapping. | When the preview grid is mirrored relative to reality. |
| **Section number offset** | Added to the z parsed from the filename. | To start numbering at 0/1, or to concatenate two acquisitions. |
| **Paths in coord files** | `relative to tile folder` (default) or `absolute`. | Relative keeps the project movable; absolute is safer when tiles live on a share mounted differently elsewhere. |

### Scan & preview

* **Scan tiles** – parses every filename, groups by section, computes positions, and reports
  `N tiles in M sections, grid R×C, tile size, nm/px, placement`. Warnings (missing tiles, duplicate
  indices, inconsistent grids) show up underneath and in the log. **Read them** — a missing tile is far
  cheaper to fix now than after stitching.
* **Preview section** + grid – tile outlines with row/column labels for one section.
* **Show overlap of two neighbouring tiles** – opens the actual overlapping strips of two adjacent
  tiles side by side. The fastest way to confirm overlap and orientation before committing to a
  multi-hour stitch.

### Write FEABAS coordinate files

**Write stitch_coord files & save volume info** does five things:

1. deletes and rewrites `stitch/stitch_coord/*.txt` (one per section);
2. saves source folder, rule, layout and volume info into `workbench_project.json`;
3. writes `configs/general_configs.yaml` (working directory, full resolution, section thickness);
4. points the coordinate files at the preprocessed tiles if you activated one (Window 2);
5. seeds sensible defaults: `alignment.matching.working_mip_level`, `thumbnail.thumbnail_mip_level`,
   `alignment.meshing.mask_mip_level`, `stitching.section_thickness`.

The line underneath shows the suggested working mip and thumbnail mip for your voxel size — those are
the numbers Windows 3–5 refer to.

If stitching outputs already exist, it asks first: rewriting coordinates makes everything downstream
stale.

### Tools

**Prefix filenames with folder name…** renames tiles in place to `<folder>_<original>` using the parent,
grandparent or great-grandparent folder name, for acquisitions where every section is a folder of
identically named tiles. It shows a preview and renames only after you confirm. This edits your raw
data — read the preview.

---

## 5. Window 2 – Preprocessing (optional)

Both tools write a **mirror** of the raw folder structure under `preprocessed/`, so you can switch the
pipeline between raw, histogram-matched and denoised tiles at any time. Skip this window entirely if
your tiles are already consistent and clean.

### Tiles used by the pipeline

Radio buttons **raw tiles / histogram-matched tiles / denoised tiles** plus **Apply to coordinate
files**. Applying rewrites the `stitch_coord` files to point at the chosen folder — that is the whole
switch. The info line counts the files available in each variant.

If stitching outputs already exist when you switch, the log warns you: clear *Match tiles* on the
Stitching page, otherwise you mix montages made from different pixel data.

### Histogram matching

Maps every tile's grey-level histogram onto a template's (monotonic CDF matching). Use it when
brightness or contrast drifts between tiles, mFoVs or sections — that drift shows up as visible tile
boundaries after rendering and can weaken matching.

| Control | Meaning | Guidance |
|---|---|---|
| **Template** | One representative tile whose histogram everything else is mapped onto. | Pick a tile with the full range of structures you care about, from the middle of the run. |
| **ignore black (0)** | Leave pixels with value 0 untouched and exclude them from the CDF. | On by default — keeps padding/beam-off regions from skewing the mapping. |
| **ignore white (max)** | Same for saturated pixels. | On for data with saturated resin or charge artefacts. |
| **workers** | Parallel processes. | About your physical core count; this is pure CPU work. |
| **Preview on a tile** | Shows raw (left) vs matched (right) for the selected tile. | Always preview before running on thousands of tiles. |
| **Match all tiles** | Runs over the whole raw folder into `preprocessed/histmatch/`. | Needs disk space equal to your raw data. |

### Denoising with CAREamics (N2V / N2V2 / StructN2V)

Self-supervised denoising: no clean ground truth needed. Runs in the deep-learning environment, on the
GPU if there is one. Useful for low-dose/fast acquisitions; unnecessary for already clean data, and it
does cost disk space and hours.

**Choosing training tiles** – pick tiles in the grid on the left, **Add selected tiles →**, or press
**Random 24**. 10–40 representative tiles is plenty; at least 4 are required. Include the different
looks in your data (dense tissue, resin, edges).

The card shows what most people need – method, run name, *Train model*; model, *Preview on a tile*,
*Denoise all tiles* – and keeps everything else under **Advanced settings** (click to expand). The
StructN2V axis and span appear next to the method only when StructN2V is selected.

| Setting | Meaning | Guidance |
|---|---|---|
| **Method** | `N2V` (default), `N2V2` (fewer checkerboard artefacts), `StructN2V` (removes line-structured scan noise). | Use StructN2V when the noise is clearly striped along the scan direction. |
| **struct axes / span** | Only for StructN2V: orientation (`horizontal`, `vertical`, `cross`) and width of the masked stripe. | Match the axis to the visible stripes; span 5 is a sensible start. |
| **Run name** | Folder under `models/n2v/`. | One name per experiment so you can compare. |
| **Train model** | Starts training in the DL environment. | |
| **Model** | Trained checkpoints found under `models/n2v/`. | |
| **Preview on a tile** | Denoises one tile and shows raw vs denoised. | Do this before the full run. |
| **Denoise all tiles** | Writes `preprocessed/denoised/`, skipping files that already exist (so it is resumable). | If the *histogram-matched tiles* radio button is selected, it denoises those instead of the raw tiles. |
| *Advanced:* **patch** | Training patch size (default 128). | 64–256. Larger patches see more context but need more VRAM. |
| *Advanced:* **batch** | Patches per step (default 12). | Lower it if you hit CUDA out-of-memory. |
| *Advanced:* **epochs / steps per epoch** | Training length (default 100 / 100). | 100×100 is a reasonable first run; watch the loss in the log. |
| *Advanced:* **ROI / masked %** | N2V blind-spot parameters: neighbourhood size (11) and percentage of pixels masked per patch (0.2). | Defaults are fine; raising masked % speeds up learning but adds noise to the gradient. |
| *Advanced:* **prediction tile / overlap / batch** | Tiled inference: tile size 512, overlap 64, batch 4. | Reduce the tile size if you run out of VRAM; increase the overlap if you see seams in the denoised output. |

---

## 6. Window 3 – Stitching

Three FEABAS steps in order: **Match tiles → Optimize montage → Render montages**. The tabs are *Run*,
*Test on subset*, *Quality check* and *All settings*.

### Run tab – the settings that matter

| Setting | Config key | What it does | When to change |
|---|---|---|---|
| **workers: matching** | `matching.num_workers` (15) | Processes used to match tile overlaps. | Set near your physical core count; lower it if matching runs out of RAM. |
| **workers: optimization** | `optimization.num_workers` (3) | Processes relaxing each section's montage. | Rarely the bottleneck; 3–8. |
| **workers: rendering** | `rendering.num_workers` (15) | Processes writing the stitched images. | I/O bound — more workers help only on fast storage. |
| **render as** | `rendering.driver` | `PNG tiles` (simple, VAST-friendly, and what fold detection at a finer mip needs) or `Neuroglancer precomputed volume` (chunked, faster downstream, best for ~TB data). | Pick PNG tiles for small/medium data or if you want to run fold detection on a finer mip; precomputed for big volumes. |
| **CLAHE** | `rendering.loader_settings.apply_CLAHE` (on) | Contrast-limited histogram equalisation while rendering. | Leave on. Turn off if you did your own histogram matching and want the raw look. |
| **invert grey** | `rendering.loader_settings.inverse` (on) | Inverts grey values (BSE images are often acquired inverted). | Turn off if your rendered sections come out as negatives. |
| **overlap blending** | `rendering.render_settings.blend` (PYRAMID) | How pixels covered by more than one tile are combined — see [Blending the overlaps](#blending-the-overlaps) below. | PYRAMID unless double exposure changes the grey level in the overlap; then NEAREST. |
| **match confidence threshold** | `matching.matcher_config.conf_thresh` (0.1) | Cross-correlations below this confidence are rejected. | Raise (0.2–0.3) when you see wrong matches / warped montages; lower only if too many overlaps are rejected in low-contrast data. |
| **search margin (px)** | `matching.margin` (1000) | Extra width searched around the nominal overlap, to absorb stage-position error. | Increase when stage positions are unreliable (costs time); decrease for very accurate stages to speed up matching. |
| **image cache (tiles)** | `matching.loader_config.cache_size` (150) | Tiles held in RAM per worker. | Lower it first when matching runs out of memory; then reduce workers. |

**Apply settings** writes them into `configs/stitching_configs.yaml`. Steps whose outputs are older than
that file immediately show as *stale*.

### Blending the overlaps

Neighbouring tiles overlap by design, so every pixel in an overlap band exists in two (at a corner, four)
source tiles. The blend mode decides what the rendered montage gets.

All modes start from the same quantity: each source tile covering a pixel carries a **weight** equal to its
distance, in source pixels, from the nearest edge of *that tile* (reduced further by `clip_lrtb`). The
weight is 0 along a tile's own border and largest in its middle, so "largest weight" means "deepest inside
a tile, furthest from its edge".

| Mode | What it computes | Consequence |
|---|---|---|
| **PYRAMID** (default) | Each tile is split by a Gaussian (σ = 2.5 px) into a low- and a high-frequency image. Low frequencies are averaged with the weights; the high-frequency detail is taken from the single tile with the largest weight. | Detail is never averaged, so residual misalignment cannot ghost, while a brightness difference between tiles is smoothed out instead of showing as a step. The best default. |
| **NEAREST** | The whole pixel comes from the tile with the largest weight. The seam runs along the line where two tiles are equally deep — the middle of the overlap. | Nothing is mixed: no ghosting, and no averaging of two different exposures. A brightness difference between the tiles shows as a hard edge along that seam. |
| **LINEAR** | Weighted average of every covering tile: `Σ(value × weight) / Σ weight`. A tile's influence fades to nothing at its own edge. | Smooth crossfade. Brightness differences become a gradient; imperfectly aligned detail blurs or ghosts in the overlap. |
| **MAX** | The brightest value among the covering tiles (weights only decide *whether* a tile covers the pixel). | Rejects dark artefacts (a shadow or charging spot present in one tile only); brightens the overlap bands relative to the rest. |
| **MIN** | The darkest value among the covering tiles. | The mirror image: rejects bright artefacts, darkens the overlap bands. |
| **NONE** | No arbitration at all: every covering tile is written into the output chunk in turn, so the last one processed wins. | Which tile that is depends on the order the spatial index happens to return, so the "winner" can change from one output tile to the next. Not the way to pick a particular tile. |

Pixels no tile covers are filled with `rendering.loader_settings.fillval` (0 = black).

**Choosing:** keep PYRAMID unless you have a reason. If the overlap band is imaged twice and the second
exposure changes the grey level (dose, contamination, charging), the mode you want is **NEAREST** — it
keeps every pixel from exactly one tile without averaging two exposures. Better still, remove the cause
first with histogram matching (Window 2), then NEAREST looks seamless. Use MAX or MIN only to reject a
specific artefact. There is no mode that always keeps a chosen tile of the pair.

> **`blend: null` renders black bands.** In the settings tree, a string key set to the text `NONE` used to
> be saved as YAML `null`. FEABAS reads a null blend as "only one tile per output chunk": it fills each
> output tile from the first source tile it finds and leaves everything else at the fill value — black
> bands exactly where the overlaps are, with the tiles that did render in the right place. Newer versions
> keep the word (only `null` and `~` clear a key), and the drop-down never produces a null. If an older
> project still has `blend: null` in `configs/stitching_configs.yaml`, pick a mode from the drop-down and
> re-render.

### Steps

| Step | Produces | Notes |
|---|---|---|
| **Match tiles** | `stitch/match_h5/*.h5` | Per section. Failures leave `*.h5_err` — use *Remove error files* and re-run after fixing settings. |
| **Optimize montage** | `stitch/tform/*` | Elastic relaxation per section; residues are reported in the log. |
| **Render montages** | `stitched_sections/…` | PNG tile folders per section, or a TensorStore volume, depending on the driver. Progress counts *finished* sections: the PNG driver writes `mip0/<section>/metadata.txt` after the last tile of a section, the precomputed driver writes `<section>/info`. |

### Test on subset tab

Creates a sandbox under `tests/<name>/` with its own `configs/` and coordinate files for the sections
(and optionally the tile rectangle) you selected. Running steps there never touches the real outputs.

1. Check one or a few sections on the left; optionally drag-select a tile region in the grid on the right.
2. Name the run and press **Create test run** (the current quick settings are copied in).
3. Run the steps in the test panel, then look at the result in *Quality check* (choose the test run as
   source). The fourth card, **Mipmaps of the montage**, is the same FEABAS step as *Make thumbnails*:
   rendering writes mip0 only, and without the downsampled levels the viewer cannot show a whole section
   at once. It builds mip1 up to `thumbnail_mip_level − 1` (at least the fine-alignment working mip).
4. **Edit this test's settings…** lets you iterate; **Save and copy to project** promotes the settings you
   like to the real project.

Do this once for every new dataset. A test run over one section and a 3×3 tile block takes minutes; a
full stitch takes hours.

### Quality check tab

Full-resolution viewer over the rendered sections (PNG tiles or precomputed volume), with a section
picker, ◀/▶ navigation, **tile outlines (nominal)** and **Open section in Fiji**. You can look while a
render is still running: **Inspect** always switches to the run whose step card you pressed it on, and a
section that has no readable tiles yet says so instead of showing an empty canvas.

What to look for: zoom into tile borders. Seams should be invisible. Visible duplication or blur along a
seam means the match there was rejected or wrong — check the log for that section, then try a larger
search margin or a lower confidence threshold, or check your overlap value in Window 1.

The info line reports how many mip levels the section has. With mip0 only, zooming out past a certain
point shows nothing — there is no coarse level to draw and reading the full-resolution section would be
hopeless — and the viewer says so. Build the levels first: the **Mipmaps of the montage** step in a test
run, or **Make thumbnails** on the Masks page for the project.

### All settings tab

The whole merged `stitching_configs.yaml` as a tree, with FEABAS's own comments as hints and overridden
keys marked. Useful branches:

* `matching.matcher_config` – `sigma` (DoG filter before matching), `coarse_downsample`/`fine_downsample`
  (speed vs accuracy), `residue_mode`/`residue_len` (how outliers are damped).
* `optimization.mesh_settings.mesh_sizes` `[100, 300]` – finer meshes absorb more tile distortion.
* `optimization.translation.residue_threshold` (0.8) – matches with residue above this are dropped as spurious.
* `optimization.disconnected_assemble.explode_factor` – how far apart tile groups that never matched are placed.
* `rendering.loader_settings.CLAHE_cliplimit` (2.0) – raise for 16-bit data.
* `rendering.render_settings.blend` (`PYRAMID`) – blending across seams; `NONE` shows you exactly where
  the seams are, which is occasionally useful for debugging.
* `rendering.tile_size` (`[4096, 4096]`) – output tile size for the PNG driver.

---

## 7. Window 4 – Masks

FEABAS needs to know where the section is (tissue vs. background) and where it is folded. Masks are
grey-value images in thumbnail space:

| Grey value | Material | Behaviour |
|---|---|---|
| `0` | default tissue | meshed, matched, rendered |
| `255` | exclude | not meshed (outside the section, holes) |
| `50` | wrinkle/fold | free to expand, resists compression |
| `100` | soft | very soft material |
| `200` | split | meshed but broken by a thin gap |
| `150` | background_lowweight | added only for structure-guided alignment (see Window 5) |

The tabs are numbered in the order they are used: **1. Thumbnails**, **2. Masks**; *Train fold model*
is optional.

### Thumbnails tab

| Setting | Config key | Guidance |
|---|---|---|
| **thumbnail mip level** | `thumbnail.thumbnail_mip_level` (2) | Aim for thumbnails of roughly 500–2000 px on the long side; the hint line computes the size for you. Bigger thumbnails make coarse matching very slow, smaller ones lose the structures the matcher needs. |
| **high-pass filter (SE images)** | `downsample.thumbnail_highpass` (on) | On for secondary-electron images (enhances somata); off for BSE images and very small mip levels. It also changes what the texture-based tissue detection sees. |
| **workers** | `downsample.num_workers` (10) | Near your core count. |

**Make thumbnails** also builds the intermediate mip levels of the stitched sections that fine alignment
reads, and writes FEABAS's default masks (everything imaged = tissue), which the next tab replaces.
Pressing **Apply** here also sets `alignment.meshing.mask_mip_level` to the same mip.

The progress bar in the status bar counts both phases: the mip levels (the slow part — one entry per
section and level, `mip levels 120/453`) and then the thumbnails themselves (`thumbnails 40/151`).

### Masks tab

The section list on the left shows what exists per section (`thumb`, `tissue`, `folds`, `mask`; a `*`
marks a hand-edited mask). The viewer shows the thumbnail with any combination of overlays — *tissue*,
*folds*, *fold probability*, *material mask*, *structures* — and an opacity slider.

**Selecting sections.** The header counts what is checked (`12 of 111 sections`). Every operation on this
page works on the checked sections, except *Preview*, which works on the section you are looking at. To
check a run of sections, highlight them in the list the usual way — click, shift-click for a range,
ctrl-click to add — and press **✓ highlighted** (or the space bar); **✗ highlighted** unchecks them.
**all** / **none** and **every N** are there for the whole stack. Computations refresh the list to show
what they produced; your checks survive that, and sections that appear for the first time start checked.

#### Tissue vs. background

Pick a **method** first; the card then shows only the settings that method uses, and a one-line
explanation of what it does. Every setting has a tooltip. Two rows are common to all methods: the
*border margin* and the *exclude regions inside the section* switches.

| Method | What it does | When |
|---|---|---|
| **the stitched tile footprint is tissue** (FEABAS default) | The tissue region is the area the tiles cover, and only what lies outside it is excluded. Three sources, best first: FEABAS's own mask (kept under `masks/roi/` the first time the workbench replaces one); the tile boxes read straight out of `stitch/tform/<sec>.h5` — the same geometry FEABAS rasterises, cached in `masks/roi/`; and, if there is no montage geometry at all, the convex hull of the image data. All three are geometric rather than a flood fill from the border, because a black fold can run from one section edge to the other. Nothing to set. | **Start here.** Whenever the section fills the imaged area. |
| **detect a uniform frame (black / white) from the outside in** | For sections that sit inside a uniform frame: black padding, a saturated white rim of empty resin or support film, a grey detector line — or several of these, one inside the other. The frames are peeled off layer by layer, starting from the area outside the imaged footprint, until textured tissue is reached; a frame that does not touch the image edge itself (a white rim behind black padding) is found too. | Single-tile sections with a white rim; montages with resin around the tissue. |
| **fixed margins from the image edge** | Type how far the frame reaches in from the **left / top / right / bottom** of the thumbnail (in thumbnail pixels; the µm equivalent is shown next to it). A dashed rectangle in the viewer updates as you type, before anything is written. The result never extends beyond the imaged footprint. | When the frame is regular and you would rather say where the tissue is than detect it. |
| **auto** | Splits by local texture only if a clear background/tissue split exists (Otsu separability above a threshold), otherwise falls back to "everything imaged". | Mixed datasets where some sections have resin background. |
| **local texture** | Always splits by local grey-level standard deviation. | Sections with genuine resin/support background. Careful on high-pass filtered thumbnails: it can split tissue by texture. |
| **intensity threshold** | Plain grey-level threshold. | Simple, high-contrast background. |

Settings of the **frame** method:

| Setting | Meaning | Guidance |
|---|---|---|
| **frame colour** | *black or white* (the usual case: black padding around a white rim), *black only*, *white only*, or *any uniform grey (auto)* – any locally flat area counts as frame, whatever its brightness. | Start with *black or white*. Use *auto* for a grey frame. Pixels of the no-data value 0 count as frame in every mode, so a white rim behind black padding is reached with *white only* too. |
| **black: grey ≤** (0) | Grey level up to which a pixel belongs to the black frame. | Raise a little if the padding is not exactly 0. |
| **white: grey ≥** (250) | Grey level from which a pixel belongs to the white frame. | 250 leaves room for a slightly blurred rim; 255 = pure white only. |
| **flatness tolerance** (4) | *auto* only: a pixel is flat when the grey range of its 3×3 neighbourhood is at most this. | EM tissue is never flat over a large area, so the default is safe. |
| **ignore streaks thinner than px** (15) | A fold of the frame colour that touches the frame would be peeled off with it; streaks thinner than this are given back and stay tissue (label them as folds below). Slivers of tissue thinner than this – the seam between two frames, a detector line – are dropped. | Set it above the width of your folds in the thumbnail and below the width of the frame. |

Clean-up, shared by the frame, auto, texture and intensity methods:

| Setting | Meaning | Guidance |
|---|---|---|
| **drop tissue pieces smaller than px** (2000) | Connected pieces of tissue with fewer pixels than this are removed: a piece has to be *at least* this big to count as tissue. | Raise to remove specks of debris or resin classified as tissue; keep it well below the size of the section. |
| **fill holes up to px** (5000) | Background holes *enclosed by tissue* that are smaller than this become tissue again; larger enclosed holes stay excluded. | Raise if pale patches or vacuoles inside the section get excluded. |
| **texture window px** (31) | auto/texture: neighbourhood over which the local grey-level variation is measured. | Larger for coarse textures, smaller for fine ones. |
| **threshold** | Manual override for the texture score or intensity; empty = automatic (Otsu). | Only when the automatic threshold is visibly wrong. |
| **dark tissue** | intensity: tissue is darker than the background. | For inverted images. |

Common to every method:

| Setting | Meaning | When |
|---|---|---|
| **border margin** (0 px) | A ring of this width just inside the section outline — it follows the outline, it is not a rectangle. The field next to it converts the width into µm, measured against the montage rather than assumed from the mip level. Applied when the material mask is composed, so it appears after *Compose*, not in the tissue preview (tick *border margin preview* above the viewer to see it). | The montage edge is where the mask is least certain and matching least reliable. Pick a width that covers the ragged rim: on a 128 nm/px thumbnail, 20 px ≈ 2.6 µm. |
| **what the margin becomes** | *soft (100)* — meshed and rendered, but its stiffness is below FEABAS's matching threshold, so no match point is ever placed there. *exclude (255)* — not meshed and **not rendered**: that strip is missing from the aligned volume. | soft for a safety margin: you keep the pixels and only stop the alignment from being driven by the uncertain edge. exclude only where there is nothing worth keeping. |
| **also exclude black regions inside the section, grey ≤** (off, 0) | Drops every connected black region of at least **at least px** pixels, wherever it is — including inside the section. | Leave off unless those black areas really carry no data. A fold is black *and* is tissue: excluding it removes it from the mesh and swallows its fold label, because fold labels are only painted inside the tissue mask. |
| **white regions, grey ≥** (off, 255) | The same for saturated white regions: empty resin, burnt spots, support film inside the section. | Same caveat. |
| **at least px** (24) | Minimum size of a black/white region before it is excluded; smaller ones are normal tissue detail. | |
| **Preview on current section** | Computes for the visible section only. | Iterate here, not on the whole stack. |
| **Compute for checked sections** | Runs over every checked section. | |

#### Folds & wrinkles

Two ways to find folds, and one decision to make before either.

**Exclude them, or mesh them?** A black region can be treated two ways, and they are mutually exclusive:

* as a **fold material** (the default) — detect the folds here and *Compose* gives them label 50 (wrinkle)
  or 100 (soft), so FEABAS meshes them and lets them expand while the alignment is driven by the tissue
  around them. Right for wrinkles and tears in a section that is otherwise intact;
* as **no data** — turn *exclude black regions* on in the tissue card and those areas become `255`,
  outside the mesh entirely. Right only where the microscope recorded nothing at all.

Fold labels are only painted inside the tissue mask, so if both are on, exclusion wins. Compose warns in
the log when most of a section's detected folds were swallowed that way.

| Setting | Meaning | Guidance |
|---|---|---|
| **method** | *dark regions* thresholds the thumbnail — no model, no GPU, seconds per section. *U-Net checkpoint* runs the bundled (or your own) network in the deep-learning environment. | Start with dark regions: folds and tears are usually black in the rendered montage. Use the U-Net when folds are grey rather than black, or textured. |
| **grey ≤** (0) | Dark-region method: a pixel counts as fold at or below this grey level. | Raise to 3–10 if the fold edges are not exactly black after downsampling. |
| **Preview on current section** | Runs the chosen method on the section you are looking at. | Check the overlay before running the stack. |
| **Checkpoint** | The fold U-Net weights. Defaults to the bundled checkpoint if present. | Replace with one you trained in the next tab. |
| **run on** | `thumbnails`, or a `stitched sections mipN` folder if PNG tiles were rendered. | Use a finer mip when folds are only a few pixels wide in the thumbnail; the workbench then also writes higher-resolution masks to `align/material_masks`. |
| **threshold** (0.5) | Probability above which a pixel counts as fold. | Lower to catch faint folds (more false positives), raise if normal tissue is flagged. Check with the *fold probability* overlay. |
| **tile / overlap** (1024 / 128, under *Advanced U-Net settings*) | Tiled inference geometry. | Lower the tile size on small GPUs; raise the overlap if you see tile-edge artefacts in the fold mask. |
| **drop fold blobs smaller than px** (50) | Fold regions with fewer pixels than this are speckle and are dropped. | Raise to remove speckle. |
| **grow folds by px** (2) | Dilate the fold regions. | A little dilation helps the mesh absorb the compression around a fold. |

**Detect folds on checked sections** runs in the deep-learning environment and writes masks plus
probability maps to `masks/folds/`.

#### Import masks made elsewhere

Point at a folder with one mask image per section (named like the sections, carrying the section number,
or simply in section order), say which **mip** of the stitched sections they were made at, and choose the
kind:

* **tissue mask** – non-zero = tissue;
* **fold mask** – non-zero = fold;
* **complete FEABAS material mask** – grey labels taken as they are (0/50/100/200/255).

Imports are resampled to the thumbnail mip, and the full-resolution copy is kept for the higher-resolution
material masks. After importing tissue or fold masks, press **Compose**. Importing a complete material
mask at a finer mip also sets `alignment.meshing.mask_mip_level` accordingly.

#### Write material masks for FEABAS

| Control | Meaning |
|---|---|
| **include folds as** | Whether detected folds enter the material mask, and as which material: *wrinkle (50)* – expands freely, resists compression (the usual choice); *exclude (255)* – cut out of the mesh entirely (for genuinely destroyed regions); *soft (100)* – very soft material. |
| **also write higher-resolution masks** | Additionally writes `align/material_masks` at the finest mip for which a mask source exists, and sets `alignment.meshing.mask_mip_level` to match. |
| **where folds and the outside overlap** | Fold detection fires on the black border around the section as readily as on a fold, so the two masks overlap there. *Outside wins* (default) clips folds to the imaged area, which keeps the border excluded (`255`). *The fold label wins* paints the fold material outside the footprint too — the 1-pixel excluded rim FEABAS needs is kept either way. |
| **keep hand-edited masks** | Composing skips sections you edited by hand (marked `*`). Leave on so you do not silently overwrite manual work. |
| **Compose for checked sections** | Writes the single material mask per section into `thumbnail_align/material_masks/`. |

The composed mask always keeps a one-pixel excluded rim, which FEABAS's matcher requires.

#### Hand editing

* **Split section: click 2 points** – paints a split line (label 200) through a broken section so the two
  pieces can move independently. Click the two end points of the line through the gap.
* **Edit mask in Fiji** – opens the material mask; paint with grey values 0 (tissue), 255 (outside),
  50 (wrinkle), save, then press **Reload mask (mark hand-edited)**.
* **Reset checked sections to FEABAS default** – a real revert: the material mask goes back to FEABAS's
  own tile-footprint mask (from `masks/roi/`), and the tissue mask, fold mask and fold probability
  computed for those sections are deleted along with the higher-resolution copy, so the next *Compose*
  starts from scratch instead of reusing them. The hand-edited flag is cleared too. It asks first and
  reports how many sections it reset.
* **Rebuild FEABAS masks…** – for sections whose FEABAS mask was already overwritten before the workbench
  kept a copy. It deletes their material masks *and thumbnails* and re-runs the thumbnail step; FEABAS
  only writes a mask for a section whose thumbnail it regenerates, which is why both have to go. The mip
  levels of the stitched sections are kept, so it takes seconds per section.

### Train fold model tab

Fine-tune the U-Net on your own data. Dataset = a folder with `images/` and `masks/` using the same
filenames, where mask > 0 means fold. Crops of the chosen patch size are sampled from full images, so any
image size works.

| Setting | Guidance |
|---|---|
| **run name** | Folder under `models/folds/`. |
| **epochs** (50) | Fine-tuning from the bundled checkpoint converges much faster than training from scratch. |
| **patch** (256) / **batch** (16) | Lower the batch on small GPUs. |
| **start from the checkpoint on the Masks tab** | On = fine-tune that checkpoint; off = train from scratch (needs much more data). |

**Use selected model** puts the trained `best.pt` into the checkpoint field.

**About the bundled checkpoint.** `fold_unet_resnet34_inference_only_fp16.ckpt` (49 MB, in
`feabas_workbench/resources/`) is an export of the original Lightning checkpoint
`best-epoch=25-val_iou=0.6638.ckpt`: every model weight is there, stored as fp16, and the detections are
identical. What was left out is the optimizer and scheduler state of that training run. Consequences:

* **fine-tuning from it works** — *start from the checkpoint on the Masks tab* loads the weights into a
  fresh fp32 U-Net and trains with a new optimizer, which is what fine-tuning is;
* **resuming the original run** (continuing epoch 26 with its optimizer momentum and learning-rate
  schedule) is the one thing it cannot do. That needs the full 280 MB Lightning checkpoint, which is not
  bundled; if it becomes available for download, the link will be added here. The workbench itself never
  resumes runs, so nothing in the GUI depends on it.

---

## 8. Window 5 – Alignment

Coarse alignment works on thumbnails, fine alignment on finite-element meshes at the working mip. The
tabs are numbered in the order they are used: **1. Coarse alignment**, **2. Fine alignment**, **3.
Quality check**, then the two full settings trees. *Test on subset (optional)* is for trying settings
before the real run — once the pipeline has run there is nothing left to test — and the experimental
*Structure-guided (optional)* tab is hidden until you switch it on under Setup → Interface.

### Coarse alignment tab

| Setting | Config key | What it does | When to change |
|---|---|---|---|
| **compare distance** | `thumbnail.alignment.compare_distance` (2) | How many neighbours each section is matched to. 1 = immediate neighbours only. | 2 makes the stack robust against a single bad section. Each extra step adds another ~N pairs to match, so raise it further only for very thin sections. |
| **match mode** | `alignment.match_mode` | `feature` (general) or `template` (block-face style). | Keep `feature` for serial sections; `template` for block-face data with almost no lateral shift. |
| **max keypoints** | `alignment.feature_matching.detect_settings.num_features` (5000) | Keypoint budget per thumbnail. | Raise for large or feature-poor thumbnails; lower to speed up. |
| **workers** | `alignment.num_workers` (15) | Parallel processes. | Near core count. |

Steps: **Match thumbnails** → **Optimize coarse stack** → **Render coarse stack** (optional but strongly
recommended, since it produces the aligned thumbnails the quality check needs).

*Match thumbnails* is the most failure-prone step in the whole pipeline; read its log. When a pair fails
you can add manual BigWarp matches in Fiji, or use structure-guided matching (optional tab).

### Structure-guided tab (optional, hidden by default)

Shown only when *Setup → Interface → show the experimental 'Structure-guided' tab* is ticked. It does not
change the normal workflow when unused. The idea: align on structures you care about (nuclei,
mitochondria, vessels) instead of anonymous texture features.

**1. Detect structures**

| Setting | Meaning | Guidance |
|---|---|---|
| **Model** | YOLO-seg weights (`.pt`). | Bring your own, or train one below. |
| **classes** | Class names or ids to keep, comma separated; empty = all. | Use it to keep only nuclei from a multi-class model. |
| **confidence** (0.25) | Detection threshold. | Raise if you get spurious detections; check the overlay in *Inspect*. |
| **tile** (1024) | Tiled inference size (overlap is tile/8, at least 64). | Lower on small GPUs. |
| **run on** | `thumbnails` or a `stitched sections mipN` folder. | Detect at the resolution where the structure is actually visible. |

**2. Coarse matches from structures**

Centroids of the same structures in neighbouring sections are matched (descriptor + RANSAC affine) and
written as FEABAS thumbnail matches. A BigWarp CSV copy goes to `thumbnail_align/manual_matches`.

| Setting | Meaning | Guidance |
|---|---|---|
| **augment** | Adds structure matches to FEABAS's own. **Run it after *Match thumbnails*.** | The safe default. |
| **replace** | Structure matches only, for the pairs where they succeed; the rest are matched normally. **Run it before *Match thumbnails*.** | For data where feature matching keeps failing. |
| **weight** (3.0) | Weight given to structure matches relative to FEABAS's. | Higher = the optimiser trusts your structures more. |
| **RANSAC tol px** (6) | Inlier tolerance for the affine fit between the two point sets. | Raise for larger section-to-section deformation. |
| **min inliers** (8) | Minimum inliers before a pair's structure match is accepted. | Raise to reject weak pairs. |

**3. Fine alignment driven by structures**

* **restrict** – tissue outside the structures becomes a `background_lowweight` material (label 150) that
  is meshed and rendered but has a stiffness multiplier below FEABAS's matching threshold, so *no fine
  matching points are placed there*. Selecting it adds the material to `configs/material_table.yaml`;
  **you must re-compose the masks** (Window 4) afterwards. **grow structures by px** dilates the
  structures before they become the matching region.
* **Re-weight fine matches by structures** – the softer alternative, applied *after* **Fine matching**:
  matches inside structures keep weight `weight inside` (1.0), the others are damped to `weight outside`
  (0.1). Run **Optimize stack** afterwards. **Undo re-weighting** restores the original weights.

**Train a YOLO-seg model** – dataset is a `data.yaml` of a YOLO-format dataset (or a folder with
`images/` and `labels/` plus the class names typed in). Choose a base model (`yolo11n-seg` is fast,
`yolo11m-seg` more accurate), epochs (100), image size (640). Results land in `models/yolo/<run>/`.

**Inspect** – shows structures on the thumbnail (green overlay + centroids), or the coarse matches to the
next section as a red/green overlay with displacement lines.

### Fine alignment tab

| Setting | Config key | What it does | When to change |
|---|---|---|---|
| **compare distance** | `align/match_name.txt` (workbench-side; *same as coarse alignment* = no file) | Which section pairs the fine matching works on. By default FEABAS reuses every pair the coarse alignment matched, i.e. it inherits the coarse compare distance. Choosing 1 (or 2, 3 …) lists only the pairs within that distance in `align/match_name.txt`, which FEABAS reads instead — its own mechanism for a custom pair list. The line under the settings says how many of the coarse pairs will be used; the list is refreshed every time a fine step is started, so pairs matched later are picked up. | Every pair costs a full block-matching pass at the working mip. With a robust coarse stack (compare distance 2) a fine distance of 1 halves the fine matching time and is usually enough; keep 2 for thin, fragile sections. |
| **working mip** | `alignment.matching.working_mip_level` (2) | Resolution at which block matching runs. | Use the suggested value: the mip whose xy pixel size is closest below the section thickness. Finer = slower and noisier, coarser = misses detail. |
| **mesh size (mip0 px)** | `alignment.meshing.mesh_size` (600) | Spacing of the finite-element mesh. | 600 is a good start. Finer meshes follow local distortion but can "fix" real biological change between sections; coarser meshes are stiffer and safer. |
| **match confidence** | `alignment.matching.matcher_config.conf_thresh` (0.35) | Rejects low-confidence block matches. | Raise if bad matches distort the stack; lower if coverage is poor (check the coverage figures). |
| **workers: matching** | `matching.matcher_config.num_workers` (15) | | Near core count; watch RAM. |
| **workers: optimization** | `optimization.num_workers` (5) | | 4–8 is plenty. |
| **chunked depth** | `optimization.chunk_settings.chunked_to_depth` (0) | `0` = sliding window over the whole stack; `>0` = align sections in chunks first, then chunks against each other. | Use chunking for very long stacks (thousands of sections) where the sliding window becomes slow. |
| **chunk size** | `chunk_settings.default_chunk_size` (16) | Sections per chunk when chunking. | |
| **window** | `optimization.slide_window.window_size` (64) | Sections optimised together in the sliding window. | Larger windows propagate constraints further at higher memory cost. |
| **buffer** | `slide_window.buffer_size` (16) | Overlap re-optimised when the window moves, to avoid fringe effects. | Roughly a quarter of the window. |

Steps: **Generate meshes** (needs the material masks) → **Fine matching** → **Optimize stack**.

**Make match-coverage figures (FEABAS tool)** overlays match positions on each thumbnail: red = matches
to the previous section, green = to the next. Areas without yellow have no matches and will only follow
the mesh — that is where distortion goes unnoticed.

### Test on subset tab

Creates a sandbox that links the stitched sections and copies thumbnails and masks of the chosen sections
(at least two), so coarse and fine alignment can be tried with different settings without touching the
project. **Edit this test's settings…** → **Save and copy to project** promotes what worked.

### Quality check tab

Choose the source (project or a test run), the section, and what to show:

* **aligned thumbnails: this section** – plain view;
* **this (red) vs next (green)** – grey means the sections agree, coloured fringes are residual
  misalignment. Some colour is normal — biology changes between sections. Systematic shifts or a
  distorted region point at missing matches or a mask problem;
* **checkerboard with next** – alternating blocks; good for spotting shear;
* **match coverage figure** – the figures made on the Fine alignment tab.

---

## 9. Window 6 – Export & view

### Render tab

| Setting | Config key | Meaning | Guidance |
|---|---|---|---|
| **PNG tile size** | `alignment.rendering.tile_size` (4096) | Output tile size for the PNG driver. | 4096 is fine for VAST. |
| **mip** | `rendering.mip_level` (0) | Resolution to render at. | 0 = full resolution. Render a coarser mip first if you just want to look at the stack. |
| **mipmaps up to** | `downsample.max_mip` (7) | How many pyramid levels to build for the PNG stack. | 7 gives smooth zooming for large volumes. |
| **interpolation** | `rendering.remap_interp` (LANCZOS) | Resampling filter. | LANCZOS for quality, LINEAR/NEAREST when speed matters or for label-like data. |
| **workers** | `rendering.num_workers` and `tensorstore_rendering.num_workers` (15) | | I/O bound; more workers help only on fast storage. |
| **output folder** | `rendering.out_dir` / `tensorstore_rendering.out_dir` | Empty = inside the project (`aligned_stack`, `aligned_tensorstore`). | Point it at a different disk when the project disk is too small. |

Two independent output paths, each with its own steps:

* **PNG tile stack (VASTlite)** – *Render aligned stack (PNG tiles)* → *Mipmaps for PNG stack*.
* **Precomputed volume (Neuroglancer/TensorStore)** – *Render aligned volume (precomputed)* →
  *Mipmaps for volume*. This is the efficient path for ~TB data.

Both read the same transforms, so you can produce either or both. Rendering is the step to distribute:
use the **subset** controls on the step card to render sections 0–499 here and 500–999 on another machine
into the same output folder.

### Export & viewers tab

| Control | Meaning |
|---|---|
| **name** | Base name of the export. |
| **what** | *VASTlite* (`.vsvi` + tile pyramid, hard-linked — instant and no extra disk space on the same drive), *OME-Zarr 0.4* (uncompressed chunks, for moderate volumes), or *both*. |
| **zarr chunk** (256) | Chunk size for the OME-Zarr output. |
| **to** | Export folder; default `<project>/exports`. |
| **Export** | Runs the export worker. Requires the PNG tile stack to exist. |
| **Open in VASTlite** | Opens the selected `.vsvi` (set the VASTlite path in Setup). |
| **Serve precomputed volume & open in Neuroglancer** | Starts a local HTTP server for the precomputed volume while the workbench is open and opens Neuroglancer in your browser. |
| **Open aligned section folder in Fiji** | Opens the current section's PNG tile folder. |

### View aligned sections tab

Tile-based viewer over the rendered PNG stack with section navigation and an **overlay next section
(red/green)** checkbox — the final check that the stack is smooth in z.

---

## 10. A first dataset, end to end

**No data at hand?** The workbench can make a small synthetic serial-section dataset (4 sections of
2×2 overlapping tiles with cell-like structures, Thermo/Maps file names) together with a ready project:

```bat
python -m feabas_workbench.core.synthetic D:\demo
```

Open `D:\demo` in the workbench and walk the windows below on it (every step takes seconds), or let the
whole pipeline run unattended — the same check CI performs on every commit:

```bat
python tools\run_demo_pipeline.py D:\demo --feabas-python C:\path\to\fw-feabas\python.exe --render
```

1. **Setup** – *Detect environments*, check the GPU line, set Fiji/VAST, **Save**.
2. **Project & data** – *New project* (empty folder, big disk) → tile folder → **Guess rule** → **Read
   from image metadata** → set section thickness → **Scan tiles** → check the preview and the warnings →
   **Show overlap of two neighbouring tiles** → **Write stitch_coord files**.
3. *(optional)* **Preprocessing** – histogram matching and/or N2V, then **Apply to coordinate files**.
4. **Stitching** – *Test on subset* with one section and a small tile block; look at the seams in
   *Quality check*; when happy, copy the settings to the project and **Run all** on the *Run* tab.
5. **Masks** – **Make thumbnails** (aim for 500–2000 px) → tissue method (start with "everything imaged
   is tissue") → **Detect folds** → **Compose**. Check a few sections with the overlays; paint split
   lines through broken sections.
6. **Alignment** – **Match thumbnails** → **Optimize coarse stack** → **Render coarse stack** → check
   red/green → **Generate meshes** → **Fine matching** → **Optimize stack**. Make coverage figures if
   something looks off.
7. **Export & view** – render PNG tiles + mipmaps → **Export** (VAST) → **Open in VASTlite**; or render
   the precomputed volume and open it in Neuroglancer.

At every stage: run the test-subset variant first on a new dataset. It costs minutes and saves hours.

---

## 11. Changing your mind: stale, clear, snapshots

**Snapshots** (Pipeline → *Create snapshot of current state…*) copy the small, expensive things —
matches, meshes, transforms, configs — into `snapshots/<timestamp>/`. Rendered images are *not* included
(they are large and reproducible). The dialog tells you how much will be copied. *Restore snapshot…*
puts them back and reloads the configs.

**Clear** (Pipeline → *Clear a step and everything after it…*, or **Clear…** on any step card) deletes a
step's outputs and everything downstream, after listing exactly what will go. This is the supported way
to re-run a FEABAS step: FEABAS skips work whose output already exists, so stale outputs would otherwise
survive the re-run.

Typical sequences:

| You changed | Clear from |
|---|---|
| coordinate files, tile source, overlap | *Match tiles* |
| stitching settings | *Match tiles* (matching-related) or *Render montages* (rendering-only) |
| thumbnail mip / high-pass | *Make thumbnails* |
| masks (tissue, folds, split lines) | *Generate meshes* — coarse matching may also be worth redoing |
| coarse matching settings | *Match thumbnails* |
| working mip, mesh size, fine matching settings | *Fine matching* |
| render settings only | *Render aligned stack* |

**Remove error files** is the lighter alternative when only a handful of sections failed: it deletes the
`*_err` markers so the next run retries exactly those sections and leaves the good ones alone.

---

## 12. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `start_gui.bat` says it found no environment | No environment with PySide6 in any of the places it scans (`start_gui.bat --envs` shows what it found, `set FW_DEBUG=1` every path it tried). Run `tools\install.bat` (it records its environment in `start_gui.local.bat`), put the folder with your environments in `FW_ENV_DIRS`, or set `FW_PYTHON` to the interpreter you want. |
| `tools\install.bat` says it found no conda | It looks on `PATH`, in `CONDA_EXE` / `MAMBA_EXE` and in the usual install folders on `C:`, `D:` and `E:` (`set FW_DEBUG=1` lists every path). Point it at your package manager: `set FW_CONDA=C:\path\to\micromamba.exe` (or `conda.exe`), then run it again. |
| Step card says **blocked** | An upstream step has no outputs. The reason line names it. |
| Step card says **stale** | A config or an upstream step is newer than these outputs. Clear this step and re-run. |
| Many `*_err` files after matching | Overlap or search margin wrong, or genuinely broken tiles. Check the log for the failing sections, fix the setting, *Remove error files*, re-run. |
| Visible seams in the montage | Matches rejected or wrong: enlarge the search margin, lower the confidence threshold, verify overlap with *Show overlap of two neighbouring tiles*. |
| Rendered sections look inverted | Turn off `rendering.loader_settings.inverse`. |
| Black bands exactly where the tiles overlap | The blend mode is null (see [Blending the overlaps](#blending-the-overlaps)). Pick a mode from the *overlap blending* drop-down and re-render. |
| Rendering dies with "unable to allocate …" | Each rendering worker holds its own image cache: `rendering.loader_settings.cache_size` × tile size × workers. Lower the cache size before lowering the worker count. |
| Coarse matching fails for some pairs | Thumbnail mip too fine or too coarse; high-pass filter setting wrong for your detector; try `compare_distance` 2, manual BigWarp matches, or structure-guided matching. |
| Red/green overlay shows a systematically distorted region | Missing matches there (check the coverage figure) or a mask problem (tissue mask cutting into real tissue). |
| Out of memory during matching | Lower `image cache (tiles)` first, then the number of workers. |
| CUDA out of memory in a deep-learning step | Lower the batch size and the prediction/inference tile size. |
| "FEABAS steps cannot run in …: cannot import feabas" | The interpreter is wrong: it is the GUI's own environment or one without FEABAS. The message names where the path came from (Setup page setting, or this project's override). Fix it on the Setup page, press *Check selected*, then *Save*. The check runs before the step starts, so nothing is written. |
| "does not import feabas" / "does not import torch" on *Check selected* | The interpreter you picked lacks the packages that step needs. |
| Quality check shows nothing when zoomed out | Only mip0 exists. Run the mipmap step (test run) or *Make thumbnails* (Masks page). |
| Folds end up as 255 instead of a fold label | The tissue mask excluded them, and fold labels are only painted inside tissue. Turn *exclude black regions* off and re-compute; Compose warns in the log when this happens. |
| A fold that reaches the section edge is cut out of the mask | Use the default tissue method (tile footprint). Anything that flood-fills from the border would follow that fold into the section. |
| The black border around the section carries a fold label | The fold detector fires there too. Set *where folds and the outside overlap* to "outside wins" and re-compose. A wrinkle-labelled border is meshed, rendered, and — unlike `soft` — matched, so the coarse matcher starts aligning montage outlines. |
| Composing brings back a mask you thought you deleted | Compose reuses the stored tissue and fold masks. *Reset checked sections to FEABAS default* deletes them. |
| The GUI is fine but nothing runs | Look at the log dock (Ctrl+L) and the console window behind the GUI; a failed job shows its last output in a dialog too. |

---

## 13. Where every setting is stored

New in this version: the fine-alignment compare distance is kept as `alignment.fine_compare_distance`
in `workbench_project.json` and materialised as `align/match_name.txt`; the tissue method with its frame
and margin settings is under `masks.tissue`; the *show structure-guided tab* switch is global, in
`%APPDATA%/FeabasWorkbench/settings.json`.

| Window | Setting group | Written to |
|---|---|---|
| 0 Setup | interpreters, Fiji, VAST, conda, PyTorch index | `%APPDATA%\FeabasWorkbench\settings.json` |
| 0 Setup | CPU budget, parallel framework, log level | `configs/general_configs.yaml` |
| 1 Project | tile folder, naming rule, layout, voxel size, path mode | `workbench_project.json` |
| 1 Project | working directory, full resolution, section thickness | `configs/general_configs.yaml` |
| 1 Project | suggested mips on write | `configs/alignment_configs.yaml`, `configs/thumbnail_configs.yaml`, `configs/stitching_configs.yaml` |
| 2 Preprocessing | template, ignore black/white, workers, N2V settings, active source | `workbench_project.json` |
| 3 Stitching | everything | `configs/stitching_configs.yaml` |
| 4 Masks | thumbnail mip, high-pass, workers | `configs/thumbnail_configs.yaml` |
| 4 Masks | tissue/fold parameters, checkpoint, compose options | `workbench_project.json` |
| 4 Masks | composed masks | `thumbnail_align/material_masks/`, optionally `align/material_masks/` |
| 5 Alignment | coarse settings | `configs/thumbnail_configs.yaml` |
| 5 Alignment | fine settings, render settings | `configs/alignment_configs.yaml` |
| 5 Alignment | YOLO model, classes, modes, weights | `workbench_project.json` |
| 5 Alignment | `background_lowweight` material | `configs/material_table.yaml` |
| 6 Export | render settings | `configs/alignment_configs.yaml` |
| 6 Export | export folder | `workbench_project.json` |

Test runs keep their own copy of `configs/` under `tests/<name>/`; **Save and copy to project** is the
only thing that promotes them to the project.
