# FEABAS Workbench – notes for coding agents

- Package: `feabas_workbench` (PySide6 GUI). Run: `python -m feabas_workbench [--project DIR]`. Offscreen page
  screenshots for checking layouts: `python -m feabas_workbench --project DIR --screenshot OUT` (set
  `QT_QPA_FONTDIR=C:/Windows/Fonts` on Windows so the offscreen platform has fonts).
- Layers: `core/` is Qt-free and unit-tested (`python -m pytest tests -q`); `workers/` are subprocess entry points
  (`python -m feabas_workbench.workers.<name> --spec file.json`, print `##PROGRESS {...}` / `##RESULT {...}`
  lines); `ui/` pages own one workflow window each; `vendor/feabas_3_0_5/` holds FEABAS's driver scripts, tools and
  default configs (MIT) – do not edit them.
- FEABAS steps run as subprocesses with **cwd = project folder**: FEABAS reads `configs/general_configs.yaml` from
  the cwd, so the project folder is the FEABAS working directory and the FEABAS install is never modified.
- Pipeline state is derived from files on disk (`core/steps.py`); never keep step state elsewhere. Output folders
  follow the project's configs (`rendering.out_dir`, `rendering.mip_level`, `align/ts_spec.json` for the volume):
  use the helpers there (`stitched_dir`, `aligned_dir`, `aligned_render_mip`, `tensorstore_dir`) instead of
  hard-coding `stitched_sections`/`aligned_stack`. Clearing (`clear_targets`) never deletes through a link or
  junction: test runs link the project's montages into their sandbox. Config staleness is per setting:
  `ConfigStore.save` records when each setting changed (`configs/.workbench_setting_changes.json`), each `Step`
  names the config sections that decide its results (`config_keys`/`config_exclude`), and `RESOURCE_KEYS`
  (workers, caches) never count; outside edits count as a change of the whole file.
- FEABAS 3.0.5 run-time fixes live in `vendor/winfix/sitecustomize.py`, put first on PYTHONPATH of every FEABAS job
  (`ui/bridge.feabas_env`) so they reach multiprocessing children too; keep that when adding new ways to launch
  FEABAS. (1) all platforms: `matcher.stitching_matcher` only assigns `phtm` when `compute_photometric` is true but
  returns it unconditionally, so every tile pair that passes the coarse confidence check raises `UnboundLocalError`
  and stitch matching yields no matches at all - the patch defaults that flag to true. (2) Windows: FEABAS builds
  `file://D:/...` URLs that TensorStore rejects, rewritten to `file:///D:/...`. New projects default to the PNG
  (`image`) render driver. The same file applies per-run settings without touching the YAML: `FW_LOCAL_RESOURCES_FILE`
  (local parallelism) and `FW_STAGE_SETTINGS` (JSON kind/field/values/cpu_budget).
- Mipmaps (`thumbnail.downsample`, `align.downsample`) default to Local parallelism "Automatic"
  (`core.local_parallel.mipmap_plan`, used by `feabas_step_spec` and `core.pipeline.run_step`): FEABAS's default
  starts a process pool per mip level per section with one job per tile of that level, which leaves a workstation
  idle, so with enough sections it runs whole sections per worker (`parallel_within_section: false`, identical
  output) and raises `cache_size` 4 -> 16 (~1/3 less CPU). Settings the project overrides are kept.
- Three interpreters: the GUI's own, a FEABAS env (feabas 3.0.5 + tensorstore; numpy 2.x works) and a deep-learning env
  (torch, careamics 0.3.2 which pins torch < 2.10, ultralytics, segmentation-models-pytorch). Paths live in
  `%APPDATA%/FeabasWorkbench/settings.json`; projects may override them.
- Heavy in-process work goes through `ui/threads.ThreadRunner`; anything that needs torch/feabas goes through a
  worker in the right env via `AppContext.worker_spec` / `feabas_step_spec`.
- Test data: `Example_data_to_stitch_and_align_and_export` (Thermo Maps tiles, 10 sections, 8×5 grid, 10 nm px;
  gitignored). Without it, `core/synthetic.py` makes a synthetic dataset + project
  (`python -m feabas_workbench.core.synthetic DIR`) and `tools/run_demo_pipeline.py` runs every FEABAS step on it
  (CI does this on Linux; locally pass `--feabas-python`). Keep that green: it is the only end-to-end test. Besides the
  output counts it checks the matches themselves (`core.pipeline.match_problems`: connected montages, several coarse
  and fine match points per pair); the demo project scales FEABAS's mesh and matching grid to its small sections
  (`core.synthetic.demo_alignment_grid`), since the defaults leave one fine match point per pair. The bundled fold U-Net is `feabas_workbench/resources/fold_unet_resnet34_inference_only_fp16.ckpt`
  (smp Unet, resnet34, fp16 weights only, 49 MB; `core.masks.bundled_fold_checkpoint()`, stored in project files as
  the sentinel `"bundled"` so a project survives a move or a different install location; inside the package so wheels ship
  it). It was exported from the 280 MB Lightning checkpoint `Fold_model_ckpt/lightning_logs/version_0/checkpoints/
  best-epoch=25-val_iou=0.6638.ckpt` (gitignored, optimizer state included) and gives identical detections;
  `workers/fold_train` fine-tunes from either, since it only ever reads the weights. `_test_projects/` and `_scratch/`
  are gitignored scratch areas.
- LRZ cluster mode (per project, opt-in; part of main since 0.3.4): `core/cluster_*` is Qt-free - `cluster_transport`
  (paramiko SSH/MFA, SFTP tree copies, the Miniforge environment script), `cluster_storage` (dssusrinfo parsing, home
  vs DSS containers, the ~5x-raw storage estimate), `cluster_workspace` (profile in `.workbench-cluster/workspace.json`,
  request bundles, the separate `view` project that shows remote results), `cluster_remote`/`cluster_runner` (run only
  inside Slurm on the compute node, never on a login node). `ui/cluster_backend` sits behind `ctx.jobs.submit`, so the
  normal Run buttons dispatch remotely while `ctx.cluster_enabled`; `ui/cluster_setup` is the setup window. Images
  travel over SSH (default) or Globus; passwords, MFA codes and tokens are never stored.
- Workers in other interpreters get `core/jobs.worker_env()` on PYTHONPATH, a staged copy of the package under
  `%APPDATA%/FeabasWorkbench/worker_pkg/<version hash>` holding only `core/`, `workers/`, `vendor/` (one folder per
  version of those files, so a running worker's files are never replaced). Never put `package_root()`
  itself there: for a wheel install that is site-packages (for a frozen build the bundle), and its compiled numpy/cv2
  would shadow the other interpreter's own. Release: bump `pyproject.toml`, `__init__.py`, `CITATION.cff`, README
  "How to cite", CHANGELOG; `python -m build` → `pip install dist/*.whl` in a clean venv; push tag `vX.Y.Z` → GitHub
  Release + PyPI (trusted publishing from `.github/workflows/release.yml`, environment `pypi`, no token).
