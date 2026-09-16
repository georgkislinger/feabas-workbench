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
- Pipeline state is derived from files on disk (`core/steps.py`); never keep step state elsewhere.
- FEABAS 3.0.5 run-time fixes live in `vendor/winfix/sitecustomize.py`, put first on PYTHONPATH of every FEABAS job
  (`ui/bridge.feabas_env`) so they reach multiprocessing children too; keep that when adding new ways to launch
  FEABAS. (1) all platforms: `matcher.stitching_matcher` only assigns `phtm` when `compute_photometric` is true but
  returns it unconditionally, so every tile pair that passes the coarse confidence check raises `UnboundLocalError`
  and stitch matching yields no matches at all - the patch defaults that flag to true. (2) Windows: FEABAS builds
  `file://D:/...` URLs that TensorStore rejects, rewritten to `file:///D:/...`. New projects default to the PNG
  (`image`) render driver.
- Three interpreters: the GUI's own, a FEABAS env (feabas + tensorstore, numpy < 2) and a deep-learning env
  (torch, careamics 0.3.2 which pins torch < 2.10, ultralytics, segmentation-models-pytorch). Paths live in
  `%APPDATA%/FeabasWorkbench/settings.json`; projects may override them.
- Heavy in-process work goes through `ui/threads.ThreadRunner`; anything that needs torch/feabas goes through a
  worker in the right env via `AppContext.worker_spec` / `feabas_step_spec`.
- Test data: `Example_data_to_stitch_and_align_and_export` (Thermo Maps tiles, 10 sections, 8×5 grid, 10 nm px;
  gitignored). The bundled fold U-Net is `feabas_workbench/resources/fold_unet_resnet34_inference_only_fp16.ckpt`
  (smp Unet, resnet34, fp16 weights only, 49 MB; `core.masks.bundled_fold_checkpoint()`, stored in project files as
  the sentinel `"bundled"` so a project survives a move or a different install location; inside the package so wheels ship
  it). It was exported from the 280 MB Lightning checkpoint `Fold_model_ckpt/lightning_logs/version_0/checkpoints/
  best-epoch=25-val_iou=0.6638.ckpt` (gitignored, optimizer state included) and gives identical detections;
  `workers/fold_train` fine-tunes from either, since it only ever reads the weights. `_test_projects/` and `_scratch/`
  are gitignored scratch areas.
- Workers in other interpreters get `core/jobs.worker_env()` on PYTHONPATH, a staged copy of the package under
  `%APPDATA%/FeabasWorkbench/worker_pkg` holding only `core/`, `workers/`, `vendor/`. Never put `package_root()`
  itself there: for a wheel install that is site-packages (for a frozen build the bundle), and its compiled numpy/cv2
  would shadow the other interpreter's own. Release: `python -m build` → `pip install dist/*.whl` in a clean venv.
