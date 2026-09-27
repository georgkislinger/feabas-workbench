"""
Regression tests for the pipeline's safety layer: what 'done', 'stale' and 'Clear' mean, test-run
sandboxes, snapshots, moved projects, settings that must not be overwritten, masks made for
earlier thumbnails, and files a cancelled job may leave behind.

Run: python -m pytest tests/test_safety.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from feabas_workbench.core.configs import ConfigStore, apply_suggested_mips
from feabas_workbench.core.project import Project, forget_cached_resolution, repair_working_directory
from feabas_workbench.core.steps import (STEPS_BY_KEY, PipelineScan, State, clear_step, clear_targets, create_snapshot,
                                         downstream, restore_snapshot)

T0 = time.time() - 100_000          # a fixed past: file times are set explicitly, no sleeping


def _project(tmp_path, n=3, name="proj"):
    p = Project.create(tmp_path / name)
    for i in range(1, n + 1):
        (p.stitch_coord_dir / f"s{i:04d}.txt").write_text("{ROOT_DIR}\t/x\n{TILE_SIZE}\t10\t10\na.tif\t0\t0\n",
                                                          encoding="utf-8")
    for f in list(p.configs_dir.glob("*.yaml")) + list(p.stitch_coord_dir.glob("*.txt")):
        os.utime(f, (T0, T0))
    return p


def _put(root: Path, rel: str, t: float, text: str = "x") -> Path:
    f = Path(root) / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
    os.utime(f, (t, t))
    return f


def _state(p, key, configs=None):
    return PipelineScan(p.root, len(p.section_names()), configs or ConfigStore(p.configs_dir))[key]


SECS = ("s0001", "s0002", "s0003")


# ----------------------------------------------------------------------------- stale detection

def test_saving_unchanged_settings_does_not_make_outputs_stale(tmp_path):
    p = _project(tmp_path)
    for s in SECS:
        _put(p.root, f"stitch/match_h5/{s}.h5", T0 + 100)
    cs = ConfigStore(p.configs_dir)
    assert _state(p, "stitch.matching", cs).state is State.COMPLETE
    cs.save("stitching")                                    # 'Apply' with nothing changed
    cs.save()
    assert _state(p, "stitch.matching").state is State.COMPLETE
    cs.set("stitching", "matching.num_workers", 3)
    cs.save("stitching")
    st = _state(p, "stitch.matching")
    assert st.state is State.STALE and "stitching config edited" in st.reasons[0]
    # back to FEABAS's defaults is a change too: the file stays, as {}, which FEABAS can read
    for s in SECS:
        _put(p.root, f"stitch/match_h5/{s}.h5", time.time() + 100)
    cs.docs["stitching"].overrides = {}
    cs.save("stitching")
    assert yaml.safe_load((p.configs_dir / "stitching_configs.yaml").read_text(encoding="utf-8")) == {}


def test_running_an_earlier_step_for_more_sections_keeps_finished_ones_current(tmp_path):
    p = _project(tmp_path)
    _put(p.root, "stitch/match_h5/s0001.h5", T0 + 100)
    _put(p.root, "stitch/tform/s0001.h5", T0 + 200)
    _put(p.root, "stitch/match_h5/s0002.h5", T0 + 300)       # a later subset run of 'Match tiles'
    st = _state(p, "stitch.optimization")
    assert st.state is State.PARTIAL, st.reasons
    _put(p.root, "stitch/match_h5/s0001.h5", T0 + 400)       # section 1 matched again: its montage is out of date
    assert _state(p, "stitch.optimization").state is State.STALE


def test_rewritten_coordinates_make_matching_and_everything_after_it_stale(tmp_path):
    p = _project(tmp_path)
    for s in SECS:
        _put(p.root, f"stitch/match_h5/{s}.h5", T0 + 100)
        _put(p.root, f"stitch/tform/{s}.h5", T0 + 200)
        _put(p.root, f"stitched_sections/mip0/{s}/metadata.txt", T0 + 300)
    scan = PipelineScan(p.root, 3, ConfigStore(p.configs_dir))
    assert all(scan[k].state is State.COMPLETE for k in ("stitch.matching", "stitch.optimization", "stitch.rendering"))
    os.utime(p.stitch_coord_dir / "s0002.txt", (T0 + 500, T0 + 500))    # e.g. switched to preprocessed tiles
    scan = PipelineScan(p.root, 3, ConfigStore(p.configs_dir))
    assert scan["stitch.matching"].state is State.STALE
    assert "coordinate files changed" in scan["stitch.matching"].reasons[0]
    # later steps inherit it, although their own inputs are older than they are
    assert scan["stitch.optimization"].state is State.STALE and "is stale" in scan["stitch.optimization"].reasons[-1]
    assert scan["stitch.rendering"].state is State.STALE


def test_meshes_follow_the_coarse_solution(tmp_path):
    assert "align.meshing" in [s.key for s in downstream("thumbnail.optimization")]
    p = _project(tmp_path)
    for s in SECS:
        _put(p.root, f"thumbnail_align/material_masks/{s}.png", T0 + 50)
        _put(p.root, f"thumbnail_align/tform/{s}.h5", T0 + 200)
        _put(p.root, f"align/mesh/{s}.h5", T0 + 300)
    for a, b in (("s0001", "s0002"), ("s0002", "s0003"), ("s0001", "s0003")):
        _put(p.root, f"thumbnail_align/matches/{a}__to__{b}.h5", T0 + 100)
    assert _state(p, "align.meshing").state is State.COMPLETE
    _put(p.root, "thumbnail_align/tform/s0002.h5", T0 + 400)       # coarse stack solved again
    assert _state(p, "align.meshing").state is State.STALE


# ----------------------------------------------------------------------------- volume and render folders

def _volume(root: Path, keys=("10_10_50", "20_20_50")) -> Path:
    vol = root / "aligned_tensorstore"
    scales = [{"key": k, "resolution": [float(k.split("_")[0])] * 2 + [50.0]} for k in keys]
    vol.mkdir(parents=True, exist_ok=True)
    (vol / "info").write_text(json.dumps({"scales": list(reversed(scales))}), encoding="utf-8")
    for k in keys:
        _put(vol, f"{k}/0-64_0-64_0-1", T0 + 100)
    return vol


def test_volume_mipmaps_are_judged_by_the_finished_levels(tmp_path):
    p = _project(tmp_path)
    _volume(p.root)                                          # the render has started: info and chunks exist
    scan = PipelineScan(p.root, 3, ConfigStore(p.configs_dir))
    assert scan["align.tsr"].done == 0 and scan["align.tsd"].done == 0
    _put(p.root, "align/ts_spec.json", T0 + 200, json.dumps({"0": {}}))     # written when the render finished
    scan = PipelineScan(p.root, 3, ConfigStore(p.configs_dir))
    assert scan["align.tsr"].state is State.COMPLETE
    assert (scan["align.tsd"].done, scan["align.tsd"].expected) == (0, 4) and scan["align.tsd"].state is not State.COMPLETE
    _put(p.root, "align/ts_spec.json", T0 + 300, json.dumps({str(m): {} for m in (0, 1, 3)}))
    scan = PipelineScan(p.root, 3, ConfigStore(p.configs_dir))
    assert scan["align.tsd"].state is State.PARTIAL and scan["align.tsd"].summary() == "2/4 mip levels"
    _put(p.root, "align/ts_spec.json", T0 + 400, json.dumps({str(m): {} for m in (0, 1, 3, 5, 7)}))
    assert _state(p, "align.tsd").state is State.COMPLETE


def test_clearing_mipmaps_keeps_the_full_resolution_render(tmp_path):
    p = _project(tmp_path)
    for m in range(0, 4):
        for s in SECS:
            _put(p.root, f"aligned_stack/mip{m}/0{SECS.index(s)}_{s}/metadata.txt", T0 + 100 + m)
    cs = ConfigStore(p.configs_dir)
    cs.set("alignment", "downsample.max_mip", 3)
    cs.save()
    scan = PipelineScan(p.root, 3, cs)
    assert scan["align.rendering"].done == 3 and scan["align.downsample"].done == 3
    targets = clear_targets(p.root, STEPS_BY_KEY["align.downsample"], configs=cs)
    assert sorted(t.name for t in targets) == ["mip1", "mip2", "mip3"]
    clear_step(p.root, STEPS_BY_KEY["align.downsample"], configs=cs)
    assert (p.root / "aligned_stack" / "mip0" / "00_s0001" / "metadata.txt").is_file()
    # the volume: only the downsampled scale goes, and the files listing the scales are cut back
    vol = _volume(p.root)
    _put(p.root, "align/ts_spec.json", T0 + 200, json.dumps({"0": {"a": 1}, "1": {"b": 2}}))
    _put(p.root, "align/mipmap_flags/mip1_f.json", T0 + 200, "[]")
    targets = clear_targets(p.root, STEPS_BY_KEY["align.tsd"])
    assert set(targets) == {vol / "20_20_50", p.root / "align" / "mipmap_flags"}
    clear_step(p.root, STEPS_BY_KEY["align.tsd"])
    assert (vol / "10_10_50").is_dir() and not (vol / "20_20_50").exists()
    assert [s["key"] for s in json.loads((vol / "info").read_text(encoding="utf-8"))["scales"]] == ["10_10_50"]
    assert json.loads((p.root / "align" / "ts_spec.json").read_text(encoding="utf-8")) == {"0": {"a": 1}}


def test_a_configured_render_folder_is_found_and_cleared_with_care(tmp_path):
    p = _project(tmp_path)
    elsewhere = tmp_path / "big_disk"
    _put(elsewhere, "notes.txt", T0)                         # not FEABAS's: must survive a clear
    for s in SECS[:2]:
        _put(elsewhere, f"mip0/0{SECS.index(s)}_{s}/metadata.txt", T0 + 100)
    cs = ConfigStore(p.configs_dir)
    cs.set("alignment", "rendering.out_dir", str(elsewhere))
    cs.save()
    st = _state(p, "align.rendering", cs)
    assert st.done == 2 and st.expected == 3
    targets = clear_targets(p.root, STEPS_BY_KEY["align.rendering"], configs=cs)
    assert elsewhere / "mip0" in targets and elsewhere not in targets and elsewhere / "notes.txt" not in targets
    clear_step(p.root, STEPS_BY_KEY["align.rendering"], configs=cs)
    assert (elsewhere / "notes.txt").is_file() and not (elsewhere / "mip0").exists()
    # a relative folder is relative to the project, and the render mip picks the level
    cs.set("alignment", "rendering.out_dir", "renders")
    cs.set("alignment", "rendering.mip_level", 2)
    cs.save()
    _put(p.root, "renders/mip2/00_s0001/metadata.txt", T0 + 100)
    assert _state(p, "align.rendering", cs).done == 1


# ----------------------------------------------------------------------------- clearing

def test_clearing_a_test_run_never_reaches_through_its_link(tmp_path):
    from feabas_workbench.core.testruns import create_align_test, delete_test_run
    p = _project(tmp_path)
    for s in SECS:
        _put(p.root, f"stitch/tform/{s}.h5", T0)
        _put(p.root, f"stitched_sections/mip0/{s}/metadata.txt", T0)
        _put(p.root, f"stitched_sections/mip1/{s}/metadata.txt", T0)
        _put(p.root, f"thumbnail_align/thumbnails/{s}.png", T0)
    try:
        tr = create_align_test(p, "t1", list(SECS[:2]))
    except RuntimeError as e:                                # no way to link on this machine
        pytest.skip(str(e))
    skipped = []
    targets = clear_targets(tr.root, STEPS_BY_KEY["thumbnail.downsample"], skipped=skipped)
    assert tr.root / "thumbnail_align" / "thumbnails" in targets
    assert not any("stitched_sections" in t.parts for t in targets)
    assert tr.root / "stitched_sections" / "mip1" in skipped
    clear_step(tr.root, STEPS_BY_KEY["thumbnail.downsample"])
    assert (p.root / "stitched_sections" / "mip1" / "s0001" / "metadata.txt").is_file()
    assert not (tr.root / "thumbnail_align" / "thumbnails").exists()
    delete_test_run(tr)
    assert not tr.root.exists() and (p.root / "stitched_sections" / "mip0" / "s0001" / "metadata.txt").is_file()


def test_clearing_thumbnails_takes_the_masks_drawn_on_them(tmp_path):
    p = _project(tmp_path)
    for rel in ("thumbnail_align/thumbnails/s0001.png", "thumbnail_align/material_masks/s0001.png",
                "masks/roi/s0001.png", "masks/tissue/s0001.png", "masks/folds/s0001.png", "masks/manifest.json",
                "masks/external/tissue/s0001.png"):
        _put(p.root, rel, T0)
    rel_targets = {t.relative_to(p.root).as_posix() for t in clear_targets(p.root, STEPS_BY_KEY["masks"])}
    assert rel_targets == {"thumbnail_align/material_masks", "masks/manifest.json"}          # tissue/folds stay
    rel_targets = {t.relative_to(p.root).as_posix() for t in clear_targets(p.root, STEPS_BY_KEY["thumbnail.downsample"])}
    assert {"masks/roi", "masks/tissue", "masks/folds", "masks/manifest.json"} <= rel_targets
    assert "masks/external" not in rel_targets                                              # imports are the user's


# ----------------------------------------------------------------------------- snapshots

def test_restoring_a_snapshot_flags_what_was_computed_from_other_inputs(tmp_path):
    p = _project(tmp_path)
    pairs = (("s0001", "s0002"), ("s0002", "s0003"), ("s0001", "s0003"))
    for s in SECS:
        for sub in ("stitch/match_h5", "stitch/tform", "thumbnail_align/tform"):
            _put(p.root, f"{sub}/{s}.h5", T0 + 100)
        _put(p.root, f"stitched_sections/mip0/{s}/metadata.txt", T0 + 100)
        _put(p.root, f"thumbnail_align/thumbnails/{s}.png", T0 + 110)
        _put(p.root, f"thumbnail_align/material_masks/{s}.png", T0 + 120)
    for a, b in pairs:
        _put(p.root, f"thumbnail_align/matches/{a}__to__{b}.h5", T0 + 150, "coarse v1")
    snap = create_snapshot(p.root, "coarse v1")
    # restoring right away changes nothing, so nothing becomes stale
    before = {f: f.stat().st_mtime for f in (p.root / "thumbnail_align" / "matches").iterdir()}
    restore_snapshot(p.root, snap)
    assert {f: f.stat().st_mtime for f in (p.root / "thumbnail_align" / "matches").iterdir()} == before
    # then: new coarse matches and a fine alignment built on them
    for a, b in pairs:
        _put(p.root, f"thumbnail_align/matches/{a}__to__{b}.h5", T0 + 300, "coarse v2")
    for s in SECS:
        _put(p.root, f"thumbnail_align/tform/{s}.h5", T0 + 310)
        _put(p.root, f"align/mesh/{s}.h5", T0 + 400)
        _put(p.root, f"align/tform/{s}.h5", T0 + 500)
    for a, b in pairs:
        _put(p.root, f"align/matches/{a}__to__{b}.h5", T0 + 450)
    assert _state(p, "align.optimization").state is State.COMPLETE
    restore_snapshot(p.root, snap)
    assert (p.root / "thumbnail_align/matches/s0001__to__s0002.h5").read_text(encoding="utf-8") == "coarse v1"
    scan = PipelineScan(p.root, 3, ConfigStore(p.configs_dir))
    for key in ("thumbnail.optimization", "align.meshing", "align.matching", "align.optimization"):
        assert scan[key].state is State.STALE, (key, scan[key].reasons)


def test_snapshot_paths_outside_the_list_are_ignored(tmp_path):
    p = _project(tmp_path)
    snap = create_snapshot(p.root, "x")
    meta = json.loads((snap / "snapshot.json").read_text(encoding="utf-8"))
    meta["paths"].append("../../escape")
    (snap / "snapshot.json").write_text(json.dumps(meta), encoding="utf-8")
    assert "../../escape" not in restore_snapshot(p.root, snap)


# ----------------------------------------------------------------------------- moved projects, general config

def test_a_copied_project_points_feabas_at_itself(tmp_path):
    from feabas_workbench.core.testruns import create_montage_test
    p = _project(tmp_path, name="original")
    create_montage_test(p, "t1", ["s0001"])
    copy = tmp_path / "copy"
    shutil.copytree(p.root, copy)
    wd = lambda root: yaml.safe_load((root / "configs" / "general_configs.yaml").read_text(encoding="utf-8"))["working_directory"]
    assert Path(wd(copy)) == p.root                            # what a plain copy carries
    Project.load(copy)
    assert Path(wd(copy)).resolve() == copy.resolve()
    assert repair_working_directory(copy / "tests" / "t1") is True
    assert Path(wd(copy / "tests" / "t1")).resolve() == (copy / "tests" / "t1").resolve()
    assert repair_working_directory(copy) is False             # nothing to do the second time


def test_general_config_takes_explicit_values_and_is_left_alone_otherwise(tmp_path):
    p = _project(tmp_path)
    g = lambda: yaml.safe_load(p.general_config_path().read_text(encoding="utf-8"))
    p.write_general_config(cpu_budget=8, parallel_framework="thread", logfile_level="DEBUG")
    assert (g()["cpu_budget"], g()["parallel_framework"], g()["logfile_level"]) == (8, "thread", "DEBUG")
    os.utime(p.general_config_path(), (T0, T0))
    p.write_general_config()                                   # e.g. writing coordinates: keeps the choices
    assert p.general_config_path().stat().st_mtime == T0 and g()["parallel_framework"] == "thread"
    p.write_general_config(cpu_budget=None)                    # "all physical cores"
    assert g()["cpu_budget"] is None and g()["logfile_level"] == "DEBUG"


def test_a_changed_pixel_size_drops_feabas_cached_resolution(tmp_path):
    p = _project(tmp_path)
    cache = p.configs_dir / "resolutions.yaml"
    cache.write_text("DATA_RESOLUTION: 4.0\n", encoding="utf-8")
    assert forget_cached_resolution(p.root, 4.0) is False and cache.is_file()
    assert forget_cached_resolution(p.root, 10.0) is True and not cache.exists()


def test_suggested_mips_never_overwrite_a_users_choice(tmp_path):
    p = _project(tmp_path)
    cs = ConfigStore(p.configs_dir)
    record, notes = apply_suggested_mips(cs, {}, thumbnail_mip=5, working_mip=2)
    assert cs.get("thumbnail", "thumbnail_mip_level") == 5 and not notes
    record, notes = apply_suggested_mips(cs, record, thumbnail_mip=4, working_mip=3)     # new pixel size
    assert cs.get("thumbnail", "thumbnail_mip_level") == 4 and cs.get("alignment", "matching.working_mip_level") == 3
    cs.set("thumbnail", "thumbnail_mip_level", 6)                                        # the user tunes it
    record, notes = apply_suggested_mips(cs, record, thumbnail_mip=4, working_mip=1)
    assert cs.get("thumbnail", "thumbnail_mip_level") == 6 and "kept your thumbnail mip 6" in notes[0]
    assert cs.get("alignment", "matching.working_mip_level") == 1


# ----------------------------------------------------------------------------- masks

def test_split_lines_use_feabas_split_material():
    from feabas_workbench.core.masks import LABEL_SPLIT, paint_split_line
    m = paint_split_line(np.zeros((40, 40), np.uint8), (2, 20), (37, 20), width=3)
    assert set(np.unique(m)) == {0, LABEL_SPLIT}


def _thumb_project(tmp_path, size=120):
    import cv2
    from feabas_workbench.core.masks import write_mask
    p = _project(tmp_path)
    rng = np.random.default_rng(0)
    img = np.zeros((size, size), np.uint8)
    img[10:-10, 10:-10] = rng.integers(60, 200, (size - 20, size - 20), dtype=np.uint8)
    d = p.root / "thumbnail_align" / "thumbnails"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "s0001.png"), img)
    mask = np.full((size, size), 255, np.uint8)
    mask[10:-10, 10:-10] = 0
    write_mask(p.root / "thumbnail_align" / "material_masks" / "s0001.png", mask)
    return p, img


def test_masks_made_for_earlier_thumbnails_are_rebuilt_not_misused(tmp_path):
    import cv2
    from feabas_workbench.core.masks import TissueParams, write_mask
    from feabas_workbench.core.maskstore import MaskStore
    p, img = _thumb_project(tmp_path)
    store = MaskStore(p)
    store.compute_tissue("s0001", TissueParams(method="all"))         # backs up FEABAS's footprint (120 px)
    old = time.time() - 50
    for f in (store.roi_dir / "s0001.png", store.tissue_dir / "s0001.png"):
        os.utime(f, (old, old))
    # the thumbnails are made again at another mip level, with FEABAS's new default mask
    small = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(p.root / "thumbnail_align" / "thumbnails" / "s0001.png"), small)
    mask = np.full((60, 60), 255, np.uint8); mask[5:-5, 5:-5] = 0
    write_mask(p.root / "thumbnail_align" / "material_masks" / "s0001.png", mask)
    assert store.tissue_is_stale("s0001")
    for method in ("manual", "border", "all"):
        m, _ = store.compute_tissue("s0001", TissueParams(method=method, crop_left=2))
        assert m.shape == (60, 60), method
    assert store.ensure_tissue("s0001", TissueParams()) is False        # fresh again
    # a mismatched footprint handed in directly no longer crashes either
    from feabas_workbench.core.masks import detect_tissue
    assert detect_tissue(small, TissueParams(method="manual"), roi=np.ones((120, 120), bool))[0].shape == (60, 60)


def test_hole_filling_matches_the_reference_loop():
    import cv2
    from feabas_workbench.core.masks import fill_holes
    rng = np.random.default_rng(3)
    mask = rng.random((80, 80)) > 0.35
    inv = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    ref = mask.copy()
    for i in range(1, n):
        if i not in border and stats[i, cv2.CC_STAT_AREA] <= 5:
            ref[lab == i] = True
    assert np.array_equal(fill_holes(mask, 5), ref)


def test_mask_writes_leave_no_countable_temp_file(tmp_path):
    from feabas_workbench.core.masks import write_mask
    write_mask(tmp_path / "s0001.png", np.zeros((4, 4), np.uint8))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s0001.png"]


# ----------------------------------------------------------------------------- tiles and metadata

def test_print_resolutions_are_not_taken_for_pixel_sizes(tmp_path):
    import tifffile
    from feabas_workbench.core.tiles import read_tile_info
    a = np.zeros((16, 16), np.uint8)
    tifffile.imwrite(tmp_path / "dpi72.tif", a, resolution=(72, 72), resolutionunit="INCH")
    tifffile.imwrite(tmp_path / "cm.tif", a, resolution=(1e6, 1e6), resolutionunit="CENTIMETER")      # 10 nm
    tifffile.imwrite(tmp_path / "ij.tif", a, imagej=True, resolution=(0.2, 0.2), metadata={"unit": "nm"})
    assert read_tile_info(tmp_path / "dpi72.tif").pixel_size_nm is None
    assert read_tile_info(tmp_path / "cm.tif").pixel_size_nm == pytest.approx(10.0)
    assert read_tile_info(tmp_path / "ij.tif").pixel_size_nm == pytest.approx(5.0)


def test_scans_skip_a_project_folder_inside_the_tile_folder(tmp_path):
    from feabas_workbench.core.tiles import list_image_files
    tiles = tmp_path / "tiles"
    for name in ("a.tif", "sub/b.tif"):
        (tiles / name).parent.mkdir(parents=True, exist_ok=True)
        (tiles / name).write_bytes(b"")
    proj = tiles / "project"
    (proj / "preprocessed" / "histmatch").mkdir(parents=True)
    (proj / "preprocessed" / "histmatch" / "a.tif").write_bytes(b"")
    assert len(list_image_files(tiles, "tif")) == 3
    assert [p.name for p in list_image_files(tiles, "tif", exclude=[proj])] == ["a.tif", "b.tif"]


def test_absolute_coordinate_files_follow_the_tile_source(tmp_path):
    from feabas_workbench.core.testruns import retarget_stitch_coords
    p = _project(tmp_path, n=0)
    raw = (tmp_path / "raw").resolve()
    p.state.source.root_dir = str(raw)
    (p.stitch_coord_dir / "s0001.txt").write_text(
        f"{{RESOLUTION}}\t4\n{{TILE_SIZE}}\t10\t10\n{raw.as_posix()}/sub/a.tif\t0\t0\n", encoding="utf-8")
    new = p.preprocessed_dir / "histmatch"
    assert retarget_stitch_coords(p, new) == 1
    assert f"{(new / 'sub' / 'a.tif').as_posix()}\t0\t0" in (p.stitch_coord_dir / "s0001.txt").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- jobs and files cut short

def test_an_interrupted_preprocessing_write_leaves_no_file_behind(tmp_path, monkeypatch):
    import tifffile
    from feabas_workbench.core import histmatch
    real = tifffile.imwrite

    def dying(path, *a, **kw):
        real(path, *a, **kw)
        raise KeyboardInterrupt                       # killed right after the bytes hit the disk
    monkeypatch.setattr(tifffile, "imwrite", dying)
    with pytest.raises(KeyboardInterrupt):
        histmatch.write_gray(tmp_path / "t.tif", np.zeros((8, 8), np.uint8))
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(tifffile, "imwrite", real)
    histmatch.write_gray(tmp_path / "t.tif", np.zeros((8, 8), np.uint8))
    assert [p.name for p in tmp_path.iterdir()] == ["t.tif"]


def test_spec_files_of_the_same_worker_do_not_overwrite_each_other(tmp_path):
    from feabas_workbench.core.jobs import write_spec_file
    a = write_spec_file(tmp_path, "fold_predict", {"n": 1})
    b = write_spec_file(tmp_path, "fold_predict", {"n": 2})
    assert a != b and json.loads(a.read_text())["n"] == 1


def test_unreadable_h5_from_a_cancelled_run_is_removed(tmp_path):
    h5py = pytest.importorskip("h5py")
    from feabas_workbench.core.steps import drop_unreadable_outputs
    p = _project(tmp_path)
    d = p.root / "stitch" / "match_h5"
    d.mkdir(parents=True)
    with h5py.File(d / "s0001.h5", "w") as f:
        f["x"] = np.arange(3)
    (d / "s0002.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 20)       # cut short by the kill
    (d / "s0003.h5").write_bytes(b"old junk")                             # from long before this run
    os.utime(d / "s0003.h5", (T0, T0))
    started = time.time() - 5
    removed = drop_unreadable_outputs(p.root, STEPS_BY_KEY["stitch.matching"], started)
    assert [r.name for r in removed] == ["s0002.h5"]
    assert (d / "s0001.h5").is_file() and (d / "s0003.h5").is_file()


def test_cancelled_jobs_run_their_clean_up(tmp_path):
    from feabas_workbench.core.jobs import Job, JobSpec
    seen, done = [], threading.Event()
    spec = JobSpec("sleeper", [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path,
                   after_cancel=lambda started: (seen.append(started), ["tidied up"])[1])
    results = []
    job = Job(spec, on_finished=lambda r: (results.append(r), done.set()))
    job.start()
    time.sleep(0.5)
    job.cancel()
    assert done.wait(30)
    assert results[0].cancelled and seen and abs(seen[0] - job.started_at) < 1e-6
    assert "tidied up" in results[0].last_lines


def test_worker_packages_are_staged_per_version(tmp_path, monkeypatch):
    from feabas_workbench.core import envs, jobs
    monkeypatch.setattr(envs, "settings_dir", lambda: tmp_path)
    first = jobs.worker_package_root()
    fake = tmp_path / "src" / "feabas_workbench"
    shutil.copytree(first / "feabas_workbench", fake)
    monkeypatch.setattr(jobs, "package_root", lambda: tmp_path / "src")
    second = jobs.worker_package_root()
    (fake / "core" / "extra.py").write_text("x = 1\n", encoding="utf-8")
    third = jobs.worker_package_root()
    assert len({first, second, third}) == 3 and first.is_dir() and second.is_dir()     # a newer one never replaces a copy in use
    assert (third / "feabas_workbench" / "core" / "extra.py").is_file()


def test_the_runtime_patch_keeps_an_environments_own_sitecustomize(tmp_path):
    import subprocess
    from feabas_workbench.core.project import VENDOR_DIR
    env_dir = tmp_path / "env_site"
    env_dir.mkdir()
    (env_dir / "sitecustomize.py").write_text(
        "import os\nopen(os.environ['FW_TEST_MARKER'], 'w').write('ran')\n", encoding="utf-8")
    marker = tmp_path / "marker"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(VENDOR_DIR.parent / "winfix"), str(env_dir)]),
               FW_TEST_MARKER=str(marker))
    subprocess.run([sys.executable, "-c", "pass"], env=env, check=True, timeout=60)
    assert marker.read_text() == "ran"


# ----------------------------------------------------------------------------- setup

def test_pytorch_build_matches_the_gpu_generation():
    from feabas_workbench.core.envs import GpuInfo
    assert GpuInfo(True, "RTX 5090", "575.51", 32607, 12.0).torch_index().endswith("/cu128")
    assert GpuInfo(True, "RTX 5090", "566.03", 32607, 12.0).torch_index().endswith("/cpu")      # driver too old
    assert GpuInfo(True, "RTX 4090", "560.94", 24564, 8.9).torch_index().endswith("/cu126")
    assert GpuInfo(True, "old driver", "470.10", 8000, 7.5).torch_index().endswith("/cu118")
    assert GpuInfo(True, "no cap field", "550.00", 8000).torch_index().endswith("/cu126")


def test_install_plans_respect_pip_configuration(tmp_path):
    from feabas_workbench.core.envs import InstallPlan
    cmds = InstallPlan("feabas", "fw-feabas", "3.12", tmp_path / "micromamba").pip_commands("python")
    assert "https://pypi.org/simple" not in cmds[0]
    cmds = InstallPlan("dl", "fw-dl", "3.11", tmp_path / "micromamba", "https://download.pytorch.org/whl/cu128").pip_commands("python")
    assert cmds[0][4:6] == ["--index-url", "https://download.pytorch.org/whl/cu128"]
    assert "https://pypi.org/simple" not in cmds[1]


def test_micromamba_download_is_checked_against_its_checksum(tmp_path, monkeypatch):
    import hashlib
    import io
    import platform
    from feabas_workbench.core import envs
    if (platform.system(), platform.machine()) not in envs.MICROMAMBA_ASSETS:
        pytest.skip("no micromamba build for this platform")
    monkeypatch.setattr(envs, "settings_dir", lambda: tmp_path)
    payload = b"\x7fELF fake micromamba"
    served = {"sum": hashlib.sha256(payload).hexdigest()}

    def fake_urlopen(url, timeout=None):
        body = (served["sum"] + "  micromamba\n").encode() if url.endswith(".sha256") else payload
        return io.BytesIO(body)
    monkeypatch.setattr(envs.urllib.request, "urlopen", fake_urlopen)
    exe = envs.download_micromamba()
    assert exe.read_bytes() == payload and exe.parent.parent == tmp_path / "micromamba" or exe.parent == tmp_path / "micromamba"
    exe.unlink()
    served["sum"] = "0" * 64
    with pytest.raises(RuntimeError, match="checksum"):
        envs.download_micromamba()
    assert not exe.exists() and not list(exe.parent.glob("*.part"))


# ----------------------------------------------------------------------------- export

def test_a_stack_rendered_at_a_coarser_mip_exports_with_that_level_as_full_resolution(tmp_path):
    import cv2
    from feabas_workbench.workers.export_vast import export_vast
    base = tmp_path / "aligned_stack"
    for m, size in ((2, 64), (3, 32)):
        d = base / f"mip{m}" / "00_s0001"
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "s0001_tr1-tc1.png"), np.zeros((size, size), np.uint8))
        (d / "metadata.txt").write_text(f"{{ROOT_DIR}}\t{d}\n{{TILE_SIZE}}\t{size}\t{size}\n"
                                        f"s0001_tr1-tc1.png\t0\t0\t{size}\t{size}\n", encoding="utf-8")
    res = export_vast(base, tmp_path / "vast", "a", (10.0, 10.0, 50.0), [2, 3], base_mip=2)
    vsvi = json.loads(Path(res["vsvi"]).read_text(encoding="utf-8"))
    assert vsvi["TargetVoxelSizeXnm"] == 40.0 and vsvi["TargetVoxelSizeZnm"] == 50.0 and vsvi["SourceMaxM"] == 1
    assert (tmp_path / "vast" / "mip0" / "slice0000" / "0000_tr1-tc1.png").is_file()
    assert (tmp_path / "vast" / "mip1" / "slice0000" / "0000_tr1-tc1.png").is_file()


def test_a_montage_test_run_leaves_the_projects_settings_alone(tmp_path):
    from feabas_workbench.core.testruns import create_montage_test
    p = _project(tmp_path)
    before = {f.name: f.stat().st_mtime for f in p.configs_dir.glob("*.yaml")}
    tr = create_montage_test(p, "t1", ["s0001"], settings={"stitching": {"matching": {"num_workers": 2}}})
    assert ConfigStore(tr.root / "configs").get("stitching", "matching.num_workers") == 2
    assert ConfigStore(p.configs_dir).get("stitching", "matching.num_workers") == 15
    assert {f.name: f.stat().st_mtime for f in p.configs_dir.glob("*.yaml")} == before


# ----------------------------------------------------------------------------- the end-to-end check

def test_match_files_without_matches_fail_the_end_to_end_check(tmp_path):
    import h5py
    from feabas_workbench.core.pipeline import match_problems
    for sub in ("stitch/match_h5", "align/matches"):
        (tmp_path / sub).mkdir(parents=True)
    with h5py.File(tmp_path / "stitch/match_h5/s0001.h5", "w") as h:          # a connected montage
        h.create_dataset("matches/1_0", data=np.zeros(8, np.float32))
        h.create_dataset("connected_subsystem", data=np.array([0, 0], np.int32))
    with h5py.File(tmp_path / "stitch/match_h5/s0002.h5", "w") as h:          # tiles that never matched
        h.create_dataset("connected_subsystem", data=np.array([0, 1], np.int32))
    with h5py.File(tmp_path / "align/matches/s0001__to__s0002.h5", "w") as h:  # one point for a whole pair
        h.create_dataset("xy0", data=np.zeros((1, 2)))
    problems = match_problems(tmp_path)
    assert len(problems) == 2, problems
    assert "s0002.h5" in problems[0] and "2 separate groups" in problems[0]
    assert "1 match point(s)" in problems[1]


def test_the_demo_project_matches_on_a_grid_its_sections_can_hold(tmp_path):
    from feabas_workbench.core.synthetic import demo_alignment_grid, make_demo_project
    p = make_demo_project(tmp_path / "demo", n_sections=2, rows=2, cols=2, tile=384)
    cs = ConfigStore(p.configs_dir)
    assert cs.get("alignment", "meshing.mesh_size") == 146          # 730 px sections
    assert cs.get("alignment", "matching.matcher_config.spacings") == [91, 26]
    assert demo_alignment_grid(40000, 2) == (8000, [5000, 1429])     # a real section keeps coarse grids


# ============================================================================= the GUI (offscreen)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(app, tmp_path, monkeypatch):
    from feabas_workbench.core import envs
    monkeypatch.setattr(envs, "SETTINGS_FILE", tmp_path / "settings.json")
    from feabas_workbench.ui.main_window import MainWindow
    s = envs.Settings()
    s.feabas_python = ""
    w = MainWindow(s)
    w.show()
    app.processEvents()
    yield w
    w.shutdown()
    w.close()
    app.processEvents()


def _answer(monkeypatch, button):
    """Make every QMessageBox.question / exec answer *button*."""
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: button))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: button)
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: QMessageBox.Ok))


def test_gui_clear_on_a_test_run_clears_only_that_test_run(app, window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from feabas_workbench.core.synthetic import make_demo_project
    from feabas_workbench.core.testruns import create_montage_test
    make_demo_project(tmp_path / "demo", n_sections=2, rows=1, cols=2, tile=96)
    window.open_project(tmp_path / "demo")
    app.processEvents()
    root = window.ctx.project.root
    secs = window.ctx.project.section_names()
    for rel in (f"stitch/match_h5/{secs[0]}.h5", f"stitch/tform/{secs[0]}.h5", f"stitched_sections/mip0/{secs[0]}/metadata.txt"):
        _put(root, rel, time.time(), "precious")
    tr = create_montage_test(window.ctx.project, "t1", secs[:1], None)
    _put(tr.root, f"stitch/match_h5/{secs[0]}.h5", time.time(), "test")
    page = window._pages[3]
    page._refresh_tests(select="t1")
    page.on_state_changed()
    app.processEvents()
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: (shown.append(self.informativeText()), QMessageBox.Yes)[1])
    page.test_steps.cards["stitch.matching"].clear_btn.click()
    app.processEvents()
    assert shown and "tests" in shown[0]
    assert (root / "stitch" / "match_h5").is_dir() and (root / "stitched_sections").is_dir()
    assert not (tr.root / "stitch" / "match_h5").exists()


def test_gui_setup_apply_writes_every_compute_setting(app, window, tmp_path):
    window.open_project(tmp_path / "p")
    page = window._pages[0]
    page.cpu.setValue(6)
    page.framework.setCurrentIndex(page.framework.findData("thread"))
    page.loglevel.setCurrentIndex(page.loglevel.findData("DEBUG"))
    page._apply_general()
    g = yaml.safe_load(window.ctx.project.general_config_path().read_text(encoding="utf-8"))
    assert (g["cpu_budget"], g["parallel_framework"], g["logfile_level"]) == (6, "thread", "DEBUG")
    page.cpu.setValue(0)
    page._apply_general()
    assert yaml.safe_load(window.ctx.project.general_config_path().read_text(encoding="utf-8"))["cpu_budget"] is None


def test_gui_opening_a_plain_folder_asks_before_making_it_a_project(app, window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    folder = tmp_path / "raw_data"
    folder.mkdir()
    (folder / "tile.tif").write_bytes(b"")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder)))
    _answer(monkeypatch, QMessageBox.No)
    window.choose_project()
    assert not Project.exists(folder) and not (folder / "configs").exists()
    _answer(monkeypatch, QMessageBox.Yes)
    window.choose_project()
    assert Project.exists(folder)


def test_gui_a_missing_recent_project_is_not_recreated(app, window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    gone = tmp_path / "gone"
    window.ctx.settings.recent_projects = [str(gone)]
    _answer(monkeypatch, QMessageBox.Yes)                       # "remove it from the list?" - yes
    window._open_recent(str(gone))
    assert not gone.exists() and window.ctx.settings.recent_projects == []


def test_gui_running_a_stale_step_offers_to_clear_first(app, window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from feabas_workbench.core.jobs import JobSpec
    p = _project(tmp_path)
    for s in SECS:
        _put(p.root, f"stitch/match_h5/{s}.h5", T0 + 100)
    cs = ConfigStore(p.configs_dir)
    cs.set("stitching", "matching.num_workers", 2)
    cs.save()                                                     # newer than the outputs: stale
    window.open_project(p.root)
    page = window._pages[3]
    page.on_state_changed()
    panel = page.steps
    assert panel.cards["stitch.matching"].status.state is State.STALE
    submitted, cleared = [], []
    monkeypatch.setattr(panel, "_spec", lambda step, *a: JobSpec(step.label, ["x"], p.root))
    monkeypatch.setattr(window.ctx.jobs, "submit", lambda spec: submitted.append(spec))
    monkeypatch.setattr(window, "confirm_clear", lambda step, cascade=True, root=None: cleared.append(step.key) or True)

    def choose(text):
        def exec_(box):
            box._choice = next((b for b in box.buttons() if b.text() == text), None)
            return 0
        monkeypatch.setattr(QMessageBox, "exec", exec_)
        monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box._choice)
    step = STEPS_BY_KEY["stitch.matching"]
    choose("Cancel")
    panel._run(step, None, None, None)
    assert not submitted and not cleared
    choose("Clear, then run")
    panel._run(step, None, None, None)
    assert cleared == ["stitch.matching"] and len(submitted) == 1
    choose("Run without clearing")
    panel._run(step, None, None, None)
    assert len(submitted) == 2 and cleared == ["stitch.matching"]


def test_gui_split_line_is_painted_as_split_material(app, window, tmp_path):
    from feabas_workbench.core.masks import LABEL_SPLIT, read_mask
    p, _img = _thumb_project(tmp_path)
    window.open_project(p.root)
    page = window._pages[4]
    page.sections.set_current("s0001")
    page.split_btn.setChecked(True)
    page._view_clicked(15.0, 60.0)
    page._view_clicked(105.0, 60.0)
    m = read_mask(p.root / "thumbnail_align" / "material_masks" / "s0001.png")
    assert (m[60, 20:100] == LABEL_SPLIT).all() and page.store.is_hand_edited("s0001")
