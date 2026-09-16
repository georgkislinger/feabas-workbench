"""Qt-free core tests. Run: python -m pytest tests -q"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feabas_workbench.core import tiles as T
from feabas_workbench.core import histmatch as H
from feabas_workbench.core.configs import ConfigStore, deep_merge, deep_diff, harvest_hints, parse_value
from feabas_workbench.core.project import Project, VENDOR_DIR
from feabas_workbench.core.steps import PipelineScan, STEPS_BY_KEY, downstream, clear_targets, create_snapshot, restore_snapshot
from feabas_workbench.core.masks import detect_tissue, compose_material_mask, TissueParams, LABEL_EXCLUDE, LABEL_DEFAULT
from feabas_workbench.core.testruns import create_montage_test, parse_stitch_coord, retarget_stitch_coords


# ----------------------------------------------------------------------------- naming rules

def test_thermo_preset_parses_row_col_section():
    r = T.NamingRule(preset="thermo")
    c = r.parse("Tile_003-002-000034_0-000.s0035_e00")
    assert (c.x, c.y, c.z) == (2, 3, 35)


def test_zeiss_preset():
    r = T.NamingRule(preset="zeiss")
    c = r.parse("Tile_r1-c2_S_0035_7")
    assert (c.x, c.y, c.z) == (2, 1, 35)


def test_sequential_and_custom():
    assert T.parse_sequential("img_3_5_120", "xyz") == T.Coord(3, 5, 120)
    assert T.parse_sequential("img_3_5_120", "rz") == T.Coord(0, 0, 5)
    r = T.NamingRule(preset="custom", template="sec#z#_row#y#_col#x#")
    assert r.parse("sec12_row3_col4") == T.Coord(4, 3, 12)
    assert r.parse("garbage") is None


def test_guess_rules_prefers_thermo(tmp_path):
    files = [tmp_path / f"Tile_{r:03d}-{c:03d}-{n:06d}_0-000.s{z:04d}_e00.tif" for z in range(3) for r in range(1, 4) for c in range(1, 3) for n in [z * 10 + r]]
    for f in files:
        f.write_bytes(b"")
    guesses = T.guess_rules(files)
    assert guesses
    best = guesses[0]
    assert best.n_z == 3 and best.n_y == 3 and best.n_x == 2


# ----------------------------------------------------------------------------- placement

def _tiles(rows=3, cols=4, z=1):
    return [T.Tile(Path(f"t_{r}_{c}.tif"), c, r, z) for r in range(rows) for c in range(cols)]


def test_grid_placement_with_percent_overlap():
    tl = _tiles()
    lay = T.LayoutParams(tile_w=1000, tile_h=500, overlap_x=10, overlap_y=20, overlap_unit="percent")
    T.place_tiles_grid(tl, lay)
    by = {(t.ix, t.iy): t for t in tl}
    assert by[(1, 0)].x_px == pytest.approx(900)
    assert by[(0, 1)].y_px == pytest.approx(400)
    assert by[(0, 0)].x_px == 0 and by[(0, 0)].y_px == 0


def test_stage_placement_recovers_rotated_grid():
    tl = _tiles(4, 5)
    px = 10e-9
    theta = np.deg2rad(-137.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rng = np.random.default_rng(0)
    for t in tl:
        nominal = np.array([t.ix * 900.0, t.iy * 400.0]) + rng.normal(0, 3, 2)
        s = R @ nominal * px
        t.stage_xy_m = (float(s[0]) + 0.01, float(s[1]) - 0.03)
    lay = T.LayoutParams(tile_w=1000, tile_h=500, mode="stage")
    assert T.place_tiles_stage(tl, lay, 10.0)
    by = {(t.ix, t.iy): t for t in tl}
    assert by[(1, 0)].x_px - by[(0, 0)].x_px == pytest.approx(900, abs=15)
    assert by[(0, 1)].y_px - by[(0, 0)].y_px == pytest.approx(400, abs=15)
    assert abs(by[(1, 0)].y_px - by[(0, 0)].y_px) < 15
    assert min(t.x_px for t in tl) == 0 and min(t.y_px for t in tl) == 0


def test_stitch_coord_format(tmp_path):
    tl = _tiles(1, 2)
    for t in tl:
        t.path = tmp_path / "raw" / t.path.name
        t.path.parent.mkdir(exist_ok=True)
        t.path.write_bytes(b"")
    T.place_tiles_grid(tl, T.LayoutParams(tile_w=100, tile_h=50, overlap_x=10, overlap_y=10))
    txt = T.format_stitch_coord(tl, tmp_path / "raw", 4.0, 50, 100)
    lines = txt.splitlines()
    assert lines[0].startswith("{ROOT_DIR}\t")
    assert lines[1] == "{RESOLUTION}\t4"
    assert lines[2] == "{TILE_SIZE}\t50\t100"
    assert lines[3].split("\t") == ["t_0_0.tif", "0.0", "0.0"]
    assert lines[4].split("\t") == ["t_0_1.tif", "90.0", "0.0"]


# ----------------------------------------------------------------------------- histogram matching

def test_histogram_matching_moves_distribution():
    rng = np.random.default_rng(1)
    tmpl = np.clip(rng.normal(150, 30, (256, 256)), 0, 255).astype(np.uint8)
    src = np.clip(rng.normal(80, 15, (256, 256)), 0, 255).astype(np.uint8)
    src[:20] = 0
    cT = H.template_cdf(tmpl, True, False)
    out = H.match_image(src, cT, True, False)
    assert out.dtype == np.uint8
    assert abs(float(out[20:].mean()) - 150) < 6
    assert (out[:20] == 0).all()


# ----------------------------------------------------------------------------- configs

def test_config_merge_diff_and_hints(tmp_path):
    cfg = tmp_path / "configs"
    cfg.mkdir()
    for f in (VENDOR_DIR / "configs").glob("default_*.yaml"):
        (cfg / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    cs = ConfigStore(cfg)
    assert cs.get("stitching", "matching.num_workers") == 15
    cs.set("stitching", "matching.num_workers", 4)
    cs.set("stitching", "rendering.driver", "image")
    cs.save()
    text = (cfg / "stitching_configs.yaml").read_text(encoding="utf-8")
    assert "num_workers: 4" in text and "driver: image" in text and "optimization" not in text
    cs2 = ConfigStore(cfg)
    assert cs2["stitching"].is_overridden("matching.num_workers")
    cs2.set("stitching", "matching.num_workers", 15)   # back to default removes override
    cs2.save()
    assert "num_workers" not in (cfg / "stitching_configs.yaml").read_text(encoding="utf-8")
    hints = harvest_hints(cfg / "default_stitching_configs.yaml")
    assert "matching.margin" in hints and "stage" in hints["matching.margin"]
    assert deep_diff({"a": {"b": 1, "c": 2}}, {"a": {"b": 1, "c": 3}}) == {"a": {"c": 3}}
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}
    assert parse_value("true", False) is True and parse_value("null", 3) is None and parse_value("[1, 2]", [0]) == [1, 2]


# ----------------------------------------------------------------------------- project + steps

def _fake_project(tmp_path, n=3):
    p = Project.create(tmp_path / "proj", "proj")
    for i in range(n):
        (p.stitch_coord_dir / f"s{i:04d}.txt").write_text("{ROOT_DIR}\t/x\n{TILE_SIZE}\t10\t10\na.tif\t0\t0\n", encoding="utf-8")
    return p


def test_project_layout_and_general_config(tmp_path):
    p = _fake_project(tmp_path)
    assert (p.configs_dir / "general_configs.yaml").is_file()
    assert (p.configs_dir / "default_stitching_configs.yaml").is_file()
    import yaml
    g = yaml.safe_load(p.general_config_path().read_text(encoding="utf-8"))
    assert Path(g["working_directory"]) == p.root
    assert p.section_names() == ["s0000", "s0001", "s0002"]
    (p.root / "section_order.txt").write_text("s0002\ns0000\ns0001\n", encoding="utf-8")
    assert p.section_names() == ["s0002", "s0000", "s0001"]
    p2 = Project.load(p.root)
    assert p2.state.name == "proj"


def test_pipeline_scan_states_and_clear(tmp_path):
    p = _fake_project(tmp_path)
    cs = ConfigStore(p.configs_dir)
    scan = PipelineScan(p.root, 3, cs)
    assert scan["stitch.matching"].state.value == "not started"
    assert scan["stitch.optimization"].state.value == "blocked"
    d = p.root / "stitch" / "match_h5"
    d.mkdir(parents=True)
    (d / "s0000.h5").write_bytes(b"x")
    (d / "s0001.h5_err").write_bytes(b"x")
    scan = PipelineScan(p.root, 3, cs)
    st = scan["stitch.matching"]
    assert st.done == 1 and st.errors == 1 and st.state.value == "errors"
    assert [s.key for s in downstream("stitch.matching")][:2] == ["stitch.optimization", "stitch.rendering"]
    (p.root / "stitch" / "tform").mkdir()
    (p.root / "stitch" / "tform" / "s0000.h5").write_bytes(b"x")
    targets = clear_targets(p.root, STEPS_BY_KEY["stitch.matching"], cascade=True)
    assert p.root / "stitch" / "match_h5" in targets and p.root / "stitch" / "tform" in targets


def test_snapshot_roundtrip(tmp_path):
    p = _fake_project(tmp_path)
    d = p.root / "stitch" / "match_h5"
    d.mkdir(parents=True)
    (d / "s0000.h5").write_bytes(b"one")
    snap = create_snapshot(p.root, "before")
    (d / "s0000.h5").write_bytes(b"two")
    restore_snapshot(p.root, snap)
    assert (d / "s0000.h5").read_bytes() == b"one"


def test_montage_test_run_subsets_tiles(tmp_path):
    p = _fake_project(tmp_path, 1)
    f = p.stitch_coord_dir / "s0000.txt"
    f.write_text("{ROOT_DIR}\t/x\n{RESOLUTION}\t4\n{TILE_SIZE}\t100\t100\na.tif\t0\t0\nb.tif\t90\t0\nc.tif\t180\t0\n", encoding="utf-8")
    p.state.volume.tile_w = p.state.volume.tile_h = 100
    tr = create_montage_test(p, "t1", ["s0000"], bbox=(0, 0, 150, 100))
    info = parse_stitch_coord(tr.root / "stitch" / "stitch_coord" / "s0000.txt")
    assert [t[0] for t in info["tiles"]] == ["a.tif", "b.tif"]
    import yaml
    g = yaml.safe_load((tr.root / "configs" / "general_configs.yaml").read_text(encoding="utf-8"))
    assert Path(g["working_directory"]) == tr.root
    n = retarget_stitch_coords(p, tmp_path / "other")
    assert n == 1 and "other" in f.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- masks

def test_tissue_detection_texture_polarity_independent():
    rng = np.random.default_rng(2)
    img = np.full((300, 300), 120, np.uint8)
    img[:, :40] = 0                                   # padding
    tissue = np.clip(rng.normal(120, 40, (200, 160)), 0, 255).astype(np.uint8)
    img[50:250, 100:260] = tissue
    m, thr = detect_tissue(img, TissueParams(method="texture", window=15, min_component_px=500, fill_holes_px=500, erode=1))
    inside = m[90:210, 140:220].mean()
    outside = m[:40, :].mean()
    assert inside > 0.85 and outside < 0.05
    # a section that is tissue everywhere: 'auto' must not split it
    full = np.clip(rng.normal(120, 40, (300, 300)), 0, 255).astype(np.uint8)
    full[:, :40] = 0
    m2, thr2 = detect_tissue(full, TissueParams(method="auto", window=15, erode=0))
    assert m2[:, 60:].mean() > 0.98 and m2[:, :30].mean() < 0.01
    mat = compose_material_mask(m, None)
    assert set(np.unique(mat)) <= {LABEL_DEFAULT, LABEL_EXCLUDE}


# ----------------------------------------------------------------------------- structure matching

def test_structure_match_recovers_known_transform():
    from feabas_workbench.workers.structure_match import match_pair
    rng = np.random.default_rng(3)
    H = W = 1200
    img0 = rng.integers(60, 200, (H, W)).astype(np.uint8)
    n = 120
    c0 = rng.uniform(80, W - 80, (n, 2))
    theta = np.deg2rad(12.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * 1.05
    c1 = c0 @ R.T + np.array([35.0, -20.0]) + rng.normal(0, 1.5, (n, 2))
    keep = rng.random(n) < 0.6                       # 40 % of structures vanish between sections
    c1 = c1[keep]
    c1 = np.vstack([c1, rng.uniform(80, W - 80, (50, 2))])   # plus unrelated new ones
    inside = (c1[:, 0] > 5) & (c1[:, 1] > 5) & (c1[:, 0] < W - 5) & (c1[:, 1] < H - 5)
    c1 = c1[inside]
    import cv2
    img1 = cv2.warpAffine(img0, np.hstack([R, np.array([[35.0], [-20.0]])]).astype(np.float32), (W, H))

    def dets(c):
        return [{"cx": float(x), "cy": float(y), "area": 80.0, "box": [x - 5, y - 5, x + 5, y + 5], "cls": 0, "conf": 0.9} for x, y in c]

    xy0, xy1, info = match_pair(img0, dets(c0), img1, dets(c1), {"ransac_tol": 5.0, "min_inliers": 8, "candidates": 6})
    assert info.get("n_pairs", 0) >= 40, info
    assert abs(info["rotation_deg"] - 12.0) < 1.5 and abs(info["scale"] - 1.05) < 0.03
    # every returned pair must be a true correspondence
    pred = xy0 @ R.T + np.array([35.0, -20.0])
    assert np.median(np.linalg.norm(pred - xy1, axis=1)) < 4


# ----------------------------------------------------------------------------- external mask import

def test_import_external_masks(tmp_path):
    import cv2
    from feabas_workbench.core.maskstore import MaskStore
    p = _fake_project(tmp_path, 2)
    cs = ConfigStore(p.configs_dir)
    cs.set("thumbnail", "thumbnail_mip_level", 3)
    cs.save()
    st = MaskStore(p)
    st.thumb_dir.mkdir(parents=True)
    for sec in ("s0000", "s0001"):
        cv2.imwrite(str(st.thumb_dir / f"{sec}.png"), np.full((100, 120), 90, np.uint8))
    ext = tmp_path / "ext"
    ext.mkdir()
    for i, sec in enumerate(("s0000", "s0001")):
        m = np.zeros((200, 240), np.uint8)          # mip2 = twice the thumbnail size
        m[20:-20, 20:-20] = 255
        cv2.imwrite(str(ext / f"tissue_{i}.png"), m)   # named by number only
    assert set(MaskStore.match_files_to_sections(ext, ["s0000", "s0001"])) == {"s0000", "s0001"}
    res = st.import_external(ext, 2, "tissue", ["s0000", "s0001"])
    assert res["imported"] == ["s0000", "s0001"] and not res["warnings"]
    assert st.hires_mip() == 2
    info = st.compose("s0000", use_folds=False, write_hires=True)
    assert info["hires"] and info["hires_mip"] == 2
    lo = cv2.imread(str(st.thumb_mask_dir / "s0000.png"), 0)
    hi = cv2.imread(str(st.align_mask_dir / "s0000.png"), 0)
    assert lo.shape == (100, 120) and hi.shape == (200, 240)
    assert lo[50, 60] == LABEL_DEFAULT and lo[2, 2] == LABEL_EXCLUDE and hi[100, 120] == LABEL_DEFAULT
    # a complete material mask is taken as-is and protected from recomposition
    mat = np.full((100, 120), 255, np.uint8); mat[10:90, 10:110] = 0; mat[40:50, 40:80] = 50
    cv2.imwrite(str(ext / "s0001.png"), mat)
    res = st.import_external(ext, 3, "material", ["s0001"])
    assert res["imported"] == ["s0001"] and st.is_hand_edited("s0001")
    out = cv2.imread(str(st.thumb_mask_dir / "s0001.png"), 0)
    assert out[45, 60] == 50 and out[0, 0] == 255


# ----------------------------------------------------------------------------- environment pre-flight

def test_check_imports_reports_the_reason():
    import sys
    from feabas_workbench.core.envs import check_imports

    assert check_imports(sys.executable, ["json", "pathlib"]) == ""
    problem = check_imports(sys.executable, ["json", "definitely_not_a_module_42"])
    assert "definitely_not_a_module_42" in problem and "ModuleNotFoundError" in problem
    assert "not found" in check_imports(str(Path(sys.executable).parent / "nope_python.exe"), ["json"])
    assert check_imports("", ["json"]) == "no interpreter configured"


def test_parse_value_keeps_string_enums():
    """'NONE' is a FEABAS blend mode, not a null: writing null makes FEABAS render one tile per chunk."""
    assert parse_value("NONE", "PYRAMID") == "NONE"
    assert parse_value("nearest", "PYRAMID") == "nearest"
    assert parse_value("null", "PYRAMID") is None          # the YAML spellings still clear a key
    assert parse_value("~", "PYRAMID") is None
    assert parse_value("none", None) is None               # a key that is null by default stays nullable
    assert parse_value("4096", 1024) == 4096


def test_rendering_counts_finished_sections(tmp_path):
    """Progress for 'Render montages' is one output per finished section, for either render driver."""
    step = STEPS_BY_KEY["stitch.rendering"]
    from feabas_workbench.core.steps import count_outputs
    base = tmp_path / "stitched_sections"
    assert count_outputs(tmp_path, step) == 0
    # PNG tiles: the section folder appears first, metadata.txt only when the section is done
    for sec in ("s0000", "s0001"):
        (base / "mip0" / sec).mkdir(parents=True)
        (base / "mip0" / sec / f"{sec}_tr1-tc1.png").write_bytes(b"")
    assert count_outputs(tmp_path, step) == 0
    (base / "mip0" / "s0000" / "metadata.txt").write_text("x", encoding="utf-8")
    assert count_outputs(tmp_path, step) == 1
    # a precomputed volume writes <section>/info instead, and mipmaps must not be counted
    (base / "s0002").mkdir()
    (base / "s0002" / "info").write_text("{}", encoding="utf-8")
    (base / "mip1" / "s0000").mkdir(parents=True)
    assert count_outputs(tmp_path, step) == 2


def test_dark_regions_and_tissue_exclusion():
    """Black holes inside a section are not tissue; specks below the size limit are kept."""
    from feabas_workbench.core.masks import dark_regions, TissueParams, detect_tissue
    img = np.full((200, 200), 120, np.uint8)
    img[:10, :] = 0                      # padding at the border
    img[98:104, 98:104] = 0              # a 36 px tear: too small for padding_mask's size rule
    img[150, 150] = 0                    # a single dark pixel of real tissue
    dark = dark_regions(img, max_value=0, min_px=24)
    assert dark[100, 100] and dark[5, 5] and not dark[150, 150]
    keep, _ = detect_tissue(img, TissueParams(method="all", exclude_dark=False, erode=0))
    assert keep[100, 100]                # the old behaviour: only the border padding is dropped
    drop, _ = detect_tissue(img, TissueParams(method="all", exclude_dark=True, erode=0, nodata_dilate=0))
    assert not drop[100, 100] and drop[150, 150] and drop[60, 60]


def test_restore_default_mask_clears_what_was_computed(tmp_path):
    import cv2
    from feabas_workbench.core.maskstore import MaskStore
    proj = Project.create(tmp_path / "p")
    st = MaskStore(proj)
    st.thumb_dir.mkdir(parents=True, exist_ok=True)
    img = np.full((60, 80), 100, np.uint8); img[:5, :] = 0
    cv2.imwrite(str(st.thumb_dir / "s0000.png"), img)
    st.compute_tissue("s0000", TissueParams(method="all"), save=True)
    cv2.imwrite(str(st.folds_dir / "s0000.png"), np.zeros((60, 80), np.uint8))
    st.set_hand_edited("s0000", True)
    assert st.restore_default_mask("s0000")
    assert st.material_mask("s0000") is not None
    assert st.tissue_mask("s0000") is None and st.fold_mask("s0000") is None
    assert not st.is_hand_edited("s0000")


def test_footprint_keeps_a_fold_that_reaches_the_edge(tmp_path):
    """A black fold crossing the whole section is connected to the outside but is still tissue."""
    import cv2
    from feabas_workbench.core.masks import footprint_mask, detect_tissue, TissueParams
    from feabas_workbench.core.maskstore import MaskStore
    img = np.full((200, 300), 130, np.uint8)
    img[:, :20] = 0; img[:, -20:] = 0          # padding around the montage
    img[:20, :] = 0; img[-20:, :] = 0
    img[:, 140:150] = 0                        # a fold from the top edge to the bottom edge
    fp = footprint_mask(img)
    assert fp[100, 145] and fp[100, 60] and not fp[100, 5]
    tissue, _ = detect_tissue(img, TissueParams(method="all", erode=0))
    assert tissue[100, 145]                    # the fold stays tissue: it gets a fold label, not 255

    # FEABAS's own mask wins when it is there, and is kept before we overwrite it
    proj = Project.create(tmp_path / "p")
    st = MaskStore(proj)
    st.thumb_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(st.thumb_dir / "s0000.png"), img)
    roi = np.full(img.shape, 255, np.uint8)
    roi[20:-20, 20:-20] = 0                    # FEABAS: tile footprint = 0, outside = 255
    st.thumb_mask_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(st.thumb_mask_dir / "s0000.png"), roi)
    assert st.backup_roi("s0000") and (st.roi_dir / "s0000.png").is_file()
    st.compose("s0000", use_folds=False)
    assert st.roi_mask("s0000")[100, 145]      # the copy survives our own mask
    base = st.default_tissue("s0000")
    assert base[100, 145] and not base[5, 5]


def test_roi_from_tform_rasterises_the_tile_boxes(tmp_path):
    """The imaged area comes from the stitched tile boxes, so black folds cannot leak into it."""
    import h5py
    from feabas_workbench.core.masks import roi_from_tform, compose_material_mask, LABEL_WRINKLE
    p = tmp_path / "s0000.h5"
    with h5py.File(p, "w") as f:                      # two tiles, 100x80 each, overlapping by 20
        f["resolution"] = 4.0
        f["moving_offsets"] = np.array([[[0.0, 0.0]], [[80.0, 0.0]]])
        f.create_dataset("moving_vertices/0", data=np.array([[0.0, 0.0], [100.0, 80.0]]))
        f.create_dataset("moving_vertices/1", data=np.array([[0.0, 0.0], [100.0, 80.0]]))
    roi = roi_from_tform(p, (82, 182))                # +2 as FEABAS sizes its own ROI image
    assert roi is not None and roi[40, 90] and roi[40, 5]
    assert not roi[-1, -1]                            # the rim outside the boxes
    assert roi_from_tform(p, (400, 100)) is None      # not the montage bounding box: refuse to guess
    assert roi_from_tform(tmp_path / "nope.h5", (10, 10)) is None

    # who wins where a fold detection overlaps the area outside the tiles
    folds = np.zeros(roi.shape, bool)
    folds[40, 90] = True                              # a real fold, inside
    folds[-2, -2] = True                              # the black rim, outside (not the 1-px border)
    clipped = compose_material_mask(roi, folds, LABEL_WRINKLE, clip_folds=True)
    kept = compose_material_mask(roi, folds, LABEL_WRINKLE, clip_folds=False)
    assert clipped[40, 90] == LABEL_WRINKLE and clipped[-2, -2] == LABEL_EXCLUDE
    assert kept[-2, -2] == LABEL_WRINKLE


def test_border_margin_band():
    """The margin follows the outline; soft keeps the pixels, exclude drops them."""
    from feabas_workbench.core.masks import border_band, compose_material_mask, LABEL_SOFT
    roi = np.zeros((60, 60), bool)
    roi[10:50, 10:50] = True
    band = border_band(roi, 3)
    assert band[10, 30] and band[12, 30] and not band[14, 30] and not band[5, 30]
    assert not border_band(roi, 0).any()
    soft = compose_material_mask(roi, None, extra={LABEL_SOFT: band})
    assert soft[10, 30] == LABEL_SOFT and soft[30, 30] == LABEL_DEFAULT and soft[5, 30] == LABEL_EXCLUDE
    cut = compose_material_mask(roi & ~band, None)
    assert cut[10, 30] == LABEL_EXCLUDE and cut[30, 30] == LABEL_DEFAULT


# ----------------------------------------------------------------------------- Zeiss / Fibics metadata

def _fibics_xml(w, h, fov_um, stage_um, mosaic_m, row, col):
    return (f'<?xml version="1.0" encoding="iso-8859-1"?><Fibics version="1.2"><Image><Width>{w}</Width><Height>{h}</Height></Image>'
            f'<Scan><FOV_X units="um">{fov_um[0]}</FOV_X><FOV_Y units="um">{fov_um[1]}</FOV_Y><ScanRot units="deg">315.7</ScanRot></Scan>'
            f'<Stage><X units="um">{stage_um[0]}</X><Y units="um">{stage_um[1]}</Y><Z units="um">1</Z></Stage>'
            f'<MosaicInfo><Row>{row}</Row><Col>{col}</Col><X>{mosaic_m[0]}</X><Y>{mosaic_m[1]}</Y></MosaicInfo></Fibics>')


def _write_zeiss_tile(path, xml, w=64, h=48):
    import tifffile
    tifffile.imwrite(str(path), np.zeros((h, w), np.uint8), extratags=[(51023, "s", 0, xml, False)])


def test_fibics_metadata_pixel_size_and_stage(tmp_path):
    f = tmp_path / "Tile_r1-c1_S_170_1.tif"
    _write_zeiss_tile(f, _fibics_xml(3290, 3346, (65.8125, 66.93), (-67797.25, -43921.58), (-9e-7, 9.2e-7), 1, 1))
    info = T.read_tile_info(f)
    assert info.vendor == "zeiss"
    assert info.pixel_size_nm == pytest.approx(65.8125 / 3290 * 1000, rel=1e-6)
    assert info.stage_xy_m == pytest.approx((-0.06779725, -0.04392158))
    assert info.mosaic_xy_m == pytest.approx((-9e-7, 9.2e-7))
    assert info.scan_rotation_rad == pytest.approx(np.deg2rad(315.7))


def test_zeiss_mosaic_offsets_used_when_stage_is_per_mosaic(tmp_path):
    # 2x2 mosaic: identical stage position in every tile, tile offsets in MosaicInfo (metres), 10 % overlap of 64 px tiles at 20 nm
    step = 0.9 * 64 * 20e-9
    tiles = []
    for r in range(1, 3):
        for c in range(1, 3):
            f = tmp_path / f"Tile_r{r}-c{c}_S_170_{r * 10 + c}.tif"
            _write_zeiss_tile(f, _fibics_xml(64, 48, (1.28, 0.96), (100.0, 200.0), ((c - 1) * step, (r - 1) * step * 0.75), r, c))
            tiles.append(f)
    plan = T.build_plan(tmp_path, T.NamingRule(preset="zeiss"), T.LayoutParams(mode="stage"))
    assert plan.placed_by_stage and any("mosaic" in w for w in plan.warnings)
    tl = {(t.ix, t.iy): t for t in plan.sections["s0170"]}
    assert tl[(2, 1)].x_px - tl[(1, 1)].x_px == pytest.approx(57.6, abs=1)
    assert tl[(1, 2)].y_px - tl[(1, 1)].y_px == pytest.approx(0.9 * 48, abs=1)


def test_border_band_is_uniform_when_tissue_touches_the_image_edge():
    from feabas_workbench.core.masks import border_band
    t = np.ones((200, 300), bool)             # tissue everywhere, including the image edges
    band = border_band(t, 10)
    assert band[:10, :].all() and band[-10:, :].all() and band[:, :10].all() and band[:, -10:].all()
    assert not band[10:-10, 10:-10].any()
    t[80:120, 140:160] = False                 # a hole inside the tissue must not get a ring
    band = border_band(t, 10)
    assert not band[60:140, 120:180].any()


# ----------------------------------------------------------------------------- worker launch environment

def test_worker_package_root_holds_only_the_package(tmp_path, monkeypatch):
    """The folder put on a worker's PYTHONPATH must contain feabas_workbench and nothing else: for a
    wheel install package_root() is site-packages, whose compiled numpy/cv2 would shadow the FEABAS
    or deep-learning interpreter's own copies."""
    from feabas_workbench.core import envs, jobs
    monkeypatch.setattr(envs, "settings_dir", lambda: tmp_path)
    root = jobs.worker_package_root()
    assert root == tmp_path / "worker_pkg"
    entries = sorted(p.name for p in root.iterdir())
    assert entries == ["feabas_workbench", "stamp.json"]
    pkg = root / "feabas_workbench"
    assert (pkg / "__init__.py").is_file()
    assert (pkg / "workers" / "fold_predict.py").is_file()
    assert (pkg / "core" / "images.py").is_file()
    assert (pkg / "vendor" / "winfix" / "sitecustomize.py").is_file()
    assert not (pkg / "ui").exists()                       # Qt is not needed by workers
    assert not list(pkg.rglob("__pycache__"))
    env = jobs.python_env_for_package_root(root)
    assert env["PYTHONPATH"].split(__import__("os").pathsep)[0] == str(root)
    # a second call with unchanged sources reuses the staged copy
    stamp = (root / "stamp.json").stat().st_mtime_ns
    assert jobs.worker_package_root() == root
    assert (root / "stamp.json").stat().st_mtime_ns == stamp


def test_cancel_probes_unblocks_a_thread_stuck_in_a_probe_subprocess():
    """Environment discovery blocks in subprocesses; closing the window must be able to end that
    thread (a QThread destroyed while running aborts the process, and on Linux terminate() cannot
    interrupt a thread blocked in a subprocess - seen in CI)."""
    import subprocess
    import sys
    import threading
    import time
    from feabas_workbench.core import envs

    envs.reset_probe_cancel()
    result = {}

    def worker():
        t0 = time.time()
        try:
            envs._run_probe([sys.executable, "-c", "import time; time.sleep(30)"], timeout=60)
            result["outcome"] = "completed"
        except envs.ProbeCancelled:
            result["outcome"] = "cancelled"
        except subprocess.TimeoutExpired:
            result["outcome"] = "timeout"
        result["seconds"] = time.time() - t0

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    deadline = time.time() + 10
    while not envs._PROBE_PROCS and time.time() < deadline:
        time.sleep(0.05)
    assert envs._PROBE_PROCS, "probe subprocess did not start"
    envs.cancel_probes()
    th.join(5)
    assert not th.is_alive(), "thread still blocked after cancel_probes()"
    assert result["outcome"] == "cancelled"
    assert result["seconds"] < 5
    assert envs.discover_environments() == []          # cancelled: no new probe starts
    envs.reset_probe_cancel()


# ----------------------------------------------------------------------------- frames, margins, progress

def _framed_section(rng, h=240, w=320):
    """Black padding, then a white rim, then textured tissue; a black fold reaches in from the rim."""
    img = np.zeros((h, w), np.uint8)                       # black padding
    img[10:h - 10, 12:w - 12] = 255                        # white frame
    tissue = np.clip(rng.normal(120, 35, (h - 80, w - 100)), 0, 255).astype(np.uint8)
    img[40:h - 40, 50:w - 50] = tissue                     # tissue inside
    img[100:104, 50:110] = 0                               # a 4 px black fold from the rim into the tissue
    img[0, :] = 107                                        # a one-pixel detector line along the top edge
    return img


def test_uniform_border_peels_frames_from_the_outside_in():
    from feabas_workbench.core.masks import uniform_border, detect_tissue, TissueParams
    rng = np.random.default_rng(0)
    img = _framed_section(rng)
    roi = np.ones(img.shape, bool)                         # the tile covers the whole image
    for mode in ("both", "auto"):
        border = uniform_border(img, roi, mode=mode, min_width=9)
        assert border[5, 5] and border[20, 20] and not border[100, 200]      # padding, rim, and not the tissue
        assert not border[102, 80]                                           # the thin fold is given back
        assert not border[150, 160]
    only_white = uniform_border(img, roi, mode="white", min_width=9)
    assert only_white[20, 20] and only_white[5, 5]                           # the rim is reached through the padding
    assert not only_white[150, 160] and not only_white[102, 80]
    grey = img.copy(); grey[img == 0] = 30                                   # padding that is dark grey, not no-data
    only_white = uniform_border(grey, roi, mode="white", min_width=9)
    assert not only_white[5, 5] and not only_white[20, 20]                   # nothing white touches the outside
    only_black = uniform_border(img, roi, mode="black", min_width=9)
    assert only_black[5, 5] and not only_black[20, 20]
    # through detect_tissue: the frame and the detector line go, the tissue with its fold stays
    m, _ = detect_tissue(img, TissueParams(method="border", border_mode="both", border_min_width=9,
                                           min_component_px=500, fill_holes_px=200, erode=0), roi=roi)
    assert m[150, 160] and m[102, 80] and not m[20, 20] and not m[0, 100]
    ys, xs = np.nonzero(m)
    assert 38 <= ys.min() <= 42 and 48 <= xs.min() <= 52


def test_manual_margins_and_white_exclusion():
    from feabas_workbench.core.masks import manual_bounds, detect_tissue, TissueParams, bright_regions
    rng = np.random.default_rng(1)
    img = _framed_section(rng)
    roi = np.ones(img.shape, bool)
    m, _ = detect_tissue(img, TissueParams(method="manual", crop_left=50, crop_top=40, crop_right=50, crop_bottom=40, erode=0), roi=roi)
    ys, xs = np.nonzero(m)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (40, 199, 50, 269)
    assert not manual_bounds((10, 10), left=6, right=6).any()               # margins that meet leave nothing
    # white regions inside the section can be excluded like black ones
    img2 = np.clip(rng.normal(120, 35, (120, 120)), 0, 254).astype(np.uint8)
    img2[40:70, 40:70] = 255                                                 # a burnt patch
    img2[100, 100] = 255                                                     # one saturated pixel of real tissue
    assert bright_regions(img2, 255, min_px=24)[50, 50] and not bright_regions(img2, 255, min_px=24)[100, 100]
    keep, _ = detect_tissue(img2, TissueParams(method="all", erode=0), roi=np.ones(img2.shape, bool))
    drop, _ = detect_tissue(img2, TissueParams(method="all", erode=0, exclude_bright=True, bright_min=255, nodata_dilate=0),
                            roi=np.ones(img2.shape, bool))
    assert keep[50, 50] and not drop[50, 50] and drop[100, 100]
    # old project files without the new keys still load
    assert TissueParams.from_dict({"method": "texture", "window": 15}).border_mode == "both"


def test_thumbnail_progress_counts_mip_levels_first(tmp_path):
    """The mip-mapping is the slow part of 'Make thumbnails' and leaves no thumbnail behind."""
    from feabas_workbench.core.steps import thumbnail_progress, thumbnail_max_mip
    p = _fake_project(tmp_path, n=4)
    cs = ConfigStore(p.configs_dir)
    cs.set("thumbnail", "thumbnail_mip_level", 4); cs.set("alignment", "matching.working_mip_level", 2); cs.save()
    assert thumbnail_max_mip(cs) == 3
    done, expected, msg = thumbnail_progress(p.root, cs, 4)
    assert (done, expected) == (0, 16)
    for m in (1, 2, 3):
        for sec in ("s0000", "s0001"):
            d = p.root / "stitched_sections" / f"mip{m}" / sec
            d.mkdir(parents=True)
            (d / "metadata.txt").write_text("x", encoding="utf-8")
    done, expected, msg = thumbnail_progress(p.root, cs, 4)
    assert done == 6 and "mip levels 6/12" in msg
    (p.root / "thumbnail_align" / "thumbnails").mkdir(parents=True)
    (p.root / "thumbnail_align" / "thumbnails" / "s0000.png").write_bytes(b"")
    done, expected, msg = thumbnail_progress(p.root, cs, 4)
    assert done == 7 and "thumbnails 1/4" in msg
    # the precomputed driver has no PNG mip folders: thumbnails only
    cs.set("stitching", "rendering.driver", "neuroglancer_precomputed"); cs.save()
    assert thumbnail_progress(p.root, cs, 4) == (1, 4, "thumbnails 1/4")


def test_fine_match_list_restricts_pairs_by_distance(tmp_path):
    from feabas_workbench.core.steps import fine_match_pairs, write_fine_match_list, read_fine_match_list
    names = [f"s{i:04d}" for i in range(5)]
    md = tmp_path / "thumbnail_align" / "matches"
    md.mkdir(parents=True)
    for k in (1, 2):
        for i in range(5 - k):
            (md / f"{names[i]}__to__{names[i + k]}.h5").write_bytes(b"")
    assert len(fine_match_pairs(tmp_path, names, 1)) == 4 and len(fine_match_pairs(tmp_path, names, 2)) == 7
    pairs = write_fine_match_list(tmp_path, names, 1)
    assert pairs == ["s0000__to__s0001", "s0001__to__s0002", "s0002__to__s0003", "s0003__to__s0004"]
    assert read_fine_match_list(tmp_path) == pairs
    assert (tmp_path / "align" / "match_name.txt").read_text(encoding="utf-8").startswith("s0000__to__s0001.h5\n")
    # the scan's expectation for the fine steps follows the list
    (tmp_path / "stitch" / "stitch_coord").mkdir(parents=True)
    scan = PipelineScan(tmp_path, 5)
    assert scan["align.matching"].expected == 4
    # a distance that keeps every pair, or none at all, means FEABAS's default: no file
    assert write_fine_match_list(tmp_path, names, 2) is None and not (tmp_path / "align" / "match_name.txt").exists()
    assert write_fine_match_list(tmp_path, names, None) is None
    assert PipelineScan(tmp_path, 5)["align.matching"].expected == 7
