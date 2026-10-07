"""Segmentation masks carried along with the alignment (core.segmentation, the slice export, the GUI card)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from feabas_workbench.core import segmentation as seg
from feabas_workbench.core.synthetic import make_demo_project


def _png(path, a):
    import cv2
    assert cv2.imwrite(str(path), a)
    return path


@pytest.fixture
def stack_project(tmp_path):
    """Three sections of one 96 x 96 image each, like an imported image stack."""
    return make_demo_project(tmp_path / "proj", n_sections=3, rows=1, cols=1, tile=96)


def _images(project):
    return [seg.read_stitch_coord(project.stitch_coord_dir / f"{s}.txt")[0][0] for s in project.section_names()]


def test_headers_of_label_images(tmp_path):
    import tifffile
    h = seg.read_image_header(_png(tmp_path / "a.png", np.zeros((5, 7), np.uint16)))
    assert (h.width, h.height, h.bits, h.channels) == (7, 5, 16, 1) and seg.check_mask_header(h, "a") is None
    rgb = seg.read_image_header(_png(tmp_path / "b.png", np.zeros((5, 7, 3), np.uint8)))
    assert "3 channels" in seg.check_mask_header(rgb, "b")
    tifffile.imwrite(tmp_path / "c.tif", np.zeros((4, 6), np.uint8))
    t = seg.read_image_header(tmp_path / "c.tif")
    assert (t.width, t.height, t.bits, t.channels) == (6, 4, 8, 1)
    tifffile.imwrite(tmp_path / "d.tif", np.zeros((4, 6), np.uint32))
    assert "32-bit" in seg.check_mask_header(seg.read_image_header(tmp_path / "d.tif"), "d")
    tifffile.imwrite(tmp_path / "e.tif", np.zeros((3, 4, 6), np.uint8), photometric="minisblack")
    with pytest.raises(ValueError, match="multi-page"):
        seg.read_image_header(tmp_path / "e.tif")
    (tmp_path / "f.jpg").write_bytes(b"\xff\xd8\xff")
    with pytest.raises(ValueError, match="JPEG"):
        seg.read_image_header(tmp_path / "f.jpg")
    from PIL import Image
    Image.fromarray(np.zeros((4, 4), np.uint8)).convert("P").save(tmp_path / "g.png")
    assert "palette" in seg.check_mask_header(seg.read_image_header(tmp_path / "g.png"), "g")


def test_masks_match_their_images_by_name_or_in_order(stack_project, tmp_path):
    p = stack_project
    images = _images(p)
    by_name = tmp_path / "by_name"; by_name.mkdir()
    for im in images:
        _png(by_name / (im.stem + ".png"), np.zeros((96, 96), np.uint8))
    plan = seg.plan_masks(p.root, p.section_names(), by_name)
    assert plan.mode == "name" and plan.bits == 8 and plan.n_masks == 3 and not plan.notes
    assert plan.sections[0] == ("s0001", {images[0].name: str(by_name / (images[0].stem + ".png"))})
    in_order = tmp_path / "in_order"; in_order.mkdir()
    for k in (10, 2, 1):                                   # natural order: seg_1, seg_2, seg_10
        _png(in_order / f"seg_{k}.png", np.full((96, 96), k, np.uint16))
    plan = seg.plan_masks(p.root, p.section_names(), in_order)
    assert plan.mode == "order" and plan.bits == 16 and plan.notes
    assert [Path(m).name for _, found in plan.sections for m in found.values()] == ["seg_1.png", "seg_2.png", "seg_10.png"]
    with pytest.raises(ValueError, match="no mask of the same name"):
        seg.plan_masks(p.root, p.section_names(), in_order, "name")
    (in_order / "seg_11.png").write_bytes((in_order / "seg_1.png").read_bytes())
    with pytest.raises(ValueError, match="4 mask files for 3 sections"):
        seg.plan_masks(p.root, p.section_names(), in_order)


def test_masks_that_do_not_fit_are_refused(stack_project, tmp_path):
    p = stack_project
    images = _images(p)
    d = tmp_path / "masks"; d.mkdir()
    for im in images:
        _png(d / (im.stem + ".png"), np.zeros((96, 96), np.uint8))
    _png(d / (images[1].stem + ".png"), np.zeros((48, 96), np.uint8))
    with pytest.raises(ValueError, match="pixel for pixel"):
        seg.plan_masks(p.root, p.section_names(), d)
    _png(d / (images[1].stem + ".png"), np.zeros((96, 96), np.uint16))
    with pytest.raises(ValueError, match="mix 8- and 16-bit"):
        seg.plan_masks(p.root, p.section_names(), d)
    _png(d / (images[1].stem + ".png"), np.zeros((96, 96), np.uint8))
    _png(d / (images[1].stem + ".tif"), np.zeros((96, 96), np.uint8))
    with pytest.raises(ValueError, match="more than one mask"):
        seg.plan_masks(p.root, p.section_names(), d)
    with pytest.raises(ValueError, match="no PNG or TIFF"):
        seg.plan_masks(p.root, p.section_names(), tmp_path / "nothing")


def test_several_images_per_section_need_names(tmp_path):
    p = make_demo_project(tmp_path / "grid", n_sections=2, rows=2, cols=2, tile=96)
    tiles = {s: seg.read_stitch_coord(p.stitch_coord_dir / f"{s}.txt")[0] for s in p.section_names()}
    d = tmp_path / "masks"; d.mkdir()
    for s, ts in tiles.items():
        for t in ts:
            _png(d / (t.stem + ".png"), np.zeros((96, 96), np.uint8))
    plan = seg.plan_masks(p.root, p.section_names(), d)
    assert plan.mode == "name" and [len(m) for _, m in plan.sections] == [4, 4]
    with pytest.raises(ValueError, match="one image per section"):
        seg.plan_masks(p.root, p.section_names(), d, "order")


def test_majority_downsampling_keeps_labels_and_is_not_shifted():
    a = np.array([[5, 5, 0, 0, 7, 8],
                  [5, 0, 0, 3, 9, 6],
                  [0, 0, 4, 4, 2, 2],
                  [0, 0, 6, 6, 1, 1]], np.uint16)
    out = seg.mode_downsample(a)
    # majority 5; three zeros beat one label; a label beats 0 on a tie; all different -> smallest label;
    # two labels tied -> the smaller; ties between labels never favour one corner
    assert out.tolist() == [[5, 0, 6], [0, 4, 1]] and out.dtype == np.uint16
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 6, (64, 64)).astype(np.uint8) * 40
    assert set(np.unique(seg.mode_downsample(labels))) <= set(np.unique(labels))
    # a vertical edge in the middle of each 2x2 block: picking one pixel per block shifts it, majority does not
    edge = np.zeros((8, 8), np.uint8); edge[:, 3:] = 9
    assert seg.mode_downsample(edge)[0].tolist() == [0, 9, 9, 9]


def _write_level(folder, tiles, tile, resolution=10.0):
    folder.mkdir(parents=True, exist_ok=True)
    meta = {}
    for (r, c), a in tiles.items():
        name = f"s0001_tr{r + 1}-tc{c + 1}.png"
        _png(folder / name, a)
        meta[name] = (c * tile, r * tile, (c + 1) * tile, (r + 1) * tile)
    seg.write_tile_metadata(folder, meta, resolution)
    return meta


def test_label_pyramid_matches_majority_of_the_whole_section(tmp_path):
    tile = 16
    rng = np.random.default_rng(2)
    full = (rng.integers(0, 4, (3 * tile, 5 * tile)) * 1000).astype(np.uint16)
    full[:, 4 * tile:] = 0                                     # an empty tile column: not written
    tiles = {(r, c): full[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] for r in range(3) for c in range(5)
             if full[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile].any()}
    base = tmp_path / "stack"
    _write_level(base / "mip0" / "00_s0001", tiles, tile)
    written = seg.build_label_pyramid(base, "00_s0001", "s0001", 0, 3, (tile, tile), "_tr{ROW_IND}-tc{COL_IND}.png",
                                      True, 10.0)
    assert written == [1, 2, 3]
    meta1 = seg.read_tile_metadata(base / "mip1" / "00_s0001" / "metadata.txt")
    assert sorted(meta1) == ["s0001_tr1-tc1.png", "s0001_tr1-tc2.png", "s0001_tr2-tc1.png", "s0001_tr2-tc2.png"]
    assert meta1["s0001_tr2-tc2.png"] == (16, 16, 32, 32)
    import cv2
    got = np.zeros((2 * tile, 3 * tile), np.uint16)
    for name, (x0, y0, x1, y1) in meta1.items():
        got[y0:y1, x0:x1] = cv2.imread(str(base / "mip1" / "00_s0001" / name), cv2.IMREAD_UNCHANGED)
    want = seg.mode_downsample(full)
    assert np.array_equal(got[:want.shape[0], :want.shape[1]], want)
    assert "{RESOLUTION}\t20.0" in (base / "mip1" / "00_s0001" / "metadata.txt").read_text()
    assert seg.build_label_pyramid(base, "00_s0001", "s0001", 0, 3, (tile, tile), "_tr{ROW_IND}-tc{COL_IND}.png",
                                   True, 10.0) == []           # finished levels are kept


def test_label_pyramid_on_an_offset_canvas(tmp_path):
    """Tiles that start off the grid (FEABAS's canvas_bbox) still land on the next level's grid."""
    tile = 8
    base = tmp_path / "stack"
    folder = base / "mip0" / "s1"; folder.mkdir(parents=True)
    a = np.full((tile, tile), 3, np.uint8)
    _png(folder / "s1_tr1-tc1.png", a)
    seg.write_tile_metadata(folder, {"s1_tr1-tc1.png": (12, 4, 20, 12)}, 10.0)
    seg.build_label_pyramid(base, "s1", "s1", 0, 1, (tile, tile), "_tr{ROW_IND}-tc{COL_IND}.png", True, 10.0)
    meta = seg.read_tile_metadata(base / "mip1" / "s1" / "metadata.txt")
    assert sorted(meta) == ["s1_tr1-tc1.png", "s1_tr1-tc2.png"]       # x 12..20 -> 6..10 at mip1: two tiles


def test_status_and_staleness_come_from_the_files(stack_project, tmp_path):
    p = stack_project
    base = seg.stack_dir(p.root, "my masks")
    assert base == p.root / "segmentation" / "my_masks"
    sections = p.section_names()
    assert seg.stack_status(p.root, base, sections, 0, 2) == seg.StackStatus(3, 0, 0)
    for k, s in enumerate(sections):
        for m in (0, 1, 2):
            seg.write_tile_metadata((base / f"mip{m}" / f"{k:02d}_{s}").mkdir(parents=True) or
                                    base / f"mip{m}" / f"{k:02d}_{s}", {}, 10.0)
    st = seg.stack_status(p.root, base, sections, 0, 2)
    assert (st.rendered, st.mipmapped, st.stale, st.done) == (3, 3, "", True)
    tform = p.root / "align" / "tform"; tform.mkdir(parents=True)
    (tform / "s0001.h5").write_bytes(b"x")
    later = time.time() + 10
    os.utime(tform / "s0001.h5", (later, later))
    assert "alignment changed" in seg.stack_status(p.root, base, sections, 0, 2).stale
    assert seg.list_stacks(p.root) == [base]


def test_slice_export_writes_whole_sections(tmp_path):
    from feabas_workbench.workers.export_vast import export_slices, export_vast
    import cv2
    base = tmp_path / "stack"
    tile = 8
    a = np.arange(64, dtype=np.uint16).reshape(8, 8) + 300
    for z, sec in ((0, "000_s0001"), (2, "002_s0003")):
        folder = base / "mip0" / sec
        _write_level(folder, {(0, 0): a + z, (1, 1): a + 1000}, tile)
    res = export_slices(base, tmp_path / "out", "masks", 0)
    assert res["count"] == 2 and res["size"] == [16, 16]
    img = cv2.imread(str(tmp_path / "out" / "masks_0002.png"), cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint16 and img.shape == (16, 16)
    assert np.array_equal(img[:8, :8], a + 2) and np.array_equal(img[8:, 8:], a + 1000) and not img[:8, 8:].any()
    info = json.loads((tmp_path / "out" / "slices.json").read_text())
    assert info["sections"]["masks_0000.png"] == "000_s0001" and info["dtype"] == "uint16"
    vsvi = export_vast(base, tmp_path / "vast", "masks", (10, 10, 50), [0])
    assert json.loads(Path(vsvi["vsvi"]).read_text())["SourceBytesPerPixel"] == 2


def test_label_overlay_colours_only_labels():
    from feabas_workbench.core.images import label_overlay, label_colors
    grey = np.full((4, 4), 100, np.uint8)
    labels = np.zeros((4, 4), np.uint16); labels[1, 1] = 7; labels[2, 2] = 40000
    out = label_overlay(grey, labels)
    assert out.shape == (4, 4, 3) and out.dtype == np.uint8
    assert (out[0, 0] == 100).all() and not (out[1, 1] == 100).all()
    assert (label_colors(np.array([[7]], np.uint16)) == label_colors(np.array([[7]], np.uint32))).all()


def test_gui_card_plans_renders_and_clears(stack_project, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from feabas_workbench.core.configs import ConfigStore
    from feabas_workbench.core.envs import Settings
    from feabas_workbench.ui.bridge import AppContext
    from feabas_workbench.ui.pages.page_export import ExportPage
    app = QApplication.instance() or QApplication([])
    p = stack_project
    ctx = AppContext(Settings()); ctx.project = ctx.local_project = p; ctx.configs = ConfigStore(p.configs_dir)
    monkeypatch.setattr(ctx, "require_feabas_python", lambda: sys.executable)
    messages = []
    for kind in ("information", "warning"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: messages.append(a[2]))
    page = ExportPage(ctx)
    submitted = []
    monkeypatch.setattr(page, "submit", lambda spec: submitted.append(spec))
    masks = tmp_path / "masks"; masks.mkdir()
    for k, im in enumerate(_images(p)):
        _png(masks / f"label_{k}.png", np.full((96, 96), k + 1, np.uint8))
    page.s_dir.setText(str(masks)); page.s_name.setText("cells")
    page._render_masks()
    assert "not aligned yet" in messages[-1] and not submitted
    tform = p.root / "align" / "tform"; tform.mkdir(parents=True)
    for s in p.section_names():
        (tform / f"{s}.h5").write_bytes(b"")
    page._render_masks()
    spec = submitted[-1]
    assert "feabas_workbench.workers.segmentation_render" in spec.argv
    payload = json.loads(Path(spec.argv[-1]).read_text())
    assert payload["out_dir"] == str(p.root / "segmentation" / "cells") and payload["plan"]["mode"] == "order"
    assert payload["workers"] >= 1 and p.state.export["segmentation"]["name"] == "cells"
    out = p.root / "segmentation" / "cells"
    for k, s in enumerate(p.section_names()):
        seg.write_tile_metadata((out / "mip0" / f"{k}_{s}").mkdir(parents=True) or out / "mip0" / f"{k}_{s}", {}, 10.0)
    page._refresh_masks()
    assert page.s_status.text().startswith("3 of 3 sections rendered")
    assert page.e_stack.findData(str(out)) > 0 and page.q_masks.findData(str(out)) > 0
    page.e_stack.setCurrentIndex(page.e_stack.findData(str(out)))
    assert page.e_name.text() == "cells"
    page._export()
    export = json.loads(Path(submitted[-1].argv[-1]).read_text())
    assert export["aligned_stack"] == str(out) and export["vast_dir"] == "vast_cells"
    monkeypatch.setattr(page, "confirm", lambda *a: True)
    page._clear_masks()
    assert not out.exists() and page.e_stack.findData(str(out)) == -1
    page.close(); app.processEvents()


def test_the_worker_gets_absolute_paths(stack_project, tmp_path, monkeypatch):
    """CI runs the demo with a relative project path; the render runs in the project folder, where
    relative mask, spec and output paths would point elsewhere."""
    p = stack_project
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel").mkdir()
    for k, im in enumerate(_images(p)):
        _png(tmp_path / "rel" / f"m{k}.png", np.zeros((96, 96), np.uint8))
    plan = seg.plan_masks(Path(os.path.relpath(p.root, tmp_path)), p.section_names(), Path("rel"))
    assert all(Path(m).is_absolute() for _, found in plan.sections for m in found.values())
    assert seg.stack_dir(Path(os.path.relpath(p.root, tmp_path)), "x", Path("out")) == p.root.resolve() / "out" / "x"
    assert seg.stack_dir(p.root, "x", tmp_path / "abs") == tmp_path / "abs" / "x"


def test_mask_level_from_the_sizes():
    assert seg.mask_level((96, 96), (96, 96)) == 0
    assert seg.mask_level((48, 48), (96, 96)) == 1 and seg.mask_level((12, 12), (96, 96)) == 3
    # a mip export rounds down or up, each side on its own
    assert seg.mask_level((48, 51), (97, 101)) == 1 and seg.mask_level((49, 50), (97, 101)) == 1
    assert seg.mask_level((13, 12), (97, 101)) == 3
    assert seg.mask_level((48, 96), (96, 96)) is None and seg.mask_level((47, 47), (96, 96)) is None
    assert seg.mask_level((0, 3), (96, 96)) is None


def test_masks_at_a_mip_level_are_planned_there(stack_project, tmp_path):
    p = stack_project
    d = tmp_path / "mip2"; d.mkdir()
    for k, im in enumerate(_images(p)):
        _png(d / f"seg_{k}.png", np.full((24, 24), k + 1, np.uint16))
    plan = seg.plan_masks(p.root, p.section_names(), d)
    assert plan.mip == 2 and plan.to_dict()["mip"] == 2 and plan.bits == 16
    assert any("mip 2" in n and "4 x 4 px block" in n for n in plan.notes)
    assert seg.first_level(plan.mip, 0) == 2 and seg.first_level(plan.mip, 3) == 3 and seg.first_level(0, 1) == 1
    _png(d / "seg_1.png", np.full((48, 48), 2, np.uint16))
    with pytest.raises(ValueError, match="different mip levels"):
        seg.plan_masks(p.root, p.section_names(), d)
    _png(d / "seg_1.png", np.full((24, 25), 2, np.uint16))
    with pytest.raises(ValueError, match="at one mip level"):
        seg.plan_masks(p.root, p.section_names(), d)
    full = tmp_path / "full"; full.mkdir()
    for k, im in enumerate(_images(p)):
        _png(full / f"seg_{k}.png", np.full((96, 96), k + 1, np.uint8))
    assert seg.plan_masks(p.root, p.section_names(), full).mip == 0


def test_reducing_labels_keeps_odd_edges():
    a = np.zeros((5, 7), np.uint16); a[:, 6] = 9; a[4, :] = 3
    r = seg.reduce_labels(a, 1)
    assert r.shape == (3, 4) and r.dtype == np.uint16
    assert r[:2, 3].tolist() == [9, 9] and r[2, :3].tolist() == [3, 3, 3]      # the odd last row/column survive
    assert seg.reduce_labels(a, 2).shape == (2, 2) and seg.reduce_labels(a, 0) is a
    rng = np.random.default_rng(3)
    even = rng.integers(0, 5, (16, 16)).astype(np.uint8)
    assert np.array_equal(seg.reduce_labels(even, 2), seg.mode_downsample(seg.mode_downsample(even)))


def test_status_of_a_stack_that_starts_at_a_coarse_level(stack_project):
    p = stack_project
    base = seg.stack_dir(p.root, "coarse")
    sections = p.section_names()
    for k, s in enumerate(sections):
        for m in (2, 3, 4):
            seg.write_tile_metadata((base / f"mip{m}" / f"{k:02d}_{s}").mkdir(parents=True) or
                                    base / f"mip{m}" / f"{k:02d}_{s}", {}, 40.0)
    (base / "mip0").mkdir()                              # an empty folder is no level
    assert seg.stack_levels(base) == [2, 3, 4] and seg.stack_first_mip(base) == 2
    assert seg.stack_first_mip(p.root / "nothing") is None
    st = seg.stack_status(p.root, base, sections, 0, 4)
    assert (st.rendered, st.mipmapped, st.first_mip, st.done) == (3, 3, 2, True)


def test_stacks_from_0_3_8_must_be_rendered_again(stack_project):
    p = stack_project
    base = seg.stack_dir(p.root, "old")
    for k, s in enumerate(p.section_names()):
        seg.write_tile_metadata((base / "mip0" / f"{k:02d}_{s}").mkdir(parents=True) or base / "mip0" / f"{k:02d}_{s}",
                                {}, 10.0)
    assert seg.legacy_problem(base) == ""                                   # no masks.json: nothing to say
    (base / "masks.json").write_text(json.dumps({"bits": 8, "match": "order", "sections": 3}))   # as 0.3.8 wrote it
    assert "0.3.8" in seg.legacy_problem(base)
    assert "0.3.8" in seg.stack_status(p.root, base, p.section_names(), 0, 0).stale
    (base / "masks.json").write_text(json.dumps({"bits": 8, "mask_mip": 0, "first_mip": 0}))
    assert seg.legacy_problem(base) == "" and seg.stack_status(p.root, base, p.section_names(), 0, 0).stale == ""


def test_overlay_enlarges_a_coarse_mask_stack_exactly(tmp_path):
    pytest.importorskip("PySide6")
    from feabas_workbench.core.images import TiledSectionSource
    from feabas_workbench.ui.pages.page_export import _MaskOverlaySource
    images, masks = tmp_path / "images", tmp_path / "masks"
    _write_level(images / "mip0" / "s0001", {(0, 0): np.full((32, 32), 50, np.uint8)}, 32)
    rng = np.random.default_rng(4)
    coarse = rng.integers(1, 9, (8, 8)).astype(np.uint16)
    _write_level(masks / "mip2" / "s0001", {(0, 0): coarse}, 8, 40.0)
    src = _MaskOverlaySource(TiledSectionSource(images, "s0001"), TiledSectionSource(masks, "s0001"))
    want = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)
    assert np.array_equal(src.labels(0, 5, 3, 17, 22), want[3:25, 5:22])     # off the 4 px grid on purpose
    assert np.array_equal(src.labels(1, 1, 3, 11, 9), np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[3:12, 1:12])
    assert np.array_equal(src.labels(2, 0, 0, 8, 8), coarse)
    assert src.read(0, 0, 0, 32, 32).shape == (32, 32, 3)


def test_gui_card_with_masks_at_a_mip_level(stack_project, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from feabas_workbench.core.configs import ConfigStore
    from feabas_workbench.core.envs import Settings
    from feabas_workbench.ui.bridge import AppContext
    from feabas_workbench.ui.pages.page_export import ExportPage
    app = QApplication.instance() or QApplication([])
    p = stack_project
    ctx = AppContext(Settings()); ctx.project = ctx.local_project = p; ctx.configs = ConfigStore(p.configs_dir)
    monkeypatch.setattr(ctx, "require_feabas_python", lambda: sys.executable)
    messages = []
    for kind in ("information", "warning"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: messages.append(a[2]))
    page = ExportPage(ctx)
    submitted = []
    monkeypatch.setattr(page, "submit", lambda spec: submitted.append(spec))
    infos = []
    monkeypatch.setattr(page, "info", lambda text: infos.append(text))
    tform = p.root / "align" / "tform"; tform.mkdir(parents=True)
    for s in p.section_names():
        (tform / f"{s}.h5").write_bytes(b"")
    masks = tmp_path / "masks"; masks.mkdir()
    for k, im in enumerate(_images(p)):
        _png(masks / f"label_{k}.png", np.full((12, 12), k + 1, np.uint8))     # 96 px images at mip 3
    page.s_dir.setText(str(masks)); page.s_name.setText("coarse")
    out = p.root / "segmentation" / "coarse"
    seg.write_tile_metadata((out / "mip0" / "0_x").mkdir(parents=True) or out / "mip0" / "0_x", {}, 10.0)
    page._render_masks()
    assert "start at mip 0; these would start at mip 3" in messages[-1] and not submitted
    (out / "masks.json").write_text(json.dumps({"bits": 8}))                # a stack from 0.3.8
    page._render_masks()
    assert "0.3.8" in messages[-1] and "Clear them first" in messages[-1] and not submitted
    import shutil
    shutil.rmtree(out)
    page._render_masks()
    payload = json.loads(Path(submitted[-1].argv[-1]).read_text())
    assert payload["plan"]["mip"] == 3 and any("at mip 3" in t and "from mip 3" in t for t in infos)
    for k, s in enumerate(p.section_names()):
        for m in (3, 4):
            seg.write_tile_metadata((out / f"mip{m}" / f"{k}_{s}").mkdir(parents=True) or out / f"mip{m}" / f"{k}_{s}",
                                    {}, 80.0)
    page._refresh_masks()
    assert page.s_status.text().startswith("3 of 3 sections rendered from mip 3")
    page.e_stack.setCurrentIndex(page.e_stack.findData(str(out)))
    page._export()
    export = json.loads(Path(submitted[-1].argv[-1]).read_text())
    assert export["base_mip"] == 3 and export["slice_mip"] == 3 and any("level 0 of this export is mip 3" in t for t in infos)
    page.e_what.setCurrentIndex(page.e_what.findData("slices")); page.e_slice_mip.setValue(5)
    page._export()
    assert json.loads(Path(submitted[-1].argv[-1]).read_text())["slice_mip"] == 5
    page.close(); app.processEvents()


def test_render_level_and_where_a_stack_starts(tmp_path):
    assert seg.render_level(0, 0) == 0 and seg.render_level(0, 3) == 0      # full-resolution masks: rendered there
    assert seg.render_level(3, 3) == 1 and seg.render_level(1, 1) == 0 and seg.render_level(6, 6) == 4
    assert seg.render_level(3, 3, supersample=0) == 3
    assert seg.render_level(1, 3) == 1                      # masks finer than the first level: at their own level
    base = tmp_path / "stack"
    assert seg.stack_start(base) is None
    seg.write_tile_metadata((base / "mip1" / "s1").mkdir(parents=True) or base / "mip1" / "s1", {}, 20.0)
    assert seg.stack_start(base) == 1                                        # from the files
    (base / "masks.json").write_text(json.dumps({"first_mip": 3}))
    assert seg.stack_start(base) == 3                                        # a render in progress: as recorded
    (base / "masks.json").write_text("{broken")
    assert seg.stack_start(base) == 1


@pytest.mark.skipif(__import__("importlib").util.find_spec("feabas") is None,
                    reason="requires the separate FEABAS runtime")
def test_the_worker_scales_the_stitching_geometry_like_feabas(stack_project):
    """Run where the worker runs: in the project folder, with the workbench's FEABAS environment."""
    import subprocess
    from feabas_workbench.core.jobs import feabas_env
    code = """
import json
import numpy as np
from feabas.spatial import scale_coordinates
from feabas.stitcher import Mesh_Info
from feabas_workbench.workers.segmentation_render import _canvas_at, _geometry
v = np.array([[3.5, 3.5], [11.5, 3.5]])
info = [Mesh_Info(v, np.array([[16.0, 8.0]]), np.array([[0, 1, 1]]), v.copy())]
same, finer = _geometry(info, 3, 3)[0], _geometry(info, 3, 1)[0]
st = {"canvas_bbox": [10, 20, 1001, 2003], "mip": 1}
print(json.dumps({"same": [same.fixed_vertices.tolist(), same.moving_vertices.tolist(), same.moving_offsets.tolist()],
                  "finer": [finer.fixed_vertices.tolist(), finer.moving_vertices.tolist(), finer.moving_offsets.tolist()],
                  "feabas": scale_coordinates(v, 0.5).tolist(),
                  "full": _geometry(info, 0, 0)[0].fixed_vertices.tolist(),
                  "canvas": [_canvas_at(st, 1), _canvas_at(st, 3), _canvas_at(st, 0), _canvas_at({"canvas_bbox": None, "mip": 0}, 3)]}))
"""
    env = dict(os.environ, **feabas_env())
    run = subprocess.run([sys.executable, "-c", code], cwd=stack_project.root, env=env, text=True, capture_output=True,
                         timeout=60)
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout.strip().splitlines()[-1])
    # the centre of each 8 x 8 block of full-resolution pixels is the centre of a mip-3 pixel
    assert np.allclose(got["same"][0], [[0, 0], [1, 0]]) and np.allclose(got["same"][1], [[0, 0], [1, 0]])
    assert np.allclose(got["same"][2], [[2, 1]])
    # mip-3 masks placed into a mip-1 montage: tile coordinates at mip 3, montage coordinates at mip 1
    assert np.allclose(got["finer"][0], [[0, 0], [1, 0]]) and np.allclose(got["finer"][1], [[1.5, 1.5], [5.5, 1.5]])
    assert np.allclose(got["finer"][2], [[8, 4]]) and np.allclose(got["feabas"], got["finer"][1])
    assert np.allclose(got["full"], [[3.5, 3.5], [11.5, 3.5]])
    assert got["canvas"] == [[10, 20, 1001, 2003], [2, 5, 251, 501], [20, 40, 2002, 4006], None]


@pytest.mark.skipif(__import__("importlib").util.find_spec("feabas") is None,
                    reason="requires the separate FEABAS runtime")
def test_an_image_stack_montage_keeps_every_mask_column(stack_project, tmp_path):
    """FEABAS places a single image 1.5 px into its montage: nearest-neighbour samples land exactly
    between two pixels, and rounding halves to even used columns 0, 2, 2, 4, 4, ... (0.3.8)."""
    import subprocess
    from feabas_workbench.core.jobs import feabas_env
    code = f"""
import json
import cv2
import numpy as np
from pathlib import Path
from feabas.stitcher import Mesh_Info, MontageRenderer
from feabas_workbench.workers.segmentation_render import LOADER, MONTAGE_PATTERN, NUDGE, _exact_montage, _nudged
out = Path({str(tmp_path)!r})
h, w = 24, 40
cv2.imwrite(str(out / "col.png"), np.tile(np.arange(1, w + 1, dtype=np.uint16), (h, 1)))
v0 = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [-0.5, h - 0.5], [w - 0.5, h - 0.5]])
o = np.array([[10.0, 10.0]])
info = [Mesh_Info(v0 + 1.5 - o, o, np.array([[0, 1, 2], [1, 3, 2]]), v0)]     # as FEABAS stitches one image
exact, fraction = _exact_montage(info, 0, 0)
picked = {{"fraction": fraction.tolist()}}
for name, geometry in (("plain", info), ("nudged", _nudged(info, NUDGE)), ("exact", _nudged(exact, NUDGE))):
    r = MontageRenderer([str(out / "col.png")], geometry, np.array([[h, w]]), resolution=10.0, loader_settings=dict(LOADER))
    d = out / name; d.mkdir()
    r.render_one_section(str(d / "s"), meta_name=str(d / "metadata.txt"), tile_size=[64, 64], driver="image",
                         num_workers=1, filename_settings={{"pattern": MONTAGE_PATTERN, "one_based": True}},
                         render_settings={{"blend": "NEAREST", "remap_interp": "NEAREST", "fillval": 0}})
    a = cv2.imread(str(next(d.glob("*.png"))), cv2.IMREAD_UNCHANGED)
    picked[name] = (a[12].astype(int) - 1).tolist()
print(json.dumps(picked))
"""
    run = subprocess.run([sys.executable, "-c", code], cwd=stack_project.root, env=dict(os.environ, **feabas_env()),
                         text=True, capture_output=True, timeout=120)
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout.strip().splitlines()[-1])
    plain = [c for c in got["plain"] if c >= 0]
    nudged = [c for c in got["nudged"] if c >= 0]
    assert plain[:6] == [0, 2, 2, 4, 4, 6]                  # what FEABAS does with exact halves
    assert nudged == list(range(len(nudged))) and len(nudged) >= 39      # every column once, in order
    # a whole-pixel copy, the half pixel left for the alignment step
    exact = [c for c in got["exact"] if c >= 0]
    assert exact == list(range(40)) and got["exact"].index(0) == 2 and np.allclose(np.abs(got["fraction"]), 0.5)
