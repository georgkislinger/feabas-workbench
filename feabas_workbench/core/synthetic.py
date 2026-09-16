"""
A tiny synthetic serial-section dataset, for tests, CI screenshots and end-to-end runs.

``make_synthetic_tiles`` writes a few sections of overlapping tiles in Thermo/Maps naming
(``Tile_<row>-<col>-<id>_0-000.s<section>_e00.tif``): cell-like blobs on a textured
background that drift and grow a little from section to section, with a dark rim around
the imaged area, so stitching has overlaps to match, alignment has structure to follow
and the mask methods have a border to find. ``make_demo_project`` turns that into a
ready workbench project (coordinate files, volume info, suggested mips) the same way the
Project page does - without Qt, so it can run in CI and in tests.

    python -m feabas_workbench.core.synthetic D:/demo            # dataset + project in D:/demo

Nothing here imports Qt.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from . import tiles as T
from .configs import ConfigStore, suggest_thumbnail_mip, suggest_working_mip, set_thumbnail_mip
from .project import Project


def synthetic_sections(n_sections: int = 4, height: int = 730, width: int = 730, seed: int = 0,
                       rim: int = 24, n_cells: int = 160) -> list[np.ndarray]:
    """
    Grey-scale sections (uint8) that look enough like EM to stitch and align: a noisy, lightly
    blurred background, dark-membraned round cells of varied size that shift by a pixel or two
    and grow or shrink per section, dark dots inside, and a rim of no-data (0) around the tissue.
    """
    import cv2
    rng = np.random.default_rng(seed)
    cx = rng.uniform(rim + 10, width - rim - 10, n_cells)
    cy = rng.uniform(rim + 10, height - rim - 10, n_cells)
    r0 = rng.uniform(7, 34, n_cells)
    dr = rng.uniform(-1.5, 1.5, n_cells)          # growth per section
    shade = rng.uniform(150, 205, n_cells)         # cell interior
    dots = [(rng.uniform(-0.6, 0.6, 6), rng.uniform(-0.6, 0.6, 6)) for _ in range(n_cells)]
    out = []
    for z in range(n_sections):
        img = np.clip(rng.normal(132, 14, (height, width)), 0, 255).astype(np.float32)
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
        dx, dy = rng.normal(0, 1.2), rng.normal(0, 1.2)       # small stage jitter between sections
        for i in range(n_cells):
            r = r0[i] + dr[i] * z
            if r < 3:
                continue
            c = (int(round(cx[i] + dx)), int(round(cy[i] + dy)))
            cv2.circle(img, c, int(r), float(shade[i]), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, c, int(r), 38.0, 2, lineType=cv2.LINE_AA)
            for ux, uy in zip(*dots[i]):
                cv2.circle(img, (int(c[0] + ux * r), int(c[1] + uy * r)), max(1, int(r / 7)), 60.0, -1,
                           lineType=cv2.LINE_AA)
        img += rng.normal(0, 6, img.shape).astype(np.float32)
        img = np.clip(img, 1, 255).astype(np.uint8)        # 0 is reserved for no-data
        img[:rim, :] = 0; img[-rim:, :] = 0; img[:, :rim] = 0; img[:, -rim:] = 0
        out.append(img)
    return out


def make_synthetic_tiles(root: os.PathLike | str, n_sections: int = 4, rows: int = 2, cols: int = 2,
                         tile: int = 384, overlap_pct: float = 10.0, seed: int = 0,
                         first_section: int = 1) -> dict:
    """
    Write the dataset under *root* and return its facts (tile size, overlap, section size ...).
    Tiles are cut from one section image with the given overlap, so they stitch back exactly.
    """
    import tifffile
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ov = int(round(tile * overlap_pct / 100.0))
    width = cols * tile - (cols - 1) * ov
    height = rows * tile - (rows - 1) * ov
    sections = synthetic_sections(n_sections, height, width, seed)
    rng = np.random.default_rng(seed + 1)
    n = 0
    for zi, img in enumerate(sections):
        z = first_section + zi
        k = 0
        for iy in range(rows):
            for ix in range(cols):
                x0, y0 = ix * (tile - ov), iy * (tile - ov)
                crop = img[y0:y0 + tile, x0:x0 + tile].astype(np.float32)
                crop = np.clip(crop * rng.uniform(0.93, 1.07) + rng.uniform(-4, 4), 0, 255)   # per-tile gain/offset
                crop[img[y0:y0 + tile, x0:x0 + tile] == 0] = 0
                name = f"Tile_{iy + 1:03d}-{ix + 1:03d}-{k:06d}_0-000.s{z:04d}_e00.tif"
                tifffile.imwrite(root / name, crop.astype(np.uint8))
                k += 1
                n += 1
    return {"root": root, "n_sections": n_sections, "rows": rows, "cols": cols, "tile": tile,
            "overlap_px": ov, "overlap_pct": overlap_pct, "section_w": width, "section_h": height, "n_tiles": n}


def make_demo_project(project_root: os.PathLike | str, tiles_root: os.PathLike | str | None = None,
                      pixel_nm: float = 10.0, thickness_nm: float = 50.0, **tile_kwargs) -> Project:
    """
    A complete workbench project on a synthetic dataset: tiles (under <project>/raw_tiles unless
    *tiles_root* is given), stitch coordinate files, volume info and the mip levels the Project
    page would suggest. Ready for 'Match tiles'.
    """
    project_root = Path(project_root)
    tiles_root = Path(tiles_root) if tiles_root else project_root / "raw_tiles"
    facts = make_synthetic_tiles(tiles_root, **tile_kwargs)
    p = Project.load(project_root) if Project.exists(project_root) else Project.create(project_root)
    rule = T.NamingRule(preset="thermo", ext="tif", recursive=True)
    layout = T.LayoutParams(overlap_x=facts["overlap_pct"], overlap_y=facts["overlap_pct"], overlap_unit="percent", mode="grid")
    plan = T.build_plan(tiles_root, rule, layout, resolution_nm=pixel_nm, read_stage=False)
    if not plan.sections:
        raise RuntimeError("synthetic tiles were not recognised: " + "; ".join(plan.warnings))
    st = p.state
    st.source.root_dir = str(tiles_root)
    st.source.rule = rule.to_dict()
    st.source.layout = layout.to_dict()
    st.source.path_mode = "relative"
    v = st.volume
    v.pixel_size_nm, v.section_thickness_nm = float(pixel_nm), float(thickness_nm)
    v.tile_w, v.tile_h = plan.tile_w, plan.tile_h
    v.n_sections, v.n_tiles = len(plan.sections), plan.n_tiles
    v.grid_rows, v.grid_cols = plan.grid_shape()
    v.section_names = list(plan.sections)
    v.dtype = "uint8"
    plan.resolution_nm = v.pixel_size_nm
    T.write_plan(plan, p.stitch_coord_dir, "relative")
    p.write_general_config()
    p.save()
    cs = ConfigStore(p.configs_dir)
    w, h = plan.section_bbox(next(iter(plan.sections)))
    tm = suggest_thumbnail_mip(w, h)
    cs.set("alignment", "matching.working_mip_level", int(suggest_working_mip(v.pixel_size_nm, v.section_thickness_nm)))
    set_thumbnail_mip(cs, tm)
    cs.set("stitching", "section_thickness", float(v.section_thickness_nm))
    cs.save()
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Make a small synthetic dataset and a workbench project on it.")
    ap.add_argument("project", help="project folder to create (tiles go to <project>/raw_tiles)")
    ap.add_argument("--sections", type=int, default=4)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--tile", type=int, default=384, help="tile edge in px")
    ap.add_argument("--overlap", type=float, default=10.0, help="tile overlap in percent")
    ap.add_argument("--pixel-nm", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    p = make_demo_project(a.project, n_sections=a.sections, rows=a.rows, cols=a.cols, tile=a.tile,
                          overlap_pct=a.overlap, seed=a.seed, pixel_nm=a.pixel_nm)
    v = p.state.volume
    print(f"{p.root}: {v.n_sections} sections x {v.grid_rows}x{v.grid_cols} tiles of {v.tile_w}x{v.tile_h} px, "
          f"{v.pixel_size_nm:g} nm/px; coordinate files in {p.stitch_coord_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
