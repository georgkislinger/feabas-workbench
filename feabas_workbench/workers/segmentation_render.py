"""
Render segmentation masks through a project's stitching and alignment transforms (FEABAS env).

For every section, in its own worker process:
  1. place the mask files like the images: a FEABAS MontageRenderer with the geometry of
     stitch/tform/<section>.h5 but the mask files and none of the images' intensity processing
     (no CLAHE, inversion or per-tile brightness/contrast correction), blending NEAREST;
  2. move that montage with the section's alignment mesh align/tform/<section>.h5 onto the canvas
     of align/tform/canvas.json - FEABAS's aligned render, with the same tile size, file names
     and folder names as the aligned PNG stack;
  3. build the mipmaps on the images' tile grid, each 2x2 block becoming its majority label
     (core.segmentation.build_label_pyramid), never an average.
Nearest-neighbour sampling throughout, so every output pixel is one of the input labels (0 outside
the masks). The masks montage of step 1 is removed once its section is done.

Spec: {"root", "out_dir", "plan": core.segmentation.MaskPlan.to_dict(), "workers"}.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import multiprocessing
import os
import shutil
import time
from pathlib import Path

from feabas_workbench.core.segmentation import MONTAGE_DIR
from feabas_workbench.workers.common import load_spec, log, progress, result, run

MONTAGE_PATTERN = "_tr{ROW_IND}-tc{COL_IND}.png"
CACHE_TILES = 16
LOADER = dict(fillval=0, apply_CLAHE=False, inverse=False, cache_size=CACHE_TILES)


def render_settings() -> dict:
    """What FEABAS's own renders of this project use (configs read from the working directory)."""
    from feabas import config
    stitch = config.stitch_configs().get("rendering", {})
    align = config.align_configs()
    ra, ds = align.get("rendering", {}), align.get("downsample", {})
    pattern = str(ra.get("pattern", "_tr{ROW_IND}-tc{COL_IND}.png"))
    return dict(montage_tile=stitch.get("tile_size", [4096, 4096]),
                montage_resolution=float(config.montage_resolution()),
                mip=int(ra.get("mip_level", 0) or 0),
                tile_size=ra.get("tile_size", [4096, 4096]),
                pattern=os.path.splitext(pattern)[0] + ".png",     # labels stay lossless whatever the images use
                one_based=bool(ra.get("one_based", True)),
                canvas_bbox=ra.get("canvas_bbox", None),
                offset_bbox=bool(ra.get("offset_bbox", True)),
                prefix_z=bool(ra.get("prefix_z_number", True)),
                max_mip=int(ds.get("max_mip", 8)))


def canvas_offset(root: Path, settings: dict):
    """The shift FEABAS's aligned render applies (align/tform/canvas.json), creating the canvas the
    way FEABAS does before its first render if it does not exist yet, so images and masks share it."""
    import numpy as np
    from feabas import common, constant as const
    from feabas.mesh import Mesh
    tform_dir = root / "align" / "tform"
    canvas = tform_dir / "canvas.json"
    tforms = sorted(tform_dir.glob("*.h5"))
    if not canvas.is_file() and settings["offset_bbox"] and tforms:
        union = None
        for t in tforms:
            M = Mesh.from_h5(str(t))
            M.change_resolution(settings["montage_resolution"])
            bbox = M.bbox(gear=const.MESH_GEAR_MOVING, offsetting=True)
            union = bbox if union is None else common.bbox_union((union, bbox))
        union = np.array(union, dtype=float)
        union[:2] = np.floor(union[:2])
        union[-2:] = np.ceil(union[-2:]) + 1
        tmp = canvas.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"mip0": [int(s) for s in union]}), encoding="utf-8")
        if not canvas.exists():
            os.replace(tmp, canvas)
        else:
            tmp.unlink()
        log(f"canvas of the aligned stack set to {[int(s) for s in union]} (align/tform/canvas.json)")
    if canvas.is_file():
        bbox = common.get_canvas_bbox(str(canvas), target_mip=0)
        return [-float(bbox[0]), -float(bbox[1])]
    return None


def z_prefixes(root: Path, settings: dict) -> dict:
    """<z>_ before each section's folder name, as FEABAS's aligned render numbers them."""
    if not settings["prefix_z"]:
        return {}
    import numpy as np
    from feabas import common
    seclist = sorted(str(p) for p in (root / "align" / "mesh").glob("*.h5"))
    if not seclist:
        return {}
    seclist, z = common.rearrange_section_order(seclist, str(root / "section_order.txt"))
    top = int(np.max(z)) if len(z) else 0
    digits = (math.ceil(math.log10(top)) + 1) if top > 0 else 1
    return {Path(s).stem: str(k).rjust(digits, "0") + "_" for k, s in zip(z, seclist) if s}


def _restart(folder: Path) -> None:
    """A level without metadata.txt is unfinished: FEABAS would keep its (possibly cut-short) tiles."""
    if folder.is_dir() and not (folder / "metadata.txt").is_file():
        for p in folder.iterdir():
            if p.is_file():
                p.unlink()


def render_section(task: dict) -> dict:
    """One section, start to finish (runs in a worker process). Returns a summary or the error."""
    t0 = time.time()
    sec = task["section"]
    try:
        return _render_section(task) | {"section": sec, "seconds": time.time() - t0}
    except Exception as e:  # noqa: BLE001 - reported per section by the main process
        import traceback
        return {"section": sec, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}


def _render_section(task: dict) -> dict:
    import numpy as np
    from feabas import constant as const, dal
    from feabas.mesh import Mesh
    from feabas.mipmap import get_image_loader
    from feabas.renderer import render_whole_mesh
    from feabas.stitcher import MontageRenderer
    from feabas_workbench.core.segmentation import build_label_pyramid, read_tile_metadata, write_tile_metadata
    root, out, sec, st = Path(task["root"]), Path(task["out_dir"]), task["section"], task["settings"]
    masks, offset, zsec = task["masks"], task["offset"], task["zprefix"] + task["section"]
    mip, max_mip = st["mip"], st["max_mip"]
    resolution = st["montage_resolution"] * 2 ** mip
    levels = [out / f"mip{m}" / zsec for m in range(mip, max(mip, max_mip) + 1)]
    if all((d / "metadata.txt").is_file() for d in levels):
        return {"tiles": 0, "skipped": True}
    target = levels[0]
    montage = out / MONTAGE_DIR / "mip0" / sec
    if not (target / "metadata.txt").is_file():
        # 1. the masks placed like the images: stitching geometry, mask files, no intensity processing
        images = MontageRenderer.from_h5(str(root / "stitch" / "tform" / f"{sec}.h5"))
        (relpaths, mesh_info, tile_sizes), _ = images.init_args()
        files = []
        for rel in relpaths:
            name = Path(rel).name
            if name not in masks:
                raise RuntimeError(f"no mask for image {name}")
            files.append(masks[name])
        labels = MontageRenderer(files, mesh_info, np.asarray(tile_sizes), resolution=images.resolution,
                                 loader_settings=dict(LOADER))
        if not (montage / "metadata.txt").is_file():
            _restart(montage)
            montage.mkdir(parents=True, exist_ok=True)
            # at the images' own resolution, never scaled: a montage scaled below 1/3 would be blurred
            labels.render_one_section(str(montage / sec), meta_name=str(montage / "metadata.txt"),
                                      tile_size=st["montage_tile"], driver="image", num_workers=1,
                                      filename_settings={"pattern": MONTAGE_PATTERN, "one_based": True},
                                      render_settings={"blend": "NEAREST", "remap_interp": "NEAREST", "fillval": 0})
        # 2. FEABAS's aligned render of that montage: same mesh, canvas, tiles and names as the images
        _restart(target)
        target.mkdir(parents=True, exist_ok=True)
        loader = get_image_loader(str(montage), resolution=images.resolution, **LOADER)
        if loader is None:                  # this section's masks are all 0: nothing to place
            write_tile_metadata(target, {}, resolution)
        else:
            M = Mesh.from_h5(str(root / "align" / "tform" / f"{sec}.h5"), locked=False)
            M.change_resolution(resolution)
            if offset is not None:
                M.apply_translation(np.array(offset) * st["montage_resolution"] / resolution,
                                    gear=const.MESH_GEAR_MOVING)
            rendered = render_whole_mesh(M, loader, str(target / sec), tile_size=st["tile_size"],
                                         pattern=st["pattern"], one_based=st["one_based"],
                                         canvas_bbox=st["canvas_bbox"], num_workers=1,
                                         remap_interp="NEAREST", fillval=0, geodesic_mask=False)
            names = sorted(rendered)
            if names:
                dal.StaticImageLoader(names, bboxes=[rendered[n] for n in names],
                                      resolution=resolution).to_coordinate_file(str(target / "metadata.txt"))
            else:
                write_tile_metadata(target, {}, resolution)
    n_tiles = len(read_tile_metadata(target / "metadata.txt"))
    # 3. mipmaps: each 2x2 block becomes its majority label, on the images' tile grid
    tile = st["tile_size"]
    tile_hw = (tile[0], tile[-1]) if isinstance(tile, (list, tuple)) else (tile, tile)
    build_label_pyramid(out, zsec, sec, mip, max(mip, max_mip), tile_hw, st["pattern"], st["one_based"], resolution)
    shutil.rmtree(montage, ignore_errors=True)
    return {"tiles": n_tiles}


def main() -> int:
    spec = load_spec("render segmentation masks through the alignment")
    root = Path(spec["root"]).resolve()
    os.chdir(root)                          # FEABAS reads configs/general_configs.yaml from here
    out = Path(spec["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = spec["plan"]
    st = render_settings()
    offset = canvas_offset(root, st)
    zp = z_prefixes(root, st)
    missing = [s for s, _ in plan["sections"] if not (root / "align" / "tform" / f"{s}.h5").is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} section(s) are not aligned yet (no align/tform/<section>.h5), e.g. "
                           + ", ".join(missing[:3]) + ": run the alignment first")
    (out / "masks.json").write_text(json.dumps(dict(spec.get("source", {}), bits=plan["bits"], match=plan["mode"],
                                                    sections=len(plan["sections"]),
                                                    written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2),
                                    encoding="utf-8")
    tasks = [dict(root=str(root), out_dir=str(out), section=s, masks=m, settings=st, offset=offset,
                  zprefix=zp.get(s, "")) for s, m in plan["sections"]]
    workers = max(1, min(len(tasks), int(spec.get("workers", 1))))
    log(f"{len(tasks)} section(s), {workers} at once; mip {st['mip']} with mipmaps up to {st['max_mip']}, "
        f"{plan['bits']}-bit labels, nearest-neighbour sampling; output {out}")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    failed, done, tiles = [], 0, 0
    progress(0, len(tasks), "starting")
    if workers == 1:
        results = map(render_section, tasks)
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        results = (f.result() for f in concurrent.futures.as_completed([pool.submit(render_section, t) for t in tasks]))
    try:
        for res in results:
            done += 1
            if "error" in res:
                failed.append(res)
                log(f"{res['section']}: FAILED - {res['error']}\n{res.get('trace', '')}")
            else:
                tiles += res.get("tiles", 0)
                state = "already done" if res.get("skipped") else f"{res.get('tiles', 0)} tiles"
                log(f"{res['section']}: {state} | {res.get('seconds', 0):.1f} s")
            progress(done, len(tasks), res["section"])
    finally:
        if workers > 1:
            pool.shutdown(wait=True, cancel_futures=True)
    montage_root = out / MONTAGE_DIR
    if montage_root.is_dir() and not any(montage_root.rglob("*.png")):
        shutil.rmtree(montage_root, ignore_errors=True)
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(tasks)} section(s) failed: "
                           + "; ".join(f"{r['section']}: {r['error']}" for r in failed[:3]))
    result({"out_dir": str(out), "sections": len(tasks), "tiles": tiles, "bits": plan["bits"]})
    return 0


if __name__ == "__main__":
    run(main)
