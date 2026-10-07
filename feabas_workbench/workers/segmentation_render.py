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
the masks). The masks montage of step 1 is removed once its section is done. For one image per
section (an image stack) step 1 is an exact copy in whole pixels and step 2 also applies the fraction
of a pixel that leaves out (_exact_montage), so such a section is resampled only once.

The stack starts at core.segmentation.first_level: the images' render level, or the masks' own
level when they were exported at a coarser mip. Steps 1 and 2 run at core.segmentation.render_level:
full resolution for full-resolution masks, else up to two levels finer than the first level
(each mask pixel covering exactly the block it stands for), so that nearest-neighbour resampling
moves label boundaries by a fraction of a mask pixel only; the majority mipmaps then bring them to
the first level, and the finer levels are removed. Coordinates are scaled with
feabas.spatial.scale_coordinates, the function FEABAS scales its meshes with (a coarse pixel is
centred on the block of fine pixels it covers - also how FEABAS's mipmaps and its aligned render at
a mip level relate to full resolution), so masks and images share one pixel grid at every level.

Spec: {"root", "out_dir", "plan": core.segmentation.MaskPlan.to_dict(), "workers", "supersample"
(optional, levels; default core.segmentation.SUPERSAMPLE)}.
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

from feabas_workbench.core.segmentation import (MONTAGE_DIR, SUPERSAMPLE, first_level, legacy_problem, render_level,
                                                stack_start)
from feabas_workbench.workers.common import load_spec, log, progress, result, run

MONTAGE_PATTERN = "_tr{ROW_IND}-tc{COL_IND}.png"
# Source coordinates are moved by this much (px of the level rendered) before nearest-neighbour
# sampling. FEABAS puts a single-image section 1.5 px into its montage, so every sample of an image
# stack falls exactly between two mask pixels, and OpenCV rounds halves to even: columns and rows
# 0, 2, 2, 4, 4, ... - every odd one dropped, every even one doubled. Nudged, they all round the
# same way: a clean shift within the unavoidable half pixel, and nothing is lost.
NUDGE = 1.0 / 64
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


def _metadata_resolution(meta: Path) -> float | None:
    for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if parts[0] == "{RESOLUTION}" and len(parts) > 1:
            return float(parts[1])
    return None


def _geometry(mesh_info: list, mask_level: int, level: int) -> list:
    """
    The stitching geometry for mask tiles at mip *mask_level*, placed into a montage at mip *level*
    (<= mask_level): tile coordinates are scaled to the masks' level, montage coordinates to the
    montage's, both with feabas.spatial.scale_coordinates as Mesh.change_resolution scales vertices
    (offsets are plain translations).
    """
    import numpy as np
    from feabas.spatial import scale_coordinates
    from feabas.stitcher import Mesh_Info
    s_tile, s_montage = 1.0 / 2 ** mask_level, 1.0 / 2 ** level
    return [Mesh_Info(scale_coordinates(mi.moving_vertices, s_montage), np.asarray(mi.moving_offsets) * s_montage,
                      mi.triangles, scale_coordinates(mi.fixed_vertices, s_tile)) for mi in mesh_info]


def _exact_montage(mesh_info: list, mask_mip: int, level: int):
    """
    For a section of one image placed by a plain shift (an imported image stack), the montage is
    made an exact copy - each mask pixel becomes 2**(mask_mip - level) whole montage pixels, no
    sample on a pixel boundary - and the fraction of a pixel that takes is returned for the
    alignment step to apply instead (it resamples anyway): every section is then sampled once,
    from the exact mapping. FEABAS places a single image 1.5 px into its montage, so otherwise every
    image of a stack would move by the same half pixel. Anything else: (mesh_info, (0, 0)).
    """
    import numpy as np
    none = np.zeros(2)
    if len(mesh_info) != 1:
        return mesh_info, none
    mi = mesh_info[0]
    scale = 2.0 ** (mask_mip - level)
    offsets = np.asarray(mi.moving_offsets, dtype=float)
    shift = mi.moving_vertices + offsets.reshape(1, 2) - scale * mi.fixed_vertices
    if np.ptp(shift, axis=0).max() > 1e-6:          # not a plain shift
        return mesh_info, none
    shift = shift.mean(axis=0)
    phase = (0.5 + scale / 2) % 1.0                 # where a shift puts block edges between pixels
    fraction = shift - (np.round(shift - phase) + phase)
    return [mi._replace(moving_offsets=offsets - fraction.reshape(offsets.shape))], fraction


def _nudged(mesh_info: list, amount: float) -> list:
    """The stitching geometry with its source (tile) coordinates moved by *amount* (in their units)."""
    return [mi._replace(fixed_vertices=mi.fixed_vertices + amount) for mi in mesh_info]


def _canvas_at(settings: dict, level: int):
    """The aligned render's canvas_bbox setting (at the images' render level) at *level*."""
    bbox = settings["canvas_bbox"]
    if bbox is None or level == settings["mip"]:
        return bbox
    f = 2 ** (level - settings["mip"])
    return [math.floor(bbox[0] / f), math.floor(bbox[1] / f), math.ceil(bbox[2] / f), math.ceil(bbox[3] / f)]


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
    from feabas_workbench.core.segmentation import (build_label_pyramid, read_image_header, read_tile_metadata,
                                                    write_tile_metadata)
    root, out, sec, st = Path(task["root"]), Path(task["out_dir"]), task["section"], task["settings"]
    masks, offset, zsec = task["masks"], task["offset"], task["zprefix"] + task["section"]
    mask_mip = int(task.get("mask_mip", 0))
    first = int(task.get("first_mip", st["mip"]))
    level = int(task.get("render_mip", first))          # rendered here, kept from *first* up
    top = max(first, st["max_mip"])
    resolution = st["montage_resolution"] * 2 ** level
    finer = [out / f"mip{m}" / zsec for m in range(level, first)]
    kept = [out / f"mip{m}" / zsec for m in range(first, top + 1)]
    if all((d / "metadata.txt").is_file() for d in kept):
        for d in finer:
            shutil.rmtree(d, ignore_errors=True)
        return {"tiles": 0, "skipped": True}
    target = out / f"mip{level}" / zsec
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
        montage_resolution = images.resolution * 2 ** level
        if mask_mip or level:               # masks at a mip level, or a montage above full resolution
            mesh_info = _geometry(mesh_info, mask_mip, level)
            tile_sizes = [(h.height, h.width) for h in map(read_image_header, map(Path, files))]
        mesh_info, fraction = _exact_montage(mesh_info, mask_mip, level)
        mesh_info = _nudged(mesh_info, NUDGE * 2.0 ** (level - mask_mip))       # NUDGE px of this level
        labels = MontageRenderer(files, mesh_info, np.asarray(tile_sizes), resolution=montage_resolution,
                                 loader_settings=dict(LOADER))
        meta = montage / "metadata.txt"
        if meta.is_file() and abs((_metadata_resolution(meta) or 0) - montage_resolution) > 1e-6 * montage_resolution:
            meta.unlink()                   # left by an interrupted run at another level
        if not meta.is_file():
            _restart(montage)
            montage.mkdir(parents=True, exist_ok=True)
            # never scaled while rendering: a montage scaled below 1/3 would be blurred
            labels.render_one_section(str(montage / sec), meta_name=str(meta),
                                      tile_size=st["montage_tile"], driver="image", num_workers=1,
                                      filename_settings={"pattern": MONTAGE_PATTERN, "one_based": True},
                                      render_settings={"blend": "NEAREST", "remap_interp": "NEAREST", "fillval": 0})
        # 2. FEABAS's aligned render of that montage: same mesh, canvas, tiles and names as the images
        _restart(target)
        target.mkdir(parents=True, exist_ok=True)
        loader = get_image_loader(str(montage), resolution=montage_resolution, **LOADER)
        if loader is None:                  # this section's masks are all 0: nothing to place
            write_tile_metadata(target, {}, resolution)
        else:
            M = Mesh.from_h5(str(root / "align" / "tform" / f"{sec}.h5"), locked=False)
            # render_whole_mesh renders at the loader's resolution: the montage's, i.e. this level's
            M.change_resolution(montage_resolution)
            # the source side: the fraction the exact montage left out, and the nudge
            M.apply_translation(np.array([NUDGE, NUDGE]) - fraction, gear=const.MESH_GEAR_INITIAL)
            if offset is not None:
                M.apply_translation(np.array(offset) * st["montage_resolution"] / montage_resolution,
                                    gear=const.MESH_GEAR_MOVING)
            rendered = render_whole_mesh(M, loader, str(target / sec), tile_size=st["tile_size"],
                                         pattern=st["pattern"], one_based=st["one_based"],
                                         canvas_bbox=_canvas_at(st, level), num_workers=1,
                                         remap_interp="NEAREST", fillval=0, geodesic_mask=False)
            names = sorted(rendered)
            if names:
                dal.StaticImageLoader(names, bboxes=[rendered[n] for n in names],
                                      resolution=resolution).to_coordinate_file(str(target / "metadata.txt"))
            else:
                write_tile_metadata(target, {}, resolution)
    # 3. mipmaps: each 2x2 block becomes its majority label, on the images' tile grid
    tile = st["tile_size"]
    tile_hw = (tile[0], tile[-1]) if isinstance(tile, (list, tuple)) else (tile, tile)
    build_label_pyramid(out, zsec, sec, level, top, tile_hw, st["pattern"], st["one_based"], resolution)
    n_tiles = len(read_tile_metadata(kept[0] / "metadata.txt"))
    for d in finer:                         # finer than the masks themselves: not kept
        shutil.rmtree(d, ignore_errors=True)
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
    mask_mip = int(plan.get("mip", 0))
    first = first_level(mask_mip, st["mip"])
    level = render_level(mask_mip, first, int(spec.get("supersample", SUPERSAMPLE)))
    existing = stack_start(out)
    if legacy_problem(out):
        raise RuntimeError(f"{out}: {legacy_problem(out)}. Clear them first, then render them again")
    if existing is not None and existing != first:
        raise RuntimeError(f"{out} holds aligned masks that start at mip {existing}; these would start at mip {first}. "
                           "Clear them first, or give the new masks another name")
    offset = canvas_offset(root, st)
    zp = z_prefixes(root, st)
    missing = [s for s, _ in plan["sections"] if not (root / "align" / "tform" / f"{s}.h5").is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} section(s) are not aligned yet (no align/tform/<section>.h5), e.g. "
                           + ", ".join(missing[:3]) + ": run the alignment first")
    (out / "masks.json").write_text(json.dumps(dict(spec.get("source", {}), bits=plan["bits"], match=plan["mode"],
                                                    mask_mip=mask_mip, first_mip=first, render_mip=level,
                                                    sections=len(plan["sections"]),
                                                    written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2),
                                    encoding="utf-8")
    tasks = [dict(root=str(root), out_dir=str(out), section=s, masks=m, settings=st, offset=offset,
                  zprefix=zp.get(s, ""), mask_mip=mask_mip, first_mip=first, render_mip=level)
             for s, m in plan["sections"]]
    workers = max(1, min(len(tasks), int(spec.get("workers", 1))))
    what = (f"masks at mip {mask_mip}, " if mask_mip else "") + f"aligned masks from mip {first}"
    if level < first:
        what += f" (rendered at mip {level}, then reduced by majority)"
    log(f"{len(tasks)} section(s), {workers} at once; {what} with mipmaps up to {max(first, st['max_mip'])}, "
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
    for m in range(level, first):           # the finer levels emptied section by section
        d = out / f"mip{m}"
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(tasks)} section(s) failed: "
                           + "; ".join(f"{r['section']}: {r['error']}" for r in failed[:3]))
    result({"out_dir": str(out), "sections": len(tasks), "tiles": tiles, "bits": plan["bits"], "mask_mip": mask_mip,
            "first_mip": first, "render_mip": level})
    return 0


if __name__ == "__main__":
    run(main)
