"""
Export the aligned PNG-tile stack (FEABAS align rendering + downsample) for
VASTlite: <out>/mipM/sliceZZZZ/ZZZZ_trR-tcC.png + <name>.vsvi, and/or as an
OME-Zarr (v0.4, zarr v2, uncompressed chunks written directly like the
TrakEM2 exporter did).

FEABAS layout (one_based, prefix_z_number):
    aligned_stack/mip0/<zzz>_<section>/<zzz>_<section>_tr{r}-tc{c}.png + metadata.txt
Tiles are square (tile_size) and padded, so they map 1:1 onto VAST tiles.
Where possible files are hard-linked instead of copied.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np

from feabas_workbench.workers.common import load_spec, progress, result, log, run
from feabas_workbench.core.images import parse_feabas_metadata, imread

_TILE_RE = re.compile(r"_tr(\d+)-tc(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def section_dirs(base: Path, mip: int) -> list[Path]:
    d = base / f"mip{mip}"
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def z_of(dirname: str) -> int:
    m = re.match(r"(\d+)_", dirname)
    return int(m.group(1)) if m else 0


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def export_vast(base: Path, out: Path, name: str, voxel: tuple[float, float, float], mips: list[int],
                fmt_bytes: int = 1, progress_cb=None) -> dict:
    secs0 = section_dirs(base, 0)
    if not secs0:
        raise RuntimeError(f"no rendered sections under {base / 'mip0'}")
    # geometry from mip0 metadata
    W = H = 0
    tile = None
    zs = []
    for d in secs0:
        meta = d / "metadata.txt"
        z = z_of(d.name)
        zs.append(z)
        if meta.is_file():
            info = parse_feabas_metadata(meta)
            if info["tiles"]:
                W = max(W, int(max(t[3] for t in info["tiles"])))
                H = max(H, int(max(t[4] for t in info["tiles"])))
                tile = tile or info["tile_size"]
    if tile is None:
        sample = next(secs0[0].glob("*_tr*-tc*.*"))
        a = imread(sample); tile = (a.shape[0], a.shape[1])
    ts = int(tile[0])
    z0, z1 = min(zs), max(zs)
    n_ops = sum(len(list(d.glob("*_tr*-tc*.*"))) for m in mips for d in section_dirs(base, m))
    done = 0
    ext = None
    for m in mips:
        for d in section_dirs(base, m):
            z = z_of(d.name)
            for p in d.iterdir():
                mm = _TILE_RE.search(p.name)
                if not mm:
                    continue
                r, c = int(mm.group(1)), int(mm.group(2))
                ext = ext or mm.group(3).lower()
                dst = out / f"mip{m}" / f"slice{z:04d}" / f"{z:04d}_tr{r}-tc{c}.{ext}"
                link_or_copy(p, dst)
                done += 1
                if progress_cb and done % 50 == 0:
                    progress_cb(done, n_ops, f"mip{m} z{z}")
    rows = int(math.ceil(H / ts)); cols = int(math.ceil(W / ts))
    vsvi = {
        "Comment": name, "ServerType": "imagetiles",
        "SourceFileNameTemplate": f".\\mip0\\slice%04d\\%04d_tr%d-tc%d.{ext or 'png'}",
        "SourceParamSequence": "ssrc", "SourceMinS": z0, "SourceMaxS": z1,
        "SourceMinR": 1, "SourceMaxR": rows, "SourceMinC": 1, "SourceMaxC": cols,
        "MipMapFileNameTemplate": f".\\mip%d\\slice%04d\\%04d_tr%d-tc%d.{ext or 'png'}",
        "MipMapParamSequence": "mssrc", "SourceMinM": 1, "SourceMaxM": max(mips) if mips else 0,
        "SourceTileSizeX": ts, "SourceTileSizeY": ts, "SourceBytesPerPixel": fmt_bytes,
        "MissingImagePolicy": "nearest",
        "TargetDataSizeX": W, "TargetDataSizeY": H, "TargetDataSizeZ": z1 - z0 + 1,
        "OffsetX": 0, "OffsetY": 0, "OffsetZ": z0, "OffsetMip": 0,
        "TargetVoxelSizeXnm": voxel[0], "TargetVoxelSizeYnm": voxel[1], "TargetVoxelSizeZnm": voxel[2],
        "TargetLayerName": name,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.vsvi").write_text(json.dumps(vsvi, indent=4, sort_keys=True), encoding="utf-8")
    return {"vsvi": str(out / f"{name}.vsvi"), "size": [W, H, z1 - z0 + 1], "tiles": done}


def export_omezarr(base: Path, out: Path, name: str, voxel: tuple[float, float, float], mips: list[int],
                   chunk: int = 256, progress_cb=None) -> dict:
    """Write OME-Zarr 0.4 (zarr v2) from the PNG tile pyramid; dtype uint8/uint16 from the tiles."""
    secs0 = section_dirs(base, 0)
    if not secs0:
        raise RuntimeError(f"no rendered sections under {base / 'mip0'}")
    zs = sorted(z_of(d.name) for d in secs0)
    z0 = zs[0]; nz = zs[-1] - z0 + 1
    sample = imread(next(secs0[0].glob("*_tr*-tc*.*")))
    dtype = np.dtype(sample.dtype)
    zdtype = "|u1" if dtype == np.uint8 else "<u2"
    sizes = {}
    for m in mips:
        W = H = 0
        for d in section_dirs(base, m):
            meta = d / "metadata.txt"
            if meta.is_file():
                info = parse_feabas_metadata(meta)
                if info["tiles"]:
                    W = max(W, int(max(t[3] for t in info["tiles"]))); H = max(H, int(max(t[4] for t in info["tiles"])))
        if W == 0:
            W = int(math.ceil(sizes[0][0] / 2 ** m)); H = int(math.ceil(sizes[0][1] / 2 ** m))
        sizes[m] = (W, H)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".zgroup").write_text(json.dumps({"zarr_format": 2}, indent=2), encoding="utf-8")
    datasets = []
    for m in mips:
        datasets.append({"path": str(m), "coordinateTransformations": [{"type": "scale",
                         "scale": [1.0, voxel[2] / 1000.0, voxel[1] * 2 ** m / 1000.0, voxel[0] * 2 ** m / 1000.0]}]})
        W, H = sizes[m]
        arr = {"chunks": [1, 1, chunk, chunk], "compressor": None, "dimension_separator": "/", "dtype": zdtype,
               "fill_value": 0, "filters": None, "order": "C", "shape": [1, nz, H, W], "zarr_format": 2}
        (out / str(m)).mkdir(exist_ok=True)
        (out / str(m) / ".zarray").write_text(json.dumps(arr, indent=2), encoding="utf-8")
        (out / str(m) / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": ["c", "z", "y", "x"]}), encoding="utf-8")
    attrs = {"multiscales": [{"version": "0.4", "name": name,
                              "axes": [{"name": "c", "type": "channel"}, {"name": "z", "type": "space", "unit": "micrometer"},
                                       {"name": "y", "type": "space", "unit": "micrometer"}, {"name": "x", "type": "space", "unit": "micrometer"}],
                              "datasets": datasets}],
             "omero": {"id": 1, "name": name, "version": "0.4", "channels": [{"active": True, "coefficient": 1, "color": "FFFFFF", "family": "linear",
                       "inverted": False, "label": name, "window": {"min": 0, "max": int(np.iinfo(dtype).max), "start": 0, "end": int(np.iinfo(dtype).max)}}],
                       "rdefs": {"defaultZ": 0, "model": "greyscale"}}}
    (out / ".zattrs").write_text(json.dumps(attrs, indent=2), encoding="utf-8")
    total = sum(len(section_dirs(base, m)) for m in mips)
    done = 0
    for m in mips:
        W, H = sizes[m]
        for d in section_dirs(base, m):
            z = z_of(d.name) - z0
            meta = d / "metadata.txt"
            tiles = parse_feabas_metadata(meta)["tiles"] if meta.is_file() else []
            if not tiles:
                for p in d.iterdir():
                    mm = _TILE_RE.search(p.name)
                    if mm:
                        a = imread(p); th, tw = a.shape[:2]
                        r, c = int(mm.group(1)), int(mm.group(2))
                        tiles.append((p.name, (c - 1) * tw, (r - 1) * th, c * tw, r * th))
            # assemble by chunk rows to bound memory: process tile by tile, write chunk files
            for tname, x0, y0, x1, y1 in tiles:
                a = imread(d / tname)
                x0, y0 = int(x0), int(y0)
                cy0, cy1 = y0 // chunk, min(H - 1, y0 + a.shape[0] - 1) // chunk
                cx0, cx1 = x0 // chunk, min(W - 1, x0 + a.shape[1] - 1) // chunk
                for cy in range(cy0, cy1 + 1):
                    for cx in range(cx0, cx1 + 1):
                        cpath = out / str(m) / "0" / str(z) / str(cy) / str(cx)
                        if cpath.exists():
                            blk = np.fromfile(cpath, dtype=dtype).reshape(chunk, chunk)
                        else:
                            blk = np.zeros((chunk, chunk), dtype)
                        by0, bx0 = cy * chunk, cx * chunk
                        sy0, sx0 = max(by0, y0), max(bx0, x0)
                        sy1, sx1 = min(by0 + chunk, y0 + a.shape[0], H), min(bx0 + chunk, x0 + a.shape[1], W)
                        if sy1 <= sy0 or sx1 <= sx0:
                            continue
                        blk[sy0 - by0:sy1 - by0, sx0 - bx0:sx1 - bx0] = a[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                        cpath.parent.mkdir(parents=True, exist_ok=True)
                        blk.tofile(cpath)
            done += 1
            if progress_cb:
                progress_cb(done, total, f"mip{m} z{z + z0}")
    return {"zarr": str(out), "levels": mips, "shape": [nz, sizes[0][1], sizes[0][0]]}


def main() -> int:
    spec = load_spec("export")
    base = Path(spec["aligned_stack"])
    out = Path(spec["out_dir"])
    name = spec.get("name", "aligned")
    voxel = tuple(float(v) for v in spec.get("voxel_nm", (4, 4, 50)))
    mips = [int(m) for m in spec.get("mips", [])] or sorted(int(p.name[3:]) for p in base.glob("mip*") if p.name[3:].isdigit())
    what = spec.get("what", "vast")
    res = {}
    if what in ("vast", "both"):
        res["vast"] = export_vast(base, out / "vast", name, voxel, mips, progress_cb=progress)
        log(f"VAST export: {res['vast']}")
    if what in ("omezarr", "both"):
        res["omezarr"] = export_omezarr(base, out / f"{name}.ome.zarr", name, voxel, mips, int(spec.get("chunk", 256)), progress_cb=progress)
        log(f"OME-Zarr export: {res['omezarr']}")
    result(res)
    return 0


if __name__ == "__main__":
    run(main)
