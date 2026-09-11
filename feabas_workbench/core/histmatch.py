"""
Batch histogram matching to a template image.

Port of the Fiji/Groovy "Histogram_matcher_v02" script:
* 8-bit and 16-bit grey images
* optional exclusion of pure black (0) and/or pure white (max) pixels from the
  histograms, and those pixels are kept unchanged in the output
* monotonic CDF matching via a lookup table
* parallel over files, original bit depth and format preserved

This module is used both in-process (previews) and by the histmatch worker.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


def read_gray(path: os.PathLike | str) -> np.ndarray:
    path = str(path)
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        img = tifffile.imread(path)
    else:
        from PIL import Image
        img = np.asarray(Image.open(path))
    if img.ndim == 3:
        img = img[..., 0]
    if img.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"{Path(path).name}: not 8/16-bit grey (dtype {img.dtype})")
    return img


def write_gray(path: os.PathLike | str, img: np.ndarray) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        tifffile.imwrite(path, img, photometric="minisblack")
    else:
        from PIL import Image
        Image.fromarray(img).save(path)


def histogram(img: np.ndarray) -> np.ndarray:
    bins = 256 if img.dtype == np.uint8 else 65536
    return np.bincount(img.ravel(), minlength=bins).astype(np.int64)


def template_cdf(tmpl: np.ndarray, ignore_black: bool, ignore_white: bool) -> np.ndarray:
    h = histogram(tmpl)
    if ignore_black:
        h[0] = 0
    if ignore_white:
        h[-1] = 0
    s = h.sum()
    if s == 0:
        raise ValueError("Template histogram is empty after ignoring backgrounds.")
    return np.cumsum(h) / float(s)


def build_lut(src: np.ndarray, cT: np.ndarray, ignore_black: bool, ignore_white: bool) -> np.ndarray:
    hS = histogram(src)
    if ignore_black:
        hS[0] = 0
    if ignore_white:
        hS[-1] = 0
    s = hS.sum()
    if s == 0:
        raise ValueError("Source histogram is empty after ignoring backgrounds.")
    cS = np.cumsum(hS) / float(s)
    # for each source level g find smallest t with cT[t] >= cS[g]  (monotonic)
    lut = np.searchsorted(cT, cS, side="left")
    maxv = len(cT) - 1
    lut = np.clip(lut, 0, maxv)
    if ignore_black:
        lut[0] = 0
    if ignore_white:
        lut[maxv] = maxv
    return lut.astype(src.dtype)


def apply_lut(src: np.ndarray, lut: np.ndarray, ignore_black: bool, ignore_white: bool) -> np.ndarray:
    out = lut[src]
    if ignore_black or ignore_white:
        maxv = np.iinfo(src.dtype).max
        keep = np.zeros(src.shape, dtype=bool)
        if ignore_black:
            keep |= src == 0
        if ignore_white:
            keep |= src == maxv
        out = np.where(keep, src, out)
    return out.astype(src.dtype)


def match_image(src: np.ndarray, cT: np.ndarray, ignore_black: bool = True, ignore_white: bool = False) -> np.ndarray:
    lut = build_lut(src, cT, ignore_black, ignore_white)
    return apply_lut(src, lut, ignore_black, ignore_white)


def _process_one(args) -> tuple[str, str | None]:
    src_path, dst_path, cT, ignore_black, ignore_white = args
    try:
        img = read_gray(src_path)
        if len(cT) != (256 if img.dtype == np.uint8 else 65536):
            return src_path, "bit depth differs from template"
        out = match_image(img, cT, ignore_black, ignore_white)
        write_gray(dst_path, out)
        return src_path, None
    except Exception as e:  # noqa: BLE001
        return src_path, str(e)


def match_folder(files: Iterable[Path], in_root: Path, out_root: Path, template: Path,
                 ignore_black: bool = True, ignore_white: bool = False, workers: int = 8,
                 progress: Callable[[int, int, str], None] | None = None,
                 skip_existing: bool = True) -> list[tuple[str, str]]:
    """Match every file; output mirrors the folder structure under out_root. Returns failures."""
    files = list(files)
    in_root, out_root = Path(in_root), Path(out_root)
    tmpl = read_gray(template)
    cT = template_cdf(tmpl, ignore_black, ignore_white)
    jobs = []
    for f in files:
        rel = Path(f).relative_to(in_root) if in_root in Path(f).parents or Path(f).parent == in_root else Path(Path(f).name)
        dst = out_root / rel
        if skip_existing and dst.is_file():
            continue
        jobs.append((str(f), str(dst), cT, ignore_black, ignore_white))
    failures: list[tuple[str, str]] = []
    total = len(jobs)
    done = 0
    if progress:
        progress(0, total, "starting")
    if workers <= 1 or total <= 1:
        for j in jobs:
            p, err = _process_one(j)
            done += 1
            if err:
                failures.append((p, err))
            if progress:
                progress(done, total, Path(p).name)
        return failures
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_one, j) for j in jobs]
        for fut in as_completed(futs):
            p, err = fut.result()
            done += 1
            if err:
                failures.append((p, err))
            if progress:
                progress(done, total, Path(p).name)
    return failures
