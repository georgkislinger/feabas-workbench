"""
Image access for the viewers: thumbnails, FEABAS PNG-tile sections at any
mip level, precomputed (TensorStore) volumes, and overlay composition.

The rule for big data: never load more pixels than the view can show. Every
source exposes ``levels`` (mip -> size) and ``read(mip, x0, y0, w, h)``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# basic IO
# ----------------------------------------------------------------------

def imread(path: os.PathLike | str) -> np.ndarray:
    path = str(path)
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        img = tifffile.imread(path)
    else:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            from PIL import Image
            img = np.asarray(Image.open(path))
    if img.ndim == 3:
        img = img[..., 0]
    return img


def imwrite(path: os.PathLike | str, img: np.ndarray) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile
        tifffile.imwrite(path, img)
    else:
        import cv2
        cv2.imwrite(path, img)


def to_uint8(img: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    if img.dtype == np.uint8 and lo is None and hi is None:
        return img
    a = img.astype(np.float32)
    if lo is None or hi is None:
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return np.zeros(img.shape, np.uint8)
        lo_, hi_ = np.percentile(finite, (0.5, 99.5))
        lo = lo_ if lo is None else lo
        hi = hi_ if hi is None else hi
    if hi <= lo:
        hi = lo + 1
    a = (a - lo) / (hi - lo)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)


def read_downsampled(path: os.PathLike | str, max_px: int = 2048) -> tuple[np.ndarray, int]:
    """Read an image and shrink it so the long side is <= max_px. Returns (img, factor)."""
    img = imread(path)
    h, w = img.shape[:2]
    f = 1
    while max(h, w) / f > max_px:
        f *= 2
    return downsample(img, f), f


# ----------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------

@dataclass
class Level:
    mip: int
    width: int
    height: int


class SingleImageSource:
    """A plain image file (thumbnail, mask, preview) with virtual mips by resizing."""

    def __init__(self, path: Path, cache: bool = True):
        self.path = Path(path)
        self._img = imread(self.path)
        h, w = self._img.shape[:2]
        self.levels = [Level(0, w, h)]
        m, ww, hh = 0, w, h
        while max(ww, hh) > 256:
            m += 1
            ww, hh = max(1, ww // 2), max(1, hh // 2)
            self.levels.append(Level(m, ww, hh))
        self._cache: dict[int, np.ndarray] = {0: self._img}

    @property
    def name(self) -> str:
        return self.path.stem

    def full(self) -> np.ndarray:
        return self._img

    def level_image(self, mip: int) -> np.ndarray:
        mip = max(0, min(mip, len(self.levels) - 1))
        if mip not in self._cache:
            self._cache[mip] = downsample(self._img, 2 ** mip)
        return self._cache[mip]

    def read(self, mip: int, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        img = self.level_image(mip)
        H, W = img.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        return img[y0:min(H, y0 + h), x0:min(W, x0 + w)]


_TILE_RE = re.compile(r"_tr(\d+)-tc(\d+)\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)


def parse_feabas_metadata(meta_file: Path) -> dict:
    """
    FEABAS writes a metadata.txt next to rendered tiles:
        {ROOT_DIR}\t<dir>
        {TILE_SIZE}\t<h>\t<w>
        <filename>\t<xmin>\t<ymin>\t<xmax>\t<ymax>
    """
    tiles = []
    root = None
    tile_size = None
    for ln in meta_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.strip():
            continue
        parts = ln.rstrip("\n").split("\t")
        if parts[0] == "{ROOT_DIR}":
            root = parts[1] if len(parts) > 1 else None
        elif parts[0] == "{TILE_SIZE}":
            tile_size = (int(parts[1]), int(parts[2])) if len(parts) > 2 else (int(parts[1]), int(parts[1]))
        elif parts[0].startswith("{"):
            continue
        elif len(parts) >= 3:
            name = parts[0]
            xmin, ymin = float(parts[1]), float(parts[2])
            if len(parts) >= 5:
                xmax, ymax = float(parts[3]), float(parts[4])
            else:
                xmax = xmin + (tile_size[1] if tile_size else 0)
                ymax = ymin + (tile_size[0] if tile_size else 0)
            tiles.append((name, xmin, ymin, xmax, ymax))
    return {"root": root, "tile_size": tile_size, "tiles": tiles}


class TiledSectionSource:
    """
    A FEABAS-rendered section stored as tiles per mip:
        <base>/mip<N>/<section>/<section>_tr{r}-tc{c}.png  (+ metadata.txt)
    """

    def __init__(self, base: Path, section: str):
        self.base = Path(base)
        self.section = section
        self.levels: list[Level] = []
        self._meta: dict[int, dict] = {}
        for mipdir in sorted(self.base.glob("mip*"), key=lambda p: int(p.name[3:]) if p.name[3:].isdigit() else 99):
            if not mipdir.name[3:].isdigit():
                continue
            mip = int(mipdir.name[3:])
            secdir = mipdir / section
            meta = secdir / "metadata.txt"
            if not secdir.is_dir():
                continue
            if meta.is_file():
                info = parse_feabas_metadata(meta)
                if info["tiles"]:
                    w = int(max(t[3] for t in info["tiles"]))
                    h = int(max(t[4] for t in info["tiles"]))
                    self._meta[mip] = info
                    self.levels.append(Level(mip, w, h))
                    continue
            # no metadata: infer from tile names and one tile size
            tiles = [p for p in secdir.iterdir() if _TILE_RE.search(p.name)]
            if not tiles:
                continue
            sample = imread(tiles[0])
            th, tw = sample.shape[:2]
            rows = cols = 0
            entries = []
            for p in tiles:
                m = _TILE_RE.search(p.name)
                r, c = int(m.group(1)), int(m.group(2))
                rows, cols = max(rows, r), max(cols, c)
                entries.append((p.name, (c - 1) * tw, (r - 1) * th, c * tw, r * th))
            self._meta[mip] = {"root": str(secdir), "tile_size": (th, tw), "tiles": entries}
            self.levels.append(Level(mip, cols * tw, rows * th))
        self.levels.sort(key=lambda l: l.mip)

    @property
    def name(self) -> str:
        return self.section

    def has_level(self, mip: int) -> bool:
        return mip in self._meta

    def read(self, mip: int, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        if mip not in self._meta:
            # nearest available coarser level then resize
            avail = [l.mip for l in self.levels]
            if not avail:
                return np.zeros((h, w), np.uint8)
            src = min(avail, key=lambda m: (abs(m - mip), m))
            f = 2 ** (src - mip)
            if f >= 1:
                img = self.read(src, int(x0 / f), int(y0 / f), max(1, int(w / f)), max(1, int(h / f)))
                import cv2
                return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            img = self.read(src, int(x0 / f), int(y0 / f), int(w / f), int(h / f))
            return downsample(img, int(1 / f))
        info = self._meta[mip]
        secdir = self.base / f"mip{mip}" / self.section
        out = None
        x1, y1 = x0 + w, y0 + h
        for name, tx0, ty0, tx1, ty1 in info["tiles"]:
            if tx1 <= x0 or ty1 <= y0 or tx0 >= x1 or ty0 >= y1:
                continue
            p = secdir / name
            if not p.is_file():
                continue
            tile = imread(p)
            if out is None:
                out = np.zeros((h, w), dtype=tile.dtype)
            sx0, sy0 = max(x0, int(tx0)), max(y0, int(ty0))
            sx1, sy1 = min(x1, int(tx0) + tile.shape[1]), min(y1, int(ty0) + tile.shape[0])
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = tile[sy0 - int(ty0):sy1 - int(ty0), sx0 - int(tx0):sx1 - int(tx0)]
        if out is None:
            out = np.zeros((h, w), np.uint8)
        return out


class TensorStoreSource:
    """
    Neuroglancer precomputed volume rendered by FEABAS, one z at a time, using every
    scale (mip) the volume has so overviews never read full resolution.
    """

    def __init__(self, spec_or_dir, z: int = 0):
        try:
            import tensorstore as ts
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError("viewing precomputed volumes needs the 'tensorstore' package in the workbench "
                               "environment (pip install tensorstore)") from e
        import math
        self.z = z
        self._ts = ts
        if isinstance(spec_or_dir, (str, os.PathLike)):
            p = Path(spec_or_dir)
            base = {"driver": "neuroglancer_precomputed", "kvstore": {"driver": "file", "path": p.as_posix() + "/"}}
            scales = find_precomputed_scales(p)
        else:
            base = dict(spec_or_dir)
            scales = []
        self._base = base
        self._stores: dict[int, object] = {}
        self._scale_of_mip: dict[int, int] = {}
        if scales:
            r0 = float(scales[0]["resolution"][0])
            for i, sc in enumerate(scales):
                mip = int(round(math.log2(float(sc["resolution"][0]) / r0)))
                self._scale_of_mip.setdefault(mip, i)
        else:
            self._scale_of_mip[0] = 0
        st0 = self._store(0)
        shape = st0.shape   # (x, y, z, c)
        self.shape = tuple(int(s) for s in shape)
        self.n_z = self.shape[2] if len(self.shape) > 2 else 1
        self.levels = []
        for mip in sorted(self._scale_of_mip):
            f = 2 ** mip
            self.levels.append(Level(mip, max(1, self.shape[0] // f), max(1, self.shape[1] // f)))

    def _store(self, mip: int):
        idx = self._scale_of_mip[mip]
        if idx not in self._stores:
            spec = dict(self._base)
            if idx:
                spec["scale_index"] = idx
            self._stores[idx] = self._ts.open(spec, open=True, read=True).result()
        return self._stores[idx]

    @property
    def name(self) -> str:
        return f"z{self.z}"

    def read(self, mip: int, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        avail = [m for m in self._scale_of_mip if m <= mip] or [min(self._scale_of_mip)]
        src_mip = max(avail)
        st = self._store(src_mip)
        f_req = 2 ** mip
        f_src = 2 ** src_mip
        sx0, sy0 = x0 * f_req // f_src, y0 * f_req // f_src
        sw, sh = max(1, w * f_req // f_src), max(1, h * f_req // f_src)
        X1, Y1 = min(int(st.shape[0]), sx0 + sw), min(int(st.shape[1]), sy0 + sh)
        if X1 <= sx0 or Y1 <= sy0:
            return np.zeros((h, w), np.uint8)
        arr = st[sx0:X1, sy0:Y1, min(self.z, int(st.shape[2]) - 1)]
        if len(st.shape) > 3:
            arr = arr[..., 0]
        img = np.asarray(arr.read().result()).T
        f = f_req // f_src
        return downsample(img, f) if f > 1 else img


def find_precomputed_scales(vol_dir: Path) -> list[dict]:
    """Read the 'info' file of a precomputed volume: list of scales with resolution/size."""
    import json
    info = vol_dir / "info"
    if not info.is_file():
        return []
    try:
        d = json.loads(info.read_text(encoding="utf-8"))
        return d.get("scales", [])
    except ValueError:
        return []


# ----------------------------------------------------------------------
# overlays
# ----------------------------------------------------------------------

def compose_two_color(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray:
    """Section A in red, B in green (grey where they agree). Returns HxWx3 uint8."""
    shape = None
    for arr in (a, b):
        if arr is not None:
            shape = arr.shape[:2] if shape is None else (min(shape[0], arr.shape[0]), min(shape[1], arr.shape[1]))
    if shape is None:
        return np.zeros((1, 1, 3), np.uint8)
    h, w = shape
    rgb = np.zeros((h, w, 3), np.uint8)
    if a is not None:
        rgb[..., 0] = to_uint8(a[:h, :w])
    if b is not None:
        rgb[..., 1] = to_uint8(b[:h, :w])
    return rgb


def colorize_labels(mask: np.ndarray, palette: dict[int, tuple[int, int, int]] | None = None,
                    alpha: int = 110) -> np.ndarray:
    """Material mask -> RGBA overlay. Default: 255 (exclude) red, 50 (wrinkle) yellow, 100 soft cyan, 200 split magenta."""
    palette = palette or {255: (230, 60, 60), 50: (250, 220, 40), 100: (60, 200, 230), 200: (230, 80, 230)}
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), np.uint8)
    for label, col in palette.items():
        hit = mask == label
        if not hit.any():
            continue
        rgba[hit, 0], rgba[hit, 1], rgba[hit, 2] = col
        rgba[hit, 3] = alpha
    return rgba


def checkerboard(a: np.ndarray, b: np.ndarray, block: int = 64) -> np.ndarray:
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    yy, xx = np.mgrid[0:h, 0:w]
    sel = ((yy // block) + (xx // block)) % 2 == 0
    out = np.where(sel, to_uint8(a[:h, :w]), to_uint8(b[:h, :w]))
    return out
