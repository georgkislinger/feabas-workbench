"""
Material masks for FEABAS.

A material mask is a PNG in the thumbnail (or a higher mip) image space whose
grey value labels the material of each pixel:

    0    default tissue (meshed, matched, rendered)
    255  exclude: outside the section / holes (not meshed)
    50   wrinkle/fold: free to expand, resists compression (default table)
    100  soft, 200 split (see default_material_table.yaml)

FEABAS writes a default mask (whole imaged ROI = tissue) in
``thumbnail_align/material_masks``; the workbench replaces it with
tissue-vs-background (the imaged footprint, a uniform frame peeled from the
outside in, fixed margins, or a texture/intensity split) plus folds from a
threshold or the fold U-Net, and can write higher-resolution masks to
``align/material_masks``.

Also here: the structure-guided material variant (see ``structure_material``)
that makes FEABAS's fine matching concentrate on user-chosen structures.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

LABEL_DEFAULT = 0
LABEL_EXCLUDE = 255
LABEL_WRINKLE = 50
LABEL_SOFT = 100
LABEL_SPLIT = 200


TISSUE_METHODS = ("all", "border", "manual", "auto", "texture", "intensity")
BORDER_MODES = ("both", "black", "white", "auto")


@dataclass
class TissueParams:
    window: int = 31            # texture window (px) in the thumbnail
    nodata_value: int = 0       # padding value in rendered sections
    nodata_dilate: int = 8
    min_component_px: int = 2000     # tissue pieces smaller than this are dropped
    fill_holes_px: int = 5000        # enclosed background holes up to this size become tissue again
    erode: int = 3              # pull back from the section edge
    method: str = "all"         # all | border | manual | auto | texture | intensity
    #   all        FEABAS default: everything the tiles cover is tissue
    #   border     a uniform frame (black and/or white, or any flat grey) is peeled off from the outside in
    #   manual     fixed margins from the image edge
    #   auto/texture/intensity   split by local texture or grey level
    min_separability: float = 0.72   # auto: below this Otsu separability everything imaged counts as tissue (unimodal ~0.64)
    invert_intensity: bool = False
    threshold: float | None = None   # manual override (texture score or intensity)
    exclude_dark: bool = False  # drop black holes/tears; off by default - a fold is black but IS tissue
    dark_max: int = 0           # grey level up to which a pixel counts as 'no data'
    dark_min_px: int = 24       # ignore dark/bright specks smaller than this (thumbnail pixels)
    exclude_bright: bool = False     # the same for saturated white regions inside the section
    bright_min: int = 255            # grey level from which a pixel counts as 'white'
    # method 'border'
    border_mode: str = "both"        # both | black | white | auto   (auto: any locally flat grey value)
    border_black_max: int = 0        # black frame: grey <= this
    border_white_min: int = 250      # white frame: grey >= this
    border_tol: int = 4              # auto: a 3x3 neighbourhood whose grey range is <= this counts as flat
    border_min_width: int = 15       # streaks of the frame colour thinner than this are folds, not border
    # method 'manual': margins from the image edge, in thumbnail pixels
    crop_left: int = 0
    crop_top: int = 0
    crop_right: int = 0
    crop_bottom: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "TissueParams":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _cv2():
    import cv2
    return cv2


def local_std(img: np.ndarray, window: int) -> np.ndarray:
    cv2 = _cv2()
    f = img.astype(np.float32)
    k = (int(window) | 1, int(window) | 1)
    m = cv2.blur(f, k)
    m2 = cv2.blur(f * f, k)
    return np.sqrt(np.clip(m2 - m * m, 0, None))


def otsu(values: np.ndarray) -> float:
    return otsu_with_separability(values)[0]


def otsu_with_separability(values: np.ndarray) -> tuple[float, float]:
    """Otsu threshold and its separability (between-class variance / total variance, 0..1)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo, 0.0
    if v.size > 2_000_000:
        v = v[:: int(v.size // 2_000_000) + 1]
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    p = hist.astype(np.float64) / max(1, hist.sum())
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * centers)
    mt = m0[-1]
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0 / w0
        mu1 = (mt - m0) / w1
        sb = w0 * w1 * (mu0 - mu1) ** 2
    sb[~np.isfinite(sb)] = 0
    k = int(np.argmax(sb))
    total = float(np.sum(p * (centers - mt) ** 2))
    sep = float(sb[k] / total) if total > 0 else 0.0
    return float(centers[k]), sep


def nodata_mask(img: np.ndarray, value: int = 0, dilate: int = 8) -> np.ndarray:
    cv2 = _cv2()
    nd = (img == value).astype(np.uint8)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        nd = cv2.dilate(nd, k)
    return nd.astype(bool)


def padding_mask(img: np.ndarray, value: int = 0, min_frac: float = 0.0005) -> np.ndarray:
    """
    The no-data padding of a rendered section: connected regions of *value* that touch
    the image border (or are large), not the scattered dark pixels inside the tissue.
    """
    cv2 = _cv2()
    zero = (img == value).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(zero, connectivity=4)
    if n <= 1:
        return np.zeros(img.shape[:2], bool)
    h, w = zero.shape
    border = np.zeros(n, bool)
    for arr in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        border[np.unique(arr)] = True
    big = stats[:, cv2.CC_STAT_AREA] >= max(64, int(min_frac * h * w))
    keep = (border | big)
    keep[0] = False
    return keep[lab]


def tile_boxes_from_tform(tform_h5: Path) -> np.ndarray | None:
    """(N, 4) tile bounding boxes in mip0 pixels, read from a FEABAS stitch tform file."""
    import h5py
    path = Path(tform_h5)
    if not path.is_file():
        return None
    try:
        with h5py.File(path, "r") as f:
            if "moving_offsets" not in f or "moving_vertices" not in f:
                return None
            offsets = f["moving_offsets"][()]
            share = f["mesh_sharing_indx"][()] if "mesh_sharing_indx" in f else None
            boxes = []
            for k in range(offsets.shape[0]):
                name = str(k)
                if name not in f["moving_vertices"]:
                    if share is None:
                        continue
                    name = str(int(share[k]))
                    if name not in f["moving_vertices"]:
                        continue
                v = f["moving_vertices"][name][()]
                o = np.asarray(offsets[k]).ravel()[:2]
                boxes.append([*(v.min(axis=0) + o), *(v.max(axis=0) + o)])
    except (OSError, KeyError, ValueError):
        return None
    if not boxes:
        return None
    return np.asarray(boxes, dtype=np.float64).clip(0, None)


def montage_extent(tform_h5: Path) -> tuple[float, float] | None:
    """Width and height of the stitched montage in mip0 pixels."""
    b = tile_boxes_from_tform(tform_h5)
    if b is None:
        return None
    ext = b[:, 2:].max(axis=0) + 2
    return float(ext[0]), float(ext[1])


def border_band(tissue: np.ndarray, margin: int) -> np.ndarray:
    """
    The *margin*-wide ring just inside the OUTER outline of the tissue (empty for margin <= 0).
    Holes inside the tissue (excluded nuclei, tears) get no ring: they are filled before eroding.
    """
    if margin <= 0:
        return np.zeros_like(np.asarray(tissue), bool)
    cv2 = _cv2()
    t = (np.asarray(tissue) > 0).astype(np.uint8)
    # fill enclosed holes: background components that do not touch the image border
    inv = (1 - t).astype(np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    filled = t.copy()
    for i in range(1, n):
        if i not in border:
            filled[lab == i] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(margin) + 1, 2 * int(margin) + 1))
    return (t > 0) & ~(cv2.erode(filled, k, borderValue=0) > 0)


def roi_from_tform(tform_h5: Path, shape: tuple[int, int], erode: int = 0) -> np.ndarray | None:
    """
    The imaged area of a section, rasterised from the stitched tile bounding boxes.

    This is what FEABAS's ``stitcher.generate_roi_mask`` does, read straight from
    ``stitch/tform/<sec>.h5`` with h5py so it needs neither the FEABAS environment nor
    the thumbnail step: the moving mesh vertices plus the per-tile offset give each
    tile's box, and their union is the area that carries image data. Purely geometric,
    so a fold that is black from one section edge to the other cannot leak into it.

    Returns a bool mask of *shape* (the thumbnail's), or None if the file is unusable.
    """
    boxes = tile_boxes_from_tform(tform_h5)
    if boxes is None:
        return None
    b = boxes
    extent = b[:, 2:].max(axis=0) + 2                      # as FEABAS sizes its own ROI image
    h, w = int(shape[0]), int(shape[1])
    sx, sy = w / max(extent[0], 1.0), h / max(extent[1], 1.0)
    if abs(sx - sy) > 0.02 * max(sx, sy):
        # the target image is not the montage bounding box (padded or cropped elsewhere):
        # scaling it anyway would put the rim in the wrong place
        return None
    out = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in b:
        out[int(round(y0 * sy)):int(round(y1 * sy)), int(round(x0 * sx)):int(round(x1 * sx))] = 1
    if erode > 0:
        cv2 = _cv2()
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(erode) + 1, 2 * int(erode) + 1))
        out = cv2.erode(out, k, borderValue=0)
    return out.astype(bool)


def footprint_mask(img: np.ndarray, value: int = 0, close: int = 9, min_px: int = 500) -> np.ndarray:
    """
    The area the microscope actually imaged: the outline of the stitched tiles.

    FEABAS derives this from the tile bounding boxes (``stitcher.generate_roi_mask``) and
    the workbench uses that mask when it is available. This is the fallback when it is not:
    the convex hull of each large blob of image data, unioned. Hulls, not flood fill, because
    a fold can be black from one edge of the section to the other - it is connected to the
    outside, but it is still inside the imaged area and must stay tissue.
    """
    cv2 = _cv2()
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    data = (img > int(value)).astype(np.uint8)
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(close) + 1, 2 * int(close) + 1))
        data = cv2.morphologyEx(data, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(data, connectivity=8)
    out = np.zeros(img.shape[:2], np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < int(min_px):
            continue
        pts = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not pts:
            continue
        hull = cv2.convexHull(np.vstack(pts))
        cv2.fillConvexPoly(out, hull, 1)
    if not out.any():                      # nothing convincing: keep everything with data
        return data.astype(bool)
    return out.astype(bool)


def value_regions(img: np.ndarray, lo: int, hi: int, min_px: int = 24, dilate: int = 0) -> np.ndarray:
    """
    Connected regions whose grey value lies in [lo, hi] and that are at least *min_px* large.

    ``min_px`` keeps single pixels of real tissue that happen to be saturated out of the result;
    ``dilate`` grows what is left so the blurred rim of a hole is dropped too.
    """
    cv2 = _cv2()
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    sel = ((img >= int(lo)) & (img <= int(hi))).astype(np.uint8)
    if min_px > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(sel, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= int(min_px)
        keep[0] = False
        sel = keep[lab].astype(np.uint8)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dilate) + 1, 2 * int(dilate) + 1))
        sel = cv2.dilate(sel, k)
    return sel.astype(bool)


def dark_regions(img: np.ndarray, max_value: int = 0, min_px: int = 24, dilate: int = 0) -> np.ndarray:
    """
    Connected regions at or below *max_value* that are at least *min_px* large.

    Rendered sections are black where nothing was imaged: the padding outside the
    section, but also holes, tears and folds inside it. Thresholding finds them all
    without a model; ``min_px`` keeps single dark pixels of real tissue out.
    """
    return value_regions(img, 0, int(max_value), min_px, dilate)


def bright_regions(img: np.ndarray, min_value: int = 255, min_px: int = 24, dilate: int = 0) -> np.ndarray:
    """Connected regions at or above *min_value*: saturated white frames, empty resin, burnt areas."""
    return value_regions(img, int(min_value), 255, min_px, dilate)


def local_range(img: np.ndarray, size: int = 3) -> np.ndarray:
    """max - min over a size x size neighbourhood: ~0 where the image is locally flat (a uniform frame)."""
    cv2 = _cv2()
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(size), int(size)))
    a = np.asarray(img)
    return cv2.dilate(a, k).astype(np.int16) - cv2.erode(a, k).astype(np.int16)


def uniform_border(img: np.ndarray, roi: np.ndarray | None = None, mode: str = "both",
                   black_max: int = 0, white_min: int = 250, tol: int = 4, min_width: int = 15,
                   max_layers: int = 6, nodata_value: int | None = 0) -> np.ndarray:
    """
    The border region of a section, found from the outside in: everything that is NOT tissue
    because it is a uniform frame reachable from the edge of the imaged area.

    Sections are often surrounded by more than one frame - black padding where nothing was
    imaged, then a white (saturated) rim of empty resin or support film, sometimes a thin grey
    line from the detector - and the inner frames do not touch the image edge themselves. So
    the frames are peeled layer by layer: starting from the area outside the imaged footprint
    (*roi*, or the image edge when there is none), connected regions of the frame colour that
    touch what is already outside are added, and that is repeated until nothing new is reached.

    *mode* says what counts as frame colour: ``black`` (grey <= black_max), ``white``
    (grey >= white_min), ``both`` (either), or ``auto`` - any locally flat area (grey range over
    a 3x3 neighbourhood <= tol), which finds a frame of any grey value. EM tissue is never flat
    over a large area, so only saturated blobs inside the tissue qualify, and those are not
    reached from the outside. Pixels of the no-data value (*nodata_value*, the padding of a
    rendered section) count as frame in every mode, so a white rim behind black padding is
    reached with ``white`` too.

    A fold that runs from the section edge inwards has the frame colour and touches the frame,
    so it would be peeled too; it is thin, though, and the frame is not. Peeled streaks
    narrower than *min_width* pixels are given back (morphological opening), so they stay
    tissue and can be labelled as folds instead.

    Returns a bool mask that is True on the border (outside the tissue).
    """
    cv2 = _cv2()
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    h, w = img.shape[:2]
    outside0 = ~np.asarray(roi).astype(bool) if roi is not None else np.zeros((h, w), bool)
    if mode == "auto":
        cand = local_range(img) <= int(tol)
    elif mode == "black":
        cand = img <= int(black_max)
    elif mode == "white":
        cand = img >= int(white_min)
    else:
        cand = (img <= int(black_max)) | (img >= int(white_min))
    if nodata_value is not None:
        cand = cand | (img == int(nodata_value))
    cand &= ~outside0
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
    outside = outside0.copy()
    edge = np.zeros((h, w), bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))   # anti-aliased 1-2 px seams between frames
    for _ in range(int(max_layers)):
        seed = cv2.dilate(outside.astype(np.uint8), bridge).astype(bool) | edge
        ids = np.unique(lab[seed & cand])
        ids = ids[ids > 0]
        if ids.size == 0:
            break
        new = np.isin(lab, ids) & ~outside
        if not new.any():
            break
        outside |= new
    peeled = outside & ~outside0
    if min_width > 0 and peeled.any():
        d = 2 * (int(min_width) // 2) + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        peeled = cv2.morphologyEx(peeled.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    return outside0 | peeled


def manual_bounds(shape: tuple[int, int], left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> np.ndarray:
    """Everything inside fixed margins from the image edge (True = tissue)."""
    h, w = int(shape[0]), int(shape[1])
    m = np.zeros((h, w), bool)
    y0, y1 = min(max(0, int(top)), h), max(0, h - max(0, int(bottom)))
    x0, x1 = min(max(0, int(left)), w), max(0, w - max(0, int(right)))
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = True
    return m


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    cv2 = _cv2()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def remove_thin(mask: np.ndarray, min_width: int) -> np.ndarray:
    """
    Drop connected pieces of *mask* that are nowhere at least *min_width* pixels thick.

    The seams between two frames (anti-aliased pixels that are neither black nor white) and
    a detector line along the image edge survive a frame detection as slivers one or two
    pixels wide; real tissue is thick somewhere.
    """
    if min_width <= 1:
        return mask
    cv2 = _cv2()
    m = mask.astype(np.uint8)
    r = max(1, int(min_width) // 2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(m, k, borderValue=1) > 0     # the image edge does not count as an edge of the piece
    n, lab = cv2.connectedComponents(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[core])] = True
    keep[0] = False
    return keep[lab]


def fill_holes(mask: np.ndarray, max_px: int) -> np.ndarray:
    inv = ~mask
    small_holes = inv & ~remove_small(inv, max_px)
    # holes touching the border are background, not holes
    cv2 = _cv2()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=4)
    h, w = mask.shape
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    out = mask.copy()
    for i in range(1, n):
        if i in border:
            continue
        if stats[i, cv2.CC_STAT_AREA] <= max_px:
            out[lab == i] = True
    return out


def tissue_score(img: np.ndarray, params: TissueParams) -> np.ndarray:
    """Higher = more tissue-like. Texture (local std) is polarity independent."""
    if params.method == "intensity":
        s = img.astype(np.float32)
        if params.invert_intensity:
            s = -s
        return s
    return local_std(img, params.window)


def detect_tissue(img: np.ndarray, params: TissueParams | None = None,
                  roi: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """
    Return (tissue mask bool, threshold used).

    *roi* is the imaged area (FEABAS's own mask of the stitched tile footprint). When it is
    given, method 'all' uses it as it is: everything the microscope imaged counts as tissue,
    including black folds that reach the section edge.
    """
    params = params or TissueParams()
    cv2 = _cv2()
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]

    def finish(m: np.ndarray) -> np.ndarray:
        """Black/white holes inside the section are not tissue either; then pull the edge back."""
        if params.exclude_dark:
            m = m & ~dark_regions(img, params.dark_max, params.dark_min_px, params.nodata_dilate)
        if params.exclude_bright:
            m = m & ~bright_regions(img, params.bright_min, params.dark_min_px, params.nodata_dilate)
        if params.erode > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * params.erode + 1, 2 * params.erode + 1))
            m = cv2.erode(m.astype(np.uint8), k, borderValue=0).astype(bool)
        return m

    def cleanup(m: np.ndarray) -> np.ndarray:
        if params.min_component_px > 0:
            m = remove_small(m, params.min_component_px)
        if params.fill_holes_px > 0:
            m = fill_holes(m, params.fill_holes_px)
        return m

    footprint = np.asarray(roi).astype(bool) if roi is not None else footprint_mask(img, params.nodata_value)
    if params.method == "all":
        return finish(footprint), float("nan")
    if params.method == "manual":
        # fixed margins from the image edge, never beyond what was imaged
        m = manual_bounds(img.shape[:2], params.crop_left, params.crop_top, params.crop_right, params.crop_bottom)
        return finish(m & footprint), float("nan")
    if params.method == "border":
        border = uniform_border(img, roi=footprint, mode=params.border_mode, black_max=params.border_black_max,
                                white_min=params.border_white_min, tol=params.border_tol,
                                min_width=params.border_min_width, nodata_value=params.nodata_value)
        return finish(cleanup(remove_thin(~border, params.border_min_width))), float("nan")
    pad = padding_mask(img, params.nodata_value)
    nd = pad.copy()
    if params.nodata_dilate or params.window:
        r = max(params.nodata_dilate, params.window // 2)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        nd = cv2.dilate(pad.astype(np.uint8), k).astype(bool)
    score = tissue_score(img, params)
    valid = ~nd
    if params.threshold is not None:
        thr = float(params.threshold)
    else:
        thr, sep = otsu_with_separability(score[valid]) if valid.any() else (0.0, 0.0)
        if params.method == "auto" and sep < params.min_separability:
            # no convincing tissue/background split: the whole imaged area is tissue
            return finish(~pad), float("nan")
    m = (score > thr) & valid
    if params.method in ("texture", "auto"):
        # smooth the decision at the window scale
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.window | 1, params.window | 1))).astype(bool)
    return finish(cleanup(m)), thr


def compose_material_mask(tissue: np.ndarray, folds: np.ndarray | None = None,
                          fold_label: int = LABEL_WRINKLE, extra: dict[int, np.ndarray] | None = None,
                          border: int = 1, clip_folds: bool = True) -> np.ndarray:
    """
    Tissue -> 0, non-tissue -> 255, folds -> fold_label, plus extra {label: mask}.

    With *clip_folds* (the default) a fold only gets its label where the tissue mask agrees:
    the area outside the imaged region stays excluded even when the fold detector marked it,
    which it usually does because that area is black too. Set it to False to let the fold
    label win everywhere it was detected.

    A *border* ring of excluded pixels is always kept: FEABAS's thumbnail matcher fails on a mask
    without any excluded region (empty tolerance list in its geometry simplification).
    """
    out = np.full(tissue.shape, LABEL_EXCLUDE, np.uint8)
    out[tissue] = LABEL_DEFAULT
    if border > 0:
        out[:border, :] = LABEL_EXCLUDE; out[-border:, :] = LABEL_EXCLUDE
        out[:, :border] = LABEL_EXCLUDE; out[:, -border:] = LABEL_EXCLUDE
        tissue = tissue.copy()
        tissue[:border, :] = False; tissue[-border:, :] = False; tissue[:, :border] = False; tissue[:, -border:] = False
    if folds is not None:
        f = folds.astype(bool)
        if clip_folds:
            f = f & tissue
        elif border > 0:                      # never paint over the excluded rim FEABAS needs
            f = f.copy()
            f[:border, :] = False; f[-border:, :] = False; f[:, :border] = False; f[:, -border:] = False
        out[f] = fold_label
    for label, m in (extra or {}).items():
        out[m.astype(bool) & tissue] = label
    return out


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    cv2 = _cv2()
    if mask.shape[:2] == tuple(shape):
        return mask
    return cv2.resize(mask, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)


def mask_stats(mask: np.ndarray) -> dict[str, float]:
    tot = float(mask.size) or 1.0
    return {
        "tissue_pct": 100.0 * float((mask == LABEL_DEFAULT).sum()) / tot,
        "exclude_pct": 100.0 * float((mask == LABEL_EXCLUDE).sum()) / tot,
        "wrinkle_pct": 100.0 * float((mask == LABEL_WRINKLE).sum()) / tot,
        "other_pct": 100.0 * float((~np.isin(mask, [LABEL_DEFAULT, LABEL_EXCLUDE, LABEL_WRINKLE])).sum()) / tot,
    }


def read_mask(path: Path) -> np.ndarray:
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise IOError(f"cannot read {path}")
    return m


def write_mask(path: Path, mask: np.ndarray) -> None:
    """Atomic PNG write (FEABAS workers may be reading the previous file)."""
    import cv2
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(mask.astype(np.uint8)))
    if not ok:
        raise IOError(f"could not encode {path}")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def paint_split_line(mask: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], width: int = 3,
                     label: int = LABEL_EXCLUDE) -> np.ndarray:
    """Draw a line of *label* through the mask: FEABAS's documented way to split a broken section."""
    import cv2
    out = mask.copy()
    cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), int(label), int(width))
    return out


# ----------------------------------------------------------------------
# structure-guided material
# ----------------------------------------------------------------------

STRUCTURE_BG_LABEL = 150   # 'background_lowweight' material added by the workbench

STRUCTURE_MATERIAL_YAML = """# Added by FEABAS Workbench for structure-guided fine alignment.
# Pixels of this label are meshed and rendered like tissue, but their stiffness
# multiplier is below matching.matcher_config.stiffness_multiplier_threshold,
# so FEABAS places NO fine-matching points there. Alignment is driven by the
# selected structures (label 0) and the rest of the tissue follows elastically.
background_lowweight:
    enable_mesh: true
    area_constraint: 1
    render: true
    render_weight: 1.0
    stiffness_multiplier: 0.09
    poisson_ratio: 0.0
    mask_label: 150
    stiffness_func_factory: null
    stiffness_func_params: {}
"""


def structure_material_mask(material: np.ndarray, structures: np.ndarray, dilate_px: int = 0) -> np.ndarray:
    """
    Convert a material mask so that tissue *outside* the structures becomes the
    low-weight material (150); structures keep label 0. Folds/exclude untouched.
    """
    import cv2
    s = structures.astype(bool)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        s = cv2.dilate(s.astype(np.uint8), k).astype(bool)
    out = material.copy()
    tissue = material == LABEL_DEFAULT
    out[tissue & ~s] = STRUCTURE_BG_LABEL
    return out
