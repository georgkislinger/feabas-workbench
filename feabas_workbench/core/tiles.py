"""
Tile discovery, filename parsing and stitch-coordinate generation.

This is the Python port of the PowerShell "TrakEM2 Helper" naming/coordinate
getter, extended with:

* FEI/Thermo metadata reading (pixel size, stage position, scan rotation), so
  voxel size and tile placement can be auto-filled from the images themselves;
* a stage-position placement mode next to the classic grid mode
  (x_index * (tile_w - overlap));
* output in FEABAS's own stitch_coord format (one TSV per section).

Nothing here imports Qt.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

IMAGE_EXTS = ("tif", "tiff", "png", "jpg", "jpeg", "bmp")

# --------------------------------------------------------------------------
# Naming rules
# --------------------------------------------------------------------------

PRESET_REGEX = {
    # Thermo Fisher Maps / Apreo:  Tile_001-002-000034_0-000.s0035_e00.tif
    #   first number = row (y), second = column (x), third = running tile id,
    #   sXXXX = section
    "thermo": (
        r"^Tile_(?P<y>\d+)-(?P<x>\d+)-(?P<r>\d+)_\d+-\d+\.s(?P<z>\d+)_e\d+$",
        "Tile_#y#-#x#-#r#_0-000.s#z#_e00",
    ),
    # Zeiss Atlas:  Tile_r1-c2_S_0035_12.tif
    "zeiss": (
        r"^Tile_r(?P<y>\d+)-c(?P<x>\d+)_S_(?P<z>\d+)(?:_(?P<r>\d+))?$",
        "Tile_r#y#-c#x#_S_#z#_#r#",
    ),
}

PRESET_LABELS = {
    "thermo": "Thermo (Maps): Tile_row-col-id_0-000.sZZZZ_e00",
    "zeiss": "Zeiss (Atlas): Tile_rY-cX_S_ZZZZ",
    "sequential": "Sequential numbers (give the order, e.g. 'yxz')",
    "custom": "Custom pattern with #x# #y# #z# #r#",
}


@dataclass
class Coord:
    x: int = 0
    y: int = 0
    z: int = 0


@dataclass
class NamingRule:
    """How to get (x_index, y_index, section) out of a filename stem."""

    preset: str = "thermo"          # thermo | zeiss | sequential | custom
    order: str = "z"                # for 'sequential': letters of x y z r (r = skip)
    template: str = ""              # for 'custom': e.g. 'Tile_#y#-#x#-#r#_0-000.s#z#_e00'
    ext: str = "tif"
    recursive: bool = True

    # -- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "NamingRule":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # -- regex ---------------------------------------------------------
    def regex(self) -> re.Pattern | None:
        if self.preset in PRESET_REGEX:
            return re.compile(PRESET_REGEX[self.preset][0], re.IGNORECASE)
        if self.preset == "custom":
            return re.compile(template_to_regex(self.template), re.IGNORECASE)
        return None

    def describe(self) -> str:
        if self.preset in PRESET_REGEX:
            return PRESET_REGEX[self.preset][1]
        if self.preset == "sequential":
            return f"sequential numbers, order '{self.order}'"
        return self.template

    # -- parsing -------------------------------------------------------
    def parse(self, stem: str) -> Coord | None:
        if self.preset == "sequential":
            return parse_sequential(stem, self.order)
        reg = self.regex()
        if reg is None:
            return None
        m = reg.match(stem)
        if not m:
            return None
        gd = m.groupdict()
        found = any(gd.get(k) is not None for k in ("x", "y", "z"))
        if not found:
            return None
        return Coord(
            x=int(gd["x"]) if gd.get("x") else 0,
            y=int(gd["y"]) if gd.get("y") else 0,
            z=int(gd["z"]) if gd.get("z") else 0,
        )


def template_to_regex(template: str) -> str:
    """'Tile_#y#-#x#-#r#_0-000.s#z#_e00' -> anchored regex with named groups."""
    tmp = template
    markers = {"#x#": "<!X!>", "#y#": "<!Y!>", "#z#": "<!Z!>", "#r#": "<!R!>"}
    for k, v in markers.items():
        tmp = tmp.replace(k, v)
    esc = re.escape(tmp)
    esc = esc.replace(re.escape("<!X!>"), r"(?P<x>\d+)")
    esc = esc.replace(re.escape("<!Y!>"), r"(?P<y>\d+)")
    esc = esc.replace(re.escape("<!Z!>"), r"(?P<z>\d+)")
    esc = esc.replace(re.escape("<!R!>"), r"\d+")
    return "^" + esc + "$"


def parse_sequential(stem: str, order: str) -> Coord | None:
    numbers = [int(m) for m in re.findall(r"\d+", stem)]
    if not numbers:
        return None
    c = Coord()
    found = False
    for i, ch in enumerate(order.lower()):
        if i >= len(numbers):
            break
        if ch == "x":
            c.x = numbers[i]; found = True
        elif ch == "y":
            c.y = numbers[i]; found = True
        elif ch == "z":
            c.z = numbers[i]; found = True
        # 'r' or anything else: skip that number
    return c if found else None


# --------------------------------------------------------------------------
# File listing
# --------------------------------------------------------------------------

def list_image_files(root: os.PathLike | str, ext: str = "tif", recursive: bool = True,
                     limit: int | None = None) -> list[Path]:
    root = Path(root)
    ext = ext.strip().lstrip(".").lower()
    if not root.is_dir():
        return []
    out: list[Path] = []
    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if not p.is_file():
            continue
        suf = p.suffix.lower().lstrip(".")
        if ext and suf != ext:
            continue
        if not ext and suf not in IMAGE_EXTS:
            continue
        out.append(p)
        if limit and len(out) >= limit:
            break
    out.sort()
    return out


# --------------------------------------------------------------------------
# Guessing (port of Guess-NumberOrder and Guess-CustomPatterns)
# --------------------------------------------------------------------------

@dataclass
class RuleGuess:
    rule: NamingRule
    score: float
    matches: int
    n_x: int
    n_y: int
    n_z: int

    def label(self) -> str:
        return (f"{self.rule.describe()}   [{self.matches} files, "
                f"x:{self.n_x} y:{self.n_y} z:{self.n_z} unique]")


def _score_rule(rule: NamingRule, stems: Sequence[str]) -> RuleGuess | None:
    """
    Score a rule on sample filenames. A good rule (1) matches most files,
    (2) never maps two files onto the same (x, y, z), and (3) yields sections
    that are real grids (many tiles per section). Presets win ties.
    """
    xs, ys, zs = set(), set(), set()
    triples: Counter = Counter()
    n = 0
    for s in stems:
        c = rule.parse(s)
        if c is None:
            continue
        n += 1
        xs.add(c.x); ys.add(c.y); zs.add(c.z)
        triples[(c.x, c.y, c.z)] += 1
    if n == 0:
        return None
    coverage = 100.0 * n / max(1, len(stems))
    dup_frac = sum(v - 1 for v in triples.values()) / n
    tiles_per_section = n / max(1, len(zs))
    score = coverage + 10.0 * math.log(max(1.0, tiles_per_section)) - 60.0 * dup_frac
    if len(zs) == 1 and n > 1:
        score -= 5.0          # a single section is less likely than a stack
    if rule.preset in PRESET_REGEX:
        score += 15.0         # a vendor pattern that matches everything is almost certainly right
    elif rule.preset == "custom":
        score += 4.0
    return RuleGuess(rule, score, n, len(xs), len(ys), len(zs))


def guess_rules(files: Sequence[Path], ext: str = "tif") -> list[RuleGuess]:
    """Try presets, sequential orders and derived custom templates; best first."""
    stems = [p.stem for p in files[:200]]
    if not stems:
        return []
    cands: list[NamingRule] = []
    for preset in PRESET_REGEX:
        cands.append(NamingRule(preset=preset, ext=ext))

    # sequential orders based on how many numbers most filenames have
    counts = Counter(len(re.findall(r"\d+", s)) for s in stems[:20])
    ncommon = counts.most_common(1)[0][0] if counts else 1
    if ncommon == 1:
        orders = ["z", "x", "y"]
    elif ncommon == 2:
        orders = ["xz", "yz", "xy", "zx", "zy", "yx"]
    elif ncommon == 3:
        orders = ["yxz", "xyz", "xzy", "yzx", "zxy", "zyx"]
    else:
        orders = ["yxrz", "xyrz", "xyz", "xyzr", "rxyz", "xzr", "yzr", "yxz"]
    for o in orders:
        cands.append(NamingRule(preset="sequential", order=o, ext=ext))

    # derived custom templates
    for tpl in _derive_templates(stems):
        cands.append(NamingRule(preset="custom", template=tpl, ext=ext))

    guesses = []
    seen = set()
    for r in cands:
        key = (r.preset, r.order, r.template)
        if key in seen:
            continue
        seen.add(key)
        g = _score_rule(r, stems)
        if g is not None:
            guesses.append(g)
    guesses.sort(key=lambda g: g.score, reverse=True)
    return guesses


def _derive_templates(stems: Sequence[str]) -> list[str]:
    out: list[str] = []
    for s in stems[:100]:
        if re.fullmatch(r"\d+", s):
            continue
        c = s
        if re.search(r"Tile_r\d+-c\d+", c):
            out.append("Tile_r#y#-c#x#_S_#z#_#r#")
        if re.search(r"Tile_\d+-\d+-\d+", c):
            out.append("Tile_#y#-#x#-#r#_0-000.s#z#_e00")
        c = re.sub(r"(?i)\bTile_r(\d+)-c(\d+)", "Tile_r#y#-c#x#", c)
        c = re.sub(r"(?i)\b(row|r)[-_]?(\d+)\b", "row#y#", c)
        c = re.sub(r"(?i)\b(col|column|c)[-_]?(\d+)\b", "col#x#", c)
        c = re.sub(r"(?i)\b(slice|sec|section|layer)[-_]?(\d+)\b", "slice#z#", c)
        c = re.sub(r"(?i)\bx[-_]?(\d+)\b", "x#x#", c)
        c = re.sub(r"(?i)\by[-_]?(\d+)\b", "y#y#", c)
        c = re.sub(r"(?i)\bz[-_]?(\d+)\b", "z#z#", c)
        c = re.sub(r"(?i)_[sS](\d+)_", "_S#z#_", c)
        c = re.sub(r"(?<!#)(\d+)(?!#)", "#r#", c)
        c = re.sub(r"(#r#[-_.]?)+", "#r#", c)
        if re.search(r"#x#|#y#|#z#", c):
            out.append(c)
    # unique, keep order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# --------------------------------------------------------------------------
# Image metadata (size, dtype, FEI/Thermo pixel size + stage)
# --------------------------------------------------------------------------

@dataclass
class TileInfo:
    width: int = 0
    height: int = 0
    dtype: str = "uint8"
    pixel_size_nm: float | None = None     # from vendor metadata if present
    stage_xy_m: tuple[float, float] | None = None
    mosaic_xy_m: tuple[float, float] | None = None   # tile position relative to the mosaic (Zeiss Atlas), if stage is per-mosaic
    scan_rotation_rad: float | None = None
    vendor: str = ""


_UNIT_TO_M = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "\u00b5m": 1e-6, "nm": 1e-9, "pm": 1e-12}


def _fibics_value(xml: str, path: str) -> tuple[float | None, str]:
    """Value and unit of the first <a><b>...</b></a> element addressed by 'a/b' (no XML parser: files are latin-1 and small)."""
    import re
    parts = path.split("/")
    seg = xml
    for i, tag in enumerate(parts):
        m = re.search(rf"<{tag}(\s[^>]*)?>", seg)
        if not m:
            return None, ""
        if i < len(parts) - 1:
            end = re.search(rf"</{tag}>", seg[m.end():])
            seg = seg[m.end(): m.end() + end.start()] if end else seg[m.end():]
        else:
            attrs = m.group(1) or ""
            um = re.search(r'units="([^"]*)"', attrs)
            end = re.search(rf"</{tag}>", seg[m.end():])
            txt = seg[m.end(): m.end() + end.start()] if end else ""
            try:
                return float(txt.strip()), (um.group(1) if um else "")
            except ValueError:
                return None, ""
    return None, ""


def _parse_fibics(xml: str, info: TileInfo) -> None:
    """
    Zeiss Atlas (Fibics AtlasEngine) TIFFs keep their metadata in an XML block (tag 51023).
    There is no pixel size: it is the field of view divided by the image size. Stage X/Y are in um.
    """
    w, _ = _fibics_value(xml, "Image/Width")
    h, _ = _fibics_value(xml, "Image/Height")
    fov_x, ux = _fibics_value(xml, "Scan/FOV_X")
    fov_y, uy = _fibics_value(xml, "Scan/FOV_Y")
    width = w or info.width
    height = h or info.height
    if fov_x and width:
        info.pixel_size_nm = fov_x * _UNIT_TO_M.get(ux or "um", 1e-6) * 1e9 / width
    elif fov_y and height:
        info.pixel_size_nm = fov_y * _UNIT_TO_M.get(uy or "um", 1e-6) * 1e9 / height
    else:
        u, _ = _fibics_value(xml, "Scan/Ux")     # um per pixel along x
        if u:
            info.pixel_size_nm = abs(u) * 1e3
    sx, usx = _fibics_value(xml, "Stage/X")
    sy, usy = _fibics_value(xml, "Stage/Y")
    if sx is not None and sy is not None:
        info.stage_xy_m = (sx * _UNIT_TO_M.get(usx or "um", 1e-6), sy * _UNIT_TO_M.get(usy or "um", 1e-6))
    mx, umx = _fibics_value(xml, "MosaicInfo/X")
    my, umy = _fibics_value(xml, "MosaicInfo/Y")
    if mx is None or my is None:
        mx, umx = _fibics_value(xml, "StitchingInfo/UpdatedStagePositionX")
        my, umy = _fibics_value(xml, "StitchingInfo/UpdatedStagePositionY")
    if mx is not None and my is not None:
        info.mosaic_xy_m = (mx * _UNIT_TO_M.get(umx or "m", 1.0), my * _UNIT_TO_M.get(umy or "m", 1.0))
    rot, _ = _fibics_value(xml, "Scan/ScanRot")
    if rot is not None:
        info.scan_rotation_rad = math.radians(rot)


def read_tile_info(path: os.PathLike | str) -> TileInfo:
    path = str(path)
    info = TileInfo()
    try:
        import tifffile
        with tifffile.TiffFile(path) as t:
            p = t.pages[0]
            info.height, info.width = int(p.shape[0]), int(p.shape[1])
            info.dtype = str(p.dtype)
            tags = p.tags
            if "FEI_HELIOS" in tags or "FEI_TITAN" in tags:
                info.vendor = "thermo"
                fei = tags["FEI_HELIOS"].value if "FEI_HELIOS" in tags else {}
                if isinstance(fei, dict):
                    scan = fei.get("Scan", {}) or {}
                    pw = scan.get("PixelWidth")
                    if pw:
                        info.pixel_size_nm = float(pw) * 1e9
                    stage = fei.get("Stage", {}) or {}
                    if "StageX" in stage and "StageY" in stage:
                        info.stage_xy_m = (float(stage["StageX"]), float(stage["StageY"]))
                    beam = fei.get("EBeam", {}) or fei.get("Beam", {}) or {}
                    rot = beam.get("ScanRotation")
                    if rot is not None:
                        info.scan_rotation_rad = float(rot)
            elif "FibicsXML" in tags or 51023 in tags:
                info.vendor = "zeiss"
                xml = tags["FibicsXML"].value if "FibicsXML" in tags else tags[51023].value
                if isinstance(xml, bytes):
                    xml = xml.decode("latin-1", errors="replace")
                _parse_fibics(str(xml), info)
            else:
                # generic TIFF resolution tag (pixels per unit)
                try:
                    xres = tags.get("XResolution")
                    unit = tags.get("ResolutionUnit")
                    if xres is not None:
                        num, den = xres.value
                        ppu = num / den if den else 0
                        if ppu > 0 and unit is not None:
                            u = unit.value
                            u = getattr(u, "value", u)
                            if u == 3:        # centimeter
                                info.pixel_size_nm = 1e7 / ppu
                            elif u == 2:      # inch
                                info.pixel_size_nm = 2.54e7 / ppu
                    # ImageJ / OME metadata could be added here
                except Exception:
                    pass
            return info
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            info.width, info.height = im.size
            info.dtype = "uint16" if im.mode in ("I;16", "I;16B", "I") else "uint8"
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------
# Tiles and layout
# --------------------------------------------------------------------------

@dataclass
class Tile:
    path: Path
    ix: int      # column index from filename
    iy: int      # row index from filename
    z: int       # section number from filename
    x_px: float = 0.0
    y_px: float = 0.0
    stage_xy_m: tuple[float, float] | None = None
    mosaic_xy_m: tuple[float, float] | None = None


def load_positions(tiles: list[Tile]) -> None:
    """Read stage (and mosaic-relative) positions from the tiles' metadata."""
    for t in tiles:
        ti = read_tile_info(t.path)
        t.stage_xy_m = ti.stage_xy_m
        t.mosaic_xy_m = ti.mosaic_xy_m


def resolve_positions(tiles: list[Tile], pixel_size_nm: float, tile_w: int) -> str:
    """
    Make ``stage_xy_m`` usable for placement. Some microscopes (Zeiss Atlas) write the *mosaic's*
    stage position into every tile and keep the tile offset in a separate mosaic entry; if the stage
    positions of a section do not spread over at least half a tile while the mosaic offsets do, the
    mosaic offsets are used instead. Returns 'stage' | 'mosaic' | 'none'.
    """
    have_stage = [t for t in tiles if t.stage_xy_m is not None]
    have_mos = [t for t in tiles if t.mosaic_xy_m is not None]
    if len(tiles) <= 1:
        return "stage" if have_stage else "none"
    px_m = pixel_size_nm * 1e-9
    half_tile = 0.5 * tile_w * px_m

    def spread(vals):
        arr = np.array(vals, float)
        return float(np.ptp(arr, axis=0).max()) if len(arr) > 1 else 0.0

    if len(have_stage) == len(tiles) and spread([t.stage_xy_m for t in tiles]) >= half_tile:
        return "stage"
    if len(have_mos) == len(tiles) and spread([t.mosaic_xy_m for t in tiles]) >= half_tile:
        for t in tiles:
            t.stage_xy_m = t.mosaic_xy_m
        return "mosaic"
    return "stage" if len(have_stage) == len(tiles) else "none"


@dataclass
class LayoutParams:
    tile_w: int = 0
    tile_h: int = 0
    overlap_x: float = 10.0        # value in overlap_unit
    overlap_y: float = 10.0
    overlap_unit: str = "percent"  # percent | pixels
    mode: str = "grid"             # grid | stage
    z_offset: int = 0
    flip_x: bool = False           # invert column direction
    flip_y: bool = False           # invert row direction

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "LayoutParams":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def overlap_px(self) -> tuple[float, float]:
        if self.overlap_unit == "percent":
            return self.tile_w * self.overlap_x / 100.0, self.tile_h * self.overlap_y / 100.0
        return float(self.overlap_x), float(self.overlap_y)


def scan_tiles(root: os.PathLike | str, rule: NamingRule) -> tuple[list[Tile], list[Path]]:
    """Return (matched tiles, unmatched files)."""
    files = list_image_files(root, rule.ext, rule.recursive)
    tiles: list[Tile] = []
    unmatched: list[Path] = []
    for p in files:
        c = rule.parse(p.stem)
        if c is None:
            unmatched.append(p)
            continue
        tiles.append(Tile(path=p, ix=c.x, iy=c.y, z=c.z))
    tiles.sort(key=lambda t: (t.z, t.iy, t.ix))
    return tiles, unmatched


def group_by_section(tiles: Iterable[Tile]) -> dict[int, list[Tile]]:
    out: dict[int, list[Tile]] = defaultdict(list)
    for t in tiles:
        out[t.z].append(t)
    return dict(sorted(out.items()))


def place_tiles_grid(tiles: list[Tile], layout: LayoutParams) -> None:
    ovx, ovy = layout.overlap_px()
    sx = layout.tile_w - ovx
    sy = layout.tile_h - ovy
    ixs = [t.ix for t in tiles]
    iys = [t.iy for t in tiles]
    if not tiles:
        return
    ix0, iy0 = min(ixs), min(iys)
    ix1, iy1 = max(ixs), max(iys)
    for t in tiles:
        cx = (ix1 - t.ix) if layout.flip_x else (t.ix - ix0)
        cy = (iy1 - t.iy) if layout.flip_y else (t.iy - iy0)
        t.x_px = cx * sx
        t.y_px = cy * sy


def place_tiles_stage(tiles: list[Tile], layout: LayoutParams, pixel_size_nm: float) -> bool:
    """
    Place tiles from microscope stage coordinates.

    The stage axes are generally rotated (and possibly mirrored) with respect
    to the image axes. The orientation is recovered from the tiles' own grid
    indices: a least-squares fit of stage position against (column, row)
    index gives the stage displacement per column step and per row step; the
    stage frame is then rotated so that the column step points to +x, and
    mirrored if the row step would then point to -y. Stage jitter is kept.

    Returns False (and falls back to grid placement) if stage data is missing
    or the fit is degenerate.
    """
    have = [t for t in tiles if t.stage_xy_m is not None]
    if len(tiles) == 1:
        tiles[0].x_px = tiles[0].y_px = 0.0        # a single-tile section needs no placement
        return True
    if len(have) < len(tiles) or len(tiles) < 2 or pixel_size_nm <= 0:
        place_tiles_grid(tiles, layout)
        return False
    px_m = pixel_size_nm * 1e-9
    S = np.array([t.stage_xy_m for t in tiles], dtype=float) / px_m   # in pixels
    IX = np.array([t.ix for t in tiles], dtype=float)
    IY = np.array([t.iy for t in tiles], dtype=float)
    A = np.stack([IX, IY, np.ones_like(IX)], axis=1)
    single_col = np.ptp(IX) == 0
    single_row = np.ptp(IY) == 0
    try:
        if single_col and single_row:
            raise np.linalg.LinAlgError
        if single_col or single_row:
            # only one direction available: fit that one, assume orthogonal other
            idx = IY if single_col else IX
            A1 = np.stack([idx, np.ones_like(idx)], axis=1)
            coef, *_ = np.linalg.lstsq(A1, S, rcond=None)
            step = coef[0]
            if single_col:
                ay = step
                ax = np.array([ay[1], -ay[0]])
            else:
                ax = step
                ay = np.array([-ax[1], ax[0]])
        else:
            coef, *_ = np.linalg.lstsq(A, S, rcond=None)
            ax = coef[0]
            ay = coef[1]
    except np.linalg.LinAlgError:
        place_tiles_grid(tiles, layout)
        return False
    if np.linalg.norm(ax) < 1e-6 or np.linalg.norm(ay) < 1e-6:
        place_tiles_grid(tiles, layout)
        return False
    theta = math.atan2(ax[1], ax[0])
    c, s = math.cos(-theta), math.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    ay_r = R @ ay
    if ay_r[1] < 0:
        R = np.array([[1, 0], [0, -1]]) @ R
    P = (R @ S.T).T
    if layout.flip_x:
        P[:, 0] = -P[:, 0]
    if layout.flip_y:
        P[:, 1] = -P[:, 1]
    P -= P.min(axis=0)
    for t, (x, y) in zip(tiles, P):
        t.x_px = float(x)
        t.y_px = float(y)
    return True


def estimate_overlap_from_stage(tiles: list[Tile], layout: LayoutParams, pixel_size_nm: float) -> tuple[float, float] | None:
    """Median overlap in pixels (x, y) implied by stage steps between neighbours."""
    work = [Tile(t.path, t.ix, t.iy, t.z, stage_xy_m=t.stage_xy_m, mosaic_xy_m=t.mosaic_xy_m) for t in tiles]
    resolve_positions(work, pixel_size_nm, layout.tile_w)
    if not place_tiles_stage(work, LayoutParams(tile_w=layout.tile_w, tile_h=layout.tile_h, mode="stage"), pixel_size_nm):
        return None
    by = {(t.ix, t.iy): t for t in work}
    dx, dy = [], []
    for t in work:
        n = by.get((t.ix + 1, t.iy))
        if n is not None:
            dx.append(abs(n.x_px - t.x_px))
        n = by.get((t.ix, t.iy + 1))
        if n is not None:
            dy.append(abs(n.y_px - t.y_px))
    if not dx and not dy:
        return None          # single tile or no neighbouring tiles: nothing to estimate
    ox = layout.tile_w - float(np.median(dx)) if dx else float("nan")
    oy = layout.tile_h - float(np.median(dy)) if dy else float("nan")
    return ox, oy


# --------------------------------------------------------------------------
# FEABAS stitch_coord output
# --------------------------------------------------------------------------

def section_name(z: int, width: int = 4) -> str:
    return f"s{z:0{width}d}"


def format_stitch_coord(tiles: list[Tile], root_dir: Path, resolution_nm: float,
                        tile_h: int, tile_w: int, path_mode: str = "relative") -> str:
    """
    FEABAS format:
        {ROOT_DIR}<TAB>path
        {RESOLUTION}<TAB>4.0
        {TILE_SIZE}<TAB>height<TAB>width
        relative/path.tif<TAB>x<TAB>y
    """
    lines = []
    root_dir = Path(root_dir)
    if path_mode == "relative":
        lines.append(f"{{ROOT_DIR}}\t{root_dir.as_posix()}")
    lines.append(f"{{RESOLUTION}}\t{resolution_nm:g}")
    lines.append(f"{{TILE_SIZE}}\t{int(tile_h)}\t{int(tile_w)}")
    for t in sorted(tiles, key=lambda t: (t.iy, t.ix)):
        if path_mode == "relative":
            try:
                rel = Path(t.path).resolve().relative_to(root_dir.resolve()).as_posix()
            except ValueError:
                rel = Path(t.path).resolve().as_posix()
        else:
            rel = Path(t.path).resolve().as_posix()
        lines.append(f"{rel}\t{t.x_px:.1f}\t{t.y_px:.1f}")
    return "\n".join(lines) + "\n"


@dataclass
class CoordPlan:
    """Everything needed to write stitch_coord files; also a preview model."""
    sections: dict[str, list[Tile]] = field(default_factory=dict)   # section name -> tiles
    tile_w: int = 0
    tile_h: int = 0
    resolution_nm: float = 4.0
    root_dir: Path = Path(".")
    placed_by_stage: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def n_tiles(self) -> int:
        return sum(len(v) for v in self.sections.values())

    def grid_shape(self) -> tuple[int, int]:
        rows = cols = 0
        for tl in self.sections.values():
            if tl:
                rows = max(rows, max(t.iy for t in tl) - min(t.iy for t in tl) + 1)
                cols = max(cols, max(t.ix for t in tl) - min(t.ix for t in tl) + 1)
        return rows, cols

    def section_bbox(self, name: str) -> tuple[float, float]:
        tl = self.sections.get(name, [])
        if not tl:
            return 0.0, 0.0
        return (max(t.x_px for t in tl) + self.tile_w, max(t.y_px for t in tl) + self.tile_h)


def build_plan(root: os.PathLike | str, rule: NamingRule, layout: LayoutParams,
               resolution_nm: float | None = None, read_stage: bool | None = None,
               z_pad: int | None = None) -> CoordPlan:
    root = Path(root)
    tiles, unmatched = scan_tiles(root, rule)
    plan = CoordPlan(root_dir=root)
    if not tiles:
        plan.warnings.append(f"No files matched rule '{rule.describe()}' under {root}.")
        return plan
    if unmatched:
        plan.warnings.append(f"{len(unmatched)} file(s) did not match the naming rule and were skipped "
                             f"(e.g. {unmatched[0].name}).")

    info = read_tile_info(tiles[0].path)
    tile_w = layout.tile_w or info.width
    tile_h = layout.tile_h or info.height
    if resolution_nm is None or resolution_nm <= 0:
        resolution_nm = info.pixel_size_nm or 4.0
    plan.tile_w, plan.tile_h, plan.resolution_nm = tile_w, tile_h, float(resolution_nm)
    layout = LayoutParams.from_dict(layout.to_dict())
    layout.tile_w, layout.tile_h = tile_w, tile_h

    use_stage = layout.mode == "stage" if read_stage is None else read_stage
    if use_stage:
        load_positions(tiles)

    zs = sorted({t.z for t in tiles})
    if z_pad is None:
        z_pad = max(4, len(str(max(zs) + layout.z_offset)))
    groups = group_by_section(tiles)
    all_stage_ok = True
    for z, tl in groups.items():
        if use_stage:
            src = resolve_positions(tl, plan.resolution_nm, tile_w)
            if src == "mosaic" and "mosaic" not in " ".join(plan.warnings):
                plan.warnings.append("Stage position is per mosaic in this metadata; tile offsets from the mosaic entry were used.")
            ok = place_tiles_stage(tl, layout, plan.resolution_nm)
            all_stage_ok = all_stage_ok and ok
        else:
            place_tiles_grid(tl, layout)
        name = section_name(z + layout.z_offset, z_pad)
        plan.sections[name] = tl
    plan.placed_by_stage = bool(use_stage and all_stage_ok)
    if use_stage and not all_stage_ok:
        plan.warnings.append("Stage coordinates missing or degenerate for some sections; those used grid placement.")

    # sanity: duplicate positions in a section
    for name, tl in plan.sections.items():
        seen = Counter((t.ix, t.iy) for t in tl)
        dups = [k for k, v in seen.items() if v > 1]
        if dups:
            plan.warnings.append(f"Section {name}: {len(dups)} duplicate grid position(s), e.g. col {dups[0][0]} row {dups[0][1]}. "
                                 f"Check the naming rule (which number is the running tile id?).")
    return plan


def write_plan(plan: CoordPlan, out_dir: os.PathLike | str, path_mode: str = "relative",
               overwrite: bool = True) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, tl in plan.sections.items():
        p = out_dir / f"{name}.txt"
        if p.exists() and not overwrite:
            continue
        p.write_text(format_stitch_coord(tl, plan.root_dir, plan.resolution_nm, plan.tile_h, plan.tile_w, path_mode),
                     encoding="utf-8")
        written.append(p)
    return written


# --------------------------------------------------------------------------
# Rename helper (port of the PowerShell "Rename files" mode)
# --------------------------------------------------------------------------

def plan_prefix_rename(root: os.PathLike | str, ext: str = "tif", level: str = "parent",
                       recursive: bool = True) -> list[tuple[Path, str]]:
    """Return (path, new_name) pairs prefixing each file with a folder name."""
    files = list_image_files(root, ext, recursive)
    out = []
    for f in files:
        d = f.parent
        if level == "grandparent":
            d = d.parent
        elif level == "greatgrand":
            d = d.parent.parent
        prefix = d.name
        if not prefix or f.name.startswith(prefix + "-"):
            continue
        out.append((f, f"{prefix}-{f.name}"))
    return out
