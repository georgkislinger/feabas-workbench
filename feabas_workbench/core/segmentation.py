"""
Segmentation masks that match an image stack, carried along when the images are (re)aligned.

Someone segmented an image stack; FEABAS improves the stack's alignment. The masks then have to
take exactly the path the images took: each mask file is placed like its image (the section's
stitching transform), then moved by the section's alignment mesh onto the same canvas, sampled
nearest-neighbour throughout so that a label is never blended with its neighbour, and without any
of the intensity processing the images get. ``workers/segmentation_render`` does that with
FEABAS's own renderers; the result sits in the layout of the aligned PNG stack (mip levels,
tiles, metadata.txt), so the stack exports (VAST, OME-Zarr, slice images) work for it too.

Mask files: 8- or 16-bit greyscale PNG or TIFF, one per input image with the same width and
height, matched to the images by file name or, with one image per section, in section order.
Everything here is Qt-free and needs no FEABAS.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

MASK_SUFFIXES = (".png", ".tif", ".tiff")
MATCH_MODES = ("auto", "name", "order")
STACKS_DIR = "segmentation"            # <project>/segmentation/<name> unless another folder is chosen
MONTAGE_DIR = "_montage"               # the masks placed like the images, before the alignment


@dataclass(frozen=True)
class ImageHeader:
    width: int
    height: int
    bits: int                          # bits per sample
    channels: int
    palette: bool = False


def read_image_header(path: Path) -> ImageHeader:
    """Size and sample format of a PNG or TIFF without reading its pixels."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        with open(path, "rb") as f:
            head = f.read(33)
        if len(head) < 33 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
            raise ValueError(f"{path.name}: not a PNG file")
        width, height, bits, color = struct.unpack(">IIBB", head[16:26])
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color, 0)
        return ImageHeader(width, height, bits, channels, palette=color == 3)
    if suffix in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) > 1:
                raise ValueError(f"{path.name}: a multi-page TIFF; save one file per section")
            page = tif.pages[0]
            samples = int(getattr(page, "samplesperpixel", 1) or 1)
            shape = page.shape
            height, width = (shape[0], shape[1]) if samples == 1 or len(shape) == 2 else \
                ((shape[1], shape[2]) if shape[0] == samples else (shape[0], shape[1]))
            return ImageHeader(int(width), int(height), int(page.dtype.itemsize * 8), samples,
                               palette=getattr(page, "photometric", None) == 3)
    raise ValueError(f"{path.name}: masks must be PNG or TIFF (lossless); JPEG would change the labels")


def check_mask_header(header: ImageHeader, name: str) -> str | None:
    """None if the file is an 8- or 16-bit single-channel label image, else what is wrong with it."""
    if header.palette:
        return f"{name}: a palette (indexed colour) image; save it as 8- or 16-bit greyscale"
    if header.channels != 1:
        return f"{name}: {header.channels} channels; masks must be single-channel greyscale label images"
    if header.bits not in (8, 16):
        return f"{name}: {header.bits}-bit samples; only 8- and 16-bit masks are supported"
    return None


def natural_key(text: str):
    """'s2' sorts before 's10'."""
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", text)]


def list_mask_files(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MASK_SUFFIXES),
                  key=lambda p: natural_key(p.name))


def read_stitch_coord(path: Path) -> tuple[list[Path], tuple[int, int] | None]:
    """The image files a section was stitched from (absolute) and their size (height, width)."""
    root, size, tiles = None, None, []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if not parts or not parts[0].strip():
            continue
        if parts[0] == "{ROOT_DIR}":
            root = Path(parts[1])
        elif parts[0] == "{TILE_SIZE}" and len(parts) >= 3:
            size = (int(float(parts[1])), int(float(parts[2])))
        elif parts[0].startswith("{"):
            continue
        else:
            p = Path(parts[0])
            tiles.append(p if p.is_absolute() or root is None else root / p)
    return tiles, size


@dataclass
class MaskPlan:
    """Which mask file goes with which input image, per section in stack order."""
    sections: list[tuple[str, dict[str, str]]] = field(default_factory=list)   # (section, {image file name: mask path})
    bits: int = 8
    mode: str = "name"
    notes: list[str] = field(default_factory=list)

    @property
    def n_masks(self) -> int:
        return sum(len(m) for _, m in self.sections)

    def to_dict(self) -> dict:
        return {"sections": [[s, m] for s, m in self.sections], "bits": self.bits, "mode": self.mode}


def plan_masks(root: Path, sections: list[str], folder: Path, mode: str = "auto") -> MaskPlan:
    """
    Match the mask files in *folder* to the images of *sections* (stack order) and check them.
    'name': a mask has the file name of its image (any of the mask suffixes); 'order': one image
    per section, the n-th mask (natural sort) goes with the n-th section; 'auto': by name when that
    finds every mask, else in order. Raises ValueError saying what does not fit.
    """
    if mode not in MATCH_MODES:
        raise ValueError(f"unknown matching mode {mode!r}")
    # absolute mask paths: the render runs in the project folder
    root, folder = Path(root).resolve(), Path(folder).resolve()
    masks = list_mask_files(folder)
    if not masks:
        raise ValueError(f"no PNG or TIFF files in {folder}")
    coord_dir = root / "stitch" / "stitch_coord"
    images: list[tuple[str, list[Path], tuple[int, int] | None]] = []
    for s in sections:
        coord = coord_dir / f"{s}.txt"
        if not coord.is_file():
            raise ValueError(f"{s}: no stitch coordinate file; set up the project's data first")
        tiles, size = read_stitch_coord(coord)
        images.append((s, tiles, size))
    if not images:
        raise ValueError("the project has no sections")
    by_stem: dict[str, list[Path]] = {}
    for m in masks:
        by_stem.setdefault(m.stem.casefold(), []).append(m)
    plan = MaskPlan()
    missing = []
    if mode in ("auto", "name"):
        for s, tiles, _ in images:
            found = {}
            for t in tiles:
                hits = by_stem.get(t.stem.casefold(), [])
                if len(hits) > 1:
                    raise ValueError(f"{t.stem}: more than one mask file with this name ({', '.join(h.name for h in hits)})")
                if hits:
                    found[t.name] = str(hits[0])
                else:
                    missing.append(t.name)
            plan.sections.append((s, found))
        plan.mode = "name"
        if missing and mode == "name":
            raise ValueError(f"{len(missing)} image(s) have no mask of the same name, e.g. "
                             + ", ".join(missing[:3]) + f" (masks in {folder})")
    if mode == "order" or (mode == "auto" and missing):
        single = all(len(tiles) == 1 for _, tiles, _ in images)
        if not single:
            raise ValueError(("masks are matched in order only with one image per section; " if mode == "order" else
                              f"{len(missing)} image(s) have no mask of the same name (e.g. {', '.join(missing[:3])}) "
                              "and the sections have several images each, so matching in order is not possible; ")
                             + "name each mask like its image")
        if len(masks) != len(images):
            raise ValueError(f"{len(masks)} mask files for {len(images)} sections: in order, every section needs exactly "
                             f"one mask" + ("" if mode == "order" else " (and matching by name found only some of them)"))
        plan.sections = [(s, {tiles[0].name: str(m)}) for (s, tiles, _), m in zip(images, masks)]
        plan.mode = "order"
        if mode == "auto":
            plan.notes.append("matched in section order (the file names differ from the images')")
    bits = set()
    sizes = {s: size for s, _, size in images}
    for s, found in plan.sections:
        for image_name, mask in found.items():
            header = read_image_header(Path(mask))
            problem = check_mask_header(header, Path(mask).name)
            if problem:
                raise ValueError(problem)
            size = sizes.get(s)
            if size is not None and (header.height, header.width) != tuple(size):
                raise ValueError(f"{Path(mask).name}: {header.width} x {header.height} px, but its image {image_name} "
                                 f"is {size[1]} x {size[0]} px; masks must match their images pixel for pixel")
            bits.add(header.bits)
    if len(bits) > 1:
        raise ValueError("the masks mix 8- and 16-bit files; save them all with one bit depth")
    plan.bits = bits.pop()
    return plan


def stack_dir(root: Path, name: str, out_dir: Path | None = None) -> Path:
    """Where the aligned masks called *name* go: <out_dir or project/segmentation>/<name>; a relative
    *out_dir* is taken relative to the project. Always absolute (the render runs in the project folder)."""
    name = safe_name(name)
    root = Path(root).resolve()
    parent = Path(out_dir) if out_dir else root / STACKS_DIR
    return (parent if parent.is_absolute() else root / parent) / name


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._")
    if not cleaned:
        raise ValueError("give the masks a name (letters, digits, '.', '_' or '-')")
    return cleaned


def section_folders(level: Path, sections: list[str]) -> dict[str, Path]:
    """The folder of each section at one mip level: FEABAS names it <z>_<section> (or <section>)."""
    found = {}
    if not level.is_dir():
        return found
    wanted = set(sections)
    for d in level.iterdir():
        if not d.is_dir():
            continue
        name = d.name if d.name in wanted else (d.name.split("_", 1)[1] if "_" in d.name else "")
        if name in wanted:
            found[name] = d
    return found


@dataclass(frozen=True)
class StackStatus:
    sections: int
    rendered: int
    mipmapped: int
    stale: str = ""

    @property
    def done(self) -> bool:
        return self.sections > 0 and self.rendered >= self.sections and self.mipmapped >= self.sections


def stack_status(root: Path, base: Path, sections: list[str], render_mip: int, max_mip: int,
                 masks: list[Path] | None = None) -> StackStatus:
    """Sections rendered and mipmapped, from the files on disk; stale when the alignment (or a
    mask file) changed after the oldest rendered section."""
    root, base = Path(root), Path(base)
    rendered = {s: d / "metadata.txt" for s, d in section_folders(base / f"mip{render_mip}", sections).items()
                if (d / "metadata.txt").is_file()}
    top = max(render_mip, int(max_mip))
    mipped = [s for s, d in section_folders(base / f"mip{top}", sections).items() if (d / "metadata.txt").is_file()]
    stale = ""
    if rendered:
        oldest = min(p.stat().st_mtime for p in rendered.values())
        tform = root / "align" / "tform"
        changed = [p for p in tform.glob("*.h5")] + [tform / "canvas.json"] if tform.is_dir() else []
        if any(p.is_file() and p.stat().st_mtime > oldest + 1 for p in changed):
            stale = "the alignment changed after these masks were rendered"
        elif masks and any(Path(m).is_file() and Path(m).stat().st_mtime > oldest + 1 for m in masks):
            stale = "mask files changed after they were rendered"
    return StackStatus(len(sections), len(rendered), len(mipped), stale)


def mode_downsample(a):
    """
    Halve a label image: each 2x2 block becomes the label that covers most of it. A label beats
    0 (background) on a tie, and the smaller label wins a tie between labels, so the result is
    not shifted towards one corner the way picking one pixel of every block is, and every output
    value is one of the input values.
    """
    import numpy as np
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    p = [a[0:h:2, 0:w:2], a[0:h:2, 1:w:2], a[1:h:2, 0:w:2], a[1:h:2, 1:w:2]]
    best = p[0].copy()
    best_n = np.zeros(best.shape, np.uint8)
    for v in p:
        n = (v == p[0]).astype(np.uint8) + (v == p[1]) + (v == p[2]) + (v == p[3])
        tie = n == best_n
        better = (n > best_n) | (tie & (v != 0) & ((best == 0) | (v < best)))
        best = np.where(better, v, best)
        best_n = np.where(better, n, best_n)
    return best


def fill_tile_pattern(pattern: str, row: int, col: int, tile_hw: tuple[int, int], one_based: bool) -> str:
    """A tile file name from FEABAS's filename pattern (row/column indices start at 0 here)."""
    th, tw = int(tile_hw[0]), int(tile_hw[-1])
    x0, y0 = col * tw, row * th
    values = {"{ROW_IND}": row + int(one_based), "{COL_IND}": col + int(one_based), "{X_MIN}": x0, "{Y_MIN}": y0,
              "{X_MAX}": x0 + tw, "{Y_MAX}": y0 + th}
    for key, value in values.items():
        pattern = pattern.replace(key, str(value))
    return pattern


def read_tile_metadata(meta: Path) -> dict[str, tuple[int, int, int, int]]:
    """Tile file name -> bbox (x0, y0, x1, y1) from a FEABAS metadata.txt."""
    out = {}
    for line in Path(meta).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) >= 5 and not parts[0].startswith("{"):
            out[Path(parts[0]).name] = tuple(int(round(float(v))) for v in parts[1:5])
    return out


def write_tile_metadata(folder: Path, tiles: dict[str, tuple[int, int, int, int]], resolution: float) -> None:
    """metadata.txt the way FEABAS writes it (written last: it marks the level as finished)."""
    lines = [f"{{ROOT_DIR}}\t{Path(folder).as_posix()}", f"{{RESOLUTION}}\t{resolution}"]
    lines += [f"{name}\t{b[0]}\t{b[1]}\t{b[2]}\t{b[3]}" for name, b in sorted(tiles.items())]
    tmp = Path(folder) / "metadata.txt.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(Path(folder) / "metadata.txt")


def build_label_pyramid(base: Path, section_dir: str, tile_prefix: str, first_mip: int, max_mip: int,
                        tile_hw: tuple[int, int], pattern: str, one_based: bool, resolution: float) -> list[int]:
    """
    Mipmaps of one section of a rendered label stack, level by level from *first_mip*, with
    mode_downsample. Tile (r, c) of a level covers [c*w, (c+1)*w) x [r*h, (r+1)*h) of that level's
    pixels - the grid FEABAS's mipmaps of the images use - and is made from whatever tiles of the
    level below cover twice that area. Empty tiles are not written (as FEABAS does); levels that
    already have a metadata.txt are kept. Returns the levels written.
    """
    import cv2
    import numpy as np
    th, tw = int(tile_hw[0]), int(tile_hw[-1])
    written = []
    for m in range(first_mip, max_mip):
        src, dst = Path(base) / f"mip{m}" / section_dir, Path(base) / f"mip{m + 1}" / section_dir
        if (dst / "metadata.txt").is_file():
            continue
        if not (src / "metadata.txt").is_file():
            raise RuntimeError(f"{src} is not finished")
        dst.mkdir(parents=True, exist_ok=True)
        for p in dst.iterdir():                      # an unfinished level from an interrupted run
            if p.is_file():
                p.unlink()
        members: dict[tuple[int, int], list[tuple[str, tuple[int, int, int, int]]]] = {}
        for name, (x0, y0, x1, y1) in read_tile_metadata(src / "metadata.txt").items():
            for r in range(y0 // (2 * th), (max(y1, y0 + 1) - 1) // (2 * th) + 1):
                for c in range(x0 // (2 * tw), (max(x1, x0 + 1) - 1) // (2 * tw) + 1):
                    members.setdefault((r, c), []).append((name, (x0, y0, x1, y1)))
        out = {}
        for r, c in sorted(members):
            block = None
            bx0, by0 = 2 * c * tw, 2 * r * th
            for name, (x0, y0, _x1, _y1) in members[(r, c)]:
                a = cv2.imread(str(src / name), cv2.IMREAD_UNCHANGED)
                if a is None:
                    raise RuntimeError(f"cannot read {src / name}")
                if block is None:
                    block = np.zeros((2 * th, 2 * tw), a.dtype)
                ix0, iy0 = max(x0, bx0), max(y0, by0)
                ix1, iy1 = min(x0 + a.shape[1], bx0 + 2 * tw), min(y0 + a.shape[0], by0 + 2 * th)
                if ix1 > ix0 and iy1 > iy0:
                    block[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0] = a[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
            small = mode_downsample(block)
            if np.any(small):
                name = tile_prefix + fill_tile_pattern(pattern, r, c, (th, tw), one_based)
                if not cv2.imwrite(str(dst / name), small):
                    raise RuntimeError(f"cannot write {dst / name}")
                out[name] = (c * tw, r * th, (c + 1) * tw, (r + 1) * th)
        write_tile_metadata(dst, out, resolution * 2 ** (m + 1 - first_mip))
        written.append(m + 1)
    return written


def list_stacks(root: Path, extra_dirs: list[Path] | None = None) -> list[Path]:
    """Aligned mask stacks in the project's segmentation folder (and in *extra_dirs*)."""
    out = []
    for parent in [Path(root) / STACKS_DIR] + [Path(d) for d in (extra_dirs or [])]:
        if parent.is_dir():
            out.extend(sorted(d for d in parent.iterdir() if d.is_dir() and any(d.glob("mip*"))))
    return out
