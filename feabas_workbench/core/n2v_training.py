"""Pixel-based N2V planning, with spatially disjoint training and validation data."""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tile:
    path: str
    height: int
    width: int
    dtype: str


def tile_header(path):
    path = Path(path).resolve()
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile
        with tifffile.TiffFile(path) as image:
            series = image.series[0]
            shape, dtype = series.shape, str(series.dtype)
            if len(shape) != 2:
                raise ValueError(f"{path.name}: select single-plane grayscale images.")
            h, w = shape
    else:
        from PIL import Image
        with Image.open(path) as image:
            if image.mode not in {"L", "I", "F", "I;16", "I;16B", "I;16L"}:
                raise ValueError(f"{path.name}: select single-plane grayscale images.")
            w, h = image.size
            dtype = image.mode
    return Tile(str(path), h, w, dtype)


def regions(tile, patch):
    """Patch-grid rectangles (y, x, rows, cols); reserve a separate spatial band."""
    rows, cols = tile.height // patch, tile.width // patch
    if rows >= 2 and cols >= 1:
        val = max(1, rows // 5)
        return (0, 0, rows - val, cols), (rows - val, 0, val, cols)
    if cols >= 2 and rows >= 1:
        val = max(1, cols // 5)
        return (0, 0, rows, cols - val), (0, cols - val, rows, val)
    return (0, 0, 0, 0), (0, 0, 0, 0)


def requirements(patch, batch):
    # Practical sampling floor, not a guarantee of specimen diversity/model quality.
    return (max(math.ceil(4 * 1024**2 / patch**2), 64, 4 * batch),
            max(math.ceil(1024**2 / patch**2), 8, batch))


def describe(tiles, patch=128, batch=12):
    train = val = 0
    for tile in tiles:
        a, b = regions(tile, patch)
        train += a[2] * a[3]
        val += b[2] * b[3]
    need_train, need_val = requirements(patch, batch)
    valid = train >= need_train and val >= need_val
    message = (f"{len(tiles)} images · usable: {train * patch**2 / 1e6:.1f} MP training + "
               f"{val * patch**2 / 1e6:.1f} MP held out. Minimum: "
               f"{need_train * patch**2 / 1e6:.1f} + {need_val * patch**2 / 1e6:.1f} MP.")
    if not valid:
        message += " Add more pixels or use a smaller patch/batch."
    return dict(tiles=tiles, train=train, val=val, valid=valid, message=message,
                need_train=need_train, need_val=need_val)


def inspect_selection(paths, patch=128, batch=12, *, auto=False, progress=None, cancelled=lambda: False):
    unique = list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
    if auto:
        random.Random(42).shuffle(unique)
    tiles = []
    for i, path in enumerate(unique):
        if cancelled():
            return None
        tiles.append(tile_header(path))
        if progress:
            progress(i + 1, len(unique), Path(path).name)
        if auto and describe(tiles, patch, batch)["valid"]:
            break
    return describe(tiles, patch, batch)


def sample_regions(tiles, patch, counts, seed=42):
    """Sample unique grid cells without enumerating huge images; record provenance."""
    rng = random.Random(seed)
    chosen = []
    for split, count in enumerate(counts):
        rects = [regions(tile, patch)[split] for tile in tiles]
        ends, total = [], 0
        for _, _, rows, cols in rects:
            total += rows * cols
            ends.append(total)
        for index in rng.sample(range(total), min(count, total)):
            tile_index = bisect.bisect_right(ends, index)
            local = index - (ends[tile_index - 1] if tile_index else 0)
            y, x, _, cols = rects[tile_index]
            chosen.append(dict(split="train" if split == 0 else "val", tile=tile_index,
                               y=(y + local // cols) * patch, x=(x + local % cols) * patch))
    return chosen


def prepare_data(paths, patch, batch, seed=42, log=print):
    """Read one source at a time; cap sampled float32 arrays at ~256 MiB normally."""
    import numpy as np
    import tifffile
    info = inspect_selection(paths, patch, batch)
    if not info["valid"]:
        raise ValueError(info["message"])
    cap = max(64 * 1024**2 // patch**2, info["need_train"] + info["need_val"])
    n_val = min(info["val"], max(info["need_val"], cap // 5))
    n_train = min(info["train"], max(info["need_train"], cap - n_val))
    records = sample_regions(info["tiles"], patch, (n_train, n_val), seed)
    arrays = {"train": np.empty((n_train, patch, patch), np.float32),
              "val": np.empty((n_val, patch, patch), np.float32)}
    positions = {"train": 0, "val": 0}
    for i, tile in enumerate(info["tiles"]):
        selected = [r for r in records if r["tile"] == i]
        if not selected:
            continue
        log(f"Reading training patches: {Path(tile.path).name}")
        if Path(tile.path).suffix.lower() in {".tif", ".tiff"}:
            try:
                image = tifffile.memmap(tile.path, mode="r")
            except ValueError:
                image = tifffile.imread(tile.path)
        else:
            from PIL import Image
            with Image.open(tile.path) as source:
                image = np.asarray(source)
        if image.shape != (tile.height, tile.width):
            raise ValueError(f"Image dimensions changed: {tile.path}")
        for record in selected:
            split, y, x = record["split"], record["y"], record["x"]
            crop = image[y:y + patch, x:x + patch]
            if not np.isfinite(crop).all():
                raise ValueError(f"Non-finite training pixels in {tile.path}")
            arrays[split][positions[split]] = crop
            record["sample"] = positions[split]
            positions[split] += 1
        del crop  # release the view before decoding the next potentially multi-GB tile
        del image
    manifest = dict(tiles=[asdict(t) for t in info["tiles"]], patches=records, patch_size=patch,
                    train_pixels=n_train * patch**2, validation_pixels=n_val * patch**2,
                    split="non-overlapping spatial bands in each image", seed=seed)
    log(info["message"])
    return arrays["train"], arrays["val"], manifest
