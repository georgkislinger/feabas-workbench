"""
Where masks live and how they are composed for a project.

    <project>/masks/tissue/<sec>.png        tissue (255) vs background (0) at thumbnail mip
    <project>/masks/folds/<sec>.png         fold mask (255) at fold mip, + <sec>_prob.png
    <project>/masks/structures/<sec>.png    structure mask (255) at structure mip (YOLO-seg)
    <project>/masks/manifest.json           provenance / hand-edited flags
    <project>/thumbnail_align/material_masks/<sec>.png   what FEABAS reads (thumbnail mip)
    <project>/align/material_masks/<sec>.png             optional hi-res masks (alignment mask_mip_level)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .masks import (TissueParams, detect_tissue, compose_material_mask, resize_mask, read_mask, write_mask,
                    LABEL_WRINKLE, LABEL_DEFAULT, LABEL_EXCLUDE, structure_material_mask, mask_stats,
                    footprint_mask, roi_from_tform, border_band, montage_extent)
from .images import imread


class MaskStore:
    def __init__(self, project):
        self.project = project
        self.root = project.root
        self.tissue_dir = project.masks_dir / "tissue"
        self.folds_dir = project.masks_dir / "folds"
        self.structures_dir = project.masks_dir / "structures"
        self.roi_dir = project.masks_dir / "roi"                # FEABAS's own tile-footprint masks
        self.external_dir = project.masks_dir / "external"      # full-resolution copies of imported masks
        self.manifest_file = project.masks_dir / "manifest.json"
        self.thumb_mask_dir = self.root / "thumbnail_align" / "material_masks"
        self.align_mask_dir = self.root / "align" / "material_masks"
        self.thumb_dir = self.root / "thumbnail_align" / "thumbnails"
        self._manifest = None

    # -- manifest ---------------------------------------------------------
    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            if self.manifest_file.is_file():
                try:
                    self._manifest = json.loads(self.manifest_file.read_text(encoding="utf-8"))
                except ValueError:
                    self._manifest = {}
            else:
                self._manifest = {}
        return self._manifest

    def save_manifest(self) -> None:
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_file.write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")

    def is_hand_edited(self, sec: str) -> bool:
        return bool(self.manifest.get(sec, {}).get("hand_edited"))

    def set_hand_edited(self, sec: str, flag: bool) -> None:
        self.manifest.setdefault(sec, {})["hand_edited"] = flag
        self.save_manifest()

    # -- inputs -----------------------------------------------------------
    def thumbnail_path(self, sec: str) -> Path | None:
        for ext in ("png", "jpg", "tif"):
            p = self.thumb_dir / f"{sec}.{ext}"
            if p.is_file():
                return p
        return None

    def thumbnail(self, sec: str) -> np.ndarray | None:
        p = self.thumbnail_path(sec)
        return imread(p) if p else None

    def thumbnail_mip(self) -> int:
        from .configs import ConfigStore
        cs = ConfigStore(self.project.configs_dir)
        return int(cs.get("thumbnail", "thumbnail_mip_level", 2))

    def sections_with_thumbnails(self) -> list[str]:
        return [s for s in self.project.section_names() if self.thumbnail_path(s)]

    def fold_mask(self, sec: str) -> np.ndarray | None:
        p = self.folds_dir / f"{sec}.png"
        return read_mask(p) > 0 if p.is_file() else None

    def fold_prob(self, sec: str) -> np.ndarray | None:
        p = self.folds_dir / f"{sec}_prob.png"
        return read_mask(p) if p.is_file() else None

    def fold_mip(self) -> int | None:
        p = self.folds_dir / "fold_stats.json"
        if p.is_file():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                for v in d.values():
                    return int(v.get("mip", 0))
            except ValueError:
                pass
        return None

    def structure_mask(self, sec: str) -> np.ndarray | None:
        p = self.structures_dir / f"{sec}.png"
        return read_mask(p) > 0 if p.is_file() else None

    def tissue_mask(self, sec: str) -> np.ndarray | None:
        p = self.tissue_dir / f"{sec}.png"
        return read_mask(p) > 0 if p.is_file() else None

    def material_mask(self, sec: str) -> np.ndarray | None:
        p = self.thumb_mask_dir / f"{sec}.png"
        return read_mask(p) if p.is_file() else None

    # -- the imaged area ----------------------------------------------------
    def roi_mask(self, sec: str) -> np.ndarray | None:
        """
        FEABAS's own mask of the stitched tile footprint (tissue where it says 0).

        The thumbnail step writes it to thumbnail_align/material_masks and never rewrites an
        existing file, so the first time the workbench is about to replace one it keeps a copy
        under masks/roi. That copy is the base for 'everything imaged is tissue': it comes from
        the tile bounding boxes, so a black fold reaching the section edge stays inside it.
        """
        p = self.roi_dir / f"{sec}.png"
        if p.is_file():
            return read_mask(p) == LABEL_DEFAULT
        src = self.thumb_mask_dir / f"{sec}.png"
        if src.is_file() and not self.manifest.get(sec, {}).get("composed"):
            m = read_mask(src)
            write_mask(p, m)
            return m == LABEL_DEFAULT
        return None

    def backup_roi(self, sec: str) -> bool:
        """Keep FEABAS's default mask before anything overwrites it. True if one is now stored."""
        return self.roi_mask(sec) is not None

    def tform_path(self, sec: str) -> Path:
        return self.root / "stitch" / "tform" / f"{sec}.h5"

    def default_tissue(self, sec: str) -> np.ndarray | None:
        """
        The imaged area, best source first:

        1. FEABAS's own mask, kept under masks/roi the first time we replaced one;
        2. the stitched tile boxes read from stitch/tform/<sec>.h5 - the same geometry
           FEABAS uses, so a black fold reaching the section edge stays inside;
        3. the convex hull of the image data, when there is no montage geometry at all.
        """
        roi = self.roi_mask(sec)
        if roi is not None:
            return roi
        img = self.thumbnail(sec)
        if img is None:
            return None
        roi = roi_from_tform(self.tform_path(sec), img.shape[:2])
        if roi is not None:
            write_mask(self.roi_dir / f"{sec}.png", np.where(roi, LABEL_DEFAULT, LABEL_EXCLUDE).astype(np.uint8))
            return roi
        return footprint_mask(img)

    # -- computation --------------------------------------------------------
    def compute_tissue(self, sec: str, params: TissueParams, save: bool = True) -> tuple[np.ndarray, float] | None:
        img = self.thumbnail(sec)
        if img is None:
            return None
        m, thr = detect_tissue(img, params, roi=self.default_tissue(sec))
        if save:
            write_mask(self.tissue_dir / f"{sec}.png", (m * 255).astype(np.uint8))
        return m, thr

    # -- external masks -------------------------------------------------------
    def _ext_meta(self) -> dict:
        p = self.external_dir / "mips.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                pass
        return {}

    def external_mip(self, kind: str) -> int | None:
        v = self._ext_meta().get(kind)
        return int(v) if v is not None else None

    def external_mask(self, kind: str, sec: str) -> np.ndarray | None:
        p = self.external_dir / kind / f"{sec}.png"
        return read_mask(p) if p.is_file() else None

    def hires_mip(self) -> int | None:
        """Finest mip at which some mask source exists below the thumbnail mip."""
        tm = self.thumbnail_mip()
        cands = [m for m in (self.external_mip("tissue"), self.external_mip("folds"), self.fold_mip()) if m is not None and m < tm]
        return min(cands) if cands else None

    @staticmethod
    def match_files_to_sections(folder: Path, sections: list[str]) -> dict[str, Path]:
        """Section name -> file. Exact stem match first, then by the trailing number, then by sorted order."""
        import re
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg"))
        by_stem = {p.stem: p for p in files}
        out: dict[str, Path] = {}
        for sec in sections:
            if sec in by_stem:
                out[sec] = by_stem[sec]
        if len(out) == len(sections) or not sections:
            return out

        def last_num(txt):
            m = re.findall(r"\d+", txt)
            return int(m[-1]) if m else None

        by_num: dict[int, Path] = {}
        for p in files:
            n = last_num(p.stem)
            if n is not None and n not in by_num:
                by_num[n] = p
        for sec in sections:
            if sec in out:
                continue
            n = last_num(sec)
            if n is not None and n in by_num:
                out[sec] = by_num[n]
        if out:
            return out
        if len(files) == len(sections):
            return dict(zip(sections, files))
        return out

    def import_external(self, folder: Path, mip: int, kind: str, sections: list[str],
                        progress=None, cancelled=None) -> dict:
        """
        Import masks made elsewhere. kind:
          'tissue'   nonzero = tissue          -> masks/tissue (thumbnail mip) + full copy in masks/external/tissue
          'folds'    nonzero = fold            -> masks/folds  (thumbnail mip) + full copy in masks/external/folds
          'material' FEABAS labels as they are -> thumbnail_align/material_masks (+ align/material_masks if finer)
        Masks are assumed to sit in the stitched-section space at the given mip; they are resampled (nearest)
        to the thumbnail size. Returns {imported: [...], missing: [...], mip: mip}.
        """
        tm = self.thumbnail_mip()
        files = self.match_files_to_sections(Path(folder), sections)
        imported, missing, warnings = [], [], []
        meta = self._ext_meta()
        for i, sec in enumerate(sections, 1):
            if cancelled and cancelled():
                break
            f = files.get(sec)
            img = self.thumbnail(sec)
            if f is None or img is None:
                missing.append(sec)
                if progress:
                    progress(i, len(sections), f"{sec}: no file/thumbnail")
                continue
            m = read_mask(f)
            th, tw = img.shape[:2]
            exp_h, exp_w = th * 2 ** (tm - mip), tw * 2 ** (tm - mip)
            if abs(m.shape[0] - exp_h) > 0.1 * exp_h or abs(m.shape[1] - exp_w) > 0.1 * exp_w:
                warnings.append(f"{sec}: mask {m.shape[1]}x{m.shape[0]} vs expected ~{exp_w:.0f}x{exp_h:.0f} at mip{mip}; check the mip level")
            small = resize_mask(m, (th, tw))
            if kind == "tissue":
                write_mask(self.tissue_dir / f"{sec}.png", (small > 0).astype(np.uint8) * 255)
                write_mask(self.external_dir / "tissue" / f"{sec}.png", (m > 0).astype(np.uint8) * 255)
            elif kind == "folds":
                write_mask(self.folds_dir / f"{sec}.png", (small > 0).astype(np.uint8) * 255)
                write_mask(self.external_dir / "folds" / f"{sec}.png", (m > 0).astype(np.uint8) * 255)
            else:
                for arr in (small, m):      # FEABAS needs at least a rim of excluded pixels
                    arr[0, :] = LABEL_EXCLUDE; arr[-1, :] = LABEL_EXCLUDE; arr[:, 0] = LABEL_EXCLUDE; arr[:, -1] = LABEL_EXCLUDE
                write_mask(self.thumb_mask_dir / f"{sec}.png", small)
                if mip < tm:
                    write_mask(self.align_mask_dir / f"{sec}.png", m)
                self.manifest.setdefault(sec, {}).update({"hand_edited": True, "imported": str(f), "composed": time.time()})
            imported.append(sec)
            if progress:
                progress(i, len(sections), sec)
        if kind in ("tissue", "folds"):
            meta[kind] = int(mip)
            self.external_dir.mkdir(parents=True, exist_ok=True)
            (self.external_dir / "mips.json").write_text(json.dumps(meta), encoding="utf-8")
        self.save_manifest()
        return {"imported": imported, "missing": missing, "mip": int(mip), "warnings": warnings, "kind": kind}

    # -- composition ----------------------------------------------------------
    def _tissue_at(self, sec: str, shape: tuple[int, int]) -> np.ndarray:
        """Tissue mask resampled to *shape*, preferring the full-resolution external copy."""
        ext = self.external_mask("tissue", sec)
        src = ext if ext is not None else self.tissue_mask(sec)
        if src is None:
            src = self.default_tissue(sec)
        if src is None:
            src = np.ones(shape, bool)
        return resize_mask(np.asarray(src).astype(np.uint8), shape) > 0

    def _folds_at(self, sec: str, shape: tuple[int, int]) -> np.ndarray | None:
        ext = self.external_mask("folds", sec)
        src = ext if ext is not None else self.fold_mask(sec)
        if src is None:
            return None
        return resize_mask(np.asarray(src).astype(np.uint8), shape) > 0

    def thumbnail_nm_per_px(self, sec: str) -> float:
        """Resolution of the thumbnail, measured against the montage rather than assumed."""
        v = self.project.state.volume
        img = self.thumbnail(sec)
        ext = montage_extent(self.tform_path(sec))
        if img is not None and ext and img.shape[1]:
            return float(v.pixel_size_nm) * ext[0] / float(img.shape[1])
        return float(v.pixel_size_nm) * 2 ** self.thumbnail_mip()

    def compose(self, sec: str, use_folds: bool = True, fold_label: int = LABEL_WRINKLE,
                structure_mode: str = "off", structure_dilate: int = 0,
                write_hires: bool = False, clip_folds: bool = True,
                margin_px: int = 0, margin_label: int = LABEL_EXCLUDE) -> dict | None:
        """
        Build the single material mask FEABAS reads (grey labels: 0 tissue, 255 outside,
        fold_label folds, 150 low-weight background). The thumbnail-mip version goes to
        thumbnail_align/material_masks; with write_hires a finer version at hires_mip()
        goes to align/material_masks (alignment_configs meshing.mask_mip_level must match).
        """
        img = self.thumbnail(sec)
        if img is None:
            return None
        self.backup_roi(sec)          # before this section's FEABAS default mask is overwritten
        shape = img.shape[:2]

        dropped = {"fold_px": 0, "outside_px": 0}

        def build(target_shape, dil_scale=1):
            tissue = self._tissue_at(sec, target_shape)
            extra = None
            m = int(round(margin_px * dil_scale))
            if m > 0:
                band = border_band(tissue, m)
                if margin_label == LABEL_EXCLUDE:
                    tissue = tissue & ~band      # the band is simply outside: not meshed, not rendered
                else:
                    extra = {int(margin_label): band}   # kept and rendered, but never matched
            folds = self._folds_at(sec, target_shape) if use_folds else None
            if folds is not None and target_shape == shape and clip_folds:
                # folds only get their label inside the tissue: count what the tissue mask swallows
                dropped["fold_px"] = int(folds.sum())
                dropped["outside_px"] = int((folds & ~tissue).sum())
            mat = compose_material_mask(tissue, folds, fold_label, extra=extra, clip_folds=clip_folds)
            if structure_mode != "off":
                st = self.structure_mask(sec)
                if st is not None:
                    st = resize_mask(st.astype(np.uint8), target_shape) > 0
                    mat = structure_material_mask(mat, st, int(structure_dilate * dil_scale))
            return mat, folds is not None

        mat, has_folds = build(shape)
        write_mask(self.thumb_mask_dir / f"{sec}.png", mat)
        info = {"section": sec, "stats": mask_stats(mat), "hires": False,
                "folds_outside_tissue": (dropped["outside_px"] / dropped["fold_px"]) if dropped["fold_px"] else 0.0}
        hm = self.hires_mip() if write_hires else None
        if hm is not None:
            f = 2 ** (self.thumbnail_mip() - hm)
            hi_shape = (shape[0] * f, shape[1] * f)
            mat_hi, _ = build(hi_shape, f)
            write_mask(self.align_mask_dir / f"{sec}.png", mat_hi)
            info["hires"] = True
            info["hires_mip"] = hm
        else:
            p = self.align_mask_dir / f"{sec}.png"
            if p.is_file() and not self.is_hand_edited(sec):
                p.unlink()
        self.manifest.setdefault(sec, {}).update({"composed": time.time(), "hand_edited": False,
                                                  "folds": has_folds, "structure_mode": structure_mode})
        self.save_manifest()
        return info

    def restore_default_mask(self, sec: str) -> bool:
        """
        Back to the FEABAS default: everything with data is tissue.

        Also drops what the workbench computed for this section (tissue mask, fold mask
        and probability, the higher-resolution copy) and clears the hand-edited flag, so
        a later 'Compose' starts from scratch instead of resurrecting the old result.
        """
        img = self.thumbnail(sec)
        if img is None:
            return False
        tissue = self.default_tissue(sec)
        if tissue is None:
            return False
        write_mask(self.thumb_mask_dir / f"{sec}.png", compose_material_mask(tissue))
        for p in (self.align_mask_dir / f"{sec}.png", self.tissue_dir / f"{sec}.png",
                  self.folds_dir / f"{sec}.png", self.folds_dir / f"{sec}_prob.png"):
            if p.is_file():
                p.unlink()
        self.manifest.setdefault(sec, {}).update({"hand_edited": False, "composed": time.time(),
                                                  "folds": False, "restored": True})
        self.save_manifest()
        return True
