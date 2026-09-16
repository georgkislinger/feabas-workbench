"""Window 4: thumbnails and material masks (tissue vs background, folds from the U-Net)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QSlider, QSplitter, QVBoxLayout, QWidget, QTabWidget)

from ...core.masks import (TissueParams, LABEL_WRINKLE, LABEL_EXCLUDE, LABEL_SOFT, bundled_fold_checkpoint,
                           resolve_fold_checkpoint, store_fold_checkpoint)
from ...core.maskstore import MaskStore
from ...core.images import colorize_labels, to_uint8
from ...core.steps import STEPS_BY_KEY
from ..widgets import (PathPicker, ImageView, SectionPicker, ConfigEditor, card, hint, form_row, spin, dspin, combo,
                       labelled, row_widget, Collapsible)
from ..widgets.steps_panel import StepsPanel
from ..threads import ThreadRunner
from .base import Page

# fp16 weights only (49 MB): identical detections to the 280 MB Lightning checkpoint it was exported
# from, fine-tunable, but with no optimizer state - the original training run cannot resume from it.
# Lives inside the package (resources/ is package data) so a wheel install ships it too. Projects
# store it as the sentinel "bundled" (core.masks.resolve_fold_checkpoint), never as this path.
DEFAULT_FOLD_CKPT = bundled_fold_checkpoint()


class MasksPage(Page):
    title = "Masks"
    subtitle = ("FEABAS needs to know where the section is (tissue vs. background) and where it is folded. Thumbnails "
                "come first (tab 1); the masks are built from them (tab 2) and written where FEABAS expects them.")
    key = "masks"

    def build(self) -> None:
        self.store: MaskStore | None = None
        self.runner = ThreadRunner(self)
        self.tabs = QTabWidget()
        self.body.addWidget(self.tabs)

        # ---- tab: thumbnails ------------------------------------------------
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Thumbnail settings")
        r = QHBoxLayout()
        self.th_mip = spin(0, 10, 2); self.th_hp = QCheckBox("high-pass filter (SE images)"); self.th_workers = spin(1, 128, 10)
        r.addWidget(labelled("thumbnail mip level", self.th_mip,
                             "Downsampling level of the thumbnails (each level halves the size). Aim for 500-2000 px "
                             "on the long side: the hint below computes it. Masks are made at this resolution."))
        self.th_hp.setToolTip("Suppress low-frequency shading before downsampling. Helps feature matching on "
                              "secondary-electron images; leave it off for backscatter images and when the "
                              "thumbnails are only one or two mip levels below the rendered sections.")
        r.addWidget(self.th_hp)
        r.addWidget(labelled("workers", self.th_workers, "Parallel processes for mip-mapping and thumbnailing."))
        apply_b = QPushButton("Apply"); apply_b.setObjectName("Primary"); r.addWidget(apply_b); r.addStretch(1)
        apply_b.setToolTip("Save these three settings to thumbnail_configs.yaml (and the mask mip level to the "
                           "alignment config).")
        lay.addLayout(r)
        self.th_hint = QLabel(""); self.th_hint.setObjectName("Hint"); self.th_hint.setWordWrap(True)
        lay.addWidget(self.th_hint)
        apply_b.clicked.connect(self._apply_thumb)
        tl.addWidget(f)
        f, lay = card("Step")
        self.steps = StepsPanel(self.ctx, ["thumbnail.downsample"], compact=True)
        self.steps.inspect_requested.connect(lambda _s: self.tabs.setCurrentIndex(1))
        lay.addWidget(self.steps)
        lay.addWidget(hint("Also builds the intermediate mip levels of the stitched sections that the fine alignment reads, "
                           "and FEABAS's default masks (everything imaged = tissue), which the next tab replaces."))
        tl.addWidget(f)
        tl.addStretch(1)
        self.tabs.addTab(t, "1. Thumbnails")

        # ---- tab: masks --------------------------------------------------------
        m = QWidget(); ml = QVBoxLayout(m); ml.setContentsMargins(6, 6, 6, 6)
        split = QSplitter()
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        self.sections = SectionPicker(checkable=True)
        ll.addWidget(self.sections, 1)
        split.addWidget(left)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        r = QHBoxLayout()
        self.ov_tissue = QCheckBox("tissue"); self.ov_folds = QCheckBox("folds"); self.ov_prob = QCheckBox("fold probability")
        self.ov_mat = QCheckBox("material mask"); self.ov_struct = QCheckBox("structures")
        self.ov_frame = QCheckBox("border margin preview")
        self.ov_tissue.setToolTip("The tissue mask (masks/tissue): everything NOT tissue is tinted red.")
        self.ov_folds.setToolTip("Detected folds (masks/folds) in yellow.")
        self.ov_prob.setToolTip("The U-Net's fold probability map, orange = likely fold.")
        self.ov_mat.setToolTip("The material mask FEABAS reads (thumbnail_align/material_masks): grey labels "
                               "coloured - excluded, wrinkle, soft, split.")
        self.ov_struct.setToolTip("Structures found by the YOLO-seg detector (Alignment page), green.")
        self.ov_frame.setToolTip("Shows the border margin from the tissue card as it will be written: cyan = soft, red = excluded.")
        self.ov_mat.setChecked(True)
        for w in (self.ov_tissue, self.ov_folds, self.ov_prob, self.ov_mat, self.ov_struct, self.ov_frame):
            r.addWidget(w); w.toggled.connect(self._show)
        r.addWidget(QLabel("opacity")); self.opacity = QSlider(Qt.Horizontal); self.opacity.setRange(10, 100); self.opacity.setValue(55); self.opacity.setMaximumWidth(120)
        self.opacity.setToolTip("Overlay opacity.")
        r.addWidget(self.opacity); self.opacity.valueChanged.connect(self._show)
        fit_b = QPushButton("fit"); fit_b.setToolTip("Zoom to fit the whole thumbnail."); fit_b.clicked.connect(lambda: self.view.fit()); r.addWidget(fit_b)
        rl.addLayout(r)
        self.view = ImageView(); self.view.setMinimumHeight(420)
        rl.addWidget(self.view, 1)
        self.view_info = QLabel(""); self.view_info.setObjectName("Hint"); self.view_info.setWordWrap(True)
        rl.addWidget(self.view_info)
        r = QHBoxLayout()
        self.split_btn = QPushButton("Split section: click 2 points"); self.split_btn.setCheckable(True)
        self.split_btn.setToolTip("Paint a split line (label 200) through a broken section so the two pieces can move "
                                  "independently: click the two end points of a line through the gap.")
        self.fiji_btn = QPushButton("Edit mask in Fiji"); self.reload_btn = QPushButton("Reload mask (mark hand-edited)")
        self.fiji_btn.setToolTip("Open this section's material mask in Fiji. Paint 0 = tissue, 255 = outside, "
                                 "50 = wrinkle, then save and press 'Reload mask'.")
        self.reload_btn.setToolTip("Re-read the material mask from disk and mark it hand-edited, so 'Compose' "
                                   "leaves it alone.")
        self.default_btn = QPushButton("Reset checked sections to FEABAS default")
        self.default_btn.setToolTip("Back to FEABAS's own mask (everything imaged = tissue). Also deletes the tissue "
                                    "and fold masks computed for the checked sections and clears the hand-edited flag.")
        self.rebuild_btn = QPushButton("Rebuild FEABAS masks…")
        self.rebuild_btn.setToolTip("Delete the material masks of the checked sections and let FEABAS write its own "
                                    "again from the tile bounding boxes (runs the thumbnail step). Use this if the "
                                    "workbench overwrote them before it kept a copy.")
        self.rebuild_btn.clicked.connect(self._rebuild_roi)
        # two short rows instead of one long one, so the page fits a 1500 px window
        r.addWidget(QLabel("hand editing:")); r.addWidget(self.split_btn); r.addWidget(self.fiji_btn); r.addWidget(self.reload_btn)
        r.addStretch(1)
        rl.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(QLabel("start over:")); r.addWidget(self.default_btn); r.addWidget(self.rebuild_btn)
        r.addStretch(1)
        rl.addLayout(r)
        split.addWidget(right)
        split.setSizes([220, 900])
        ml.addWidget(split, 1)
        self.sections.current_changed.connect(lambda _n: self._show())
        self.split_btn.toggled.connect(self._split_toggle)
        self.view.clicked.connect(self._view_clicked)
        self.fiji_btn.clicked.connect(self._edit_fiji)
        self.reload_btn.clicked.connect(self._reload_mask)
        self.default_btn.clicked.connect(self._restore_default)
        self._split_pts: list[tuple[float, float]] = []

        # tissue card: one method selector, and only the settings that method uses
        f, lay = card("Tissue vs. background")
        r = QHBoxLayout()
        self.ti_method = combo([("the stitched tile footprint is tissue (FEABAS default, recommended)", "all"),
                                ("detect a uniform frame (black / white) from the outside in", "border"),
                                ("fixed margins from the image edge (left / top / right / bottom)", "manual"),
                                ("auto: texture split only if a clear background/tissue split exists", "auto"),
                                ("local texture (resin vs. tissue)", "texture"),
                                ("intensity threshold", "intensity")], "all")
        self.ti_method.setToolTip("How the tissue outline is found. Everything outside it is excluded (255): not "
                                  "meshed, not matched, not rendered.")
        r.addWidget(QLabel("method")); r.addWidget(self.ti_method, 1)
        lay.addLayout(r)
        self.ti_method_hint = hint("")
        lay.addWidget(self.ti_method_hint)

        # -- border method
        self.ti_border_mode = combo([("black or white frame", "both"), ("black frame only", "black"),
                                     ("white frame only", "white"), ("any uniform grey (auto)", "auto")], "both")
        self.ti_border_mode.setToolTip("What counts as frame colour. 'black or white' handles the usual case of black "
                                       "padding around a white (saturated) rim. 'auto' takes any locally flat grey "
                                       "value - a frame of any brightness - and needs no threshold.")
        self.ti_border_black = spin(0, 255, 0); self.ti_border_white = spin(0, 255, 250); self.ti_border_tol = spin(0, 64, 4)
        self.ti_border_width = spin(0, 200, 15)
        self.ti_border_black_w = labelled("black: grey ≤", self.ti_border_black,
                                          "A pixel belongs to the black frame at or below this grey level. 0 = only "
                                          "pure black; raise it a little if the padding is not exactly 0.")
        self.ti_border_white_w = labelled("white: grey ≥", self.ti_border_white,
                                          "A pixel belongs to the white frame at or above this grey level. 250 leaves "
                                          "room for a slightly blurred rim; 255 = only pure white.")
        self.ti_border_tol_w = labelled("flatness tolerance", self.ti_border_tol,
                                        "auto only: a pixel is 'flat' when the grey range of its 3x3 neighbourhood is "
                                        "at most this. Real tissue is never flat over a large area.")
        self.ti_border_width_w = labelled("ignore streaks thinner than px", self.ti_border_width,
                                          "A fold of the frame colour that reaches the frame would be peeled off with "
                                          "it. Streaks thinner than this stay tissue (and can be labelled as folds "
                                          "below); slivers of tissue thinner than this are dropped.")
        self.ti_border_row = row_widget(labelled("frame colour", self.ti_border_mode), self.ti_border_black_w,
                                        self.ti_border_white_w, self.ti_border_tol_w, self.ti_border_width_w)
        lay.addWidget(self.ti_border_row)

        # -- manual method
        self.ti_left = spin(0, 100_000, 0, 10); self.ti_top = spin(0, 100_000, 0, 10)
        self.ti_right = spin(0, 100_000, 0, 10); self.ti_bottom = spin(0, 100_000, 0, 10)
        crop_tip = ("Margin from the {} edge of the thumbnail, in thumbnail pixels (the µm equivalent is shown on "
                    "the right). Everything inside the four margins - and inside the imaged footprint - is tissue. "
                    "The dashed preview updates as you type; press Preview/Compute to write the mask.")
        self.ti_manual_info = QLabel(""); self.ti_manual_info.setObjectName("Hint")
        self.ti_manual_row = row_widget(labelled("left", self.ti_left, crop_tip.format("left")),
                                        labelled("top", self.ti_top, crop_tip.format("top")),
                                        labelled("right", self.ti_right, crop_tip.format("right")),
                                        labelled("bottom", self.ti_bottom, crop_tip.format("bottom")),
                                        QLabel("px"), self.ti_manual_info)
        lay.addWidget(self.ti_manual_row)
        for w in (self.ti_left, self.ti_top, self.ti_right, self.ti_bottom):
            w.valueChanged.connect(self._manual_changed)

        # -- texture / intensity methods
        self.ti_window = spin(5, 301, 31, 2)
        self.ti_thr = QLineEdit(""); self.ti_thr.setPlaceholderText("auto"); self.ti_thr.setMaximumWidth(80)
        self.ti_invert = QCheckBox("dark tissue")
        self.ti_window_w = labelled("texture window px", self.ti_window,
                                    "Size of the neighbourhood over which the local grey-level variation is measured. "
                                    "Larger for coarse textures, smaller for fine ones.")
        self.ti_thr_w = labelled("threshold", self.ti_thr,
                                 "Manual threshold on the texture score (or on the grey level for the intensity "
                                 "method). Leave empty for automatic (Otsu).")
        self.ti_invert.setToolTip("Intensity method: tissue is darker than the background (inverted images).")
        self.ti_texture_row = row_widget(self.ti_window_w, self.ti_thr_w, self.ti_invert)
        lay.addWidget(self.ti_texture_row)

        # -- clean-up shared by every method that classifies pixels
        self.ti_min = spin(0, 10_000_000, 2000, 500); self.ti_holes = spin(0, 10_000_000, 5000, 500)
        self.ti_min_w = labelled("drop tissue pieces smaller than px", self.ti_min,
                                 "Connected pieces of tissue with fewer pixels than this are removed (debris, specks "
                                 "of resin classified as tissue). A piece has to be at least this big to count.")
        self.ti_holes_w = labelled("fill holes up to px", self.ti_holes,
                                   "Background holes enclosed by tissue that are smaller than this become tissue "
                                   "again (pale patches, vacuoles). Larger enclosed holes stay excluded.")
        self.ti_cleanup_row = row_widget(self.ti_min_w, self.ti_holes_w)
        lay.addWidget(self.ti_cleanup_row)

        # -- border margin: applied to every method when the material mask is composed
        r = QHBoxLayout()
        self.ti_margin = spin(0, 500, 0)
        self.ti_margin_label = combo([("keep it, but never match there (soft, 100)", LABEL_SOFT),
                                      ("cut it out (exclude, 255)", LABEL_EXCLUDE)], LABEL_SOFT)
        self.ti_margin_info = QLabel(""); self.ti_margin_info.setObjectName("Hint")
        self.ti_margin.setToolTip("A ring of this width just inside the section outline, following its shape. "
                                  "The edge of a montage is where the tissue mask is least certain and where "
                                  "matching is least reliable. 0 = no margin. Applied when the material mask is "
                                  "composed, so it shows up after 'Compose' (tick 'border margin preview' to see it).")
        self.ti_margin_label.setToolTip("soft (100) is meshed and rendered but its stiffness is below FEABAS's "
                                        "matching threshold, so no match point is placed there and you keep the "
                                        "image data. exclude (255) is not meshed and NOT RENDERED: that strip is "
                                        "missing from the aligned volume.")
        r.addWidget(QLabel("border margin")); r.addWidget(self.ti_margin); r.addWidget(QLabel("px"))
        r.addWidget(self.ti_margin_label, 1); r.addWidget(self.ti_margin_info)
        r.addStretch(1)
        lay.addLayout(r)
        self.ti_margin.valueChanged.connect(self._update_margin_info)
        self.ti_margin.valueChanged.connect(self._frame_changed)
        self.ti_margin_label.currentIndexChanged.connect(self._frame_changed)

        # -- holes inside the section, black and/or white
        r = QHBoxLayout()
        self.ti_dark = QCheckBox("also exclude black regions inside the section, grey ≤")
        self.ti_dark_max = spin(0, 255, 0)
        self.ti_bright = QCheckBox("white regions, grey ≥")
        self.ti_bright_min = spin(0, 255, 255)
        self.ti_dark_min = spin(0, 1_000_000, 24, 8)
        self.ti_dark.setToolTip("Off by default: a fold is black but it is still tissue, and excluding it removes it "
                                "from the mesh and swallows its fold label. Turn this on only where the black areas "
                                "carry no data at all (holes, missing tiles).")
        self.ti_dark_max.setToolTip("A pixel counts as black at or below this grey level.")
        self.ti_bright.setToolTip("Also exclude saturated white regions inside the section: empty resin, burnt "
                                  "spots, support film. Same caveat as for black regions.")
        self.ti_bright_min.setToolTip("A pixel counts as white at or above this grey level.")
        self.ti_dark_min.setToolTip("Only black/white regions of at least this many pixels are excluded; smaller "
                                    "ones are normal tissue detail (single saturated pixels, small vesicles).")
        r.addWidget(self.ti_dark); r.addWidget(self.ti_dark_max)
        r.addWidget(self.ti_bright); r.addWidget(self.ti_bright_min)
        r.addWidget(QLabel("at least px")); r.addWidget(self.ti_dark_min); r.addStretch(1)
        lay.addLayout(r)

        r = QHBoxLayout()
        b1 = QPushButton("Preview on current section"); b2 = QPushButton("Compute for checked sections"); b2.setObjectName("Primary")
        b1.setToolTip("Compute the tissue mask for the section in the viewer only and show it (red = not tissue).")
        b2.setToolTip("Compute and save the tissue mask (masks/tissue) for every checked section.")
        r.addWidget(b1); r.addWidget(b2); r.addStretch(1)
        lay.addLayout(r)
        lay.addWidget(hint("The tissue mask says where the section is; 'Compose' below turns it into the material mask "
                           "FEABAS reads and adds the border margin and the folds. Black areas inside the section stay "
                           "tissue with every method: mark those as folds below instead. Better masks from other tools "
                           "can be imported further down."))
        b1.clicked.connect(lambda: self._tissue(preview=True))
        b2.clicked.connect(lambda: self._tissue(preview=False))
        self.ti_method.currentIndexChanged.connect(self._ti_method_changed)
        self.ti_border_mode.currentIndexChanged.connect(self._ti_method_changed)
        self._ti_method_changed()
        ml.addWidget(f)

        # folds card
        f, lay = card("Folds & wrinkles")
        r = QHBoxLayout()
        self.fold_method = combo([("dark regions (threshold, no model)", "dark"),
                                  ("U-Net checkpoint (deep-learning environment)", "unet")], "dark")
        self.fold_method.setToolTip("Folds and tears are usually black in the rendered montage, so a threshold finds "
                                    "them in seconds without a model. The U-Net is for folds that are grey or "
                                    "textured rather than black; it runs in the deep-learning environment.")
        self.fold_dark_max = spin(0, 255, 0)
        self.fold_dark_w = labelled("grey ≤", self.fold_dark_max,
                                    "A pixel counts as fold at or below this grey level. 0 = pure black; raise to "
                                    "3-10 if fold edges are not exactly black after downsampling.")
        r.addWidget(QLabel("method")); r.addWidget(self.fold_method, 1)
        r.addWidget(self.fold_dark_w); r.addStretch(1)
        lay.addLayout(r)
        self.fold_ckpt = PathPicker("file", "checkpoint (.ckpt / .pt)", "Checkpoints (*.ckpt *.pt *.pth)")
        self.fold_ckpt_row = form_row("Checkpoint", self.fold_ckpt,
                                      "The fold U-Net weights. The bundled checkpoint is used unless you pick one you "
                                      "trained on the 'Train fold model' tab.")
        lay.addWidget(self.fold_ckpt_row)
        self.fold_src = combo([("thumbnails", "thumb")], "thumb")
        self.fold_thr = dspin(0.05, 0.95, 0.5, 0.05, 2); self.fold_tile = spin(256, 4096, 1024, 256); self.fold_ov = spin(0, 1024, 128, 32)
        self.fold_unet_row = row_widget(
            labelled("run on", self.fold_src, "The thumbnails, or a finer mip of the stitched sections (PNG-tile "
                                              "render driver) when folds are only a few pixels wide in the thumbnail; "
                                              "higher-resolution masks are then written to align/material_masks."),
            labelled("threshold", self.fold_thr, "Probability above which a pixel counts as fold. Lower catches "
                                                 "fainter folds (more false positives); check with the 'fold "
                                                 "probability' overlay."))
        lay.addWidget(self.fold_unet_row)
        self.fold_adv = Collapsible("Advanced U-Net settings")
        self.fold_adv.addWidget(row_widget(
            labelled("tile px", self.fold_tile, "Tiled inference geometry: lower on small GPUs."),
            labelled("overlap px", self.fold_ov, "Overlap between inference tiles; raise if tile edges show in the mask.")))
        lay.addWidget(self.fold_adv)
        self.fold_min = spin(0, 1_000_000, 50, 10); self.fold_dil = spin(0, 100, 2)
        lay.addWidget(row_widget(
            labelled("drop fold blobs smaller than px", self.fold_min, "Fold regions with fewer pixels than this are speckle and are dropped."),
            labelled("grow folds by px", self.fold_dil, "Dilate the fold regions a little so the mesh can absorb the "
                                                        "compression around a fold.")))
        r = QHBoxLayout()
        b3 = QPushButton("Detect folds on checked sections"); b3.setObjectName("Primary")
        b3p = QPushButton("Preview on current section")
        b3.setToolTip("Run the chosen method on every checked section and save the fold masks (masks/folds).")
        b3p.setToolTip("Run the chosen method on the section in the viewer only and show the result in yellow.")
        r.addWidget(b3); r.addWidget(b3p); r.addStretch(1)
        lay.addLayout(r)
        lay.addWidget(hint("A black region is either a fold (meshed as fold material, the default) or missing data "
                           "(exclude it in the tissue card): fold labels are only painted inside the tissue mask, so if "
                           "both are on, exclusion wins."))
        b3.clicked.connect(lambda: self._detect_folds(preview=False))
        b3p.clicked.connect(lambda: self._detect_folds(preview=True))
        self.fold_method.currentIndexChanged.connect(self._fold_method_changed)
        self._fold_method_changed()
        ml.addWidget(f)

        # import card
        f, lay = card("Import masks made elsewhere")
        lay.addWidget(hint("Point at a folder with one mask per section (named like the sections, e.g. s0035.png, or "
                           "carrying the section number, or in section order) and say at which mip level of the stitched "
                           "sections they were made. They are resampled to the thumbnail mip; the full-resolution copy is "
                           "kept for the higher-resolution material masks."))
        r = QHBoxLayout()
        self.imp_dir = PathPicker("dir", "folder with mask images")
        self.imp_mip = spin(0, 10, 0); self.imp_kind = combo([("tissue mask (non-zero = tissue)", "tissue"),
                                                              ("fold mask (non-zero = fold)", "folds"),
                                                              ("complete FEABAS material mask (grey labels as they are)", "material")], "tissue")
        self.imp_kind.setToolTip("What the images are. Tissue and fold masks go through 'Compose'; a complete "
                                 "material mask is written for FEABAS as it is.")
        r.addWidget(self.imp_dir, 1)
        r.addWidget(labelled("made at mip", self.imp_mip, "Mip level of the stitched sections the masks were drawn on "
                                                          "(0 = full resolution). They are resampled to the thumbnail mip."))
        r.addWidget(self.imp_kind, 1)
        b_imp = QPushButton("Import for checked sections"); b_imp.setObjectName("Primary"); r.addWidget(b_imp)
        lay.addLayout(r)
        b_imp.clicked.connect(self._import_masks)
        ml.addWidget(f)

        # compose card
        f, lay = card("Write material masks for FEABAS")
        r = QHBoxLayout()
        self.use_folds = QCheckBox("include folds as"); self.use_folds.setChecked(True)
        self.use_folds.setToolTip("Whether the detected folds enter the material mask at all.")
        self.fold_label = combo([("wrinkle (50): expands freely, resists compression", LABEL_WRINKLE),
                                 ("exclude (255): cut out of the mesh", LABEL_EXCLUDE), ("soft (100): very soft material", LABEL_SOFT)], LABEL_WRINKLE)
        self.fold_label.setToolTip("The material a fold becomes. wrinkle is the usual choice; exclude only for "
                                   "genuinely destroyed regions (they will be missing from the rendered volume).")
        self.hires = QCheckBox("also write higher-resolution masks (align/material_masks)")
        self.hires.setToolTip("Additionally write masks at the finest mip for which a mask source exists (fold "
                              "detection on a finer mip, or imported masks) and point the alignment config at them.")
        r.addWidget(self.use_folds); r.addWidget(self.fold_label, 1); r.addWidget(self.hires)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.clip_folds = combo([("outside the imaged area wins: folds are clipped to the tissue (recommended)", True),
                                 ("the fold label wins: keep it outside the imaged area too", False)], True)
        self.clip_folds.setToolTip("Fold detection also fires on the black border around the section. Clipping keeps "
                                   "that border excluded (255), which is what FEABAS expects; the other way round the "
                                   "border becomes fold material, which is meshed, rendered and - for the wrinkle "
                                   "material - even matched.")
        r.addWidget(QLabel("where folds and the outside overlap")); r.addWidget(self.clip_folds, 1)
        lay.addLayout(r)
        r = QHBoxLayout()
        b4 = QPushButton("Compose for checked sections"); b4.setObjectName("Primary")
        b4.setToolTip("Write the material mask FEABAS reads (thumbnail_align/material_masks) for every checked "
                      "section: tissue mask + border margin + folds. Sections without a tissue mask get one with "
                      "the settings above first.")
        self.skip_edited = QCheckBox("keep hand-edited masks"); self.skip_edited.setChecked(True)
        self.skip_edited.setToolTip("Skip sections whose mask you edited by hand (marked * in the list).")
        r.addWidget(b4); r.addWidget(self.skip_edited); r.addStretch(1)
        lay.addLayout(r)
        self.compose_info = QLabel(""); self.compose_info.setObjectName("Hint"); self.compose_info.setWordWrap(True)
        lay.addWidget(self.compose_info)
        b4.clicked.connect(self._compose)
        ml.addWidget(f)
        self.tabs.addTab(m, "2. Masks")

        # ---- tab: train fold model ----------------------------------------------
        tr = QWidget(); trl = QVBoxLayout(tr); trl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Train / fine-tune the fold U-Net")
        lay.addWidget(hint("Dataset folder with images/ and masks/ (same filenames; mask > 0 = fold). Crops of the given "
                           "patch size are sampled from the full images, so any image size works. Starting from the current "
                           "checkpoint fine-tunes it on your data."))
        self.tr_data = PathPicker("dir", "dataset folder")
        lay.addWidget(form_row("Dataset", self.tr_data))
        r = QHBoxLayout()
        self.tr_name = QLineEdit("folds_run1"); self.tr_name.setMaximumWidth(180)
        self.tr_epochs = spin(1, 1000, 50); self.tr_patch = spin(64, 1024, 256, 64); self.tr_batch = spin(1, 64, 16)
        self.tr_init = QCheckBox("start from the checkpoint on the Masks tab"); self.tr_init.setChecked(True)
        self.tr_init.setToolTip("On = fine-tune the current checkpoint on your data (converges fast). Off = train "
                                "from scratch, which needs much more data. The bundled checkpoint can be fine-tuned: "
                                "it holds every model weight (fp16); only the original run's optimizer state was "
                                "left out, and fine-tuning starts a fresh optimizer anyway. Resuming that original "
                                "training run would need the full 280 MB Lightning checkpoint, which is not bundled.")
        r.addWidget(labelled("run name", self.tr_name, "Folder name under models/folds/."))
        r.addWidget(labelled("epochs", self.tr_epochs, "Passes over the sampled crops. Fine-tuning needs far fewer than training from scratch."))
        r.addWidget(labelled("patch px", self.tr_patch, "Size of the random crops sampled from the images."))
        r.addWidget(labelled("batch", self.tr_batch, "Crops per training step; lower on small GPUs."))
        r.addWidget(self.tr_init); r.addStretch(1)
        lay.addLayout(r)
        r = QHBoxLayout()
        b5 = QPushButton("Train"); b5.setObjectName("Primary")
        self.tr_models = QComboBox(); self.tr_models.setMinimumWidth(260)
        b6 = QPushButton("Use selected model")
        r.addWidget(b5); r.addSpacing(20); r.addWidget(QLabel("trained models")); r.addWidget(self.tr_models); r.addWidget(b6); r.addStretch(1)
        lay.addLayout(r)
        b5.clicked.connect(self._train)
        b6.clicked.connect(lambda: self.fold_ckpt.setText(self.tr_models.currentData() or ""))
        trl.addWidget(f)
        trl.addStretch(1)
        self.tabs.addTab(tr, "Train fold model (optional)")

        # ---- tab: settings -----------------------------------------------------
        self.editor = ConfigEditor()
        self.editor.changed.connect(self._editor_changed)
        self.tabs.addTab(self.editor, "Thumbnail settings")
        self.ctx.jobs.job_finished.connect(self._job_finished)

    # ------------------------------------------------------------------
    def on_project_changed(self, project) -> None:
        if project is None:
            self.store = None
            self.editor.set_doc(None)
            return
        self.store = MaskStore(project)
        self.editor.set_doc(self.ctx.configs["thumbnail"])
        cs = self.ctx.configs
        self.th_mip.setValue(int(cs.get("thumbnail", "thumbnail_mip_level", 2)))
        self.th_hp.setChecked(bool(cs.get("thumbnail", "downsample.thumbnail_highpass", True)))
        self.th_workers.setValue(int(cs.get("thumbnail", "downsample.num_workers", 10)))
        ms = project.state.masks
        self.fold_ckpt.setText(resolve_fold_checkpoint(ms.get("fold_ckpt")))
        tp = TissueParams.from_dict(ms.get("tissue"))
        self.ti_method.setCurrentIndex(max(0, self.ti_method.findData(tp.method)))
        self.ti_window.setValue(tp.window); self.ti_min.setValue(tp.min_component_px); self.ti_holes.setValue(tp.fill_holes_px)
        self.ti_invert.setChecked(tp.invert_intensity)
        self.ti_thr.setText("" if tp.threshold is None else str(tp.threshold))
        self.ti_dark.setChecked(tp.exclude_dark); self.ti_dark_max.setValue(tp.dark_max); self.ti_dark_min.setValue(tp.dark_min_px)
        self.ti_bright.setChecked(tp.exclude_bright); self.ti_bright_min.setValue(tp.bright_min)
        self.ti_border_mode.setCurrentIndex(max(0, self.ti_border_mode.findData(tp.border_mode)))
        self.ti_border_black.setValue(tp.border_black_max); self.ti_border_white.setValue(tp.border_white_min)
        self.ti_border_tol.setValue(tp.border_tol); self.ti_border_width.setValue(tp.border_min_width)
        for w, v in ((self.ti_left, tp.crop_left), (self.ti_top, tp.crop_top), (self.ti_right, tp.crop_right), (self.ti_bottom, tp.crop_bottom)):
            w.blockSignals(True); w.setValue(int(v)); w.blockSignals(False)
        self._ti_method_changed()
        # projects made before the margin existed carry it as the tissue mask's erode radius
        self.ti_margin.setValue(int(ms.get("margin_px", tp.erode)))
        self.ti_margin_label.setCurrentIndex(max(0, self.ti_margin_label.findData(int(ms.get("margin_label", LABEL_SOFT)))))
        self.fold_method.setCurrentIndex(max(0, self.fold_method.findData(ms.get("fold_method", "dark"))))
        self.fold_dark_max.setValue(int(ms.get("fold_dark_max", 0)))
        self._fold_method_changed()
        self.fold_thr.setValue(float(ms.get("fold_threshold", 0.5))); self.fold_min.setValue(int(ms.get("fold_min_area", 50)))
        self.fold_dil.setValue(int(ms.get("fold_dilate", 2)))
        self.use_folds.setChecked(bool(ms.get("use_folds", True)))
        self.fold_label.setCurrentIndex(max(0, self.fold_label.findData(int(ms.get("fold_label", LABEL_WRINKLE)))))
        self.hires.setChecked(bool(ms.get("hires", False)))
        self.clip_folds.setCurrentIndex(0 if ms.get("clip_folds", True) else 1)
        self._refresh_models()
        self._refresh_sections()
        self._update_hint()
        self._update_margin_info()

    def on_state_changed(self) -> None:
        self.steps.refresh(self.ctx.scan())
        self._refresh_sections()
        self._refresh_fold_sources()

    def on_running_changed(self, running: bool) -> None:
        self.steps.refresh(self.ctx.scan())

    def on_shown(self) -> None:
        self._refresh_sections()

    # -- thumbnails ------------------------------------------------------
    def _update_hint(self) -> None:
        p = self.project
        if not p:
            return
        v = p.state.volume
        mip = self.th_mip.value()
        res = v.pixel_size_nm * 2 ** mip
        w = v.tile_w * max(1, v.grid_cols) * 0.95 / 2 ** mip
        h = v.tile_h * max(1, v.grid_rows) * 0.95 / 2 ** mip
        self.th_hint.setText(f"mip{mip}: {res:g} nm/px, thumbnails about {w:.0f} × {h:.0f} px (FEABAS works best with roughly 500–2000 px thumbnails; larger ones make coarse matching slow). "
                             f"Masks are made at this resolution.")

    def _apply_thumb(self) -> None:
        cs = self.ctx.configs
        if not cs:
            return
        from ...core.configs import set_thumbnail_mip
        if set_thumbnail_mip(cs, self.th_mip.value(), self.th_hp.isChecked()):
            self.th_hp.setChecked(False)
            self.warn("thumbnail mip 0: the high-pass filter is switched off - FEABAS builds it from a coarser "
                      "mip level, which does not exist at mip 0")
        cs.set("thumbnail", "downsample.num_workers", self.th_workers.value())
        cs.save()
        self.editor.rebuild()
        self._update_hint()
        self.info("thumbnail settings saved")
        self.ctx.state_changed.emit()

    def _editor_changed(self) -> None:
        if self.ctx.configs:
            self.ctx.configs.save("thumbnail")
            self.ctx.state_changed.emit()

    # -- sections / viewer ----------------------------------------------
    def _refresh_sections(self) -> None:
        if not self.store:
            self.sections.set_sections([])
            return
        names = self.project.section_names()
        status = {}
        for n in names:
            bits = []
            if self.store.thumbnail_path(n):
                bits.append("thumb")
            if (self.store.tissue_dir / f"{n}.png").is_file():
                bits.append("tissue")
            if (self.store.folds_dir / f"{n}.png").is_file():
                bits.append("folds")
            if (self.store.thumb_mask_dir / f"{n}.png").is_file():
                bits.append("mask" + ("*" if self.store.is_hand_edited(n) else ""))
            status[n] = " ".join(bits)
        self.sections.set_sections(names, status)
        self._show()

    def _refresh_fold_sources(self) -> None:
        cur = self.fold_src.currentData()
        self.fold_src.blockSignals(True)
        self.fold_src.clear()
        self.fold_src.addItem("thumbnails", "thumb")
        if self.project:
            base = self.project.root / "stitched_sections"
            for d in sorted(base.glob("mip*")) if base.is_dir() else []:
                if d.name[3:].isdigit():
                    self.fold_src.addItem(f"stitched sections {d.name}", int(d.name[3:]))
        self.fold_src.blockSignals(False)
        i = self.fold_src.findData(cur)
        self.fold_src.setCurrentIndex(max(0, i))

    def _show(self) -> None:
        sec = self.sections.current()
        if not self.store or not sec:
            self.view.clear()
            return
        img = self.store.thumbnail(sec)
        if img is None:
            self.view.clear()
            self.view_info.setText(f"{sec}: no thumbnail yet (run the Thumbnails step)")
            return
        self.view.set_image(to_uint8(img), fit=(self.view.level0_size() != (img.shape[1], img.shape[0])))
        self.view.clear_overlays()
        a = int(255 * self.opacity.value() / 100)
        h, w = img.shape[:2]
        info = [f"{sec}: {w}×{h} px"]
        if self.ov_mat.isChecked():
            m = self.store.material_mask(sec)
            if m is not None:
                m = self._fit(m, (h, w))
                self.view.set_overlay("mat", colorize_labels(m, alpha=a))
                from ...core.masks import mask_stats
                st = mask_stats(m)
                info.append(f"material: tissue {st['tissue_pct']:.1f}% · excluded {st['exclude_pct']:.1f}% · wrinkle {st['wrinkle_pct']:.2f}%")
                if self.store.is_hand_edited(sec):
                    info.append("(hand-edited)")
        if self.ov_tissue.isChecked():
            t = self.store.tissue_mask(sec)
            if t is not None:
                rgba = np.zeros((h, w, 4), np.uint8); t = self._fit(t.astype(np.uint8), (h, w)) > 0
                rgba[~t] = (230, 60, 60, a)
                self.view.set_overlay("tissue", rgba)
        if self.ov_folds.isChecked():
            fm = self.store.fold_mask(sec)
            if fm is not None:
                fm = self._fit(fm.astype(np.uint8), (h, w)) > 0
                rgba = np.zeros((h, w, 4), np.uint8); rgba[fm] = (250, 220, 40, a)
                self.view.set_overlay("folds", rgba)
                info.append(f"folds {100 * fm.mean():.2f}%")
        if self.ov_prob.isChecked():
            pr = self.store.fold_prob(sec)
            if pr is not None:
                pr = self._fit(pr, (h, w))
                rgba = np.zeros((h, w, 4), np.uint8); rgba[..., 0] = 255; rgba[..., 1] = 120; rgba[..., 3] = (pr.astype(np.uint16) * a // 255).astype(np.uint8)
                self.view.set_overlay("prob", rgba)
        if self.ov_struct.isChecked():
            s = self.store.structure_mask(sec)
            if s is not None:
                s = self._fit(s.astype(np.uint8), (h, w)) > 0
                rgba = np.zeros((h, w, 4), np.uint8); rgba[s] = (60, 220, 120, a)
                self.view.set_overlay("struct", rgba)
        if self.ov_frame.isChecked():
            from ...core.masks import border_band, padding_mask
            margin = self.ti_margin.value()
            t = self.store.tissue_mask(sec)
            t = self._fit(t.astype(np.uint8), (h, w)) > 0 if t is not None else ~padding_mask(img)
            band = border_band(t, margin)
            soft = int(self.ti_margin_label.currentData()) == LABEL_SOFT
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[band] = (60, 200, 230, min(255, a + 60)) if soft else (230, 60, 60, min(255, a + 60))
            self.view.set_overlay("frame", rgba)
            nm = self._thumb_nm()
            info.append(f"margin {margin} px" + (f" ≈ {margin * nm / 1000:.1f} µm" if nm else "") +
                        f" → {'soft (kept, not matched)' if soft else 'excluded'}; {100 * band.mean():.1f}% of the image"
                        + ("" if self.store.tissue_mask(sec) is not None else " (from the imaged footprint; compute tissue for the exact outline)"))
        self.view_info.setText("   ".join(info))
        self._manual_changed()

    def _frame_changed(self, *_a) -> None:
        if self.ov_frame.isChecked():
            self._show()

    def _thumb_nm(self) -> float:
        try:
            return float(self.project.state.volume.pixel_size_nm) * 2 ** self.store.thumbnail_mip()
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _fit(m: np.ndarray, shape) -> np.ndarray:
        from ...core.masks import resize_mask
        return resize_mask(m, shape)

    # -- tissue -----------------------------------------------------------
    def _update_margin_info(self) -> None:
        """The margin is in thumbnail pixels; say what that is in the sample."""
        n = self.ti_margin.value()
        if not self.store or not n:
            self.ti_margin_info.setText("")
            return
        sec = self.sections.current() or (self.sections.names()[:1] or [""])[0]
        nm = self.store.thumbnail_nm_per_px(sec) if sec else 0.0
        self.ti_margin_info.setText(f"= {n * nm / 1000:.1f} µm ({nm:.0f} nm per thumbnail pixel)" if nm else "")

    def _tissue_params(self) -> TissueParams:
        thr = self.ti_thr.text().strip()
        # the margin is applied when the material mask is composed, so the stored tissue mask
        # stays the full footprint and the band can be soft instead of excluded
        return TissueParams(window=self.ti_window.value(), min_component_px=self.ti_min.value(), fill_holes_px=self.ti_holes.value(),
                            erode=0, method=self.ti_method.currentData(), invert_intensity=self.ti_invert.isChecked(),
                            threshold=float(thr) if thr else None, exclude_dark=self.ti_dark.isChecked(),
                            dark_max=self.ti_dark_max.value(), dark_min_px=self.ti_dark_min.value(),
                            exclude_bright=self.ti_bright.isChecked(), bright_min=self.ti_bright_min.value(),
                            border_mode=self.ti_border_mode.currentData(), border_black_max=self.ti_border_black.value(),
                            border_white_min=self.ti_border_white.value(), border_tol=self.ti_border_tol.value(),
                            border_min_width=self.ti_border_width.value(),
                            crop_left=self.ti_left.value(), crop_top=self.ti_top.value(),
                            crop_right=self.ti_right.value(), crop_bottom=self.ti_bottom.value())

    TISSUE_METHOD_HINTS = {
        "all": "Everything the stitched tiles cover is tissue and only what lies outside is excluded: the same mask "
               "FEABAS builds from the tile bounding boxes. Nothing to set. Use it whenever the section fills the "
               "imaged area.",
        "border": "For sections that sit inside a uniform frame - black padding, a saturated white rim of empty "
                  "resin or support film, or both. The frame is peeled off from the outside in, layer by layer, "
                  "until real texture is reached; frames that do not touch the image edge themselves are found too.",
        "manual": "When the frame is regular, simply say how far it reaches in from each side of the thumbnail. "
                  "The dashed rectangle in the viewer shows the result before anything is written.",
        "auto": "Splits by local texture only when the section clearly contains two populations (textured tissue "
                "and flat background); otherwise everything imaged is tissue. For mixed stacks.",
        "texture": "Always splits by local grey-level variation: tissue is textured, resin is flat. Careful on "
                   "high-pass filtered thumbnails, which can split tissue by texture.",
        "intensity": "Plain grey-level threshold on the thumbnail: for a bright, uniform background (tick 'dark "
                     "tissue' for the opposite polarity).",
    }

    def _ti_method_changed(self, *_a) -> None:
        """Show only the settings the chosen method uses."""
        m = self.ti_method.currentData()
        self.ti_method_hint.setText(self.TISSUE_METHOD_HINTS.get(m, ""))
        self.ti_border_row.setVisible(m == "border")
        bm = self.ti_border_mode.currentData()
        self.ti_border_black_w.setVisible(bm in ("both", "black"))
        self.ti_border_white_w.setVisible(bm in ("both", "white"))
        self.ti_border_tol_w.setVisible(bm == "auto")
        self.ti_manual_row.setVisible(m == "manual")
        self.ti_texture_row.setVisible(m in ("auto", "texture", "intensity"))
        self.ti_window_w.setVisible(m in ("auto", "texture"))
        self.ti_invert.setVisible(m == "intensity")
        self.ti_cleanup_row.setVisible(m in ("border", "auto", "texture", "intensity"))
        self._manual_changed()

    def _manual_changed(self, *_a) -> None:
        """Live preview of the fixed margins as a dashed frame; nothing is written."""
        if not hasattr(self, "view"):
            return
        sec = self.sections.current() if self.store else None
        img = self.store.thumbnail(sec) if sec else None
        if self.ti_method.currentData() != "manual" or img is None:
            self.view.set_overlay("manual", None)
            self.ti_manual_info.setText("")
            return
        from ...core.masks import manual_bounds
        h, w = img.shape[:2]
        inside = manual_bounds((h, w), self.ti_left.value(), self.ti_top.value(), self.ti_right.value(), self.ti_bottom.value())
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[~inside] = (230, 60, 60, 70)
        edge = inside ^ (np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) & np.roll(inside, -1, 1))
        rgba[edge & inside] = (255, 255, 255, 230)
        self.view.set_overlay("manual", rgba)
        nm = self._thumb_nm()
        if nm:
            self.ti_manual_info.setText("= %.1f / %.1f / %.1f / %.1f µm" % tuple(v * nm / 1000 for v in (
                self.ti_left.value(), self.ti_top.value(), self.ti_right.value(), self.ti_bottom.value())))
        else:
            self.ti_manual_info.setText("")

    def _save_mask_settings(self) -> None:
        if not self.project:
            return
        ms = self.project.state.masks
        ms["tissue"] = self._tissue_params().to_dict()
        ms["fold_ckpt"] = store_fold_checkpoint(self.fold_ckpt.text())
        ms["fold_threshold"] = self.fold_thr.value(); ms["fold_min_area"] = self.fold_min.value(); ms["fold_dilate"] = self.fold_dil.value()
        ms["fold_method"] = self.fold_method.currentData(); ms["fold_dark_max"] = self.fold_dark_max.value()
        ms["use_folds"] = self.use_folds.isChecked(); ms["fold_label"] = int(self.fold_label.currentData()); ms["hires"] = self.hires.isChecked()
        ms["clip_folds"] = bool(self.clip_folds.currentData())
        ms["margin_px"] = self.ti_margin.value(); ms["margin_label"] = int(self.ti_margin_label.currentData())
        self.project.save()

    def _tissue(self, preview: bool) -> None:
        if not self.store:
            return
        self._save_mask_settings()
        params = self._tissue_params()
        secs = [self.sections.current()] if preview else self.sections.checked()
        secs = [s for s in secs if s and self.store.thumbnail_path(s)]
        if not secs:
            QMessageBox.information(self, "Tissue", "No thumbnails for the selected sections yet.")
            return

        def work(progress=None, cancelled=None):
            out = {}
            for i, s in enumerate(secs, 1):
                if cancelled and cancelled():
                    break
                r = self.store.compute_tissue(s, params, save=True)
                if r:
                    out[s] = r[1]
                progress(i, len(secs), s)
            return out

        self._start_thread(work, "tissue masks", after=lambda res: (self.ov_tissue.setChecked(True), self._refresh_sections()))

    def _start_thread(self, fn, label: str, after=None) -> None:
        if self.runner.running:
            self.warn("already computing")
            return
        self.compose_info.setText(f"{label}: working…")

        def prog(d, t, m):
            self.compose_info.setText(f"{label}: {d}/{t} {m}")

        def done(res, err):
            if err:
                self.error(f"{label} failed: {err}")
            else:
                self.compose_info.setText(f"{label}: done ({len(res) if hasattr(res, '__len__') else ''})")
                self.info(f"{label} done")
                if after:
                    after(res)
            self.ctx.state_changed.emit()

        self.runner.start(fn, on_progress=prog, on_done=done)

    # -- folds -------------------------------------------------------------
    def _fold_method_changed(self) -> None:
        unet = self.fold_method.currentData() == "unet"
        self.fold_ckpt_row.setVisible(unet)
        self.fold_unet_row.setVisible(unet)
        self.fold_adv.setVisible(unet)
        self.fold_dark_w.setVisible(not unet)

    def _detect_folds_dark(self, preview: bool) -> None:
        """Folds/tears as black regions: no model, no GPU, runs here."""
        from ...core.masks import dark_regions, write_mask
        secs = [self.sections.current()] if preview else self.sections.checked()
        secs = [s for s in secs if s and self.store.thumbnail_path(s)]
        if not secs:
            QMessageBox.information(self, "Folds", "No thumbnails for the selected sections yet.")
            return
        store = self.store
        max_grey, min_px, dil = self.fold_dark_max.value(), self.fold_min.value(), self.fold_dil.value()

        def work(progress=None, cancelled=None):
            out = {}
            for i, s in enumerate(secs, 1):
                if cancelled and cancelled():
                    break
                img = store.thumbnail(s)
                if img is not None:
                    m = dark_regions(img, max_grey, min_px, dil)
                    write_mask(store.folds_dir / f"{s}.png", (m * 255).astype(np.uint8))
                    out[s] = float(m.mean())
                progress(i, len(secs), s)
            return out

        def after(res):
            self.ov_folds.setChecked(True)
            if res:
                worst = max(res.items(), key=lambda kv: kv[1])
                self.info(f"dark-region folds: {100 * sum(res.values()) / len(res):.2f}% of the thumbnail on average, "
                          f"most in {worst[0]} ({100 * worst[1]:.2f}%)")
            self._refresh_sections()

        self._start_thread(work, "fold masks (dark regions)", after=after)

    def _detect_folds(self, preview: bool = False) -> None:
        if not self.store:
            return
        self._save_mask_settings()
        if self.fold_method.currentData() == "dark":
            self._detect_folds_dark(preview)
            return
        ck = self.fold_ckpt.path()
        if not ck or not ck.is_file():
            QMessageBox.information(self, "Checkpoint", "Choose a fold model checkpoint.")
            return
        if not self.ctx.dl_python():
            QMessageBox.information(self, "Environment", "Configure the deep-learning Python on the Setup page.")
            return
        secs = [self.sections.current()] if preview else self.sections.checked()
        secs = [s for s in secs if s]
        src = self.fold_src.currentData()
        items = []
        for s in secs:
            if src == "thumb":
                p = self.store.thumbnail_path(s)
                if p:
                    items.append({"section": s, "image": str(p), "mip": self.store.thumbnail_mip()})
            else:
                items.append({"section": s, "tiled_base": str(self.project.root / "stitched_sections"), "mip": int(src)})
        if not items:
            QMessageBox.information(self, "Folds", "No input images for the checked sections.")
            return
        payload = {"checkpoint": str(ck), "items": items, "out_dir": str(self.store.folds_dir), "threshold": self.fold_thr.value(),
                   "tile": self.fold_tile.value(), "overlap": self.fold_ov.value(), "min_area": self.fold_min.value(),
                   "dilate": self.fold_dil.value()}
        spec = self.ctx.worker_spec("fold_predict", payload, f"Fold detection ({len(items)} sections)",
                                    count_outputs=lambda: len(list(self.store.folds_dir.glob("*_prob.png"))), expected=len(items))
        self.submit(spec)

    def _train(self) -> None:
        if not self.require_project():
            return
        d = self.tr_data.path()
        if not d or not d.is_dir():
            QMessageBox.information(self, "Dataset", "Choose a dataset folder with images/ and masks/.")
            return
        name = self.tr_name.text().strip() or "folds_run"
        out = self.project.models_dir / "folds" / name
        payload = {"dataset": str(d), "out_dir": str(out), "epochs": self.tr_epochs.value(), "patch": self.tr_patch.value(),
                   "batch": self.tr_batch.value(), "init_checkpoint": self.fold_ckpt.text() if self.tr_init.isChecked() and self.fold_ckpt.text() else None}
        self.submit(self.ctx.worker_spec("fold_train", payload, f"Fold U-Net training '{name}'"))

    def _refresh_models(self) -> None:
        self.tr_models.clear()
        if not self.project:
            return
        d = self.project.models_dir / "folds"
        for run in sorted(d.iterdir()) if d.is_dir() else []:
            if (run / "best.pt").is_file():
                self.tr_models.addItem(run.name, str(run / "best.pt"))

    def _job_finished(self, res) -> None:
        if res.spec.name.startswith("Fold detection"):
            self.ov_folds.setChecked(True)
            self._refresh_sections()
        elif res.spec.name.startswith("Fold U-Net training"):
            self._refresh_models()

    # -- compose ----------------------------------------------------------
    def _compose(self) -> None:
        if not self.store:
            return
        self._save_mask_settings()
        secs = [s for s in self.sections.checked() if self.store.thumbnail_path(s)]
        if not secs:
            QMessageBox.information(self, "Compose", "No thumbnails for the checked sections yet.")
            return
        use_folds, label, hires = self.use_folds.isChecked(), int(self.fold_label.currentData()), self.hires.isChecked()
        clip = bool(self.clip_folds.currentData())
        margin, margin_label = self.ti_margin.value(), int(self.ti_margin_label.currentData())
        skip = self.skip_edited.isChecked()
        smode = self.project.state.structure.get("material_mode", "off")
        sdil = int(self.project.state.structure.get("material_dilate", 0))
        store = self.store

        def work(progress=None, cancelled=None):
            out = []
            for i, s in enumerate(secs, 1):
                if cancelled and cancelled():
                    break
                if skip and store.is_hand_edited(s):
                    progress(i, len(secs), f"{s} (kept)")
                    continue
                if store.tissue_mask(s) is None:
                    store.compute_tissue(s, self._tissue_params(), save=True)
                r = store.compose(s, use_folds, label, smode, sdil, hires, clip_folds=clip,
                                  margin_px=margin, margin_label=margin_label)
                out.append(r)
                progress(i, len(secs), s)
            return out

        def after(res):
            swallowed = [r["section"] for r in res if r and r.get("folds_outside_tissue", 0) > 0.5]
            if swallowed:
                self.warn(f"{len(swallowed)} section(s) (e.g. {swallowed[0]}): most detected folds lie outside the tissue "
                          f"mask, so they stay excluded (255) instead of becoming {self.fold_label.currentText().split(':')[0]}. "
                          f"Turn off 'exclude black regions' in the tissue card if you want them meshed as a fold material.")
            n_hi = sum(1 for r in res if r and r.get("hires"))
            if n_hi:
                fm = self.store.hires_mip()
                if fm is not None and self.ctx.configs:
                    self.ctx.configs.set("alignment", "meshing.mask_mip_level", int(fm))
                    self.ctx.configs.save("alignment")
                    self.info(f"alignment meshing.mask_mip_level set to {fm} for the higher-resolution masks")
            self.ov_mat.setChecked(True)
            self._refresh_sections()

        self._start_thread(work, "material masks", after=after)

    def _import_masks(self) -> None:
        if not self.store:
            return
        d = self.imp_dir.path()
        if not d or not d.is_dir():
            QMessageBox.information(self, "Import", "Choose the folder with the mask images.")
            return
        secs = [s for s in self.sections.checked() if self.store.thumbnail_path(s)]
        if not secs:
            QMessageBox.information(self, "Import", "No thumbnails for the checked sections yet (run the Thumbnails step first).")
            return
        kind, mip, store = self.imp_kind.currentData(), self.imp_mip.value(), self.store
        files = store.match_files_to_sections(d, secs)
        if not files:
            QMessageBox.information(self, "Import", "No mask files could be matched to the section names.")
            return

        def work(progress=None, cancelled=None):
            return store.import_external(d, mip, kind, secs, progress=progress, cancelled=cancelled)

        def after(res):
            for w in res.get("warnings", []):
                self.warn(w)
            self.info(f"imported {len(res['imported'])} {kind} mask(s) at mip{mip}; missing: {res['missing'] or 'none'}")
            if kind == "material":
                if mip < store.thumbnail_mip() and self.ctx.configs:
                    self.ctx.configs.set("alignment", "meshing.mask_mip_level", int(mip))
                    self.ctx.configs.save("alignment")
                    self.info(f"alignment meshing.mask_mip_level set to {mip}")
                self.ov_mat.setChecked(True)
            else:
                (self.ov_tissue if kind == "tissue" else self.ov_folds).setChecked(True)
                self.compose_info.setText("imported; press 'Compose for checked sections' to write the FEABAS material masks")
            self._refresh_sections()

        self._start_thread(work, f"import {kind} masks", after=after)

    # -- editing -----------------------------------------------------------
    def _edit_fiji(self) -> None:
        sec = self.sections.current()
        if not self.store or not sec:
            return
        p = self.store.thumb_mask_dir / f"{sec}.png"
        if not p.is_file():
            self.warn("no material mask for this section yet")
            return
        from ..external import open_in_fiji
        open_in_fiji(self.ctx.settings, p, self.info)
        self.info("paint with grey values 0 (tissue), 255 (outside), 50 (wrinkle); save, then press 'Reload mask'")

    def _reload_mask(self) -> None:
        sec = self.sections.current()
        if self.store and sec:
            self.store.set_hand_edited(sec, True)
            self._refresh_sections()

    def _restore_default(self) -> None:
        if not self.store:
            return
        secs = [s for s in self.sections.checked() if self.store.thumbnail_path(s)]
        if not secs:
            QMessageBox.information(self, "Reset", "No checked section has a thumbnail yet.")
            return
        if not self.confirm("Reset masks", f"Reset {len(secs)} section(s) to the FEABAS default (everything imaged is "
                                           f"tissue)?\n\nThis also deletes the tissue and fold masks computed for them "
                                           f"and clears the hand-edited flag, so 'Compose' starts from scratch."):
            return
        n = sum(1 for s in secs if self.store.restore_default_mask(s))
        self.info(f"reset {n} section(s) to the FEABAS default mask")
        self.compose_info.setText(f"reset {n} section(s) to the FEABAS default mask")
        self._refresh_sections()

    def _rebuild_roi(self) -> None:
        """Let FEABAS write its own material masks again: it skips sections that still have one."""
        if not self.store:
            return
        secs = [s for s in self.sections.checked() if self.store.thumbnail_path(s)]
        if not secs:
            QMessageBox.information(self, "Rebuild", "No checked section has a thumbnail yet.")
            return
        if not self.confirm("Rebuild FEABAS masks",
                            f"Delete the material masks and thumbnails of {len(secs)} section(s) and re-run the "
                            f"thumbnail step so FEABAS rebuilds them from the tile bounding boxes?\n\nAnything you "
                            f"composed or painted for those sections is replaced. The mip levels of the stitched "
                            f"sections are kept, so this takes seconds per section."):
            return
        n = 0
        for s in secs:
            thumb = self.store.thumbnail_path(s)
            # FEABAS only writes a mask for sections whose thumbnail it (re)generates
            for p in (self.store.thumb_mask_dir / f"{s}.png", self.store.roi_dir / f"{s}.png",
                      self.store.align_mask_dir / f"{s}.png", thumb):
                if p is not None and p.is_file():
                    p.unlink()
                    n += 1
            self.store.manifest.pop(s, None)
        self.store.save_manifest()
        self.info(f"removed {n} mask file(s); running the thumbnail step so FEABAS writes its defaults again")
        try:
            spec = self.ctx.feabas_step_spec(STEPS_BY_KEY["thumbnail.downsample"])
        except RuntimeError as e:
            self.error(str(e))
            return
        self.submit(spec)

    def _split_toggle(self, on: bool) -> None:
        self._split_pts = []
        if on:
            self.view_info.setText("click the two end points of the split line (through the gap between the pieces)")

    def _view_clicked(self, x: float, y: float) -> None:
        if not self.split_btn.isChecked() or not self.store:
            return
        self._split_pts.append((x, y))
        if len(self._split_pts) == 2:
            sec = self.sections.current()
            m = self.store.material_mask(sec)
            if m is not None:
                from ...core.masks import paint_split_line
                m2 = paint_split_line(m, self._split_pts[0], self._split_pts[1], width=max(3, m.shape[0] // 300))
                from ...core.masks import write_mask
                write_mask(self.store.thumb_mask_dir / f"{sec}.png", m2)
                self.store.set_hand_edited(sec, True)
                self.info(f"{sec}: split line painted (mask marked hand-edited)")
            self.split_btn.setChecked(False)
            self._refresh_sections()
