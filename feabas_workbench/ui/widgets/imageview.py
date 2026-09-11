"""
Pan/zoom image viewer with level-of-detail loading, overlays and clickable rectangles.

Two ways to feed it:
* ``set_image(array)`` for anything that fits in memory (thumbnails, previews);
* ``set_source(source)`` for big sections: the source exposes ``levels`` and
  ``read(mip, x0, y0, w, h)`` and only the visible region at the appropriate
  mip is loaded (debounced on view changes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal, QThreadPool, QRunnable, QObject, Slot
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QFont
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem)

from ..theme import ACCENT, OK, WARN


def np_to_qimage(arr: np.ndarray) -> QImage:
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            from ...core.images import to_uint8
            arr = to_uint8(arr)
        a = np.ascontiguousarray(arr)
        h, w = a.shape
        img = QImage(a.data, w, h, w, QImage.Format_Grayscale8)
        return img.copy()
    if arr.shape[2] == 3:
        a = np.ascontiguousarray(arr.astype(np.uint8))
        h, w, _ = a.shape
        return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    a = np.ascontiguousarray(arr.astype(np.uint8))
    h, w, _ = a.shape
    return QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


@dataclass
class RectSpec:
    x: float
    y: float
    w: float
    h: float
    label: str = ""
    color: str = ACCENT
    key: object = None
    selected: bool = False


class _ReadSignals(QObject):
    done = Signal(object, int, int, int, int, int, int)   # array, mip, x0, y0, w, h, generation


class _ReadTask(QRunnable):
    def __init__(self, source, mip, x0, y0, w, h, generation, signals):
        super().__init__()
        self.args = (source, mip, x0, y0, w, h, generation)
        self.signals = signals

    def run(self) -> None:
        source, mip, x0, y0, w, h, gen = self.args
        try:
            arr = source.read(mip, x0, y0, w, h)
        except Exception:  # noqa: BLE001
            arr = None
        self.signals.done.emit(arr, mip, x0, y0, w, h, gen)


class ImageView(QGraphicsView):
    clicked = Signal(float, float)            # scene (level-0) coordinates
    rect_clicked = Signal(object)             # RectSpec.key
    hovered = Signal(float, float)
    view_changed = Signal()
    note = Signal(str)                        # why the view is not showing anything ("" = fine)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QBrush(QColor("#0B1220")))
        self.setMouseTracking(True)
        self._base = QGraphicsPixmapItem()
        self._base.setZValue(0)
        self._scene.addItem(self._base)
        self._overlays: dict[str, QGraphicsPixmapItem] = {}
        self._rect_items: list[tuple[QGraphicsRectItem, RectSpec]] = []
        self._point_items: list[QGraphicsItem] = []
        self._source = None
        self._image: np.ndarray | None = None
        self._level0_size = (0, 0)
        self._gen = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._request_region)
        self._signals = _ReadSignals()
        self._signals.done.connect(self._region_ready)
        self._pool = QThreadPool.globalInstance()
        self.pinned_mip: int | None = None
        self.current_mip = 0
        self._press_pos = None
        self._nearest = False

    # -- content -----------------------------------------------------------
    def clear(self) -> None:
        self._source = None
        self._image = None
        self._base.setPixmap(QPixmap())
        self.clear_overlays()
        self.clear_rects()
        self.clear_points()
        self._level0_size = (0, 0)
        self._scene.setSceneRect(QRectF(0, 0, 1, 1))

    def set_image(self, arr: np.ndarray | None, fit: bool = True, nearest: bool = False) -> None:
        self._source = None
        self._image = arr
        self._nearest = nearest
        if arr is None:
            self._base.setPixmap(QPixmap())
            return
        h, w = arr.shape[:2]
        self._level0_size = (w, h)
        self._base.setPixmap(QPixmap.fromImage(np_to_qimage(arr)))
        self._base.setPos(0, 0)
        self._base.setScale(1.0)
        self._base.setTransformationMode(Qt.FastTransformation if nearest else Qt.SmoothTransformation)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self.current_mip = 0
        if fit:
            self.fit()

    def set_source(self, source, fit: bool = True) -> None:
        self._image = None
        self._source = source
        if source is None or not getattr(source, "levels", None):
            self.clear()
            return
        l0 = source.levels[0]
        w, h = l0.width * (2 ** l0.mip), l0.height * (2 ** l0.mip)
        self._level0_size = (w, h)
        self._base.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        if fit:
            self.fit()
        self._timer.start()

    def image(self) -> np.ndarray | None:
        return self._image

    def level0_size(self) -> tuple[int, int]:
        return self._level0_size

    # -- overlays ----------------------------------------------------------
    def set_overlay(self, name: str, rgba: np.ndarray | None, opacity: float = 1.0, scale: float = 1.0,
                    offset: tuple[float, float] = (0, 0)) -> None:
        item = self._overlays.get(name)
        if rgba is None:
            if item is not None:
                self._scene.removeItem(item)
                del self._overlays[name]
            return
        if item is None:
            item = QGraphicsPixmapItem()
            item.setZValue(5)
            self._scene.addItem(item)
            self._overlays[name] = item
        item.setPixmap(QPixmap.fromImage(np_to_qimage(rgba)))
        item.setOpacity(opacity)
        item.setScale(scale)
        item.setPos(*offset)
        item.setTransformationMode(Qt.FastTransformation)

    def set_overlay_visible(self, name: str, visible: bool) -> None:
        item = self._overlays.get(name)
        if item is not None:
            item.setVisible(visible)

    def clear_overlays(self) -> None:
        for it in self._overlays.values():
            self._scene.removeItem(it)
        self._overlays.clear()

    # -- rectangles (tile grid, bboxes) -------------------------------------
    def set_rects(self, rects: list[RectSpec], show_labels: bool = True, pen_width: float = 0.0) -> None:
        self.clear_rects()
        for spec in rects:
            item = QGraphicsRectItem(QRectF(spec.x, spec.y, spec.w, spec.h))
            pen = QPen(QColor(spec.color))
            pen.setCosmetic(True)
            pen.setWidthF(3 if spec.selected else 1.2)
            item.setPen(pen)
            col = QColor(spec.color)
            col.setAlpha(90 if spec.selected else 25)
            item.setBrush(QBrush(col))
            item.setZValue(10)
            item.setData(0, spec.key)
            self._scene.addItem(item)
            self._rect_items.append((item, spec))
            if show_labels and spec.label:
                t = QGraphicsSimpleTextItem(spec.label, item)
                t.setBrush(QBrush(QColor("#FFFFFF")))
                f = QFont()
                f.setPointSize(9)
                t.setFont(f)
                t.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                t.setPos(spec.x + 4, spec.y + 4)
                t.setZValue(11)

    def update_rect_selection(self, selected_keys: set) -> None:
        for item, spec in self._rect_items:
            spec.selected = spec.key in selected_keys
            pen = item.pen()
            pen.setWidthF(3 if spec.selected else 1.2)
            item.setPen(pen)
            col = QColor(spec.color)
            col.setAlpha(90 if spec.selected else 25)
            item.setBrush(QBrush(col))

    def clear_rects(self) -> None:
        for item, _ in self._rect_items:
            self._scene.removeItem(item)
        self._rect_items.clear()

    # -- points / lines (matches) ------------------------------------------
    def set_points(self, pts: np.ndarray | None, color: str = OK, radius: float = 3.0,
                   lines_to: np.ndarray | None = None, line_color: str = WARN) -> None:
        self.clear_points()
        if pts is None or len(pts) == 0:
            return
        pen = QPen(QColor(color))
        pen.setCosmetic(True)
        brush = QBrush(QColor(color))
        lpen = QPen(QColor(line_color))
        lpen.setCosmetic(True)
        for i, (x, y) in enumerate(np.asarray(pts)[:, :2]):
            e = QGraphicsEllipseItem(x - radius, y - radius, 2 * radius, 2 * radius)
            e.setPen(pen)
            e.setBrush(brush)
            e.setZValue(12)
            e.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
            self._scene.addItem(e)
            self._point_items.append(e)
            if lines_to is not None and i < len(lines_to):
                x2, y2 = lines_to[i][:2]
                ln = QGraphicsLineItem(x, y, x2, y2)
                ln.setPen(lpen)
                ln.setZValue(11)
                self._scene.addItem(ln)
                self._point_items.append(ln)

    def clear_points(self) -> None:
        for it in self._point_items:
            self._scene.removeItem(it)
        self._point_items.clear()

    # -- navigation --------------------------------------------------------
    def fit(self) -> None:
        r = self._scene.sceneRect()
        if r.width() > 1 and r.height() > 1:
            self.fitInView(r, Qt.KeepAspectRatio)
            self._on_view_changed()

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)
        self._on_view_changed()

    def scale_factor(self) -> float:
        return self.transform().m11()

    def center_on_scene(self, x: float, y: float, scale: float | None = None) -> None:
        if scale is not None:
            self.resetTransform()
            self.scale(scale, scale)
        self.centerOn(QPointF(x, y))
        self._on_view_changed()

    def visible_scene_rect(self) -> QRectF:
        return self.mapToScene(self.viewport().rect()).boundingRect()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        self._on_view_changed()
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._on_view_changed()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._on_view_changed()

    def mousePressEvent(self, event) -> None:
        self._press_pos = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if self._press_pos is not None and (event.position() - self._press_pos).manhattanLength() < 4:
            p = self.mapToScene(event.position().toPoint())
            hit = None
            for item, spec in self._rect_items:
                if item.rect().contains(p):
                    hit = spec.key
                    break
            if hit is not None:
                self.rect_clicked.emit(hit)
            self.clicked.emit(p.x(), p.y())
        self._press_pos = None

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        p = self.mapToScene(event.position().toPoint())
        self.hovered.emit(p.x(), p.y())

    def _on_view_changed(self) -> None:
        self.view_changed.emit()
        if self._source is not None:
            self._timer.start()

    # -- LOD loading -------------------------------------------------------
    def _choose_mip(self) -> int:
        if self.pinned_mip is not None:
            return self.pinned_mip
        s = max(self.scale_factor(), 1e-6)
        mip = int(math.floor(math.log2(1.0 / s))) if s < 1 else 0
        levels = [l.mip for l in self._source.levels]
        return max(min(levels), min(max(levels), mip))

    def _request_region(self) -> None:
        if self._source is None:
            return
        mip = self._choose_mip()
        self.current_mip = mip
        f = 2 ** mip
        vis = self.visible_scene_rect()
        W, H = self._level0_size
        margin = 0.25
        x0 = max(0.0, vis.x() - vis.width() * margin)
        y0 = max(0.0, vis.y() - vis.height() * margin)
        x1 = min(float(W), vis.x() + vis.width() * (1 + margin))
        y1 = min(float(H), vis.y() + vis.height() * (1 + margin))
        if x1 <= x0 or y1 <= y0:
            return
        rx0, ry0 = int(x0 // f), int(y0 // f)
        rw, rh = int(math.ceil((x1 - x0) / f)) + 1, int(math.ceil((y1 - y0) / f)) + 1
        if rw * rh > 40_000_000:
            # nothing coarse enough to draw this view: say so instead of leaving it blank
            coarsest = max(l.mip for l in self._source.levels)
            self.note.emit(f"Too much data for this zoom level: the coarsest level available is mip{coarsest}. "
                           f"Build the mipmaps for this stack, or zoom in.")
            return
        self.note.emit("")
        self._gen += 1
        task = _ReadTask(self._source, mip, rx0, ry0, rw, rh, self._gen, self._signals)
        self._pool.start(task)

    @Slot(object, int, int, int, int, int, int)
    def _region_ready(self, arr, mip, x0, y0, w, h, gen) -> None:
        if gen != self._gen or arr is None or self._source is None:
            return
        f = 2 ** mip
        self._base.setPixmap(QPixmap.fromImage(np_to_qimage(arr)))
        self._base.setPos(x0 * f, y0 * f)
        self._base.setScale(f)
        self._base.setTransformationMode(Qt.SmoothTransformation)

    def refresh_source(self) -> None:
        if self._source is not None:
            self._timer.start()

    def grab_array(self) -> np.ndarray:
        img = self.grab().toImage().convertToFormat(QImage.Format_RGB888)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
        return arr.copy()
