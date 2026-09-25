"""Small dependency-free training chart."""
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath
from PySide6.QtWidgets import QWidget
from ..theme import PANEL, TEXT


class LossPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.setMinimumHeight(190)

    def set_history(self, history):
        self.history = history
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(PANEL))
        painter.setPen(QColor(TEXT))
        plot = QRectF(65, 30, max(1, self.width() - 85), max(1, self.height() - 65))
        values = [r[k] for r in self.history for k in ("train_loss", "val_loss")
                  if r.get(k) is not None and math.isfinite(r[k])]
        if not values:
            painter.drawText(self.rect(), Qt.AlignCenter, "Loss appears after the first completed epoch")
            return
        low, high = min(values), max(values)
        padding = max((high - low) * .08, abs(high) * .01, 1e-6)
        low -= padding; high += padding
        epochs = max(2, max(r["epoch"] for r in self.history))
        painter.drawRect(plot)
        painter.drawText(QRectF(0, plot.top() - 8, 60, 20), Qt.AlignRight, f"{high:.3g}")
        painter.drawText(QRectF(0, plot.bottom() - 12, 60, 20), Qt.AlignRight, f"{low:.3g}")
        painter.drawText(QRectF(plot.left(), plot.bottom() + 4, plot.width(), 25), Qt.AlignCenter,
                         f"Epoch 1 → {max(r['epoch'] for r in self.history)}")
        for key, label, line_color, xpos in (("train_loss", "Training", "#479ee5", 70),
                                            ("val_loss", "Validation", "#eea94d", 180)):
            painter.setPen(QPen(QColor(line_color), 2))
            painter.drawText(xpos, 19, label)
            path = QPainterPath()
            connected = False
            for row in self.history:
                value = row.get(key)
                if value is None or not math.isfinite(value):
                    connected = False
                    continue
                point = QPointF(plot.left() + (row["epoch"] - 1) / (epochs - 1) * plot.width(),
                                plot.bottom() - (value - low) / (high - low) * plot.height())
                if connected:
                    path.lineTo(point)
                else:
                    path.moveTo(point)
                painter.drawEllipse(point, 2, 2)
                connected = True
            painter.drawPath(path)
