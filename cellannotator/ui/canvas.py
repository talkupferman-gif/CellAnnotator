"""Image canvas. Draw and erase share one gesture: right-click, then move the pointer."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from cellannotator.masks import add_closed_outline, erase_closed_outline, interior_outline
from cellannotator.segio import MaskSession


OUTLINE_COLORS = (
    (255, 77, 109),
    (77, 171, 247),
    (105, 219, 124),
    (255, 212, 59),
    (177, 151, 252),
    (255, 146, 43),
    (34, 184, 207),
    (240, 101, 149),
    (116, 143, 252),
    (255, 107, 107),
    (99, 230, 190),
    (255, 169, 77),
)


def rgb_for_key(key: str) -> tuple[int, int, int]:
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return OUTLINE_COLORS[digest[0] % len(OUTLINE_COLORS)]


def to_qimage(array: np.ndarray) -> QImage:
    array = np.ascontiguousarray(array)
    height, width = array.shape[:2]
    if array.ndim == 2:
        image = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
    elif array.shape[2] == 3:
        image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)
    else:
        image = QImage(array.data, width, height, 4 * width, QImage.Format_RGBA8888)
    return image.copy()


class Canvas(QWidget):
    edited = Signal()
    labelsTouched = Signal(object)
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.session: MaskSession | None = None
        self.mode = "draw"
        self.brush_radius = 8
        self.show_fill = True
        self.show_outlines = False
        self.masks_visible = True
        self.selected: set[int] = set()
        self.color_keys: dict[int, str] = {}
        self._rgb: np.ndarray | None = None
        self._base: QImage | None = None
        self._overlay: QImage | None = None
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._panning = False
        self._press = QPointF()
        self._pan_origin = QPointF()
        self._moved = False
        self._stroke: list[tuple[float, float]] = []
        self._stroking = False
    def set_mode(self, mode: str) -> None:
        self._cancel_stroke()
        self.mode = mode
        self.status.emit("Draw" if mode == "draw" else "Erase")
        self.update()

    def set_image(self, session: MaskSession, rgb: np.ndarray) -> None:
        self._cancel_stroke()
        self.session = session
        self._rgb = rgb
        self._base = to_qimage(rgb)
        self.selected.clear()
        self.color_keys = {}
        self._rebuild_overlay()
        self.fit()

    def set_color_keys(self, keys: dict[int, str]) -> None:
        self.color_keys = dict(keys)
        self._rebuild_overlay()
        self.update()

    def set_display(self, rgb: np.ndarray) -> None:
        self._rgb = rgb
        self._base = to_qimage(rgb)
        self.update()

    def refresh(self) -> None:
        self.selected = {label for label in self.selected if self.session is not None and (self.session.masks == label).any()}
        self._rebuild_overlay()
        self.update()

    def fit(self) -> None:
        if self.session is None or self.width() < 2 or self.height() < 2:
            return
        height, width = self.session.masks.shape
        self._zoom = min(self.width() / width, self.height() / height) * 0.98
        self._pan = QPointF(width / 2, height / 2)
        self.update()

    def zoom_by(self, factor: float) -> None:
        self._zoom = min(80.0, max(0.02, self._zoom * factor))
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.session is not None and self._zoom == 1.0 and self._pan == QPointF(0, 0):
            self.fit()

    def _screen_to_image(self, point: QPointF) -> tuple[float, float]:
        x = (point.x() - self.width() / 2) / self._zoom + self._pan.x()
        y = (point.y() - self.height() / 2) / self._zoom + self._pan.y()
        return x, y

    def _close_distance(self) -> float:
        return max(1.0, 12.0 / self._zoom)

    def _label_at(self, x: float, y: float) -> int:
        if self.session is None:
            return 0
        height, width = self.session.masks.shape
        column = int(round(x))
        row = int(round(y))
        if not (0 <= row < height and 0 <= column < width):
            return 0
        return int(self.session.masks[row, column])

    def _rebuild_overlay(self) -> None:
        if self.session is None:
            self._overlay = None
            return
        masks = self.session.masks
        height, width = masks.shape
        rgba = np.zeros((height, width, 4), np.uint8)
        if not self.masks_visible:
            self._overlay = to_qimage(rgba)
            return
        labels = [int(value) for value in np.unique(masks) if int(value) > 0]
        if labels:
            table = np.zeros((max(labels) + 1, 3), np.uint8)
            for label in labels:
                table[label] = rgb_for_key(self.color_keys.get(label, str(label)))
            if self.show_fill:
                foreground = masks > 0
                rgba[foreground, :3] = table[masks[foreground]]
                rgba[foreground, 3] = 80
                if self.selected:
                    chosen = np.isin(masks, list(self.selected))
                    rgba[chosen, 3] = 150
            if self.show_outlines:
                edges = interior_outline(masks, width=3)
                rgba[edges, :3] = table[masks[edges]]
                rgba[edges, 3] = 255
                if self.selected:
                    selected = interior_outline(masks, width=4) & np.isin(masks, list(self.selected))
                    rgba[selected, :3] = (255, 255, 255)
                    rgba[selected, 3] = 255
        self._overlay = to_qimage(rgba)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0F172A"))
        if self._base is None:
            painter.end()
            return
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(self._zoom, self._zoom)
        painter.translate(-self._pan.x(), -self._pan.y())
        painter.drawImage(0, 0, self._base)
        if self._overlay is not None:
            painter.drawImage(0, 0, self._overlay)
        if len(self._stroke) >= 2:
            pen = QPen(QColor("#22C55E") if self.mode == "draw" else QColor("#EF4444"))
            pen.setWidthF(max(2.0, 3.0 / self._zoom))
            painter.setPen(pen)
            for start, end in zip(self._stroke, self._stroke[1:]):
                painter.drawLine(QPointF(*start), QPointF(*end))
        if self._stroking and self._stroke:
            pen = QPen(QColor("#F8FAFC"))
            pen.setWidthF(max(1.5, 2.0 / self._zoom))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            radius = self._close_distance()
            start = self._stroke[0]
            painter.drawEllipse(QPointF(*start), radius, radius)
        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        before = self._screen_to_image(event.position())
        self._zoom = min(80.0, max(0.02, self._zoom * factor))
        self._pan = QPointF(
            before[0] - (event.position().x() - self.width() / 2) / self._zoom,
            before[1] - (event.position().y() - self.height() / 2) / self._zoom,
        )
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.setFocus()
        if self.session is None:
            return
        if event.button() == Qt.RightButton:
            self._start_stroke(self._screen_to_image(event.position()))
            return
        if event.button() == Qt.LeftButton:
            self._panning = True
            self._moved = False
            self._press = event.position()
            self._pan_origin = QPointF(self._pan)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            delta = event.position() - self._press
            if delta.manhattanLength() > 3:
                self._moved = True
            self._pan = QPointF(
                self._pan_origin.x() - delta.x() / self._zoom,
                self._pan_origin.y() - delta.y() / self._zoom,
            )
            self.update()
            return
        if not self._stroking:
            return
        point = self._screen_to_image(event.position())
        if self._stroke:
            last = self._stroke[-1]
            if math.hypot(point[0] - last[0], point[1] - last[1]) < 0.5:
                return
        self._stroke.append(point)
        if self._ready_to_close():
            if self.mode == "erase":
                self._close_erase()
            else:
                self._close_outline()
            return
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._panning:
            self._panning = False
            if not self._moved:
                self._select_at(self._screen_to_image(event.position()), event.modifiers() & Qt.ShiftModifier)
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self._cancel_stroke()
            return
        super().keyPressEvent(event)

    def _start_stroke(self, point: tuple[float, float]) -> None:
        if self.session is None or self._stroking:
            return
        self._stroking = True
        self._stroke = [point]
        self.update()

    def _close_erase(self) -> None:
        if self.session is None:
            return
        labels = set(self.selected) if self.selected else None
        self.session.checkpoint()
        updated, touched = erase_closed_outline(self.session.masks, self._stroke, labels)
        self._stroking = False
        self._stroke = []
        if not touched:
            self.session.drop_checkpoint()
            self.status.emit("That loop did not cut a cell")
        else:
            self.session.masks = updated
            for label in touched:
                if (self.session.masks == label).any():
                    self.session.set_manual(label, True)
            self.session.dirty = True
            self.labelsTouched.emit(touched)
            self.edited.emit()
            self.status.emit("Cut the inside of the loop")
        self._rebuild_overlay()
        self.update()

    def _ready_to_close(self) -> bool:
        if len(self._stroke) < 8:
            return False
        start = self._stroke[0]
        end = self._stroke[-1]
        if math.hypot(end[0] - start[0], end[1] - start[1]) > self._close_distance():
            return False
        length = 0.0
        for left, right in zip(self._stroke, self._stroke[1:]):
            length += math.hypot(right[0] - left[0], right[1] - left[1])
        return length > self._close_distance() * 3

    def _close_outline(self) -> None:
        if self.session is None:
            return
        self.session.checkpoint()
        updated, label = add_closed_outline(self.session.masks, self._stroke)
        self._stroking = False
        self._stroke = []
        if label == 0:
            self.session.drop_checkpoint()
            self.status.emit("That outline did not add a cell")
        else:
            self.session.masks = updated
            self.session.set_manual(label, True)
            self.session.dirty = True
            self.selected = {label}
            self.labelsTouched.emit({label})
            self.edited.emit()
            self.status.emit(f"Added cell {label}")
        self._rebuild_overlay()
        self.update()

    def _cancel_stroke(self) -> None:
        self._stroking = False
        self._stroke = []
        self.update()

    def _select_at(self, point: tuple[float, float], extend: bool) -> None:
        label = self._label_at(*point)
        if extend and label:
            if label in self.selected:
                self.selected.remove(label)
            else:
                self.selected.add(label)
        else:
            self.selected = {label} if label else set()
        self._rebuild_overlay()
        names = ", ".join(str(item) for item in sorted(self.selected)) or "none"
        self.status.emit(f"Selected {names}")
        self.update()
