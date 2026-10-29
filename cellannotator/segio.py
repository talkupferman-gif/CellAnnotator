"""Cellpose ``*_seg.npy`` files and the working copy that Save writes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cellannotator.masks import edge_mask


QUALITY_GRADES = (1, 2, 3, 4, 5)
GRADE_MEANINGS = {
    1: "Punish this mask. The model is pushed not to create it.",
    2: "Do not learn from this mask.",
    3: "A cell is under this mask, but the shape is incomplete.",
    4: "An okay mask. Its shape can be learned.",
    5: "A perfect mask. Train on it the most.",
}


OWNED_KEYS = {
    "outlines",
    "masks",
    "chan_choose",
    "ismanual",
    "quality",
    "filename",
    "flows",
    "diameter",
    "readiness",
}


def seg_pair(image_path: Path) -> tuple[Path, Path]:
    """``image.tif`` -> ``image_seg.npy`` and ``image_seg.working.npy``."""
    base = os.path.splitext(str(image_path))[0]
    return Path(base + "_seg.npy"), Path(base + "_seg.working.npy")


def working_rel_for_committed(relpath: str) -> str:
    if relpath.endswith("_seg.npy"):
        return relpath[: -len(".npy")] + ".working.npy"
    if relpath == "annotation_meta.json":
        return "annotation_meta.working.json"
    raise ValueError(f"Not a project file: {relpath}")


class MaskSession:
    def __init__(self, masks: np.ndarray, filename: str = ""):
        self.masks = np.asarray(masks, dtype=np.int32)
        self.filename = filename
        self.ismanual = np.zeros(int(self.masks.max()), dtype=bool)
        self.quality = np.zeros(int(self.masks.max()), dtype=np.uint8)
        self.chan_choose = [0, 0]
        self.flows = None
        self.diameter = float("nan")
        self.readiness: int | None = None
        self.extras: dict = {}
        self.dirty = False
        self.warning = ""
        self._undo: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._redo: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    @classmethod
    def empty(cls, height: int, width: int, filename: str = "") -> MaskSession:
        return cls(np.zeros((height, width), np.int32), filename)

    def is_manual(self, label: int) -> bool:
        index = label - 1
        return label > 0 and index < self.ismanual.size and bool(self.ismanual[index])

    def set_manual(self, label: int, value: bool) -> None:
        if label <= 0:
            return
        self._grow_manual(label)
        self.ismanual[label - 1] = value

    def _grow_manual(self, label: int) -> None:
        if self.ismanual.size < label:
            grown = np.zeros(label, dtype=bool)
            grown[: self.ismanual.size] = self.ismanual
            self.ismanual = grown

    def quality_of(self, label: int) -> int:
        index = label - 1
        if label <= 0 or index >= self.quality.size:
            return 0
        return int(self.quality[index])

    def set_quality(self, label: int, value: int) -> None:
        if label <= 0:
            return
        rating = int(value)
        if rating not in (0, *QUALITY_GRADES):
            raise ValueError("Quality is 1, 2, 3, 4, or 5")
        self._grow_quality(label)
        self.quality[label - 1] = rating

    def _grow_quality(self, label: int) -> None:
        if self.quality.size < label:
            grown = np.zeros(label, dtype=np.uint8)
            grown[: self.quality.size] = self.quality
            self.quality = grown

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.masks.copy(), self.ismanual.copy(), self.quality.copy())

    def checkpoint(self) -> None:
        self._undo.append(self._snapshot())
        self._redo.clear()
        if len(self._undo) > 30:
            self._undo.pop(0)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self.masks, self.ismanual, self.quality = self._undo.pop()
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self.masks, self.ismanual, self.quality = self._redo.pop()
        self.dirty = True
        return True

    def drop_checkpoint(self) -> None:
        if self._undo:
            self._undo.pop()

    def to_payload(self) -> dict:
        outlines = self.masks * edge_mask(self.masks)
        limit = int(self.masks.max())
        manual = np.zeros(limit, dtype=bool)
        quality = np.zeros(limit, dtype=np.uint8)
        if limit:
            copy_n = min(limit, self.ismanual.size)
            manual[:copy_n] = self.ismanual[:copy_n]
            quality_n = min(limit, self.quality.size)
            quality[:quality_n] = self.quality[:quality_n]
        width = np.uint16 if max(int(self.masks.max()), int(outlines.max() if outlines.size else 0)) < 2**16 - 1 else np.uint32
        payload = {
            "outlines": outlines.astype(width),
            "masks": self.masks.astype(width),
            "chan_choose": list(self.chan_choose),
            "ismanual": manual,
            "quality": quality,
            "filename": self.filename,
            "flows": self.flows if self.flows is not None else [],
            "diameter": self.diameter,
            "readiness": self.readiness,
        }
        for key, value in self.extras.items():
            if key not in OWNED_KEYS:
                payload[key] = value
        return payload


def save_session(path: Path, session: MaskSession) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, session.to_payload())


def load_session(path: Path, shape: tuple[int, int], filename: str = "") -> MaskSession:
    if not path.exists():
        return MaskSession.empty(shape[0], shape[1], filename)
    data = np.load(path, allow_pickle=True).item()
    masks = np.asarray(data.get("masks", np.zeros(shape, np.int32)))
    session = MaskSession(masks, data.get("filename") or filename)
    if session.masks.shape != tuple(shape):
        session = MaskSession.empty(shape[0], shape[1], filename)
        session.warning = f"{path.name} does not match this image and was not applied."
        return session
    manual = np.asarray(data.get("ismanual", []), dtype=bool).reshape(-1)
    session.ismanual = manual
    session._grow_manual(int(session.masks.max()))
    quality = np.asarray(data.get("quality", []), dtype=np.int32).reshape(-1)
    session.quality = np.clip(quality, 0, max(QUALITY_GRADES)).astype(np.uint8)
    session._grow_quality(int(session.masks.max()))
    channels = data.get("chan_choose", [0, 0])
    try:
        values = [int(value) for value in list(channels)]
    except TypeError:
        values = [int(channels)]
    session.chan_choose = (values + [0, 0])[:2]
    session.flows = data.get("flows", [])
    session.diameter = data.get("diameter", float("nan"))
    readiness = data.get("readiness", None)
    session.readiness = int(readiness) if readiness is not None and not (isinstance(readiness, float) and np.isnan(readiness)) else None
    session.extras = {key: value for key, value in data.items() if key not in OWNED_KEYS}
    return session


def load_session_for_image(image_path: Path, shape: tuple[int, int]) -> MaskSession:
    committed, working = seg_pair(image_path)
    source = working if working.exists() else committed
    session = load_session(source, shape, image_path.name)
    session.dirty = False
    return session
