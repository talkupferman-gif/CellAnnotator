"""Cellpose ``*_seg.npy`` files and the working copy that Save writes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cellannotator.masks import edge_mask


OWNED_KEYS = {"outlines", "masks", "chan_choose", "ismanual", "filename", "flows", "diameter", "readiness"}


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
        self.chan_choose = [0, 0]
        self.flows = None
        self.diameter = float("nan")
        self.readiness: int | None = None
        self.extras: dict = {}
        self.dirty = False
        self.warning = ""
        self._undo: list[tuple[np.ndarray, np.ndarray]] = []

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

    def checkpoint(self) -> None:
        self._undo.append((self.masks.copy(), self.ismanual.copy()))
        if len(self._undo) > 30:
            self._undo.pop(0)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.masks, self.ismanual = self._undo.pop()
        self.dirty = True
        return True

    def drop_checkpoint(self) -> None:
        if self._undo:
            self._undo.pop()

    def to_payload(self) -> dict:
        outlines = self.masks * edge_mask(self.masks)
        limit = int(self.masks.max())
        manual = np.zeros(limit, dtype=bool)
        if limit:
            copy_n = min(limit, self.ismanual.size)
            manual[:copy_n] = self.ismanual[:copy_n]
        width = np.uint16 if max(int(self.masks.max()), int(outlines.max() if outlines.size else 0)) < 2**16 - 1 else np.uint32
        payload = {
            "outlines": outlines.astype(width),
            "masks": self.masks.astype(width),
            "chan_choose": list(self.chan_choose),
            "ismanual": manual,
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
