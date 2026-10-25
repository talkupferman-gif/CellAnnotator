"""A local image folder, its working copies, and the last synced commit."""

from __future__ import annotations

import json
from pathlib import Path

from cellannotator.files import natural_key, sha256_file
from cellannotator.images import IMAGE_EXTENSIONS, load_array, plane_shape
from cellannotator.meta import ProjectMeta, load_meta, meta_paths, save_meta
from cellannotator.segio import MaskSession, load_session_for_image, save_session, seg_pair


STATE_DIR = ".cellannotator"
STATE_NAME = "state.json"


class Project:
    def __init__(self, root: Path, meta: ProjectMeta | None = None):
        self.root = Path(root)
        self.meta = meta if meta is not None else load_meta(self.root)
        self.images = scan_images(self.root)
        self.state = load_state(self.root)
        self._shapes: dict[str, tuple[int, int]] = {}
        self._arrays: dict[str, object] = {}

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @property
    def rels(self) -> list[str]:
        return [self.rel(path) for path in self.images]

    def image_path(self, relpath: str) -> Path:
        return self.root / relpath

    def shape_of(self, relpath: str) -> tuple[int, int]:
        if relpath not in self._shapes:
            array = self.array_of(relpath)
            self._shapes[relpath] = plane_shape(array)
        return self._shapes[relpath]

    def array_of(self, relpath: str):
        if relpath not in self._arrays:
            self._arrays[relpath] = load_array(self.image_path(relpath))
        return self._arrays[relpath]

    def load_session(self, relpath: str) -> MaskSession:
        path = self.image_path(relpath)
        return load_session_for_image(path, self.shape_of(relpath))

    def write_working(self, relpath: str, session: MaskSession) -> None:
        _committed, working = seg_pair(self.image_path(relpath))
        save_session(working, session)
        session.dirty = False

    def write_working_meta(self) -> None:
        _committed, working = meta_paths(self.root)
        save_meta(working, self.meta)

    def write_committed_bytes(self, relpath: str, data: bytes) -> None:
        target = self.root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        working_name = None
        if relpath.endswith("_seg.npy"):
            working_name = relpath[: -len(".npy")] + ".working.npy"
        elif relpath == "annotation_meta.json":
            working_name = "annotation_meta.working.json"
        if working_name:
            working = self.root / working_name
            if working.exists():
                working.unlink()

    def iter_publishable(self) -> list[tuple[str, Path]]:
        """Files that differ from the committed copy and should be uploaded."""
        found: list[tuple[str, Path]] = []
        for image in self.images:
            committed, working = seg_pair(image)
            if _differs(working, committed):
                found.append((self.rel(committed), working))
        committed_meta, working_meta = meta_paths(self.root)
        if _differs(working_meta, committed_meta):
            found.append((self.rel(committed_meta), working_meta))
        return found

    def file_manifest(self) -> list[dict]:
        rows = []
        for image in self.images:
            rows.append(
                {
                    "path": self.rel(image),
                    "sha256": sha256_file(image),
                    "bytes": image.stat().st_size,
                }
            )
        return rows

    def save_state(self) -> None:
        folder = self.root / STATE_DIR
        folder.mkdir(exist_ok=True)
        (folder / STATE_NAME).write_text(json.dumps(self.state, indent=2), encoding="utf-8")


def scan_images(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if STATE_DIR in path.relative_to(root).parts:
            continue
        name = path.name
        if name.startswith("annotation_meta"):
            continue
        if name.endswith("_seg.npy") or name.endswith("_seg.working.npy"):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            found.append(path)
    found.sort(key=lambda item: natural_key(item.relative_to(root).as_posix()))
    return found


def _differs(working: Path, committed: Path) -> bool:
    if not working.exists():
        return False
    if not committed.exists():
        return True
    return sha256_file(working) != sha256_file(committed)


def load_state(root: Path) -> dict:
    path = root / STATE_DIR / STATE_NAME
    if not path.exists():
        return {"project_id": None, "base_commit_id": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("project_id", None)
    data.setdefault("base_commit_id", None)
    return data
