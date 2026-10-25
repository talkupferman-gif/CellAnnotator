"""Project metadata: z-stack links. Readiness scores live in each ``*_seg.npy``."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path


META_NAME = "annotation_meta.json"
WORKING_META_NAME = "annotation_meta.working.json"


@dataclass
class CellLink:
    id: str
    anchors: dict[str, int] = field(default_factory=dict)
    filled: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "anchors": {key: int(value) for key, value in self.anchors.items()},
            "filled": {key: int(value) for key, value in self.filled.items()},
        }


@dataclass
class ProjectMeta:
    kind: str = "unrelated"
    links: list[CellLink] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"kind": self.kind, "links": [link.to_json() for link in self.links]}

    @classmethod
    def from_json(cls, data: dict) -> ProjectMeta:
        links = []
        for raw in data.get("links", []):
            links.append(
                CellLink(
                    id=str(raw["id"]),
                    anchors={str(k): int(v) for k, v in dict(raw.get("anchors", {})).items()},
                    filled={str(k): int(v) for k, v in dict(raw.get("filled", {})).items()},
                )
            )
        kind = data.get("kind", "unrelated")
        if kind not in {"unrelated", "zstack"}:
            kind = "unrelated"
        return cls(kind=kind, links=links)

    def link_by_id(self, link_id: str) -> CellLink | None:
        return next((link for link in self.links if link.id == link_id), None)

    def link_for_anchor(self, relpath: str, label: int) -> CellLink | None:
        for link in self.links:
            if link.anchors.get(relpath) == label:
                return link
        return None

    def new_link(self, relpath: str, label: int) -> CellLink:
        link = CellLink(id=uuid.uuid4().hex[:8], anchors={relpath: int(label)})
        self.links.append(link)
        return link

    def promote_filled(self, relpath: str, label: int) -> bool:
        for link in self.links:
            if link.filled.get(relpath) == label:
                link.anchors[relpath] = label
                del link.filled[relpath]
                return True
        return False

    def remove_label(self, relpath: str, label: int) -> None:
        for link in self.links:
            if link.anchors.get(relpath) == label:
                del link.anchors[relpath]
            if link.filled.get(relpath) == label:
                del link.filled[relpath]
        self.links = [link for link in self.links if link.anchors]

    def drop_slice(self, relpath: str) -> None:
        for link in self.links:
            link.anchors.pop(relpath, None)
            link.filled.pop(relpath, None)
        self.links = [link for link in self.links if link.anchors]

    def rewrite_label(self, relpath: str, mapping: dict[int, int]) -> None:
        """Point links at surviving labels. The first link keeps a duplicated anchor."""
        claimed: set[int] = set()
        for link in self.links:
            for bucket in (link.anchors, link.filled):
                if relpath not in bucket:
                    continue
                label = mapping.get(bucket[relpath], bucket[relpath])
                if label <= 0 or label in claimed and bucket is link.anchors:
                    del bucket[relpath]
                    continue
                bucket[relpath] = label
                if bucket is link.anchors:
                    claimed.add(label)
        self.links = [link for link in self.links if link.anchors]


def meta_paths(root: Path) -> tuple[Path, Path]:
    return root / META_NAME, root / WORKING_META_NAME


def load_meta(root: Path) -> ProjectMeta:
    committed, working = meta_paths(root)
    path = working if working.exists() else committed
    if not path.exists():
        return ProjectMeta()
    return ProjectMeta.from_json(json.loads(path.read_text(encoding="utf-8")))


def save_meta(path: Path, meta: ProjectMeta) -> None:
    path.write_text(json.dumps(meta.to_json(), indent=2), encoding="utf-8")
