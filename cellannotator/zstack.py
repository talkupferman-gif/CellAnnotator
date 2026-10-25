"""Fill slices that sit between hand-chosen anchors of the same cell."""

from __future__ import annotations

from cellannotator.masks import interpolate_binary
from cellannotator.meta import ProjectMeta
from cellannotator.segio import MaskSession


def fill_links(meta: ProjectMeta, sessions: dict[str, MaskSession], ordered: list[str]) -> list[str]:
    """Write interpolated masks into slices between anchors.

    Slices that already have this cell are left alone, including filled slices
    the user later edited (those are anchors and are not recomputed).
    """
    index = {rel: position for position, rel in enumerate(ordered)}
    warnings: list[str] = []
    for link in list(meta.links):
        keys: list[tuple[int, str, int]] = []
        for relpath, label in link.anchors.items():
            if relpath not in index or relpath not in sessions:
                warnings.append(f"Missing slice for cell {link.id}: {relpath}")
                continue
            session = sessions[relpath]
            if label <= 0 or not (session.masks == label).any():
                warnings.append(f"Cell {link.id} no longer has a mask on {relpath}")
                continue
            keys.append((index[relpath], relpath, int(label)))
        keys.sort()
        for (index_a, rel_a, label_a), (index_b, rel_b, label_b) in zip(keys, keys[1:]):
            source = sessions[rel_a].masks == label_a
            target = sessions[rel_b].masks == label_b
            if source.shape != target.shape:
                warnings.append(f"Cannot fill between {rel_a} and {rel_b}: different sizes")
                continue
            span = index_b - index_a
            for mid in range(index_a + 1, index_b):
                rel_m = ordered[mid]
                if rel_m in link.anchors or rel_m in link.filled:
                    continue
                session = sessions[rel_m]
                if session.masks.shape != source.shape:
                    warnings.append(f"Cannot fill {rel_m}: different size")
                    continue
                t = (mid - index_a) / span
                blob = interpolate_binary(source, target, t)
                blob &= session.masks == 0
                if not blob.any():
                    warnings.append(f"No free pixels to fill on {rel_m}")
                    continue
                session.checkpoint()
                new_label = int(session.masks.max()) + 1
                session.masks[blob] = new_label
                session.set_manual(new_label, False)
                session.dirty = True
                link.filled[rel_m] = new_label
    return warnings
