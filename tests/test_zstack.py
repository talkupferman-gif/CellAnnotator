import numpy as np

from cellannotator.meta import ProjectMeta
from cellannotator.segio import MaskSession
from cellannotator.zstack import fill_links


def _session(label_box):
    masks = np.zeros((30, 30), np.int32)
    if label_box is not None:
        top, left, label = label_box
        masks[top : top + 8, left : left + 8] = label
    session = MaskSession(masks, "slice")
    if label_box is not None:
        session.set_manual(label_box[2], True)
    return session


def test_fill_writes_between_anchors_and_skips_a_later_edit():
    meta = ProjectMeta(kind="zstack")
    link = meta.new_link("a.png", 1)
    link.anchors["c.png"] = 1
    sessions = {
        "a.png": _session((4, 4, 1)),
        "b.png": _session(None),
        "c.png": _session((4, 16, 1)),
    }
    warnings = fill_links(meta, sessions, ["a.png", "b.png", "c.png"])
    assert warnings == []
    assert "b.png" in link.filled
    filled_label = link.filled["b.png"]
    assert (sessions["b.png"].masks == filled_label).any()
    assert sessions["b.png"].is_manual(filled_label) is False

    sessions["b.png"].masks[sessions["b.png"].masks == filled_label] = 0
    sessions["b.png"].masks[10:18, 10:18] = filled_label
    meta.promote_filled("b.png", filled_label)
    sessions["b.png"].set_manual(filled_label, True)
    edited = sessions["b.png"].masks.copy()
    fill_links(meta, sessions, ["a.png", "b.png", "c.png"])
    assert np.array_equal(sessions["b.png"].masks, edited)
