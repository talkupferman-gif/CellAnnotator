import numpy as np

from cellannotator.masks import (
    add_closed_outline,
    delete_labels,
    erase_closed_outline,
    erase_disk,
    interpolate_binary,
    merge_labels,
    split_label,
)
from cellannotator.segio import MaskSession


def _square(masks, top, left, size, label=None):
    points = [
        (left, top),
        (left + size, top),
        (left + size, top + size),
        (left, top + size),
    ]
    updated, new_label = add_closed_outline(masks, points)
    return updated, new_label


def test_outline_keeps_existing_pixels():
    masks = np.zeros((40, 40), np.int32)
    masks, first = _square(masks, 5, 5, 15)
    masks, second = _square(masks, 10, 10, 20)
    assert first == 1 and second == 2
    assert masks[8, 8] == 1
    assert masks[25, 25] == 2


def test_mask_change_counts_added_deleted_and_edited():
    from cellannotator.masks import mask_change_counts

    before = np.zeros((4, 4), np.int32)
    before[0, 0] = 1
    after = before.copy()
    after[0, 0] = 0
    after[1, 1] = 2
    after[2, 2] = 1
    counts = mask_change_counts(before, after)
    assert counts == {"added": 1, "deleted": 0, "edited": 1}


def test_closed_erase_cuts_only_the_inside():
    masks = np.zeros((40, 40), np.int32)
    masks[5:30, 5:30] = 1
    points = [(10, 10), (22, 10), (22, 22), (10, 22)]
    updated, touched = erase_closed_outline(masks, points, None)
    assert touched == {1}
    assert updated[16, 16] == 0
    assert updated[6, 6] == 1


def test_undo_then_redo_restores_the_edit():
    session = MaskSession.empty(4, 4)
    session.masks[0, 0] = 1
    session.checkpoint()
    session.masks[0, 0] = 2
    assert session.undo()
    assert session.masks[0, 0] == 1
    assert session.redo()
    assert session.masks[0, 0] == 2


def test_erase_selected_cell_only():
    masks = np.zeros((30, 30), np.int32)
    masks[5:15, 5:15] = 1
    masks[5:15, 16:25] = 2
    updated, touched = erase_disk(masks, 10, 10, 4, {1})
    assert touched == {1}
    assert updated[10, 10] == 0
    assert (updated == 2).sum() == (masks == 2).sum()


def test_erase_uses_cell_under_pointer_when_nothing_is_selected():
    masks = np.zeros((20, 20), np.int32)
    masks[2:10, 2:10] = 4
    updated, touched = erase_disk(masks, 5, 5, 2, {4})
    assert 4 in touched
    assert updated[5, 5] == 0


def test_split_leaves_one_piece_until_asked():
    masks = np.zeros((20, 40), np.int32)
    masks[2:8, 2:8] = 3
    masks[2:8, 20:28] = 3
    unchanged, created = split_label(masks, 0)
    assert created == []
    split, created = split_label(masks, 3)
    assert (split == 3).any()
    assert created
    assert (split == created[0]).any()


def test_merge_and_delete():
    masks = np.zeros((10, 20), np.int32)
    masks[:, :5] = 2
    masks[:, 8:12] = 5
    merged, keep = merge_labels(masks, {5, 2})
    assert keep == 2
    assert (merged == 5).sum() == 0
    assert (merged[:, 8:12] == 2).all()
    deleted = delete_labels(merged, {2})
    assert deleted.max() == 0


def test_interpolate_lands_between_two_disks():
    shape = (40, 60)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    left = (yy - 20) ** 2 + (xx - 15) ** 2 <= 6**2
    right = (yy - 20) ** 2 + (xx - 45) ** 2 <= 6**2
    middle = interpolate_binary(left, right, 0.5)
    band = middle[:, 25:35]
    assert band.any()
    assert middle[:, :8].sum() == 0
