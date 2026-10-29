import numpy as np
import pytest
import torch

from cellannotator.ranktrain import (
    build_grade_maps,
    build_rank_net,
    graded_loss,
    image_can_train,
    recover_instances,
    train_from_rankings,
)


def test_grades_four_and_five_share_targets_at_different_weights():
    masks = np.zeros((40, 40), np.int32)
    masks[5:15, 5:15] = 1
    masks[5:15, 22:32] = 2
    maps = build_grade_maps(masks, np.array([5, 4]))
    assert maps["fg_target"][9, 9] == 1
    assert maps["fg_target"][9, 26] == 1
    assert maps["fg_weight"][9, 9] == pytest.approx(1)
    assert maps["fg_weight"][9, 26] == pytest.approx(0.5)
    assert maps["hover_weight"][9, 9] == pytest.approx(1)
    assert maps["hover_weight"][9, 26] == pytest.approx(0.5)
    assert maps["dice_weight"][9, 9] == pytest.approx(1)
    assert maps["dice_weight"][9, 26] == pytest.approx(0.5)
    assert maps["hover"][0, 9, 6] > 0
    assert maps["hover"][1, 6, 9] > 0
    assert maps["hover"][0, 9, 23] > 0
    assert maps["boundary_target"][5, 9] == 1
    assert maps["boundary_target"][9, 9] == 0
    assert maps["boundary_weight"][9, 9] == pytest.approx(1)
    assert maps["fg_target"][4, 9] == 0
    assert maps["fg_weight"][4, 9] == pytest.approx(1)


def test_grade_three_covers_the_interior_and_leaves_the_halo():
    masks = np.zeros((40, 40), np.int32)
    masks[10:20, 10:20] = 1
    maps = build_grade_maps(masks, np.array([3]))
    assert maps["fg_target"][15, 15] == 1
    assert maps["fg_weight"][15, 15] == pytest.approx(0.5)
    assert maps["hover_weight"][15, 15] == pytest.approx(0.5)
    assert maps["hover"][0, 15, 11] > 0
    assert maps["dice_weight"][15, 15] == 0
    assert maps["boundary_weight"][15, 15] == 0
    assert maps["boundary_target"][10, 15] == 0
    assert maps["fg_weight"][8, 15] == 0
    assert maps["hover_weight"][8, 15] == 0
    assert maps["boundary_weight"][8, 15] == 0
    assert maps["fg_weight"][0, 0] == pytest.approx(0.1)

    foreground = torch.tensor(maps["fg_target"])
    hover = torch.tensor(maps["hover"])
    boundary = torch.zeros((40, 40), dtype=torch.float32)
    base = graded_loss(foreground, hover, boundary, maps)
    changed = foreground.clone()
    changed[8, 15] = 1
    again = graded_loss(changed, hover, boundary, maps)
    assert torch.allclose(base, again)
    assert image_can_train(masks, np.array([3]))


def test_grade_two_and_unrated_are_excluded():
    masks = np.zeros((40, 40), np.int32)
    masks[5:12, 5:12] = 1
    masks[5:12, 25:32] = 2
    quality = np.array([2, 0])
    maps = build_grade_maps(masks, quality)
    for name in ("fg_weight", "hover_weight", "boundary_weight", "dice_weight"):
        assert maps[name][8, 8] == 0
        assert maps[name][8, 28] == 0
        assert maps[name][12, 8] == 0
    assert maps["fg_weight"][35, 35] == pytest.approx(0.1)
    assert not image_can_train(masks, quality)


def test_grade_one_punishes_foreground_hover_and_boundary():
    masks = np.ones((8, 8), np.int32)
    maps = build_grade_maps(masks, np.array([1]))
    assert maps["fg_weight"].sum() == 0
    assert maps["hover_weight"].sum() == 0
    assert maps["boundary_weight"].sum() == 0
    assert maps["punish"].sum() == 64
    quiet = graded_loss(torch.zeros(8, 8), torch.zeros(2, 8, 8), torch.zeros(8, 8), maps)
    loud_foreground = graded_loss(torch.ones(8, 8), torch.zeros(2, 8, 8), torch.zeros(8, 8), maps)
    loud_hover = graded_loss(torch.zeros(8, 8), torch.ones(2, 8, 8), torch.zeros(8, 8), maps)
    loud_boundary = graded_loss(torch.zeros(8, 8), torch.zeros(2, 8, 8), torch.ones(8, 8), maps)
    assert float(loud_foreground) > float(quiet)
    assert float(loud_hover) > float(quiet)
    assert float(loud_boundary) > float(quiet)
    assert image_can_train(masks, np.array([1]))


def test_small_network_returns_full_resolution_maps():
    net = build_rank_net(1, full=False, img_size=64)
    net.train()
    net.use_checkpoint = True
    batch = torch.rand(1, 1, 64, 64)
    output = net(batch)
    assert output["foreground"].shape == (1, 1, 64, 64)
    assert output["hover"].shape == (1, 2, 64, 64)
    assert output["boundary"].shape == (1, 1, 64, 64)
    loss = output["foreground"].sum() + output["hover"].sum() + output["boundary"].sum()
    loss.backward()


def test_hover_splits_two_touching_cells():
    foreground = np.zeros((48, 48), np.float32)
    hover = np.zeros((2, 48, 48), np.float32)
    boundary = np.zeros((48, 48), np.float32)

    def paint(top, bottom, left, right):
        foreground[top:bottom, left:right] = 1
        center_row = (top + bottom - 1) / 2
        center_col = (left + right - 1) / 2
        rows, cols = np.mgrid[top:bottom, left:right]
        hover[0, top:bottom, left:right] = (center_col - cols) / 128
        hover[1, top:bottom, left:right] = (center_row - rows) / 128

    paint(8, 28, 6, 22)
    paint(8, 28, 22, 40)
    labels = recover_instances(foreground, hover, boundary, min_size=15)
    assert int(labels.max()) == 2
    left = labels[8:28, 6:22]
    right = labels[8:28, 22:40]
    left_id = int(np.bincount(left.ravel()).argmax())
    right_id = int(np.bincount(right.ravel()).argmax())
    assert left_id != right_id
    assert float((left == left_id).mean()) > 0.8
    assert float((right == right_id).mean()) > 0.8


def test_training_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    image = np.zeros((8, 8), np.float32)
    masks = np.ones((8, 8), np.int32)
    with pytest.raises(RuntimeError, match="CUDA"):
        train_from_rankings([image], [masks], [np.array([5])], [0, 0], "ranking.pt", n_epochs=1)
