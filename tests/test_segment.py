import numpy as np
import pytest

from cellannotator.segment import image_for_model, resolve_pretrained


def test_gray_image_stays_one_plane():
    image = np.zeros((8, 6), np.uint16)
    prepared = image_for_model(image)
    assert prepared.shape == (8, 6)


def test_two_layers_stay_on_the_last_axis():
    image = np.zeros((2, 32, 24), np.uint8)
    image[0] = 10
    image[1] = 20
    prepared = image_for_model(image)
    assert prepared.shape == (32, 24, 2)
    assert prepared[0, 0, 0] == 10
    assert prepared[0, 0, 1] == 20


def test_custom_weights_must_exist(tmp_path):
    missing = tmp_path / "missing_model"
    with pytest.raises(RuntimeError, match="Cannot find the weights file"):
        resolve_pretrained("cyto3", str(missing))


def test_cellpose4_name_is_rejected():
    with pytest.raises(RuntimeError, match="not a Cellpose 3 model"):
        resolve_pretrained("cpsam", None)


def test_custom_weights_path_is_used(tmp_path):
    weights = tmp_path / "my_model"
    weights.write_bytes(b"not-a-real-model")
    assert resolve_pretrained("cyto3", str(weights)) == str(weights)
