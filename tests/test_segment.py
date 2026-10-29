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


def test_training_keeps_every_mask_and_skips_empty_images():
    from cellannotator.segment import training_examples

    blank = np.zeros((32, 24), np.uint8)
    labels = np.zeros((32, 24), np.int32)
    labels[1:5, 1:5] = 1
    labels[8:12, 8:12] = 2
    prepared, kept = training_examples([blank, blank], [labels, np.zeros((32, 24), np.int32)])
    assert len(prepared) == 1
    assert set(np.unique(kept[0])) >= {1, 2}


def test_train_calls_cellpose_without_quality(monkeypatch, tmp_path):
    from cellannotator.segment import ModelRunner

    captured = {}

    class _Net:
        pass

    class _Model:
        def __init__(self, gpu=False, pretrained_model=False, model_type=None, **_kwargs):
            captured["model_type"] = model_type
            captured["pretrained_model"] = pretrained_model
            self.net = _Net()

    def _train_seg(_net, **kwargs):
        captured["kwargs"] = kwargs
        return (str(tmp_path / "models" / "finetuned"), [0.2])

    monkeypatch.setattr("cellpose.models.CellposeModel", _Model)
    monkeypatch.setattr("cellpose.train.train_seg", _train_seg)
    image = np.zeros((32, 24), np.uint8)
    labels = np.zeros((32, 24), np.int32)
    labels[0, 0] = 4
    saved = ModelRunner().train(
        [image],
        [labels],
        model_name="cyto3",
        channels=[1, 2],
        save_to=str(tmp_path / "finetuned"),
        n_epochs=3,
        learning_rate=0.2,
    )
    assert saved.endswith("finetuned")
    assert captured["model_type"] == "cyto3"
    assert captured["kwargs"]["channels"] == [1, 2]
    assert captured["kwargs"]["min_train_masks"] == 1
    assert captured["kwargs"]["n_epochs"] == 3
    assert captured["kwargs"]["learning_rate"] == 0.2
    assert captured["kwargs"]["SGD"] is True
    assert "train_probs" not in captured["kwargs"]
    assert int(captured["kwargs"]["train_labels"][0].max()) == 4


def test_run_passes_thresholds_to_eval(monkeypatch):
    from cellannotator.segment import ModelRunner

    captured = {}

    class _Model:
        def eval(self, image, **kwargs):
            captured.update(kwargs)
            return np.zeros(image.shape[:2], np.int32)

    monkeypatch.setattr("cellannotator.segment._build_model", lambda *_args: _Model())
    monkeypatch.setattr("cellannotator.segment._gpu_available", lambda: False)
    monkeypatch.setattr("cellannotator.segment._is_cellpose4_file", lambda _path: False)
    ModelRunner().run(
        np.zeros((32, 24), np.uint8),
        model_name="cyto3",
        flow_threshold=0.8,
        cellprob_threshold=-1.0,
        min_size=20,
    )
    assert captured["flow_threshold"] == 0.8
    assert captured["cellprob_threshold"] == -1.0
    assert captured["min_size"] == 20
