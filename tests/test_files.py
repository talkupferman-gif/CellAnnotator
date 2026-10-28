import numpy as np

from cellannotator.files import natural_key
from cellannotator.images import composite_layers
from cellannotator.project import Project, scan_images
from cellannotator.segio import MaskSession, load_session_for_image, save_session, seg_pair


def test_natural_sort_orders_numbers():
    assert natural_key("img2.tif") < natural_key("img10.tif")


def test_working_copy_does_not_replace_committed(tmp_path):
    image = tmp_path / "cell.tif"
    image.write_bytes(b"not-a-real-tiff")
    committed, working = seg_pair(image)
    original = MaskSession.empty(4, 4, "cell.tif")
    original.masks[0, 0] = 1
    original.readiness = 3
    save_session(committed, original)

    edited = load_session_for_image(image, (4, 4))
    edited.masks[1, 1] = 2
    edited.readiness = 4
    edited.dirty = True
    save_session(working, edited)

    reloaded_committed = load_session_for_image(image, (4, 4))
    # Working file wins while it exists, which is the copy you are editing.
    assert reloaded_committed.masks[1, 1] == 2
    stored = np.load(committed, allow_pickle=True).item()
    assert stored["masks"][1, 1] == 0
    assert stored["readiness"] == 3
    assert np.load(working, allow_pickle=True).item()["readiness"] == 4


def test_scan_skips_mask_files(tmp_path):
    (tmp_path / "b.tif").write_bytes(b"b")
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "a_seg.npy").write_bytes(b"x")
    (tmp_path / "annotation_meta.json").write_text("{}", encoding="utf-8")
    names = [path.name for path in scan_images(tmp_path)]
    assert names == ["a.png", "b.tif"]
    project = Project(tmp_path)
    assert project.rels == ["a.png", "b.tif"]


def test_two_layers_keep_their_colors():
    red = np.zeros((8, 8), np.uint8)
    red[:, :4] = 255
    blue = np.zeros((8, 8), np.uint8)
    blue[:, 4:] = 255
    image = composite_layers([red, blue], ["#FF0000", "#0000FF"])
    assert image[0, 1, 0] > 200
    assert image[0, 1, 2] < 30
    assert image[0, 6, 2] > 200
    assert image[0, 6, 0] < 30
