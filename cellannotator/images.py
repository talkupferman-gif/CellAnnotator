"""Load microscopy images and turn them into an 8-bit RGB view."""

from __future__ import annotations

from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def page_count(path: Path) -> int:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return 1
    import tifffile

    with tifffile.TiffFile(path) as tiff:
        return len(tiff.pages)


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile

        array = tifffile.imread(path)
    else:
        from PIL import Image

        with Image.open(path) as handle:
            array = np.array(handle)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] > 4 and array.shape[-1] not in {3, 4}:
        # A multi-page file is not a folder z-stack. Show the first page.
        array = array[0]
    return array


def plane_shape(array: np.ndarray) -> tuple[int, int]:
    if array.ndim == 2:
        return int(array.shape[0]), int(array.shape[1])
    if array.ndim == 3 and array.shape[-1] <= 4:
        return int(array.shape[0]), int(array.shape[1])
    if array.ndim == 3:
        return int(array.shape[1]), int(array.shape[2])
    raise ValueError(f"Unsupported image shape {array.shape}")


def _to_uint8(plane: np.ndarray) -> np.ndarray:
    plane = np.asarray(plane)
    if plane.dtype == np.uint8:
        return plane
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros(plane.shape, np.uint8)
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        high = low + 1
    scaled = np.clip((plane.astype(np.float32) - low) / (high - low), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _planes(array: np.ndarray) -> list[np.ndarray]:
    if array.ndim == 2:
        gray = _to_uint8(array)
        return [gray, gray, gray]
    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape {array.shape}")
    if array.shape[-1] <= 4 and array.shape[0] > 4 and array.shape[1] > 4:
        count = array.shape[-1]
        planes = [_to_uint8(array[..., index]) for index in range(min(3, count))]
    elif array.shape[0] <= 4:
        planes = [_to_uint8(array[index]) for index in range(min(3, array.shape[0]))]
    else:
        gray = _to_uint8(array[0])
        planes = [gray, gray, gray]
    while len(planes) < 3:
        planes.append(np.zeros_like(planes[0]))
    return planes[:3]


def display_rgb(array: np.ndarray, channel: str = "gray") -> np.ndarray:
    red, green, blue = _planes(array)
    if channel == "r":
        return np.stack([red, red, red], axis=-1)
    if channel == "g":
        return np.stack([green, green, green], axis=-1)
    if channel == "b":
        return np.stack([blue, blue, blue], axis=-1)
    return np.stack([red, green, blue], axis=-1)


def for_model(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] <= 4 and array.shape[0] < array.shape[-1]:
        return np.moveaxis(array, 0, -1)
    return array
