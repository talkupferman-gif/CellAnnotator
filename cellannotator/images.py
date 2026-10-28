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
    return np.squeeze(array)


def split_layers(array: np.ndarray) -> list[np.ndarray]:
    """Return color layers. A long page stack stays the first page."""
    array = np.squeeze(np.asarray(array))
    if array.ndim == 2:
        return [array]
    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape {array.shape}")
    shape = array.shape
    if shape[-1] <= 16 and shape[0] > 16 and shape[1] > 16:
        return [array[..., index] for index in range(shape[-1])]
    if shape[0] <= 16 and shape[1] > 16 and shape[2] > 16:
        return [array[index] for index in range(shape[0])]
    if shape[0] > 16:
        return [array[0]]
    return [array[0]]


def plane_shape(array: np.ndarray) -> tuple[int, int]:
    plane = split_layers(array)[0]
    if plane.ndim != 2:
        raise ValueError(f"Unsupported image shape {array.shape}")
    return int(plane.shape[0]), int(plane.shape[1])


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


LAYER_PALETTE = (
    "#FF4D6D",
    "#4DABF7",
    "#69DB7C",
    "#FFD43B",
    "#B197FC",
    "#FF922B",
    "#22B8CF",
    "#F06595",
)


def default_layer_colors(count: int) -> list[str | None]:
    if count <= 1:
        return ["#F8FAFC"]
    return [LAYER_PALETTE[index % len(LAYER_PALETTE)] for index in range(count)]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))


def color_name(value: str | None) -> str:
    """A short name for a saved channel color, so a picker can say which stain it is."""
    if not value:
        return "hidden"
    red, green, blue = hex_to_rgb(value)
    spread = max(red, green, blue) - min(red, green, blue)
    if spread < 28:
        return "white" if red > 180 else "gray"
    hue = ((green - blue) / spread) % 6 if red >= green and red >= blue else None
    if hue is None and green >= red and green >= blue:
        hue = 2 + (blue - red) / spread
    if hue is None:
        hue = 4 + (red - green) / spread
    degrees = hue * 60
    if 15 <= degrees < 45:
        return "orange"
    if 45 <= degrees < 70:
        return "yellow"
    if 70 <= degrees < 165:
        return "green"
    if 165 <= degrees < 200:
        return "cyan"
    if 200 <= degrees < 255:
        return "blue"
    if 255 <= degrees < 320:
        return "purple"
    if blue > green + 15:
        return "pink"
    return "red"


def composite_layers(planes: list[np.ndarray], colors: list[str | None]) -> np.ndarray:
    """Add each visible layer, tinted by its color. None hides a layer."""
    height, width = planes[0].shape[:2]
    total = np.zeros((height, width, 3), np.float32)
    for plane, color in zip(planes, colors):
        if not color:
            continue
        source = plane if plane.ndim == 2 else plane.mean(axis=-1)
        weight = _to_uint8(source).astype(np.float32) / 255.0
        red, green, blue = hex_to_rgb(color)
        total[..., 0] += weight * red
        total[..., 1] += weight * green
        total[..., 2] += weight * blue
    return np.clip(total, 0, 255).astype(np.uint8)


def render_display(array: np.ndarray, channel: str = "layers", colors: list[str | None] | None = None) -> np.ndarray:
    if channel != "layers":
        return display_rgb(array, channel)
    planes = split_layers(array)
    chosen = list(colors) if colors else default_layer_colors(len(planes))
    if len(chosen) < len(planes):
        chosen = chosen + default_layer_colors(len(planes))[len(chosen) :]
    return composite_layers(planes, chosen[: len(planes)])


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
    planes = split_layers(array)
    if len(planes) == 1:
        return planes[0]
    return np.stack([plane if plane.ndim == 2 else plane.mean(axis=-1) for plane in planes], axis=-1)
