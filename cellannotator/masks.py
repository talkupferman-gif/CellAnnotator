"""Pixel operations on label masks. 0 is background. Labels are positive ints."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def rasterize_polygon(shape: tuple[int, int], points: list[tuple[float, float]]) -> np.ndarray:
    height, width = shape
    canvas = Image.new("L", (width, height), 0)
    if len(points) >= 3:
        ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in points], outline=1, fill=1)
    return np.array(canvas, dtype=bool)


def add_closed_outline(masks: np.ndarray, points: list[tuple[float, float]]) -> tuple[np.ndarray, int]:
    """Fill a closed outline. Pixels that already belong to a cell stay with that cell."""
    polygon = rasterize_polygon(masks.shape, points)
    polygon &= masks == 0
    if not polygon.any():
        return masks, 0
    out = masks.copy()
    new_label = int(out.max()) + 1
    out[polygon] = new_label
    return out, new_label


def _disk(view_origin: tuple[int, int], y: float, x: float, radius: float, view_shape: tuple[int, int]) -> np.ndarray:
    y0, x0 = view_origin
    height, width = view_shape
    yy, xx = np.ogrid[y0 : y0 + height, x0 : x0 + width]
    return (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2


def erase_disk(masks: np.ndarray, y: float, x: float, radius: float, labels: set[int]) -> tuple[np.ndarray, set[int]]:
    if radius <= 0 or not labels:
        return masks, set()
    height, width = masks.shape
    reach = int(math.ceil(radius))
    y0 = max(0, int(math.floor(y)) - reach)
    y1 = min(height, int(math.floor(y)) + reach + 1)
    x0 = max(0, int(math.floor(x)) - reach)
    x1 = min(width, int(math.floor(x)) + reach + 1)
    if y0 >= y1 or x0 >= x1:
        return masks, set()
    circle = _disk((y0, x0), y, x, radius, (y1 - y0, x1 - x0))
    view = masks[y0:y1, x0:x1]
    hit = circle & np.isin(view, list(labels))
    if not hit.any():
        return masks, set()
    touched = {int(value) for value in np.unique(view[hit])}
    out = masks.copy()
    region = out[y0:y1, x0:x1]
    region[hit] = 0
    return out, touched


def erase_segment(
    masks: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
    labels: set[int],
) -> tuple[np.ndarray, set[int]]:
    x0, y0 = start
    x1, y1 = end
    distance = math.hypot(x1 - x0, y1 - y0)
    step = max(radius * 0.5, 1.0)
    count = max(1, int(distance / step))
    out = masks
    touched: set[int] = set()
    for index in range(count + 1):
        t = index / count
        out, hit = erase_disk(out, y0 + (y1 - y0) * t, x0 + (x1 - x0) * t, radius, labels)
        touched |= hit
    return out, touched


def merge_labels(masks: np.ndarray, labels: set[int]) -> tuple[np.ndarray, int]:
    use = sorted(label for label in labels if label > 0 and (masks == label).any())
    if len(use) < 2:
        return masks, 0
    keep = use[0]
    out = masks.copy()
    for label in use[1:]:
        out[masks == label] = keep
    return out, keep


def delete_labels(masks: np.ndarray, labels: set[int]) -> np.ndarray:
    out = masks.copy()
    for label in labels:
        if label > 0:
            out[out == label] = 0
    return out


def split_label(masks: np.ndarray, label: int) -> tuple[np.ndarray, list[int]]:
    """Break disconnected pieces of one label. The largest piece keeps the id."""
    if label <= 0:
        return masks, []
    region = masks == label
    if not region.any():
        return masks, []
    structure = ndimage.generate_binary_structure(2, 1)
    components, count = ndimage.label(region, structure=structure)
    if count <= 1:
        return masks, []
    sizes = [(int((components == index).sum()), index) for index in range(1, count + 1)]
    sizes.sort(reverse=True)
    out = masks.copy()
    next_id = int(out.max()) + 1
    created: list[int] = []
    for rank, (_, index) in enumerate(sizes):
        piece = components == index
        if rank == 0:
            out[piece] = label
        else:
            out[piece] = next_id
            created.append(next_id)
            next_id += 1
    return out, created


def signed_distance(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    if not binary.any():
        return -np.ones(binary.shape, np.float32)
    if binary.all():
        return np.ones(binary.shape, np.float32)
    inside = ndimage.distance_transform_edt(binary)
    outside = ndimage.distance_transform_edt(~binary)
    return (inside - outside).astype(np.float32)


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def _shift_bool(mask: np.ndarray, delta_row: float, delta_col: float) -> np.ndarray:
    shifted = ndimage.shift(mask.astype(np.float32), (delta_row, delta_col), order=0, cval=0.0)
    return shifted > 0.5


def interpolate_binary(mask_a: np.ndarray, mask_b: np.ndarray, t: float) -> np.ndarray:
    """Blend two slices of one cell.

    The masks are moved to the in-between position first. A plain blend leaves
    a hole when the cell moved farther than its radius between slices.
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError("Cannot interpolate masks with different shapes")
    center_a = _centroid(mask_a)
    center_b = _centroid(mask_b)
    if center_a is None and center_b is None:
        return np.zeros(mask_a.shape, dtype=bool)
    if center_a is None:
        return mask_b.astype(bool).copy()
    if center_b is None:
        return mask_a.astype(bool).copy()
    target = (1.0 - t) * center_a + t * center_b
    aligned_a = _shift_bool(mask_a, *(target - center_a))
    aligned_b = _shift_bool(mask_b, *(target - center_b))
    blended = (1.0 - t) * signed_distance(aligned_a) + t * signed_distance(aligned_b)
    return blended > 0


def edge_mask(masks: np.ndarray) -> np.ndarray:
    edge = np.zeros(masks.shape, dtype=bool)
    edge[1:] |= masks[1:] != masks[:-1]
    edge[:-1] |= masks[:-1] != masks[1:]
    edge[:, 1:] |= masks[:, 1:] != masks[:, :-1]
    edge[:, :-1] |= masks[:, :-1] != masks[:, 1:]
    edge &= masks != 0
    return edge
