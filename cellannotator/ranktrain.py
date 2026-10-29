"""Train and run a ranking model. Grades 1–5 are different teaching signals."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn

from cellannotator.masks import edge_mask
from cellannotator.segment import image_for_model


TILE = 384
PATCH = 16
HOVER_SCALE = 128.0
FULL_TAPS = (5, 11, 17, 23)
BACKGROUND_WEIGHT = 0.1
RING_RADIUS = 3
HALO_RADIUS = 8
EXCLUDE_RADIUS = 3
CROPS_PER_IMAGE = 4
MIN_INSTANCE_SIZE = 15
OVERLAP = 64
TRAINABLE_GRADES = (1, 3, 4, 5)
_STRUCTURE = ndimage.generate_binary_structure(2, 2)


def image_can_train(masks: np.ndarray, quality: np.ndarray) -> bool:
    """True when a grade 1, 3, 4, or 5 cell is present."""
    grades = _grades_present(masks, quality)
    return any(grade in TRAINABLE_GRADES for grade in grades)


def model_channels(image: np.ndarray, channels: list[int] | None) -> np.ndarray:
    """Return float32 channels shaped (C, H, W), each scaled to 0..1."""
    prepared = image_for_model(image)
    if prepared.ndim == 2:
        planes = [prepared]
    else:
        planes = [prepared[..., index] for index in range(prepared.shape[-1])]
    cyto, nucleus = (list(channels or [0, 0]) + [0, 0])[:2]
    chosen: list[np.ndarray] = []
    if int(cyto) > 0:
        chosen.append(_plane(planes, int(cyto)))
    if int(nucleus) > 0 and int(nucleus) != int(cyto):
        chosen.append(_plane(planes, int(nucleus)))
    if not chosen:
        chosen = planes
    return np.stack([_unit_interval(plane) for plane in chosen], axis=0).astype(np.float32)


def build_grade_maps(masks: np.ndarray, quality: np.ndarray) -> dict[str, np.ndarray]:
    """Targets and weights for foreground, hover, and boundary."""
    labels = np.asarray(masks)
    grades = np.asarray(quality, dtype=np.int32).reshape(-1)
    height, width = labels.shape
    fg_target = np.zeros((height, width), np.float32)
    fg_weight = np.full((height, width), BACKGROUND_WEIGHT, np.float32)
    hover = np.zeros((2, height, width), np.float32)
    hover_weight = np.full((height, width), BACKGROUND_WEIGHT, np.float32)
    boundary_target = np.zeros((height, width), np.float32)
    boundary_weight = np.full((height, width), BACKGROUND_WEIGHT, np.float32)
    dice_weight = np.zeros((height, width), np.float32)
    punish = np.zeros((height, width), np.float32)

    cells = [int(value) for value in np.unique(labels) if int(value) > 0]
    by_grade = {label: _grade(label, grades) for label in cells}
    exclude = np.zeros((height, width), dtype=bool)
    halo = np.zeros((height, width), dtype=bool)
    protected = np.zeros((height, width), dtype=bool)
    for label, grade in by_grade.items():
        cell = labels == label
        if grade in (0, 2):
            exclude |= _dilate(cell, EXCLUDE_RADIUS)
        elif grade == 3:
            halo |= _dilate(cell, HALO_RADIUS) & ~cell
            protected |= cell
        elif grade in (4, 5):
            protected |= cell

    occupied = labels > 0
    for label, grade in by_grade.items():
        cell = labels == label
        if grade not in (4, 5, 3):
            continue
        weight = 1.0 if grade == 5 else 0.5
        rows, cols = np.nonzero(cell)
        if rows.size == 0:
            continue
        center_row = float(rows.mean())
        center_col = float(cols.mean())
        hover[0, rows, cols] = np.clip((center_col - cols) / HOVER_SCALE, -1, 1)
        hover[1, rows, cols] = np.clip((center_row - rows) / HOVER_SCALE, -1, 1)
        fg_target[cell] = 1
        fg_weight[cell] = weight
        hover_weight[cell] = weight
        if grade == 3:
            boundary_weight[cell] = 0
            continue
        rim = edge_mask(labels) & cell
        boundary_target[rim] = 1
        boundary_weight[rim] = weight
        boundary_target[cell & ~rim] = 0
        boundary_weight[cell & ~rim] = weight
        ring = _dilate(cell, RING_RADIUS) & ~occupied & ~halo & ~exclude
        fg_target[ring] = 0
        fg_weight[ring] = weight
        boundary_target[ring] = 0
        boundary_weight[ring] = weight
        dice_weight[cell | ring] = weight

    unlabeled = (halo | exclude | (_punish_region(labels, by_grade))) & ~protected
    for weight_map in (fg_weight, hover_weight, boundary_weight, dice_weight):
        weight_map[unlabeled] = 0
    punish = _punish_region(labels, by_grade).astype(np.float32)
    fg_weight[punish > 0] = 0
    hover_weight[punish > 0] = 0
    boundary_weight[punish > 0] = 0
    dice_weight[punish > 0] = 0
    return {
        "fg_target": fg_target,
        "fg_weight": fg_weight,
        "hover": hover,
        "hover_weight": hover_weight,
        "boundary_target": boundary_target,
        "boundary_weight": boundary_weight,
        "dice_weight": dice_weight,
        "punish": punish,
    }


def _punish_region(labels: np.ndarray, by_grade: dict[int, int]) -> np.ndarray:
    region = np.zeros(labels.shape, dtype=bool)
    for label, grade in by_grade.items():
        if grade == 1:
            region |= labels == label
    return region


def graded_loss(
    foreground: torch.Tensor,
    hover: torch.Tensor,
    boundary: torch.Tensor,
    maps: dict[str, np.ndarray],
) -> torch.Tensor:
    """Weighted foreground, hover, and boundary loss, plus the grade-1 penalty."""
    fg_target = _as_tensor(maps["fg_target"], foreground)
    fg_weight = _as_tensor(maps["fg_weight"], foreground)
    hover_target = _as_tensor(maps["hover"], hover)
    hover_weight = _as_tensor(maps["hover_weight"], foreground)
    boundary_target = _as_tensor(maps["boundary_target"], boundary)
    boundary_weight = _as_tensor(maps["boundary_weight"], boundary)
    dice_weight = _as_tensor(maps["dice_weight"], foreground)
    punish = _as_tensor(maps["punish"], foreground) > 0

    fg_loss = _weighted_bce(foreground, fg_target, fg_weight)
    dice_loss = _soft_dice(foreground, fg_target, dice_weight)
    hover_error = ((hover - hover_target) ** 2).mean(dim=0)
    hover_loss = (hover_error * hover_weight).sum() / hover_weight.sum().clamp(min=1)
    boundary_loss = _weighted_bce(boundary, boundary_target, boundary_weight)
    if bool(punish.any()):
        magnitude = torch.linalg.vector_norm(hover, dim=0)
        punish_loss = foreground[punish].mean() + magnitude[punish].mean() + boundary[punish].mean()
    else:
        punish_loss = foreground.sum() * 0
    return fg_loss + dice_loss + hover_loss + boundary_loss + punish_loss


class FuseUp(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            _group_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            _group_norm(channels),
            nn.GELU(),
        )

    def forward(self, current: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        current = F.interpolate(current, scale_factor=2, mode="bilinear", align_corners=False)
        skip = F.interpolate(skip, size=current.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([current, skip], dim=1))


class RankNet(nn.Module):
    """ViT encoder, full-resolution decoder, foreground, hover, and boundary heads."""

    def __init__(self, encoder: nn.Module, tap: tuple[int, ...], decoder_dim: int, use_checkpoint: bool):
        super().__init__()
        self.encoder = encoder
        self.tap = tuple(tap)
        self.use_checkpoint = use_checkpoint
        embed_dim = int(encoder.embed_dim)
        patch = encoder.patch_embed.patch_size
        self.patch = int(patch[0] if isinstance(patch, tuple) else patch)
        self.projections = nn.ModuleList(nn.Conv2d(embed_dim, decoder_dim, 1) for _ in tap)
        self.fusions = nn.ModuleList(FuseUp(decoder_dim) for _ in range(len(tap) - 1))
        self.final = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            _group_norm(decoder_dim),
            nn.GELU(),
        )
        self.foreground_head = _head(decoder_dim, 1)
        self.hover_head = _head(decoder_dim, 2)
        self.boundary_head = _head(decoder_dim, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        height, width = images.shape[-2:]
        pad_h = (self.patch - height % self.patch) % self.patch
        pad_w = (self.patch - width % self.patch) % self.patch
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h))
        maps = self._token_maps(images)
        current = maps[-1]
        shallower = list(reversed(maps[:-1]))
        for fusion, skip in zip(self.fusions, shallower):
            current = fusion(current, skip)
        current = F.interpolate(current, scale_factor=2, mode="bilinear", align_corners=False)
        current = self.final(current)
        if current.shape[-2:] != (height, width):
            current = F.interpolate(current, size=(height, width), mode="bilinear", align_corners=False)
        return {
            "foreground": torch.sigmoid(self.foreground_head(current)),
            "hover": torch.tanh(self.hover_head(current)),
            "boundary": torch.sigmoid(self.boundary_head(current)),
        }

    def _token_maps(self, images: torch.Tensor) -> list[torch.Tensor]:
        encoder = self.encoder
        tokens = encoder.patch_embed(images)
        tokens = encoder._pos_embed(tokens)
        if hasattr(encoder, "patch_drop"):
            tokens = encoder.patch_drop(tokens)
        if hasattr(encoder, "norm_pre"):
            tokens = encoder.norm_pre(tokens)
        wanted = set(self.tap)
        taken: list[torch.Tensor] = []
        for index, block in enumerate(encoder.blocks):
            if self.use_checkpoint and self.training:
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in wanted:
                taken.append(tokens)
        if len(taken) != len(self.tap):
            raise RuntimeError("The encoder did not return a feature map at every requested block.")
        grid_h = images.shape[-2] // self.patch
        grid_w = images.shape[-1] // self.patch
        prefix = int(getattr(encoder, "num_prefix_tokens", 1))
        maps = []
        for tokens, projection in zip(taken, self.projections):
            spatial = tokens[:, prefix:, :]
            batch, _count, channels = spatial.shape
            grid = spatial.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
            maps.append(projection(grid))
        return maps


def build_rank_net(in_channels: int, *, full: bool = True, img_size: int = TILE, pretrained: bool = True) -> RankNet:
    """Full builds ViT-Large. Tests pass full=False and skip the pretrained weights."""
    if full:
        import timm

        encoder = timm.create_model(
            "vit_large_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            img_size=TILE,
            in_chans=in_channels,
        )
        return RankNet(encoder, FULL_TAPS, decoder_dim=512, use_checkpoint=True)
    from timm.models.vision_transformer import VisionTransformer

    encoder = VisionTransformer(
        img_size=img_size,
        patch_size=PATCH,
        in_chans=in_channels,
        num_classes=0,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    return RankNet(encoder, (0, 1, 2, 3), decoder_dim=32, use_checkpoint=False)


def train_from_rankings(
    images: list[np.ndarray],
    labels: list[np.ndarray],
    qualities: list[np.ndarray],
    channels: list[int] | None,
    save_to: str,
    n_epochs: int = 100,
    learning_rate: float = 0.0001,
    on_epoch: Callable[[str], None] | None = None,
) -> str:
    """Train a ViT-Large ranking model. Cellpose's trainer is not used."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training from rankings needs a CUDA GPU.")
    prepared, kept_labels, kept_quality = _training_sets(images, labels, qualities, channels)
    in_channels = int(prepared[0].shape[0])
    destination = _weights_path(save_to)
    device = torch.device("cuda")
    net = build_rank_net(in_channels, full=True).to(device)
    encoder_ids = {id(param) for param in net.encoder.parameters()}
    encoder_params = [param for param in net.parameters() if id(param) in encoder_ids]
    other_params = [param for param in net.parameters() if id(param) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(learning_rate) * 0.1},
            {"params": other_params, "lr": float(learning_rate)},
        ],
        weight_decay=0.05,
    )
    rng = np.random.default_rng()
    net.train()
    for epoch in range(int(n_epochs)):
        if on_epoch is not None:
            on_epoch(f"Training from rankings… epoch {epoch + 1} of {int(n_epochs)}")
        order = rng.permutation(len(prepared))
        for index in order:
            _train_image(
                net,
                optimizer,
                device,
                prepared[int(index)],
                kept_labels[int(index)],
                kept_quality[int(index)],
                rng,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "in_channels": in_channels,
            "channels": [int(value) for value in (channels or [0, 0])],
            "img_size": TILE,
            "architecture": "vit_large_patch16_224",
        },
        destination,
    )
    return str(destination)


def run_ranking_model(image: np.ndarray, checkpoint: str, min_size: int = MIN_INSTANCE_SIZE) -> np.ndarray:
    """Tile a ranking checkpoint and recover one mask per cell."""
    if not torch.cuda.is_available():
        raise RuntimeError("Running a ranking model needs a CUDA GPU.")
    try:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(checkpoint, map_location="cpu")
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise RuntimeError("That file is not a ranking model.")
    in_channels = int(blob["in_channels"])
    channels = [int(value) for value in blob.get("channels", [0, 0])]
    prepared = model_channels(image, channels)
    if prepared.shape[0] != in_channels:
        raise RuntimeError(
            f"This ranking model was trained on {in_channels} channels. This image produced {prepared.shape[0]}."
        )
    device = torch.device("cuda")
    net = build_rank_net(in_channels, full=True, pretrained=False).to(device)
    try:
        net.load_state_dict(blob["state_dict"])
    except Exception as exc:
        raise RuntimeError("That file is not a ranking model. " + str(exc)) from exc
    net.eval()
    foreground, hover, boundary = predict_maps(net, prepared, device)
    return recover_instances(foreground, hover, boundary, min_size=min_size)


def predict_maps(
    net: RankNet,
    image: np.ndarray,
    device: torch.device,
    tile: int = TILE,
    overlap: int = OVERLAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average overlapping tiles of foreground, hover, and boundary."""
    _channels, height, width = image.shape
    foreground = np.zeros((height, width), np.float32)
    hover = np.zeros((2, height, width), np.float32)
    boundary = np.zeros((height, width), np.float32)
    counts = np.zeros((height, width), np.float32)
    step = max(tile - overlap, 1)
    row_origins = _origins(height, tile, step)
    col_origins = _origins(width, tile, step)
    net.eval()
    with torch.inference_mode():
        for top in row_origins:
            for left in col_origins:
                bottom = min(top + tile, height)
                right = min(left + tile, width)
                crop = np.zeros((image.shape[0], tile, tile), np.float32)
                crop[:, : bottom - top, : right - left] = image[:, top:bottom, left:right]
                batch = torch.from_numpy(crop).unsqueeze(0).to(device)
                prediction = net(batch)
                fg = prediction["foreground"][0, 0, : bottom - top, : right - left].detach().cpu().numpy()
                hv = prediction["hover"][0, :, : bottom - top, : right - left].detach().cpu().numpy()
                bd = prediction["boundary"][0, 0, : bottom - top, : right - left].detach().cpu().numpy()
                foreground[top:bottom, left:right] += fg
                hover[:, top:bottom, left:right] += hv
                boundary[top:bottom, left:right] += bd
                counts[top:bottom, left:right] += 1
    counts = np.maximum(counts, 1)
    return foreground / counts, hover / counts, boundary / counts


def recover_instances(
    foreground: np.ndarray,
    hover: np.ndarray,
    boundary: np.ndarray,
    min_size: int = MIN_INSTANCE_SIZE,
    foreground_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
) -> np.ndarray:
    """Split the foreground where hover vectors meet, and drop tiny fragments."""
    region = (np.asarray(foreground) >= foreground_threshold) & (np.asarray(boundary) < boundary_threshold)
    if not region.any():
        return np.zeros(np.asarray(foreground).shape, np.int32)
    vectors = np.asarray(hover, dtype=np.float32)
    magnitude = np.sqrt(vectors[0] ** 2 + vectors[1] ** 2)
    smooth = ndimage.gaussian_filter(magnitude, 0.6)
    ridge = _separation_ridge(vectors) & region
    window = np.where(region & ~ridge, smooth, np.inf)
    local = ndimage.minimum_filter(window, size=9)
    minima = region & ~ridge & (smooth <= local + 1e-8)
    markers, _count = ndimage.label(minima)
    if int(markers.max()) == 0:
        components, count = ndimage.label(region & ~ridge)
        markers = components
        if count == 0:
            return np.zeros(region.shape, np.int32)
    from skimage.segmentation import watershed

    height = magnitude.copy()
    if ridge.any():
        height[ridge] = float(magnitude.max()) + 1.0
    labels = watershed(np.where(region, height, 0), markers, mask=region)
    labels = labels.astype(np.int32)
    for value in np.unique(labels):
        if int(value) <= 0:
            continue
        if int((labels == value).sum()) < int(min_size):
            labels[labels == value] = 0
    compact = np.zeros_like(labels)
    next_label = 1
    for value in np.unique(labels):
        if int(value) <= 0:
            continue
        compact[labels == value] = next_label
        next_label += 1
    return compact


def _separation_ridge(vectors: np.ndarray) -> np.ndarray:
    """Pixels where neighboring hover vectors point away from each other."""
    horizontal = vectors[0]
    vertical = vectors[1]
    ridge = np.zeros(horizontal.shape, dtype=bool)
    split_x = (horizontal[:, :-1] < -1e-3) & (horizontal[:, 1:] > 1e-3)
    split_y = (vertical[:-1, :] < -1e-3) & (vertical[1:, :] > 1e-3)
    ridge[:, 1:] |= split_x
    ridge[:, :-1] |= split_x
    ridge[1:, :] |= split_y
    ridge[:-1, :] |= split_y
    return ridge


def _train_image(net, optimizer, device, image, masks, quality, rng) -> None:
    trainable = _trainable_mask(masks, quality)
    rows, cols = np.nonzero(trainable)
    for _crop in range(CROPS_PER_IMAGE):
        top, left = _crop_origin(image.shape[1], image.shape[2], TILE, rows, cols, rng)
        crop_image, crop_masks = _take_crop(image, masks, top, left, TILE)
        maps = build_grade_maps(crop_masks, quality)
        batch = torch.from_numpy(np.ascontiguousarray(crop_image)).unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = net(batch)
        loss = graded_loss(prediction["foreground"][0, 0], prediction["hover"][0], prediction["boundary"][0, 0], maps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()


def _training_sets(images, labels, qualities, channels):
    prepared: list[np.ndarray] = []
    kept_labels: list[np.ndarray] = []
    kept_quality: list[np.ndarray] = []
    for image, label, quality in zip(images, labels, qualities):
        masks = np.asarray(label)
        grades = np.asarray(quality, dtype=np.int32).reshape(-1)
        if not image_can_train(masks, grades):
            continue
        prepared.append(model_channels(image, channels))
        kept_labels.append(masks.astype(np.int32))
        kept_quality.append(grades)
    if not prepared:
        raise RuntimeError("None of those images have a grade to train on. Use 1, 3, 4, or 5.")
    counts = {array.shape[0] for array in prepared}
    if len(counts) > 1:
        raise RuntimeError("Train on images that all have the same number of channels.")
    return prepared, kept_labels, kept_quality


def _grades_present(masks: np.ndarray, quality: np.ndarray) -> list[int]:
    grades = np.asarray(quality, dtype=np.int32).reshape(-1)
    found = []
    for label in np.unique(masks):
        label = int(label)
        if label > 0:
            found.append(_grade(label, grades))
    return found


def _grade(label: int, quality: np.ndarray) -> int:
    index = label - 1
    if index < 0 or index >= int(quality.size):
        return 0
    return int(quality[index])


def _plane(planes: list[np.ndarray], index: int) -> np.ndarray:
    if index > len(planes):
        raise RuntimeError(f"This image has {len(planes)} channels. Channel {index} is not there.")
    return planes[index - 1]


def _unit_interval(plane: np.ndarray) -> np.ndarray:
    values = np.asarray(plane, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, np.float32)
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        high = low + 1
    return np.clip((values - low) / (high - low), 0, 1)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return ndimage.binary_dilation(mask, structure=_STRUCTURE, iterations=radius)


def _as_tensor(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), device=like.device, dtype=like.dtype)


def _weighted_bce(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    clipped = prediction.clamp(1e-6, 1 - 1e-6)
    loss = -(target * clipped.log() + (1 - target) * (1 - clipped).log())
    return (loss * weight).sum() / weight.sum().clamp(min=1)


def _soft_dice(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if float(weight.sum()) == 0:
        return prediction.sum() * 0
    intersection = (prediction * target * weight).sum()
    denominator = (prediction * weight).sum() + (target * weight).sum()
    return 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(max(groups, 1), channels)


def _head(channels: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(channels, outputs, 1),
    )


def _weights_path(save_to: str) -> Path:
    destination = Path(save_to)
    if not destination.name:
        raise RuntimeError("Choose a file name for the new weights.")
    if destination.suffix.lower() not in {".pt", ".pth"}:
        destination = destination.with_suffix(".pt")
    return destination


def _origins(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, step))
    last = length - tile
    if origins[-1] != last:
        origins.append(last)
    return origins


def _trainable_mask(masks: np.ndarray, quality: np.ndarray) -> np.ndarray:
    region = np.zeros(masks.shape, dtype=bool)
    for label in np.unique(masks):
        label = int(label)
        if label > 0 and _grade(label, quality) in TRAINABLE_GRADES:
            region |= masks == label
    return region


def _crop_origin(height, width, size, rows, cols, rng) -> tuple[int, int]:
    if rows.size:
        pick = int(rng.integers(0, rows.size))
        center_row = int(rows[pick])
        center_col = int(cols[pick])
    else:
        center_row = height // 2
        center_col = width // 2
    if height <= size:
        top = 0
    else:
        top = int(np.clip(center_row - size // 2, 0, height - size))
    if width <= size:
        left = 0
    else:
        left = int(np.clip(center_col - size // 2, 0, width - size))
    return top, left


def _take_crop(image: np.ndarray, masks: np.ndarray, top: int, left: int, size: int):
    bottom = min(top + size, image.shape[1])
    right = min(left + size, image.shape[2])
    crop_image = image[:, top:bottom, left:right]
    crop_masks = masks[top:bottom, left:right]
    pad_h = size - crop_image.shape[1]
    pad_w = size - crop_image.shape[2]
    if pad_h or pad_w:
        crop_image = np.pad(crop_image, ((0, 0), (0, pad_h), (0, pad_w)))
        crop_masks = np.pad(crop_masks, ((0, pad_h), (0, pad_w)))
    return crop_image, crop_masks
