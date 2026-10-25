"""Run a Cellpose model. Training is intentionally not implemented yet."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class ModelRunner:
    """One place the canvas asks for masks. A later training command can sit beside ``run``."""

    def run(
        self,
        image: np.ndarray,
        model_name: str = "cyto3",
        weights: str | None = None,
        diameter: float | None = None,
        channels: list[int] | None = None,
    ) -> np.ndarray:
        try:
            from cellpose import models
        except ImportError as exc:
            raise RuntimeError(
                "Cellpose is not installed. From the project folder run: pip install cellpose"
            ) from exc

        gpu = _gpu_available()
        pretrained = weights or model_name
        model = _build_model(models, gpu, pretrained)
        kwargs = {"channels": channels or [0, 0]}
        if diameter is not None:
            kwargs["diameter"] = diameter
        result = model.eval(image, **kwargs)
        masks = result[0] if isinstance(result, tuple) else result
        masks = np.asarray(masks)
        if masks.ndim != 2:
            raise RuntimeError(f"Cellpose returned masks with shape {masks.shape}, expected a 2D label image")
        return masks.astype(np.int32)


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _build_model(models, gpu: bool, pretrained: str):
    errors: list[str] = []
    attempts = (
        lambda: models.CellposeModel(gpu=gpu, pretrained_model=pretrained),
        lambda: models.CellposeModel(gpu=gpu, model_type=pretrained),
        lambda: models.Cellpose(gpu=gpu, model_type=pretrained),
    )
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:  # noqa: BLE001 - try the next Cellpose constructor
            errors.append(str(exc))
    raise RuntimeError("Could not start Cellpose. " + " | ".join(errors))


def weights_label(path: str | None) -> str:
    return Path(path).name if path else ""
