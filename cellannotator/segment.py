"""Run and fine-tune a Cellpose 3 model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cellannotator.images import split_layers

# Cellpose 3 names. The saved lab models are Cellpose 3 files, which Cellpose 4 refuses to load.
BUILTIN_MODELS = ("cyto3", "nuclei", "cyto2", "cyto")


class ModelRunner:
    """One place the canvas asks for masks, and for a fine-tune of those masks."""

    def run(
        self,
        image: np.ndarray,
        model_name: str = "cyto3",
        weights: str | None = None,
        diameter: float | None = None,
        channels: list[int] | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
    ) -> np.ndarray:
        try:
            from cellpose import models
        except ImportError as exc:
            raise RuntimeError(
                "Cellpose is not installed. From the project folder run: "
                ".\\.venv\\Scripts\\python.exe -m pip install \"cellpose>=3.1,<4\""
            ) from exc

        prepared = image_for_model(image)
        pretrained = resolve_pretrained(model_name, weights)
        if Path(pretrained).is_file() and _is_cellpose4_file(pretrained):
            raise RuntimeError(
                "That file is a Cellpose 4 model. This app runs Cellpose 3 so your own models still open. "
                "Choose cyto3, nuclei, or a Cellpose 3 weights file."
            )
        try:
            model = _build_model(models, _gpu_available(), pretrained)
            kwargs = {
                "channels": list(channels or [0, 0]),
                "flow_threshold": float(flow_threshold),
                "cellprob_threshold": float(cellprob_threshold),
                "min_size": int(min_size),
            }
            if prepared.ndim == 3:
                kwargs["channel_axis"] = -1
            if diameter is not None:
                kwargs["diameter"] = float(diameter)
            result = model.eval(prepared, **kwargs)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Could not run that Cellpose model. " + str(exc)) from exc
        masks = result[0] if isinstance(result, tuple) else result
        masks = np.squeeze(np.asarray(masks))
        if masks.ndim == 3:
            masks = masks[0]
        if masks.ndim != 2:
            raise RuntimeError(f"Cellpose returned masks with shape {masks.shape}, expected a 2D label image")
        if masks.shape != prepared.shape[:2]:
            raise RuntimeError(
                f"Cellpose returned masks of size {masks.shape}, but the image is {prepared.shape[:2]}"
            )
        return masks.astype(np.int32)

    def train(
        self,
        images: list[np.ndarray],
        labels: list[np.ndarray],
        model_name: str = "cyto3",
        weights: str | None = None,
        channels: list[int] | None = None,
        save_to: str = "",
        n_epochs: int = 100,
        learning_rate: float = 0.2,
    ) -> str:
        """Fine-tune a Cellpose 3 model. Quality ratings are not read here."""
        try:
            from cellpose import models
            from cellpose import train as cellpose_train
        except ImportError as exc:
            raise RuntimeError(
                "Cellpose is not installed. From the project folder run: "
                ".\\.venv\\Scripts\\python.exe -m pip install \"cellpose>=3.1,<4\""
            ) from exc

        prepared, train_labels = training_examples(images, labels)
        pretrained = resolve_pretrained(model_name, weights)
        if Path(pretrained).is_file() and _is_cellpose4_file(pretrained):
            raise RuntimeError(
                "That file is a Cellpose 4 model. This app trains Cellpose 3 so your own models still open."
            )
        destination = Path(save_to)
        if not destination.name:
            raise RuntimeError("Choose a file name for the new weights.")
        if destination.parent.name == "models":
            save_path = destination.parent.parent
        else:
            save_path = destination.parent
        try:
            model = _build_trainable(models, _gpu_available(), pretrained)
            channel_axis = -1 if prepared[0].ndim == 3 else None
            saved, _losses = cellpose_train.train_seg(
                model.net,
                train_data=prepared,
                train_labels=train_labels,
                channels=list(channels or [0, 0]),
                channel_axis=channel_axis,
                min_train_masks=1,
                save_path=str(save_path),
                nimg_per_epoch=max(8, len(prepared)),
                learning_rate=float(learning_rate),
                n_epochs=int(n_epochs),
                SGD=True,
                model_name=destination.name,
            )[:2]
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Could not train that Cellpose model. " + str(exc)) from exc
        return str(saved)


def training_examples(images: list[np.ndarray], labels: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Keep every mask in images that have cells. Quality is not consulted."""
    prepared: list[np.ndarray] = []
    kept: list[np.ndarray] = []
    for image, label in zip(images, labels):
        masks = np.asarray(label)
        if masks.size == 0 or int(masks.max()) <= 0:
            continue
        prepared.append(image_for_model(image))
        kept.append(masks.astype(np.int32))
    if not prepared:
        raise RuntimeError("None of those images have cells to train on.")
    channel_counts = {image.shape[-1] if image.ndim == 3 else 1 for image in prepared}
    if len(channel_counts) > 1:
        raise RuntimeError("Train on images that all have the same number of channels.")
    return prepared, kept


def resolve_pretrained(model_name: str, weights: str | None) -> str:
    """Return a weights path or a built-in Cellpose 3 name."""
    if weights:
        path = Path(weights)
        if not path.is_file():
            raise RuntimeError(f"Cannot find the weights file: {path}")
        return str(path)
    if model_name not in BUILTIN_MODELS:
        raise RuntimeError(
            f"{model_name} is not a Cellpose 3 model. Choose cyto3, nuclei, or Custom weights."
        )
    return model_name


def image_for_model(image: np.ndarray) -> np.ndarray:
    """Return one plane, or layers stacked on the last axis, for Cellpose 3."""
    planes = [np.squeeze(plane) for plane in split_layers(np.asarray(image))]
    if not planes or planes[0].ndim != 2:
        raise RuntimeError(f"Cannot run a model on an image with shape {getattr(image, 'shape', None)}")
    if len(planes) == 1:
        return planes[0]
    return np.stack(planes, axis=-1)


def _is_cellpose4_file(path: str) -> bool:
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return isinstance(state, dict) and "W2" in state


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _build_model(models, gpu: bool, pretrained: str):
    if Path(pretrained).is_file():
        return models.CellposeModel(gpu=gpu, pretrained_model=pretrained)
    return models.Cellpose(gpu=gpu, model_type=pretrained)


def _build_trainable(models, gpu: bool, pretrained: str):
    if Path(pretrained).is_file():
        return models.CellposeModel(gpu=gpu, pretrained_model=pretrained)
    return models.CellposeModel(gpu=gpu, model_type=pretrained)


def weights_label(path: str | None) -> str:
    return Path(path).name if path else ""
