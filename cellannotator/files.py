from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_rel(value: str) -> str:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError(f"Unsafe path: {value}")
    text = path.as_posix()
    if text in {"", "."}:
        raise ValueError(f"Unsafe path: {value}")
    return text


_NATURAL = re.compile(r"(\d+)")


def natural_key(value: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in _NATURAL.split(value)]
