"""The app symbol used on the window, the home page, and the desktop shortcut."""

from __future__ import annotations

import sys
from pathlib import Path


def icon_path(name: str = "icon.png") -> Path:
    bundled = Path(__file__).resolve().parent / "assets" / name
    if bundled.is_file():
        return bundled
    root = getattr(sys, "_MEIPASS", None)
    if root:
        extracted = Path(root) / "cellannotator" / "assets" / name
        if extracted.is_file():
            return extracted
    return bundled
