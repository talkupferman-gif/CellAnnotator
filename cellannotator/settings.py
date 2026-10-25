from __future__ import annotations

import json
import os
from pathlib import Path


def settings_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home()))
    folder = base / "CellAnnotator"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "settings.json"


class Settings:
    def __init__(self, server_url: str = "http://127.0.0.1:8765", token: str | None = None, email: str = ""):
        self.server_url = server_url
        self.token = token
        self.email = email

    @classmethod
    def load(cls) -> Settings:
        path = settings_path()
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            server_url=data.get("server_url") or "http://127.0.0.1:8765",
            token=data.get("token"),
            email=data.get("email") or "",
        )

    def save(self) -> None:
        settings_path().write_text(
            json.dumps({"server_url": self.server_url, "token": self.token, "email": self.email}, indent=2),
            encoding="utf-8",
        )
