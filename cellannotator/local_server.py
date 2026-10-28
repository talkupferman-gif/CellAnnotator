"""Start the account server on this computer when it is not already running."""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_URL = "http://127.0.0.1:8765"
_thread: threading.Thread | None = None


def server_data_dir() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home()))
    folder = base / "CellAnnotator" / "server_data"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost"}


def server_is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=0.4) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def ensure_local_server(url: str = DEFAULT_URL) -> bool:
    """Return True when a local account server is reachable."""
    global _thread
    if server_is_up(url):
        return True
    if not is_local_url(url):
        return False
    if _thread is None or not _thread.is_alive():
        from cellannotator.server.api import create_app

        import uvicorn

        app = create_app(server_data_dir())
        port = urlparse(url).port or 8765
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        _thread = threading.Thread(target=server.run, name="cellannotator-server", daemon=True)
        _thread.start()
    for _ in range(50):
        if server_is_up(url):
            return True
        time.sleep(0.1)
    return False
