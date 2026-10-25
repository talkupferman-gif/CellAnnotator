from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from cellannotator.server.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="CellAnnotator project server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", default="server_data", help="Folder for the database and mask blobs")
    args = parser.parse_args()
    uvicorn.run(create_app(Path(args.data)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
