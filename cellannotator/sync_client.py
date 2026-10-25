"""HTTP client for the project server."""

from __future__ import annotations

import json
from pathlib import Path

import httpx


class SyncError(Exception):
    def __init__(self, message: str, status: int = 0, payload: dict | None = None):
        super().__init__(message)
        self.status = status
        self.payload = payload or {}


class SyncClient:
    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = httpx.request(method, self.base_url + path, headers=self._headers(), timeout=120, **kwargs)
        except httpx.HTTPError as exc:
            raise SyncError(f"Cannot reach the server at {self.base_url}") from exc
        if response.status_code >= 400:
            payload: dict = {}
            detail = response.text
            try:
                body = response.json()
                detail = body.get("detail", body)
                if isinstance(detail, dict):
                    payload = detail
                    detail = payload.get("error") or "The server rejected the request"
                elif isinstance(detail, str):
                    payload = {"error": detail}
            except Exception:
                payload = {}
            raise SyncError(str(detail), response.status_code, payload)
        return response

    def register(self, email: str, password: str) -> dict:
        return self._request("POST", "/register", json={"email": email, "password": password}).json()

    def login(self, email: str, password: str) -> dict:
        return self._request("POST", "/login", json={"email": email, "password": password}).json()

    def list_projects(self) -> list[dict]:
        return self._request("GET", "/projects").json()

    def create_project(self, name: str, kind: str, manifest: dict) -> dict:
        return self._request("POST", "/projects", json={"name": name, "kind": kind, "manifest": manifest}).json()

    def get_project(self, project_id: str) -> dict:
        return self._request("GET", f"/projects/{project_id}").json()

    def update_manifest(self, project_id: str, manifest: dict) -> dict:
        return self._request("POST", f"/projects/{project_id}/manifest", json={"manifest": manifest}).json()

    def invite(self, project_id: str, email: str) -> dict:
        return self._request("POST", f"/projects/{project_id}/invites", json={"email": email}).json()

    def list_commits(self, project_id: str) -> list[dict]:
        return self._request("GET", f"/projects/{project_id}/commits").json()

    def commit(
        self,
        project_id: str,
        parent_id: str | None,
        files: list[tuple[str, Path]],
        resolutions: dict[str, str] | None = None,
        message: str = "Update masks",
    ) -> dict:
        uploads = []
        handles = []
        try:
            for relpath, path in files:
                handle = path.open("rb")
                handles.append(handle)
                uploads.append(("uploads", (Path(relpath).name, handle, "application/octet-stream")))
            response = self._request(
                "POST",
                f"/projects/{project_id}/commits",
                data={
                    "parent_id": parent_id or "",
                    "message": message,
                    "paths": json.dumps([relpath for relpath, _path in files]),
                    "resolutions": json.dumps(resolutions or {}),
                },
                files=uploads,
            )
        finally:
            for handle in handles:
                handle.close()
        return response.json()

    def archive(self, project_id: str, commit_id: str) -> bytes:
        return self._request("GET", f"/projects/{project_id}/commits/{commit_id}/archive").content
