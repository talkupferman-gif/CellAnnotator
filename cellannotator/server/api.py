"""Account and project server. Images stay on each computer; this stores masks."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from cellannotator.files import safe_rel
from cellannotator.masks import mask_change_counts
from cellannotator.server.db import Database
from cellannotator.sync_logic import merge_files


class RegisterBody(BaseModel):
    email: str
    password: str


class ProjectBody(BaseModel):
    name: str
    kind: str
    manifest: dict
    description: str = ""


class InviteBody(BaseModel):
    email: str
    role: str = "helper"


class ManifestBody(BaseModel):
    manifest: dict


class DescriptionBody(BaseModel):
    description: str


class LockBody(BaseModel):
    locked: bool


def _masks(data: bytes):
    try:
        loaded = np.load(io.BytesIO(data), allow_pickle=True)
        item = loaded.item() if getattr(loaded, "shape", ()) == () else loaded
        if isinstance(item, dict) and "masks" in item:
            return np.asarray(item["masks"])
    except Exception:
        return None
    return None


def _stats_json(database: Database, previous_id: str | None, files: dict[str, str]) -> str:
    previous = database.commit_files(previous_id)
    rows = []
    for path, blob_hash in files.items():
        if not path.endswith("_seg.npy") or previous.get(path) == blob_hash:
            continue
        after = _masks(database.read_blob(blob_hash))
        if after is None:
            continue
        before_hash = previous.get(path)
        before = _masks(database.read_blob(before_hash)) if before_hash else None
        rows.append({"path": path, **mask_change_counts(before, after)})
    return json.dumps(rows)


def create_app(data_dir: str | Path) -> FastAPI:
    database = Database(Path(data_dir))
    app = FastAPI(title="CellAnnotator")
    app.state.db = database

    def current_user(authorization: str | None = Header(default=None)):
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Login required")
        user = database.user_for_token(authorization.split(" ", 1)[1].strip())
        if user is None:
            raise HTTPException(status_code=401, detail="Login required")
        return user

    def member_project(project_id: str, user=Depends(current_user)):
        if database.get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Unknown project")
        if not database.is_member(project_id, user["id"]):
            raise HTTPException(status_code=403, detail="You are not on this project")
        return user

    def require_role(project_id: str, user, allowed: set[str]) -> str:
        role = database.role_of(project_id, user["id"])
        if role not in allowed:
            raise HTTPException(status_code=403, detail="You cannot do that on this project")
        return role or ""

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/register")
    def register(body: RegisterBody):
        email = body.email.strip().lower()
        if "@" not in email:
            raise HTTPException(status_code=400, detail="Enter an email address")
        if len(body.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        try:
            database.create_user(email, body.password)
        except Exception as exc:  # noqa: BLE001
            if "UNIQUE" in str(exc).upper():
                raise HTTPException(status_code=409, detail="An account with that email already exists") from exc
            raise
        token = database.authenticate(email, body.password)
        return {"token": token, "email": email}

    @app.post("/login")
    def login(body: RegisterBody):
        email = body.email.strip().lower()
        token = database.authenticate(email, body.password)
        if not token:
            raise HTTPException(status_code=401, detail="Email or password is wrong")
        return {"token": token, "email": email}

    @app.get("/projects")
    def list_projects(user=Depends(current_user)):
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "kind": row["kind"],
                "head_id": row["head_id"],
                "description": row["description"],
                "status": row["status"],
                "role": row["role"],
                "owner_email": row["owner_email"],
            }
            for row in database.projects_for(user["id"])
        ]

    @app.post("/projects")
    def create_project(body: ProjectBody, user=Depends(current_user)):
        if body.kind not in {"unrelated", "zstack"}:
            raise HTTPException(status_code=400, detail="Kind must be unrelated or zstack")
        if not body.name.strip():
            raise HTTPException(status_code=400, detail="Name the project")
        project_id = database.create_project(
            user["id"], body.name.strip(), body.kind, json.dumps(body.manifest), body.description.strip()
        )
        return {"id": project_id}

    @app.get("/projects/{project_id}")
    def get_project(project_id: str, user=Depends(member_project)):
        project = database.get_project(project_id)
        owner = database.conn.execute("SELECT email FROM users WHERE id = ?", (project["owner_id"],)).fetchone()
        return {
            "id": project["id"],
            "name": project["name"],
            "kind": project["kind"],
            "head_id": project["head_id"],
            "description": project["description"],
            "status": project["status"],
            "owner_email": owner["email"] if owner else "",
            "my_role": database.role_of(project_id, user["id"]),
            "staff": [{"email": row["email"], "role": row["role"]} for row in database.staff(project_id)],
            "manifest": json.loads(database.manifest(project_id)),
        }

    @app.post("/projects/{project_id}/description")
    def update_description(project_id: str, body: DescriptionBody, user=Depends(member_project)):
        require_role(project_id, user, {"owner", "admin"})
        database.set_description(project_id, body.description.strip())
        return {"ok": True}

    @app.post("/projects/{project_id}/lock")
    def set_lock(project_id: str, body: LockBody, user=Depends(member_project)):
        require_role(project_id, user, {"owner"})
        database.set_status(project_id, "locked" if body.locked else "open")
        return {"status": "locked" if body.locked else "open"}

    @app.post("/projects/{project_id}/manifest")
    def update_manifest(project_id: str, body: ManifestBody, user=Depends(member_project)):
        database.set_manifest(project_id, json.dumps(body.manifest))
        return {"ok": True}

    @app.post("/projects/{project_id}/invites")
    def invite(project_id: str, body: InviteBody, user=Depends(member_project)):
        email = body.email.strip().lower()
        if "@" not in email:
            raise HTTPException(status_code=400, detail="Enter an email address")
        if body.role not in {"admin", "helper"}:
            raise HTTPException(status_code=400, detail="Role must be admin or helper")
        require_role(project_id, user, {"owner", "admin"})
        return {"status": database.invite(project_id, email, body.role)}

    @app.get("/projects/{project_id}/commits")
    def list_commits(project_id: str, user=Depends(member_project)):
        return [
            {
                "id": row["id"],
                "parent_id": row["parent_id"],
                "message": row["message"],
                "created_at": row["created_at"],
                "author_email": row["author_email"],
            }
            for row in database.list_commits(project_id)
        ]

    @app.post("/projects/{project_id}/commits")
    async def create_commit(
        project_id: str,
        parent_id: str = Form(default=""),
        message: str = Form(default="Update masks"),
        paths: str = Form(...),
        resolutions: str = Form(default="{}"),
        uploads: list[UploadFile] = File(default=[]),
        user=Depends(member_project),
    ):
        project = database.get_project(project_id)
        if project["status"] == "locked":
            raise HTTPException(status_code=403, detail="This project is locked")
        role = require_role(project_id, user, {"owner", "admin", "helper"})
        try:
            relpaths = [safe_rel(path) for path in json.loads(paths)]
            choices = {safe_rel(key): value for key, value in json.loads(resolutions).items()}
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid file list") from exc
        if len(relpaths) != len(uploads):
            raise HTTPException(status_code=400, detail="Each file needs a path")
        for relpath in relpaths:
            if relpath != "annotation_meta.json" and not relpath.endswith("_seg.npy"):
                raise HTTPException(status_code=400, detail=f"Unsupported project file: {relpath}")
        for choice in choices.values():
            if choice not in {"mine", "theirs"}:
                raise HTTPException(status_code=400, detail="Resolution must be mine or theirs")

        head_id = project["head_id"]
        parent = parent_id or None
        if parent is None and head_id:
            raise HTTPException(status_code=409, detail={"error": "pull_first", "head_id": head_id, "conflicts": []})
        if parent and database.get_commit(project_id, parent) is None:
            raise HTTPException(status_code=400, detail="Unknown parent commit")

        changes: dict[str, str] = {}
        for relpath, upload in zip(relpaths, uploads):
            data = await upload.read()
            changes[relpath] = database.write_blob(data)

        if role == "helper":
            branch_id = database.add_branch(
                project_id, parent, user["id"], message.strip() or "Update masks", changes
            )
            return {"id": branch_id, "parent_id": parent, "branch": True}

        base = database.commit_files(parent)
        head = database.commit_files(head_id)
        conflicts, merged = merge_files(base, head, changes, choices)
        if conflicts:
            raise HTTPException(status_code=409, detail={"conflicts": conflicts, "head_id": head_id})
        assert merged is not None
        stats = _stats_json(database, head_id, merged)
        commit_id = database.add_commit(
            project_id, head_id, user["id"], message.strip() or "Update masks", merged, stats
        )
        return {"id": commit_id, "parent_id": head_id, "branch": False}

    @app.get("/projects/{project_id}/commits/{commit_id}/archive")
    def archive(project_id: str, commit_id: str, user=Depends(member_project)):
        if database.get_commit(project_id, commit_id) is None:
            raise HTTPException(status_code=404, detail="Unknown commit")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive_file:
            for path, blob_hash in database.commit_files(commit_id).items():
                archive_file.writestr(path, database.read_blob(blob_hash))
        return Response(content=buffer.getvalue(), media_type="application/zip")

    @app.get("/projects/{project_id}/changes")
    def changes(project_id: str, user=Depends(member_project)):
        require_role(project_id, user, {"owner"})
        result = []
        for row in database.list_commits(project_id):
            result.append(
                {
                    "id": row["id"],
                    "parent_id": row["parent_id"],
                    "message": row["message"],
                    "created_at": row["created_at"],
                    "author_email": row["author_email"],
                    "images": json.loads(row["stats"] or "[]"),
                }
            )
        return result

    @app.get("/projects/{project_id}/branches")
    def branches(project_id: str, user=Depends(member_project)):
        require_role(project_id, user, {"owner", "admin"})
        found = []
        for row in database.list_branches(project_id):
            found.append(
                {
                    "id": row["id"],
                    "parent_id": row["parent_id"],
                    "message": row["message"],
                    "created_at": row["created_at"],
                    "author_email": row["author_email"],
                    "paths": list(database.branch_files(row["id"])),
                }
            )
        return found

    @app.delete("/projects/{project_id}/branches/{branch_id}")
    def discard_branch(project_id: str, branch_id: str, user=Depends(member_project)):
        require_role(project_id, user, {"owner", "admin"})
        if not database.discard_branch(project_id, branch_id):
            raise HTTPException(status_code=404, detail="Unknown helper commit")
        return {"ok": True}

    @app.get("/projects/{project_id}/branches/{branch_id}/archive")
    def branch_archive(project_id: str, branch_id: str, user=Depends(member_project)):
        require_role(project_id, user, {"owner", "admin"})
        if database.get_branch(project_id, branch_id) is None:
            raise HTTPException(status_code=404, detail="Unknown helper commit")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive_file:
            for path, blob_hash in database.branch_files(branch_id).items():
                archive_file.writestr(path, database.read_blob(blob_hash))
        return Response(content=buffer.getvalue(), media_type="application/zip")

    return app
