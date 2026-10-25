"""SQLite storage for accounts, projects, and mask commits."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex() + ":" + digest.hex()


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split(":")
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000)
    return secrets.compare_digest(digest.hex(), digest_hex)


class Database:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.data_dir / "cellannotator.sqlite", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                owner_id TEXT NOT NULL REFERENCES users(id),
                head_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS members (
                project_id TEXT NOT NULL REFERENCES projects(id),
                user_id TEXT NOT NULL REFERENCES users(id),
                PRIMARY KEY (project_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS invites (
                project_id TEXT NOT NULL REFERENCES projects(id),
                email TEXT NOT NULL,
                PRIMARY KEY (project_id, email)
            );
            CREATE TABLE IF NOT EXISTS manifests (
                project_id TEXT PRIMARY KEY REFERENCES projects(id),
                body TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS commits (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id),
                parent_id TEXT,
                author_id TEXT NOT NULL REFERENCES users(id),
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS commit_files (
                commit_id TEXT NOT NULL REFERENCES commits(id),
                path TEXT NOT NULL,
                blob_hash TEXT NOT NULL,
                PRIMARY KEY (commit_id, path)
            );
            """
        )
        self.conn.commit()

    def create_user(self, email: str, password: str) -> str:
        user_id = str(uuid.uuid4())
        with self.lock:
            self.conn.execute(
                "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, email, hash_password(password), _now()),
            )
            self._accept_invites(user_id, email)
            self.conn.commit()
        return user_id

    def _accept_invites(self, user_id: str, email: str) -> None:
        rows = self.conn.execute("SELECT project_id FROM invites WHERE email = ?", (email,)).fetchall()
        for row in rows:
            self.conn.execute(
                "INSERT OR IGNORE INTO members (project_id, user_id) VALUES (?, ?)",
                (row["project_id"], user_id),
            )
        self.conn.execute("DELETE FROM invites WHERE email = ?", (email,))

    def authenticate(self, email: str, password: str) -> str | None:
        row = self.conn.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,)).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        token = secrets.token_urlsafe(32)
        with self.lock:
            self.conn.execute("INSERT INTO tokens (token, user_id) VALUES (?, ?)", (token, row["id"]))
            self.conn.commit()
        return token

    def user_for_token(self, token: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT users.id, users.email FROM tokens
            JOIN users ON users.id = tokens.user_id
            WHERE tokens.token = ?
            """,
            (token,),
        ).fetchone()

    def create_project(self, owner_id: str, name: str, kind: str, manifest_json: str) -> str:
        project_id = str(uuid.uuid4())
        with self.lock:
            self.conn.execute(
                "INSERT INTO projects (id, name, kind, owner_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (project_id, name, kind, owner_id, _now()),
            )
            self.conn.execute(
                "INSERT INTO members (project_id, user_id) VALUES (?, ?)",
                (project_id, owner_id),
            )
            self.conn.execute(
                "INSERT INTO manifests (project_id, body) VALUES (?, ?)",
                (project_id, manifest_json),
            )
            self.conn.commit()
        return project_id

    def projects_for(self, user_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT projects.id, projects.name, projects.kind, projects.head_id
            FROM projects JOIN members ON members.project_id = projects.id
            WHERE members.user_id = ?
            ORDER BY projects.created_at
            """,
            (user_id,),
        ).fetchall()

    def is_member(self, project_id: str, user_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM members WHERE project_id = ? AND user_id = ?",
            (project_id, user_id),
        ).fetchone()
        return row is not None

    def get_project(self, project_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()

    def set_manifest(self, project_id: str, body: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO manifests (project_id, body) VALUES (?, ?) "
                "ON CONFLICT(project_id) DO UPDATE SET body = excluded.body",
                (project_id, body),
            )
            self.conn.commit()

    def manifest(self, project_id: str) -> str:
        row = self.conn.execute("SELECT body FROM manifests WHERE project_id = ?", (project_id,)).fetchone()
        return row["body"] if row else '{"files": []}'

    def invite(self, project_id: str, email: str) -> str:
        existing = self.conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        with self.lock:
            if existing:
                self.conn.execute(
                    "INSERT OR IGNORE INTO members (project_id, user_id) VALUES (?, ?)",
                    (project_id, existing["id"]),
                )
                status = "joined"
            else:
                self.conn.execute(
                    "INSERT OR IGNORE INTO invites (project_id, email) VALUES (?, ?)",
                    (project_id, email),
                )
                status = "invited"
            self.conn.commit()
        return status

    def commit_files(self, commit_id: str | None) -> dict[str, str]:
        if not commit_id:
            return {}
        rows = self.conn.execute(
            "SELECT path, blob_hash FROM commit_files WHERE commit_id = ?",
            (commit_id,),
        ).fetchall()
        return {row["path"]: row["blob_hash"] for row in rows}

    def add_commit(
        self,
        project_id: str,
        parent_id: str | None,
        author_id: str,
        message: str,
        files: dict[str, str],
    ) -> str:
        commit_id = str(uuid.uuid4())
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute(
                    "INSERT INTO commits (id, project_id, parent_id, author_id, message, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (commit_id, project_id, parent_id, author_id, message, _now()),
                )
                self.conn.executemany(
                    "INSERT INTO commit_files (commit_id, path, blob_hash) VALUES (?, ?, ?)",
                    [(commit_id, path, blob) for path, blob in files.items()],
                )
                self.conn.execute("UPDATE projects SET head_id = ? WHERE id = ?", (commit_id, project_id))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return commit_id

    def list_commits(self, project_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT commits.id, commits.parent_id, commits.message, commits.created_at, users.email AS author_email
            FROM commits JOIN users ON users.id = commits.author_id
            WHERE commits.project_id = ?
            ORDER BY commits.created_at
            """,
            (project_id,),
        ).fetchall()

    def get_commit(self, project_id: str, commit_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM commits WHERE project_id = ? AND id = ?",
            (project_id, commit_id),
        ).fetchone()

    def blob_path(self, blob_hash: str) -> Path:
        folder = self.data_dir / "blobs" / blob_hash[:2]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / blob_hash

    def write_blob(self, data: bytes) -> str:
        import hashlib

        blob_hash = hashlib.sha256(data).hexdigest()
        path = self.blob_path(blob_hash)
        if not path.exists():
            path.write_bytes(data)
        return blob_hash

    def read_blob(self, blob_hash: str) -> bytes:
        return self.blob_path(blob_hash).read_bytes()
