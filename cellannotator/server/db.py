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
                created_at TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open'
            );
            CREATE TABLE IF NOT EXISTS members (
                project_id TEXT NOT NULL REFERENCES projects(id),
                user_id TEXT NOT NULL REFERENCES users(id),
                role TEXT NOT NULL DEFAULT 'helper',
                PRIMARY KEY (project_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS invites (
                project_id TEXT NOT NULL REFERENCES projects(id),
                email TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'helper',
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
                created_at TEXT NOT NULL,
                stats TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS commit_files (
                commit_id TEXT NOT NULL REFERENCES commits(id),
                path TEXT NOT NULL,
                blob_hash TEXT NOT NULL,
                PRIMARY KEY (commit_id, path)
            );
            CREATE TABLE IF NOT EXISTS branches (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id),
                author_id TEXT NOT NULL REFERENCES users(id),
                parent_id TEXT,
                message TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS branch_files (
                branch_id TEXT NOT NULL REFERENCES branches(id),
                path TEXT NOT NULL,
                blob_hash TEXT NOT NULL,
                PRIMARY KEY (branch_id, path)
            );
            """
        )
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        self._add_column("projects", "description", "TEXT NOT NULL DEFAULT ''")
        self._add_column("projects", "status", "TEXT NOT NULL DEFAULT 'open'")
        self._add_column("members", "role", "TEXT NOT NULL DEFAULT 'helper'")
        self._add_column("invites", "role", "TEXT NOT NULL DEFAULT 'helper'")
        self._add_column("commits", "stats", "TEXT NOT NULL DEFAULT '[]'")
        self.conn.execute(
            """
            UPDATE members SET role = 'owner'
            WHERE role != 'owner' AND EXISTS (
                SELECT 1 FROM projects
                WHERE projects.id = members.project_id AND projects.owner_id = members.user_id
            )
            """
        )

    def _add_column(self, table: str, column: str, declaration: str) -> None:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(row["name"] == column for row in rows):
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

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
        rows = self.conn.execute("SELECT project_id, role FROM invites WHERE email = ?", (email,)).fetchall()
        for row in rows:
            self.conn.execute(
                "INSERT OR IGNORE INTO members (project_id, user_id, role) VALUES (?, ?, ?)",
                (row["project_id"], user_id, row["role"] or "helper"),
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

    def create_project(self, owner_id: str, name: str, kind: str, manifest_json: str, description: str = "") -> str:
        project_id = str(uuid.uuid4())
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO projects (id, name, kind, owner_id, created_at, description, status)
                VALUES (?, ?, ?, ?, ?, ?, 'open')
                """,
                (project_id, name, kind, owner_id, _now(), description),
            )
            self.conn.execute(
                "INSERT INTO members (project_id, user_id, role) VALUES (?, ?, 'owner')",
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
            SELECT projects.id, projects.name, projects.kind, projects.head_id,
                   projects.description, projects.status, members.role,
                   owner.email AS owner_email
            FROM projects
            JOIN members ON members.project_id = projects.id
            JOIN users AS owner ON owner.id = projects.owner_id
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

    def role_of(self, project_id: str, user_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT role FROM members WHERE project_id = ? AND user_id = ?",
            (project_id, user_id),
        ).fetchone()
        return row["role"] if row else None

    def set_description(self, project_id: str, description: str) -> None:
        with self.lock:
            self.conn.execute("UPDATE projects SET description = ? WHERE id = ?", (description, project_id))
            self.conn.commit()

    def set_status(self, project_id: str, status: str) -> None:
        with self.lock:
            self.conn.execute("UPDATE projects SET status = ? WHERE id = ?", (status, project_id))
            self.conn.commit()

    def staff(self, project_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT users.email, members.role FROM members
            JOIN users ON users.id = members.user_id
            WHERE members.project_id = ? AND members.role IN ('owner', 'admin')
            ORDER BY CASE members.role WHEN 'owner' THEN 0 ELSE 1 END, users.email
            """,
            (project_id,),
        ).fetchall()

    def invite(self, project_id: str, email: str, role: str = "helper") -> str:
        existing = self.conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        with self.lock:
            if existing:
                self.conn.execute(
                    """
                    INSERT INTO members (project_id, user_id, role) VALUES (?, ?, ?)
                    ON CONFLICT(project_id, user_id) DO UPDATE SET role = excluded.role
                    WHERE members.role != 'owner'
                    """,
                    (project_id, existing["id"], role),
                )
                status = "joined"
            else:
                self.conn.execute(
                    """
                    INSERT INTO invites (project_id, email, role) VALUES (?, ?, ?)
                    ON CONFLICT(project_id, email) DO UPDATE SET role = excluded.role
                    """,
                    (project_id, email, role),
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
        stats: str = "[]",
    ) -> str:
        commit_id = str(uuid.uuid4())
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute(
                    """
                    INSERT INTO commits
                    (id, project_id, parent_id, author_id, message, created_at, stats)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (commit_id, project_id, parent_id, author_id, message, _now(), stats),
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
            SELECT commits.id, commits.parent_id, commits.message, commits.created_at,
                   commits.stats, users.email AS author_email
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

    def add_branch(
        self,
        project_id: str,
        parent_id: str | None,
        author_id: str,
        message: str,
        files: dict[str, str],
    ) -> str:
        branch_id = str(uuid.uuid4())
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute(
                    """
                    INSERT INTO branches
                    (id, project_id, author_id, parent_id, message, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'open', ?)
                    """,
                    (branch_id, project_id, author_id, parent_id, message, _now()),
                )
                self.conn.executemany(
                    "INSERT INTO branch_files (branch_id, path, blob_hash) VALUES (?, ?, ?)",
                    [(branch_id, path, blob) for path, blob in files.items()],
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return branch_id

    def list_branches(self, project_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT branches.id, branches.parent_id, branches.message, branches.created_at,
                   users.email AS author_email
            FROM branches JOIN users ON users.id = branches.author_id
            WHERE branches.project_id = ? AND branches.status = 'open'
            ORDER BY branches.created_at
            """,
            (project_id,),
        ).fetchall()

    def branch_files(self, branch_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT path, blob_hash FROM branch_files WHERE branch_id = ?",
            (branch_id,),
        ).fetchall()
        return {row["path"]: row["blob_hash"] for row in rows}

    def get_branch(self, project_id: str, branch_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM branches WHERE project_id = ? AND id = ? AND status = 'open'",
            (project_id, branch_id),
        ).fetchone()

    def discard_branch(self, project_id: str, branch_id: str) -> bool:
        branch = self.get_branch(project_id, branch_id)
        if branch is None:
            return False
        hashes = list(self.branch_files(branch_id).values())
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute("DELETE FROM branch_files WHERE branch_id = ?", (branch_id,))
                self.conn.execute(
                    "UPDATE branches SET status = 'discarded' WHERE id = ?",
                    (branch_id,),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        for blob_hash in hashes:
            self._drop_blob_if_unused(blob_hash)
        return True

    def _drop_blob_if_unused(self, blob_hash: str) -> None:
        used = self.conn.execute(
            """
            SELECT 1 FROM commit_files WHERE blob_hash = ?
            UNION SELECT 1 FROM branch_files WHERE blob_hash = ?
            """,
            (blob_hash, blob_hash),
        ).fetchone()
        if used is not None:
            return
        path = self.blob_path(blob_hash)
        if path.exists():
            path.unlink()
