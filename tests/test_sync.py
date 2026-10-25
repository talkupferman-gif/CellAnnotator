import json
import zipfile
from io import BytesIO

from fastapi.testclient import TestClient

from cellannotator.server.api import create_app
from cellannotator.sync_logic import merge_files


def test_merge_keeps_both_sides_until_the_same_file_changes():
    base = {"a_seg.npy": "a0", "b_seg.npy": "b0"}
    head = {"a_seg.npy": "a0", "b_seg.npy": "b1"}
    changes = {"a_seg.npy": "a2"}
    conflicts, merged = merge_files(base, head, changes)
    assert conflicts == []
    assert merged == {"a_seg.npy": "a2", "b_seg.npy": "b1"}

    both = {"a_seg.npy": "a9"}
    head2 = {"a_seg.npy": "a3"}
    conflicts, merged = merge_files({"a_seg.npy": "a0"}, head2, both)
    assert conflicts == ["a_seg.npy"]
    assert merged is None
    _conflicts, merged = merge_files({"a_seg.npy": "a0"}, head2, both, {"a_seg.npy": "theirs"})
    assert merged == {"a_seg.npy": "a3"}


def _token(client: TestClient, email: str) -> str:
    response = client.post("/register", json={"email": email, "password": "password1"})
    assert response.status_code == 200, response.text
    return response.json()["token"]


def _commit(client, token, project_id, parent, name, content, resolutions=None):
    return client.post(
        f"/projects/{project_id}/commits",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "parent_id": parent or "",
            "message": "edit",
            "paths": f'["{name}"]',
            "resolutions": "{}" if resolutions is None else json.dumps(resolutions),
        },
        files=[("uploads", (name, content, "application/octet-stream"))],
    )


def test_two_people_merge_different_files_and_conflict_on_one(tmp_path):
    client = TestClient(create_app(tmp_path))
    owner = _token(client, "owner@lab.test")
    created = client.post(
        "/projects",
        headers={"Authorization": f"Bearer {owner}"},
        json={"name": "plates", "kind": "zstack", "manifest": {"files": []}},
    )
    assert created.status_code == 200, created.text
    project_id = created.json()["id"]
    first = _commit(client, owner, project_id, None, "a_seg.npy", b"base-a")
    assert first.status_code == 200, first.text
    parent = first.json()["id"]

    guest = _token(client, "guest@lab.test")
    invited = client.post(
        f"/projects/{project_id}/invites",
        headers={"Authorization": f"Bearer {owner}"},
        json={"email": "guest@lab.test"},
    )
    assert invited.json()["status"] == "joined"

    other = _commit(client, guest, project_id, parent, "b_seg.npy", b"guest-b")
    assert other.status_code == 200, other.text

    clash = _commit(client, owner, project_id, parent, "b_seg.npy", b"owner-b")
    assert clash.status_code == 409
    assert clash.json()["detail"]["conflicts"] == ["b_seg.npy"]

    resolved = _commit(
        client,
        owner,
        project_id,
        parent,
        "b_seg.npy",
        b"owner-b",
        {"b_seg.npy": "mine"},
    )
    assert resolved.status_code == 200, resolved.text
    archive = client.get(
        f"/projects/{project_id}/commits/{resolved.json()['id']}/archive",
        headers={"Authorization": f"Bearer {owner}"},
    )
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content)) as bundle:
        names = set(bundle.namelist())
        assert "a_seg.npy" in names
        assert bundle.read("b_seg.npy") == b"owner-b"

    history = client.get(f"/projects/{project_id}/commits", headers={"Authorization": f"Bearer {owner}"})
    assert len(history.json()) == 3
