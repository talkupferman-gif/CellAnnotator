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
        json={"email": "guest@lab.test", "role": "admin"},
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


def _npy(masks) -> bytes:
    import numpy as np
    from io import BytesIO

    buffer = BytesIO()
    np.save(buffer, {"masks": np.asarray(masks, dtype=np.int32)})
    return buffer.getvalue()


def test_helper_commit_stays_off_main_until_discarded(tmp_path):
    import numpy as np

    client = TestClient(create_app(tmp_path))
    owner = _token(client, "owner@lab.test")
    created = client.post(
        "/projects",
        headers={"Authorization": f"Bearer {owner}"},
        json={"name": "plates", "kind": "unrelated", "manifest": {"files": []}, "description": "Round 1"},
    )
    project_id = created.json()["id"]
    base = _npy(np.zeros((4, 4), np.int32))
    first = _commit(client, owner, project_id, None, "a_seg.npy", base)
    assert first.status_code == 200, first.text
    parent = first.json()["id"]
    assert first.json()["branch"] is False

    helper = _token(client, "helper@lab.test")
    joined = client.post(
        f"/projects/{project_id}/invites",
        headers={"Authorization": f"Bearer {owner}"},
        json={"email": "helper@lab.test", "role": "helper"},
    )
    assert joined.json()["status"] == "joined"
    denied = client.post(
        f"/projects/{project_id}/invites",
        headers={"Authorization": f"Bearer {helper}"},
        json={"email": "other@lab.test", "role": "helper"},
    )
    assert denied.status_code == 403

    edited = np.zeros((4, 4), np.int32)
    edited[0, 0] = 1
    branch = _commit(client, helper, project_id, parent, "a_seg.npy", _npy(edited))
    assert branch.status_code == 200, branch.text
    assert branch.json()["branch"] is True
    detail = client.get(f"/projects/{project_id}", headers={"Authorization": f"Bearer {owner}"})
    assert detail.json()["head_id"] == parent
    assert detail.json()["staff"][0]["role"] == "owner"
    waiting = client.get(f"/projects/{project_id}/branches", headers={"Authorization": f"Bearer {owner}"})
    assert len(waiting.json()) == 1
    assert waiting.json()[0]["paths"] == ["a_seg.npy"]

    removed = client.delete(
        f"/projects/{project_id}/branches/{branch.json()['id']}",
        headers={"Authorization": f"Bearer {owner}"},
    )
    assert removed.status_code == 200, removed.text
    assert client.get(f"/projects/{project_id}/branches", headers={"Authorization": f"Bearer {owner}"}).json() == []


def test_owner_commit_records_cell_counts_and_lock_blocks_commits(tmp_path):
    import numpy as np

    client = TestClient(create_app(tmp_path))
    owner = _token(client, "owner@lab.test")
    created = client.post(
        "/projects",
        headers={"Authorization": f"Bearer {owner}"},
        json={"name": "plates", "kind": "unrelated", "manifest": {"files": []}},
    )
    project_id = created.json()["id"]
    blank = np.zeros((4, 4), np.int32)
    first = _commit(client, owner, project_id, None, "a_seg.npy", _npy(blank))
    parent = first.json()["id"]
    grown = blank.copy()
    grown[1, 1] = 1
    grown[1, 2] = 1
    second = _commit(client, owner, project_id, parent, "a_seg.npy", _npy(grown))
    assert second.status_code == 200, second.text
    history = client.get(f"/projects/{project_id}/changes", headers={"Authorization": f"Bearer {owner}"})
    images = history.json()[-1]["images"]
    assert images[0]["path"] == "a_seg.npy"
    assert images[0]["added"] == 1
    assert images[0]["deleted"] == 0

    guest = _token(client, "guest@lab.test")
    client.post(
        f"/projects/{project_id}/invites",
        headers={"Authorization": f"Bearer {owner}"},
        json={"email": "guest@lab.test", "role": "admin"},
    )
    forbidden = client.get(f"/projects/{project_id}/changes", headers={"Authorization": f"Bearer {guest}"})
    assert forbidden.status_code == 403

    locked = client.post(
        f"/projects/{project_id}/lock",
        headers={"Authorization": f"Bearer {owner}"},
        json={"locked": True},
    )
    assert locked.json()["status"] == "locked"
    blocked = _commit(client, owner, project_id, second.json()["id"], "a_seg.npy", _npy(grown))
    assert blocked.status_code == 403
    client.post(
        f"/projects/{project_id}/lock",
        headers={"Authorization": f"Bearer {owner}"},
        json={"locked": False},
    )
    opened = _commit(client, owner, project_id, second.json()["id"], "b_seg.npy", _npy(blank))
    assert opened.status_code == 200, opened.text
