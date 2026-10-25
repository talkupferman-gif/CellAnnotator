"""Three-way merge of project files. Hashes stand in for file contents."""

from __future__ import annotations


def merge_files(
    base: dict[str, str],
    head: dict[str, str],
    changes: dict[str, str],
    resolutions: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str] | None]:
    """Merge ``changes`` (path -> hash) onto ``head``, using ``base`` as the parent.

    A file only one side changed is kept from that side. A file both sides
    changed must have a resolution of ``mine`` (the incoming change) or
    ``theirs`` (the current head). Returns conflict paths and, when clean,
    the resulting hash map.
    """
    resolutions = resolutions or {}
    result = dict(head)
    conflicts: list[str] = []
    for path, new_hash in changes.items():
        base_hash = base.get(path)
        head_hash = head.get(path)
        if head_hash == new_hash or head_hash == base_hash:
            result[path] = new_hash
            continue
        choice = resolutions.get(path)
        if choice == "mine":
            result[path] = new_hash
        elif choice == "theirs":
            result[path] = head_hash
        else:
            conflicts.append(path)
    if conflicts:
        return sorted(conflicts), None
    return [], result
