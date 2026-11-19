"""Contract tests for run-scoped verifier evidence resolution."""

from pathlib import Path

import pytest

from rpent.harness.evidence import EvidenceResolver


def test_resolver_derives_timezone_timestamp_hash_and_media_type(tmp_path: Path):
    artifact = tmp_path / "state.json"
    artifact.write_text('{"supported": true}\n', encoding="utf-8")

    item = EvidenceResolver(tmp_path).resolve_many(["state.json"])[0]

    assert item.path == artifact.resolve()
    assert item.media_type == "application/json"
    assert item.captured_at.endswith("+00:00")
    assert len(item.sha256) == 64


@pytest.mark.parametrize("ref", ("../outside.json", "/etc/passwd"))
def test_resolver_rejects_paths_outside_root(tmp_path: Path, ref: str):
    with pytest.raises(ValueError, match="outside run output"):
        EvidenceResolver(tmp_path).resolve_many([ref])


def test_resolver_rejects_symlink_escape(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "escape.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="outside run output"):
        EvidenceResolver(tmp_path).resolve_many(["escape.txt"])


def test_resolver_enforces_extension_whitelist(tmp_path: Path):
    (tmp_path / "payload.bin").write_bytes(b"payload")

    with pytest.raises(ValueError, match="unsupported evidence type"):
        EvidenceResolver(tmp_path).resolve_many(["payload.bin"])


def test_resolver_rejects_missing_and_oversized_text(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        EvidenceResolver(tmp_path).resolve_many(["missing.json"])

    (tmp_path / "large.txt").write_text("12345", encoding="utf-8")
    with pytest.raises(ValueError, match="text evidence exceeds"):
        EvidenceResolver(tmp_path, max_text_bytes=4).resolve_many(["large.txt"])


def test_resolver_deduplicates_refs_in_original_order(tmp_path: Path):
    (tmp_path / "first.txt").write_text("first", encoding="utf-8")
    (tmp_path / "second.txt").write_text("second", encoding="utf-8")

    resolved = EvidenceResolver(tmp_path).resolve_many(
        ["first.txt", "second.txt", "first.txt"]
    )

    assert [item.ref for item in resolved] == ["first.txt", "second.txt"]
