"""Resolve verifier evidence inside one run-owned directory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".txt": "text/plain",
}
_TEXT_TYPES = {"application/json", "application/x-ndjson", "text/plain"}


@dataclass(frozen=True)
class ResolvedEvidence:
    ref: str
    path: Path
    captured_at: str
    media_type: str
    sha256: str


class EvidenceResolver:
    """Canonicalize a request-scoped evidence whitelist without path escape."""

    def __init__(self, root: Path, *, max_text_bytes: int = 1_000_000) -> None:
        if isinstance(max_text_bytes, bool) or max_text_bytes < 0:
            raise ValueError("max_text_bytes must be non-negative")
        self.root = Path(root).resolve()
        self.max_text_bytes = int(max_text_bytes)

    def resolve_many(self, refs: list[str]) -> tuple[ResolvedEvidence, ...]:
        if not refs or not all(isinstance(ref, str) and ref.strip() for ref in refs):
            raise ValueError("observation_refs must be a non-empty array of strings")
        return tuple(
            self._resolve(ref) for ref in dict.fromkeys(item.strip() for item in refs)
        )

    def _resolve(self, ref: str) -> ResolvedEvidence:
        supplied = Path(ref)
        candidate = supplied if supplied.is_absolute() else self.root / supplied
        self._require_contained(candidate.resolve(strict=False), ref)
        try:
            path = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"evidence does not exist: {ref}") from exc
        self._require_contained(path, ref)
        if not path.is_file():
            raise ValueError(f"evidence is not a file: {ref}")
        media_type = _MEDIA_TYPES.get(path.suffix.lower())
        if media_type is None:
            raise ValueError(f"unsupported evidence type: {ref}")
        raw = path.read_bytes()
        if media_type in _TEXT_TYPES and len(raw) > self.max_text_bytes:
            raise ValueError(f"text evidence exceeds size limit: {ref}")
        captured_at = datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat()
        return ResolvedEvidence(
            ref=ref,
            path=path,
            captured_at=captured_at,
            media_type=media_type,
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def _require_contained(self, path: Path, ref: str) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"evidence is outside run output: {ref}") from exc


__all__ = ["EvidenceResolver", "ResolvedEvidence"]
