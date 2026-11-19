"""Small durable-file and media helpers used by the Case recorder."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO


class JsonlRollbackError(RuntimeError):
    """An append and its compensating truncate both failed."""

    def __init__(
        self, append_error: BaseException, rollback_error: BaseException
    ) -> None:
        super().__init__("JSONL rollback failed; journal cannot continue")
        self.append_error = append_error
        self.rollback_error = rollback_error


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(path: Path) -> None:
    """Create directories and persist each newly linked directory entry."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise NotADirectoryError(current)
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        else:
            _fsync_directory(directory.parent)


def camera_storage_component(camera_id: str) -> str:
    """Encode a native camera ID as exactly one portable path component."""
    digest = hashlib.sha256(
        b"rpent-camera-storage-v1\0" + camera_id.encode("utf-8")
    ).hexdigest()
    return f"camera-sha256-{digest}"


def operation_storage_component(operation_id: str) -> str:
    """Encode an operation ID as exactly one portable path component."""
    digest = hashlib.sha256(
        b"rpent-operation-storage-v1\0" + operation_id.encode("utf-8")
    ).hexdigest()
    return f"operation-sha256-{digest}"


def action_camera_artifact_id(operation_id: str, camera_id: str) -> str:
    """Return the versioned artifact identity for one operation Camera clip."""
    operation_bytes = operation_id.encode("utf-8")
    camera_bytes = camera_id.encode("utf-8")
    digest = hashlib.sha256(
        b"rpent.action-camera.v1\0"
        + len(operation_bytes).to_bytes(8, "big", signed=False)
        + operation_bytes
        + len(camera_bytes).to_bytes(8, "big", signed=False)
        + camera_bytes
    ).hexdigest()
    return f"action-camera-sha256-{digest}"


def legacy_camera_storage_component(camera_id: str) -> str:
    """Return the pre-digest component used by retained recordings."""
    return "camera-" + camera_id.encode("utf-8").hex()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    """Durably replace one file without exposing a partial value."""
    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, _json_bytes(value))


def _write_all(stream: BinaryIO, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise OSError("JSONL append made no write progress")
        remaining = remaining[written:]


def _rollback_append(stream: BinaryIO, original_size: int, *, created: bool) -> None:
    del created
    stream.seek(original_size)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def append_jsonl(path: Path, value: Any) -> None:
    """Durably append one record or restore the journal's original length."""
    ensure_directory(path.parent)
    created = not path.exists()
    mode = "w+b" if created else "r+b"
    stream = path.open(mode)
    try:
        original_size = stream.seek(0, os.SEEK_END)
    except BaseException:
        stream.close()
        raise
    try:
        _write_all(stream, _json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
        if created:
            _fsync_directory(path.parent)
        stream.close()
    except BaseException as append_error:
        try:
            if not stream.closed:
                _rollback_append(stream, original_size, created=created)
                stream.close()
            if created and original_size == 0:
                path.unlink()
                _fsync_directory(path.parent)
        except BaseException as rollback_error:
            try:
                stream.close()
            except BaseException:
                pass
            raise JsonlRollbackError(append_error, rollback_error) from append_error
        raise


def replace_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically replace a journal with its complete records."""
    atomic_write_bytes(path, b"".join(_json_bytes(record) for record in records))


def read_complete_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read complete objects, ignoring only an invalid final tail."""
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    incomplete = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index != len(lines) - 1:
                raise ValueError(f"invalid JSONL record {index + 1} in {path}")
            incomplete = True
            break
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record {index + 1} in {path} is not an object")
        records.append(value)
    if raw and not raw.endswith(b"\n"):
        incomplete = True
    return records, incomplete


def write_parquet(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically materialize transition rows as Parquet."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet finalization requires pyarrow") from exc

    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pylist(list(records)), temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def finalize_video(path: Path, frames: Iterable[Path], fps: float) -> None:
    """Atomically encode PNG segments into one MP4."""
    import imageio.v2 as imageio

    frame_paths = tuple(frames)
    if not frame_paths:
        return
    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".mp4", dir=path.parent
    )
    os.close(fd)
    try:
        with imageio.get_writer(temporary, fps=fps, macro_block_size=1) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def video_frame_count(path: Path) -> int:
    """Decode a video fully and return its frame count."""
    import imageio.v2 as imageio

    count = 0
    with imageio.get_reader(path) as reader:
        for _frame in reader:
            count += 1
    return count


def finalize_validated_video(
    path: Path,
    frames: Iterable[Path],
    fps: float,
    *,
    expected_frames: int,
) -> None:
    """Atomically encode and publish an MP4 only after full decode validation."""
    import imageio.v2 as imageio

    frame_paths = tuple(frames)
    if len(frame_paths) != expected_frames:
        raise ValueError("source frame count does not match expected frame count")
    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".mp4", dir=path.parent
    )
    os.close(fd)
    try:
        with imageio.get_writer(temporary, fps=fps, macro_block_size=1) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        decoded = video_frame_count(Path(temporary))
        if decoded != expected_frames:
            raise ValueError(
                f"decoded frame count {decoded} does not match expected "
                f"frame count {expected_frames}"
            )
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "action_camera_artifact_id",
    "append_jsonl",
    "atomic_write_bytes",
    "atomic_write_json",
    "camera_storage_component",
    "ensure_directory",
    "finalize_validated_video",
    "finalize_video",
    "JsonlRollbackError",
    "legacy_camera_storage_component",
    "operation_storage_component",
    "read_complete_jsonl",
    "replace_jsonl",
    "video_frame_count",
    "write_parquet",
]
