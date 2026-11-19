"""Bounded, UTF-8-safe access to one worker Console file."""

from __future__ import annotations

import codecs
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from rpent.ui.schemas import ConsolePageView

MAX_CONSOLE_PAGE_BYTES = 256 * 1024
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


def read_console_page(
    path: str | Path,
    *,
    offset: int,
    limit: int,
    complete: bool,
    root: str | Path | None = None,
) -> ConsolePageView:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("console offset must be a non-negative integer")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
        or limit > MAX_CONSOLE_PAGE_BYTES
    ):
        raise ValueError("console limit is outside the allowed range")
    with _open_console(path, root=root) as stream:
        size = os.fstat(stream.fileno()).st_size
        truncated = offset > size
        selected_offset = 0 if truncated else offset
        stream.seek(selected_offset)
        buffered = stream.read(
            min(max(limit, 4), MAX_CONSOLE_PAGE_BYTES, size - selected_offset)
        )
    if buffered and buffered[0] & 0xC0 == 0x80:
        raise ValueError("console offset is not a UTF-8 boundary")
    raw = buffered[:limit]
    text, consumed = _decode_page(raw)
    if raw and consumed == 0:
        raw = buffered[: max(limit, _utf8_scalar_width(buffered[0]))]
        text, consumed = _decode_page(raw)
    next_offset = selected_offset + consumed
    return ConsolePageView(
        offset=selected_offset,
        next_offset=next_offset,
        text=text,
        truncated=truncated,
        complete=bool(complete and next_offset == size),
    )


def read_console_text(
    path: str | Path, *, root: str | Path | None = None
) -> str:
    with _open_console(path, root=root) as stream:
        raw = stream.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("console output is not valid UTF-8") from exc


def open_console_download(
    path: str | Path, *, root: str | Path | None = None
) -> tuple[BinaryIO, int]:
    stream = _open_console(path, root=root)
    size = os.fstat(stream.fileno()).st_size
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    remaining = size
    try:
        while remaining:
            chunk = stream.read(min(_DOWNLOAD_CHUNK_BYTES, remaining))
            if not chunk:
                raise OSError("console output changed during validation")
            remaining -= len(chunk)
            decoder.decode(chunk, final=remaining == 0)
        if size == 0:
            decoder.decode(b"", final=True)
        stream.seek(0)
        return stream, size
    except UnicodeDecodeError as exc:
        stream.close()
        raise ValueError("console output is not valid UTF-8") from exc
    except BaseException:
        stream.close()
        raise


def iter_console_download(stream: BinaryIO, size: int) -> Iterator[bytes]:
    remaining = size
    try:
        while remaining:
            chunk = stream.read(min(_DOWNLOAD_CHUNK_BYTES, remaining))
            if not chunk:
                raise OSError("console output changed during download")
            remaining -= len(chunk)
            yield chunk
    finally:
        stream.close()


def _decode_page(raw: bytes) -> tuple[str, int]:
    try:
        return raw.decode("utf-8"), len(raw)
    except UnicodeDecodeError as exc:
        if exc.end == len(raw) and exc.reason == "unexpected end of data":
            prefix = raw[: exc.start]
            try:
                return prefix.decode("utf-8"), len(prefix)
            except UnicodeDecodeError:
                pass
        raise ValueError("console output is not valid UTF-8") from exc


def _utf8_scalar_width(first: int) -> int:
    if first < 0x80:
        return 1
    if 0xC2 <= first <= 0xDF:
        return 2
    if 0xE0 <= first <= 0xEF:
        return 3
    if 0xF0 <= first <= 0xF4:
        return 4
    return 1


def _open_console(
    path: str | Path, *, root: str | Path | None
):
    selected = Path(path).absolute()
    boundary = Path(root).resolve() if root is not None else selected.parent.resolve()
    try:
        selected.relative_to(boundary)
    except ValueError as exc:
        raise ValueError("console path escapes the attempt root") from exc
    try:
        metadata = selected.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(metadata.st_mode):
        raise OSError("console path must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("console path must be a regular file")
    resolved = selected.resolve()
    if resolved != selected or not resolved.is_relative_to(boundary):
        raise ValueError("console path escapes the attempt root")
    relative = selected.relative_to(boundary)
    if not relative.parts:
        raise OSError("console path must be a regular file")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow
    directories = [os.open(boundary, directory_flags)]
    try:
        for part in relative.parts[:-1]:
            directories.append(
                os.open(part, directory_flags, dir_fd=directories[-1])
            )
        descriptor = os.open(
            relative.parts[-1], os.O_RDONLY | nofollow, dir_fd=directories[-1]
        )
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise OSError("console path must be a regular file")
            return os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        for directory in reversed(directories):
            os.close(directory)


__all__ = [
    "MAX_CONSOLE_PAGE_BYTES",
    "iter_console_download",
    "open_console_download",
    "read_console_page",
    "read_console_text",
]
