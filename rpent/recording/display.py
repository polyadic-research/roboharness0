"""Simulator-owned display camera boundary."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from rpent.recording.contracts import CameraDescriptor

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class DisplayAdapter(Protocol):
    @property
    def descriptors(self) -> tuple[CameraDescriptor, ...]:
        raise NotImplementedError

    def frames(self, observation: Mapping[str, Any]) -> Mapping[str, bytes]:
        raise NotImplementedError

    def frame(self, camera_id: str, observation: Mapping[str, Any]) -> bytes:
        raise NotImplementedError


class NullDisplayAdapter:
    """Explicit adapter for callers that do not declare display cameras."""

    @property
    def descriptors(self) -> tuple[CameraDescriptor, ...]:
        return ()

    def frames(self, observation: Mapping[str, Any]) -> Mapping[str, bytes]:
        del observation
        return {}

    def frame(self, camera_id: str, observation: Mapping[str, Any]) -> bytes:
        del observation
        raise ValueError(f"unknown display camera {camera_id!r}")


def png_bytes(image: Any) -> bytes:
    """Encode one HWC RGB image with the existing dependency-free PNG path."""
    if isinstance(image, bytes):
        _png_metadata(image)
        return image
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError("display images must use uint8 pixels")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("display images must be HWC RGB arrays")
    height, width, _channels = array.shape
    if height <= 0 or width <= 0:
        raise ValueError("display image width and height must be positive")
    array = np.ascontiguousarray(array)
    scanlines = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    return b"".join(
        (
            _PNG_SIGNATURE,
            chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            ),
            chunk(b"IDAT", zlib.compress(scanlines)),
            chunk(b"IEND", b""),
        )
    )


def _png_metadata(image: bytes) -> tuple[int, int, int]:
    if not image.startswith(_PNG_SIGNATURE):
        raise ValueError("display frame bytes must be PNG")
    offset = len(_PNG_SIGNATURE)
    ihdr: tuple[int, int, int] | None = None
    idat = bytearray()
    idat_started = False
    idat_ended = False
    saw_iend = False
    chunk_index = 0
    while offset < len(image):
        if len(image) - offset < 12:
            raise ValueError("PNG contains a truncated chunk")
        length = struct.unpack(">I", image[offset : offset + 4])[0]
        kind = image[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        chunk_end = data_end + 4
        if chunk_end > len(image):
            raise ValueError("PNG contains a truncated chunk")
        payload = image[data_start:data_end]
        expected_crc = struct.unpack(">I", image[data_end:chunk_end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        if chunk_index == 0 and kind != b"IHDR":
            raise ValueError("PNG IHDR must be the first chunk")
        if kind == b"IDAT":
            if idat_ended:
                raise ValueError("PNG IDAT chunks must be consecutive")
            idat_started = True
        elif idat_started:
            idat_ended = True
        if kind == b"IHDR":
            if ihdr is not None:
                raise ValueError("PNG must contain exactly one IHDR")
            if length != 13:
                raise ValueError("PNG IHDR must contain 13 bytes")
            width, height, bit_depth, color_type, compression, filter_method, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if width <= 0 or height <= 0:
                raise ValueError("PNG width and height must be positive")
            if bit_depth != 8 or color_type != 2:
                raise ValueError("PNG display frames must be 8-bit RGB")
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError("PNG uses unsupported encoding methods")
            ihdr = (width, height, 3)
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            if length != 0:
                raise ValueError("PNG IEND must be empty")
            saw_iend = True
            offset = chunk_end
            break
        offset = chunk_end
        chunk_index += 1
    if ihdr is None or not idat or not saw_iend:
        raise ValueError("PNG must contain IHDR, IDAT, and IEND chunks")
    if offset != len(image):
        raise ValueError("PNG contains trailing bytes after IEND")
    width, height, channels = ihdr
    row_length = 1 + width * channels
    expected_length = height * row_length
    decompressor = zlib.decompressobj()
    try:
        scanlines = decompressor.decompress(bytes(idat), expected_length + 1)
    except zlib.error as exc:
        raise ValueError("PNG IDAT is not a valid zlib stream") from exc
    if len(scanlines) > expected_length or decompressor.unconsumed_tail:
        raise ValueError("PNG IDAT expands beyond the expected image size")
    if (
        not decompressor.eof
        or decompressor.unused_data
    ):
        raise ValueError("PNG IDAT contains incomplete or trailing data")
    if len(scanlines) != expected_length:
        raise ValueError("PNG IDAT scanline length does not match IHDR")
    if any(scanlines[offset] > 4 for offset in range(0, expected_length, row_length)):
        raise ValueError("PNG scanline contains an invalid filter byte")
    return ihdr


def ordered_frames(
    descriptors: tuple[CameraDescriptor, ...],
    observation: Mapping[str, Any],
) -> Mapping[str, bytes]:
    """Extract validated native camera values in descriptor order."""
    return {
        descriptor.camera_id: camera_frame(
            descriptors, descriptor.camera_id, observation
        )
        for descriptor in descriptors
    }


def camera_frame(
    descriptors: tuple[CameraDescriptor, ...],
    camera_id: str,
    observation: Mapping[str, Any],
) -> bytes:
    """Extract and validate one declared camera without touching its siblings."""
    descriptor = next(
        (item for item in descriptors if item.camera_id == camera_id), None
    )
    if descriptor is None:
        raise ValueError(f"unknown display camera {camera_id!r}")
    if camera_id not in observation:
        raise ValueError(f"display observation is missing camera {camera_id!r}")
    value = observation[camera_id]
    expected = (descriptor.height, descriptor.width, descriptor.channels)
    if isinstance(value, bytes):
        width, height, channels = _png_metadata(value)
        actual = (height, width, channels)
        if actual != expected:
            raise ValueError(
                f"display camera {camera_id!r} PNG has shape {actual}, expected {expected}"
            )
    else:
        array = np.asarray(value)
        if array.shape != expected or array.dtype != np.uint8:
            raise ValueError(
                f"display camera {camera_id!r} must have shape {expected} and dtype uint8"
            )
    return png_bytes(value)


__all__ = ["DisplayAdapter", "NullDisplayAdapter"]
