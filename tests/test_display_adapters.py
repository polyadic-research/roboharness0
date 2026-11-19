from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping

import numpy as np
import pytest

import rpent.recording.display as display_module
from robots.robocasa.display import RoboCasaDisplayAdapter
from rpent.recording.display import DisplayAdapter, png_bytes
from tests.synthetic_tasks import SyntheticDisplayAdapter


def _image(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def assert_display_adapter(
    adapter: DisplayAdapter,
    observation: Mapping[str, object],
    expected_ids: tuple[str, ...],
) -> None:
    descriptors = adapter.descriptors
    assert tuple(item.camera_id for item in descriptors) == expected_ids
    assert sum(item.role == "primary" for item in descriptors) == 1
    frames = adapter.frames(observation)
    assert tuple(frames) == expected_ids
    assert all(
        isinstance(value, bytes) and value.startswith(b"\x89PNG")
        for value in frames.values()
    )
    for camera_id in expected_ids:
        assert adapter.frame(camera_id, observation) == frames[camera_id]




def test_robocasa_display_exposes_overview_then_policy_and_wrist_cameras() -> None:
    adapter = RoboCasaDisplayAdapter(
        width=2,
        height=2,
        overview_width=3,
        overview_height=2,
    )
    observation = {
        "state": np.zeros(16, dtype=np.float32),
        "robot0_agentview_center": np.full((2, 3, 3), 7, dtype=np.uint8),
        "robot0_agentview_left": _image(11),
        "robot0_agentview_right": _image(22),
        "robot0_eye_in_hand": _image(33),
    }

    assert_display_adapter(
        adapter,
        observation,
        (
            "robot0_agentview_center",
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ),
    )
    assert [
        (item.camera_id, item.label, item.role, item.width, item.height)
        for item in adapter.descriptors
    ] == [
        ("robot0_agentview_center", "Overview", "primary", 3, 2),
        ("robot0_agentview_left", "Policy Left", None, 2, 2),
        ("robot0_agentview_right", "Policy Right", None, 2, 2),
        ("robot0_eye_in_hand", "Wrist", "wrist", 2, 2),
    ]




def test_display_passes_through_png_bytes() -> None:
    adapter = SyntheticDisplayAdapter(width=2, height=2)
    produced = adapter.frames(
        {"main_images": _image(11), "wrist_images": _image(22)}
    )
    external = _external_png(2, 2)

    frames = adapter.frames(
        {"main_images": produced["main_images"], "wrist_images": external}
    )

    assert frames["main_images"] is produced["main_images"]
    assert frames["wrist_images"] is external


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _external_png(
    width: int,
    height: int,
    *,
    bit_depth: int = 8,
    color_type: int = 2,
    compression: int = 0,
    filter_method: int = 0,
    interlace: int = 0,
    idat_payload: bytes | None = None,
) -> bytes:
    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        color_type,
        compression,
        filter_method,
        interlace,
    )
    scanlines = b"".join(b"\x00" + bytes(width * 3) for _ in range(height))
    if idat_payload is None:
        idat_payload = zlib.compress(scanlines)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", idat_payload),
            _chunk(b"IEND", b""),
        )
    )


def _with_bad_ihdr_crc(png: bytes) -> bytes:
    mutated = bytearray(png)
    mutated[29] ^= 0x01
    return bytes(mutated)


@pytest.mark.parametrize(
    "invalid",
    [
        _external_png(2, 2)[:-1],
        _external_png(2, 2) + b"trailing",
        _with_bad_ihdr_crc(_external_png(2, 2)),
        _external_png(3, 2),
        _external_png(2, 2, bit_depth=16),
        _external_png(2, 2, color_type=6),
        _external_png(2, 2, compression=1),
        _external_png(2, 2, filter_method=1),
        _external_png(2, 2, interlace=1),
        _external_png(2, 2, interlace=2),
    ],
)
def test_display_rejects_invalid_png_bytes(invalid: bytes) -> None:
    adapter = SyntheticDisplayAdapter(width=2, height=2)

    with pytest.raises(ValueError, match="PNG|display camera"):
        adapter.frames({"main_images": invalid, "wrist_images": _external_png(2, 2)})


_SCANLINES_2X2 = b"\x00" + bytes(6) + b"\x00" + bytes(6)


@pytest.mark.parametrize(
    "invalid",
    [
        _external_png(2, 2, idat_payload=b"not-zlib"),
        _external_png(2, 2, idat_payload=b""),
        _external_png(2, 2, idat_payload=zlib.compress(_SCANLINES_2X2[:-1])),
        _external_png(
            2,
            2,
            idat_payload=zlib.compress(_SCANLINES_2X2) + b"trailing",
        ),
        _external_png(
            2,
            2,
            idat_payload=zlib.compress(_SCANLINES_2X2) + zlib.compress(b"extra"),
        ),
        _external_png(
            2,
            2,
            idat_payload=zlib.compress(b"\x05" + bytes(6) + b"\x00" + bytes(6)),
        ),
    ],
)
def test_display_rejects_invalid_png_image_data(invalid: bytes) -> None:
    adapter = SyntheticDisplayAdapter(width=2, height=2)

    with pytest.raises(ValueError, match="PNG"):
        adapter.frames({"main_images": invalid, "wrist_images": _external_png(2, 2)})


def test_display_rejects_zlib_bomb_without_flushing_unconsumed_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _external_png(2, 2, idat_payload=zlib.compress(bytes(1_000_000)))
    real_decompressobj = zlib.decompressobj

    class GuardedDecompressor:
        def __init__(self) -> None:
            self.inner = real_decompressobj()

        def decompress(self, data: bytes, max_length: int) -> bytes:
            return self.inner.decompress(data, max_length)

        @property
        def eof(self) -> bool:
            return self.inner.eof

        @property
        def unused_data(self) -> bytes:
            return self.inner.unused_data

        @property
        def unconsumed_tail(self) -> bytes:
            return self.inner.unconsumed_tail

        def flush(self) -> bytes:
            if self.unconsumed_tail:
                raise AssertionError("must not flush unconsumed bomb data")
            return self.inner.flush()

    monkeypatch.setattr(
        display_module.zlib,
        "decompressobj",
        GuardedDecompressor,
    )
    adapter = SyntheticDisplayAdapter(width=2, height=2)

    with pytest.raises(ValueError, match="PNG"):
        adapter.frames({"main_images": invalid, "wrist_images": _external_png(2, 2)})


def test_display_rejects_non_contiguous_idat_chunks() -> None:
    compressed = zlib.compress(_SCANLINES_2X2)
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    invalid = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", compressed[:5]),
            _chunk(b"tEXt", b"key\x00value"),
            _chunk(b"IDAT", compressed[5:]),
            _chunk(b"IEND", b""),
        )
    )
    adapter = SyntheticDisplayAdapter(width=2, height=2)

    with pytest.raises(ValueError, match="PNG"):
        adapter.frames({"main_images": invalid, "wrist_images": _external_png(2, 2)})


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((2, 2, 3), dtype=np.float32),
        np.full((2, 2, 3), -1, dtype=np.int8),
        np.zeros((0, 2, 3), dtype=np.uint8),
    ],
)
def test_png_encoder_rejects_non_uint8_or_empty_arrays(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uint8|positive"):
        png_bytes(invalid)
