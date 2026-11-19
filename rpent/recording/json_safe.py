"""Strict JSON conversion without embedding binary observation payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Return a strict-JSON value, replacing binary blobs with metadata."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numbers must be finite")
        return number
    if isinstance(value, bytes):
        return {
            "_binary_omitted": True,
            "bytes": len(value),
            "prefix_hex": value[:8].hex(),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    raise ValueError(f"value of type {type(value).__name__} is not JSON-safe")


__all__ = ["json_safe"]
