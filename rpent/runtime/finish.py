"""Backend-independent completion-sentinel helpers."""

from __future__ import annotations

import json
from typing import Any


def contains_finish_sentinel(value: Any) -> bool:
    """Return True only when an executed tool result carries ``_finish``."""
    value = getattr(value, "root", value)
    if isinstance(value, str):
        try:
            return contains_finish_sentinel(json.loads(value))
        except (TypeError, json.JSONDecodeError):
            return False
    if isinstance(value, dict):
        if value.get("_finish") is True:
            return True
        return any(contains_finish_sentinel(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_finish_sentinel(item) for item in value)
    if hasattr(value, "model_dump"):
        return contains_finish_sentinel(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return contains_finish_sentinel(vars(value))
    return False


__all__ = ["contains_finish_sentinel"]
