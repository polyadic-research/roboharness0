"""Private registry primitives for late-bound episode presets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimePreset:
    """A named factory kept internal to runtime selection."""

    name: str
    factory: Callable[..., Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("preset name must be a non-empty string")
        if not callable(self.factory):
            raise ValueError("preset factory must be callable")


class PresetRegistry:
    """Reject ambiguous preset registration and unknown lookup."""

    def __init__(self) -> None:
        self._presets: dict[str, RuntimePreset] = {}

    def register(self, preset: RuntimePreset) -> None:
        if not isinstance(preset, RuntimePreset):
            raise TypeError("preset must be a RuntimePreset")
        if preset.name in self._presets:
            raise ValueError(f"duplicate runtime preset: {preset.name!r}")
        self._presets[preset.name] = preset

    def get(self, name: str) -> RuntimePreset:
        try:
            return self._presets[name]
        except KeyError as exc:
            raise KeyError(f"unknown runtime preset: {name!r}") from exc
