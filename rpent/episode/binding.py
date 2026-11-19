"""Explicit mappings between simulator-native and policy feature schemas."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


class FeatureValidationError(ValueError):
    """Raised when a feature or action violates an explicit policy binding."""


@dataclass(frozen=True, slots=True)
class FeatureBinding:
    """Map one simulator-native feature to one policy input feature."""

    native_key: str
    policy_key: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    def __post_init__(self) -> None:
        if not isinstance(self.native_key, str) or not self.native_key:
            raise _feature_error(
                "native_key",
                "a non-empty string",
                _describe_key(self.native_key),
            )
        if not isinstance(self.policy_key, str) or not self.policy_key:
            raise _feature_error(
                self.native_key,
                "a non-empty policy key string",
                _describe_key(self.policy_key),
            )

        shape = _normalize_shape(
            self.shape,
            binding_name="<unbound>",
            feature_key=self.native_key,
            expected="positive integer dimensions",
        )
        object.__setattr__(self, "shape", shape)

        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise _feature_error(
                self.native_key,
                "a valid numeric dtype",
                repr(self.dtype),
            ) from exc
        if not np.issubdtype(dtype, np.number):
            raise _feature_error(
                self.native_key,
                "a numeric dtype",
                f"dtype {dtype.name}",
            )
        object.__setattr__(self, "dtype", dtype)


@dataclass(frozen=True, slots=True)
class PolicyBinding:
    """Validate observations and action chunks at a policy boundary."""

    name: str
    inputs: tuple[FeatureBinding, ...]
    action_shape: tuple[int, ...]
    action_dtype: np.dtype[Any]
    action_low: float
    action_high: float
    allowed_missing_policy_inputs: frozenset[str] = frozenset()
    clip_actions_to_bounds: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise _binding_error(
                self.name if isinstance(self.name, str) else "<invalid>",
                "name",
                "a non-empty string",
                _describe_key(self.name),
            )

        try:
            inputs = tuple(self.inputs)
        except TypeError as exc:
            raise self._error(
                "inputs",
                "an iterable of FeatureBinding values",
                type(self.inputs).__name__,
            ) from exc
        invalid_input = next(
            (feature for feature in inputs if not isinstance(feature, FeatureBinding)),
            None,
        )
        if invalid_input is not None:
            raise self._error(
                "inputs",
                "FeatureBinding values",
                type(invalid_input).__name__,
            )
        object.__setattr__(self, "inputs", inputs)

        action_shape = _normalize_shape(
            self.action_shape,
            binding_name=self.name,
            feature_key="action_shape",
            expected="one positive integer vector dimension",
        )
        if len(action_shape) != 1:
            raise self._error(
                "action_shape",
                "one positive integer vector dimension",
                repr(action_shape),
            )
        object.__setattr__(self, "action_shape", action_shape)

        try:
            action_dtype = np.dtype(self.action_dtype)
        except (TypeError, ValueError) as exc:
            raise self._error(
                "action_dtype",
                "a valid floating dtype",
                repr(self.action_dtype),
            ) from exc
        if not np.issubdtype(action_dtype, np.floating):
            raise self._error(
                "action_dtype",
                "a floating dtype",
                action_dtype.name,
            )
        object.__setattr__(self, "action_dtype", action_dtype)

        if not _valid_action_bounds(self.action_low, self.action_high):
            raise self._error(
                "action_bounds",
                "finite real values with low <= high",
                f"low={self.action_low!r}, high={self.action_high!r}",
            )
        object.__setattr__(self, "action_low", float(self.action_low))
        object.__setattr__(self, "action_high", float(self.action_high))

        if not isinstance(self.clip_actions_to_bounds, bool):
            raise self._error(
                "clip_actions_to_bounds",
                "a boolean",
                type(self.clip_actions_to_bounds).__name__,
            )

        if isinstance(self.allowed_missing_policy_inputs, (str, bytes)):
            raise self._error(
                "allowed-missing",
                "a collection of non-empty string keys",
                type(self.allowed_missing_policy_inputs).__name__,
            )
        try:
            allowed_missing = frozenset(self.allowed_missing_policy_inputs)
        except TypeError as exc:
            raise self._error(
                "allowed-missing",
                "an iterable of non-empty string keys",
                type(self.allowed_missing_policy_inputs).__name__,
            ) from exc
        invalid_allowed_key = next(
            (key for key in allowed_missing if not isinstance(key, str)),
            None,
        )
        if invalid_allowed_key is not None:
            raise self._error(
                "allowed-missing",
                "string keys",
                type(invalid_allowed_key).__name__,
            )
        object.__setattr__(self, "allowed_missing_policy_inputs", allowed_missing)

        native_keys = [feature.native_key for feature in inputs]
        policy_keys = [feature.policy_key for feature in inputs]
        duplicate_native = _first_duplicate(native_keys)
        if duplicate_native is not None:
            raise self._error(
                duplicate_native,
                "unique native feature keys",
                f"duplicate native key '{duplicate_native}'",
            )
        duplicate_policy = _first_duplicate(policy_keys)
        if duplicate_policy is not None:
            raise self._error(
                duplicate_policy,
                "unique policy feature keys",
                f"duplicate policy key '{duplicate_policy}'",
            )

        empty_allowed_key = next((key for key in allowed_missing if not key), None)
        if empty_allowed_key is not None:
            raise self._error(
                "allowed-missing",
                "non-empty allowed-missing policy keys",
                "empty key",
            )

        overlap = set(policy_keys) & allowed_missing
        if overlap:
            key = sorted(overlap)[0]
            raise self._error(
                key,
                "a key mapped by inputs or declared allowed-missing, but not both",
                f"'{key}' appears in both",
            )

    def bind_observation(
        self, native: Mapping[str, Any]
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Validate and map native observation values without copying arrays."""

        supplied_missing = set(native) & self.allowed_missing_policy_inputs
        if supplied_missing:
            key = sorted(supplied_missing)[0]
            raise self._error(
                key,
                "absent because it is a checkpoint-declared allowed-missing input",
                "caller supplied allowed-missing policy input",
            )

        mapped: dict[str, np.ndarray[Any, Any]] = {}
        for feature in self.inputs:
            if feature.native_key not in native:
                raise self._error(feature.native_key, "present", "missing")

            value = np.asarray(native[feature.native_key])
            if value.shape != feature.shape:
                raise self._error(
                    feature.native_key,
                    str(feature.shape),
                    str(value.shape),
                )
            if value.dtype != feature.dtype:
                raise self._error(
                    feature.native_key,
                    f"dtype {feature.dtype}",
                    f"dtype {value.dtype}",
                )
            if not np.all(np.isfinite(value)):
                raise self._error(
                    feature.native_key,
                    "finite values",
                    "non-finite values",
                )
            mapped[feature.policy_key] = value
        return mapped

    def validate_action_chunk(self, actions: Any) -> np.ndarray[Any, Any]:
        """Validate and, when declared, saturate actions at controller bounds."""

        value = np.asarray(actions)
        action_dimensions = ", ".join(str(dimension) for dimension in self.action_shape)
        expected_shape = f"(chunk_length, {action_dimensions})"
        if value.ndim != 2 or value.shape[1:] != self.action_shape:
            raise self._error(
                "actions",
                expected_shape,
                str(value.shape),
            )
        if value.dtype != self.action_dtype:
            raise self._error(
                "actions",
                f"dtype {self.action_dtype}",
                f"dtype {value.dtype}",
            )
        if value.shape[0] == 0:
            raise self._error(
                "actions",
                "a non-empty action chunk",
                f"shape {value.shape}",
            )
        if not np.all(np.isfinite(value)):
            raise self._error("actions", "finite values", "non-finite values")
        if self.clip_actions_to_bounds:
            value = np.clip(value, self.action_low, self.action_high)
        if np.any(value < self.action_low) or np.any(value > self.action_high):
            actual_low = float(np.min(value))
            actual_high = float(np.max(value))
            raise self._error(
                "actions",
                f"values in [{self.action_low}, {self.action_high}]",
                f"range [{actual_low}, {actual_high}]",
            )
        return value

    def _error(
        self, feature_key: str, expected: str, actual: str
    ) -> FeatureValidationError:
        return FeatureValidationError(
            f"binding '{self.name}' feature '{feature_key}': "
            f"expected {expected}, actual {actual}"
        )


def _first_duplicate(keys: list[str]) -> str | None:
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            return key
        seen.add(key)
    return None


def _normalize_shape(
    shape: Any,
    *,
    binding_name: str,
    feature_key: str,
    expected: str,
) -> tuple[int, ...]:
    try:
        normalized = tuple(shape)
    except TypeError as exc:
        raise _binding_error(
            binding_name,
            feature_key,
            expected,
            repr(shape),
        ) from exc
    if any(
        not isinstance(dimension, Integral)
        or isinstance(dimension, (bool, np.bool_))
        or dimension <= 0
        for dimension in normalized
    ):
        raise _binding_error(
            binding_name,
            feature_key,
            expected,
            repr(normalized),
        )
    return tuple(int(dimension) for dimension in normalized)


def _valid_action_bounds(low: Any, high: Any) -> bool:
    try:
        for value in (low, high):
            if (
                not isinstance(value, Real)
                or isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
            ):
                return False
        return bool(low <= high)
    except (TypeError, ValueError, OverflowError):
        return False


def _feature_error(
    feature_key: str, expected: str, actual: str
) -> FeatureValidationError:
    return _binding_error("<unbound>", feature_key, expected, actual)


def _binding_error(
    binding_name: str, feature_key: str, expected: str, actual: str
) -> FeatureValidationError:
    return FeatureValidationError(
        f"binding '{binding_name}' feature '{feature_key}': "
        f"expected {expected}, actual {actual}"
    )


def _describe_key(value: Any) -> str:
    if isinstance(value, str) and value == "":
        return "empty string"
    return f"{type(value).__name__} {value!r}"
