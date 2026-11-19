"""Validate Pi0 checkpoint assets and trace non-strict parameter loading."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def capture_load_state_dict(
    module_class: type | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Capture missing and unexpected keys reported by ``load_state_dict``."""
    if module_class is None:
        import torch

        module_class = torch.nn.Module

    original = module_class.load_state_dict
    records: list[dict[str, Any]] = []

    def traced(module: Any, state_dict: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(module, state_dict, *args, **kwargs)
        missing_keys = list(result.missing_keys)
        records.append(
            {
                "module": type(module).__name__,
                "checkpoint_key_count": len(state_dict),
                "missing_keys": missing_keys,
                "unexpected_keys": list(result.unexpected_keys),
                "missing_action_keys": [key for key in missing_keys if "action" in key],
            }
        )
        return result

    module_class.load_state_dict = traced
    try:
        yield records
    finally:
        module_class.load_state_dict = original


def require_loaded_action_parameters(records: list[dict[str, Any]]) -> None:
    """Reject checkpoint loads that leave action-producing parameters missing."""
    missing = [
        key for record in records for key in record.get("missing_action_keys", [])
    ]
    if missing:
        raise RuntimeError(
            "checkpoint did not load action head parameters: " + ", ".join(missing)
        )


def require_norm_stats_dimensions(model_path: str | Path, *, expected_dim: int) -> None:
    """Require LIBERO normalization assets to match the Pi0 checkpoint contract."""
    stats_path = (
        Path(model_path) / "physical-intelligence" / "libero" / "norm_stats.json"
    )
    payload = json.loads(stats_path.read_text(encoding="utf-8"))["norm_stats"]
    state_dim = len(payload["state"]["mean"])
    action_dim = len(payload["actions"]["mean"])
    if state_dim != expected_dim or action_dim != expected_dim:
        raise RuntimeError(
            "checkpoint norm stats dimension mismatch: "
            f"expected {expected_dim}, state={state_dim}, actions={action_dim}; "
            "weights and normalization assets may come from different checkpoints"
        )
