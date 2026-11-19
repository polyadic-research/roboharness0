from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rpent.utils.model_load_diagnostics import (
    capture_load_state_dict,
    require_loaded_action_parameters,
    require_norm_stats_dimensions,
)


def _write_norm_stats(path, *, state_dim: int = 32, action_dim: int = 32) -> None:
    payload = {
        "norm_stats": {
            "state": {"mean": [0.0] * state_dim, "std": [1.0] * state_dim},
            "actions": {
                "mean": [0.0] * action_dim,
                "std": [1.0] * action_dim,
            },
        }
    }
    stats_path = path / "physical-intelligence" / "libero" / "norm_stats.json"
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_pi05_norm_stats_requires_metadata_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="norm_stats.json"):
        require_norm_stats_dimensions(tmp_path, expected_dim=32)


@pytest.mark.parametrize(
    ("field", "state_dim", "action_dim"),
    [
        ("state", 31, 32),
        ("actions", 32, 7),
    ],
)
def test_validate_pi05_norm_stats_rejects_wrong_dimensions(
    tmp_path, field, state_dim, action_dim
):
    _write_norm_stats(tmp_path, state_dim=state_dim, action_dim=action_dim)

    with pytest.raises(RuntimeError, match=rf"expected 32.*{field}"):
        require_norm_stats_dimensions(tmp_path, expected_dim=32)


def test_validate_pi05_norm_stats_accepts_expected_dimensions(tmp_path):
    _write_norm_stats(tmp_path)

    require_norm_stats_dimensions(tmp_path, expected_dim=32)


def test_capture_load_state_dict_diagnostics_records_keys_and_restores_method():
    class Loadable:
        def load_state_dict(self, state_dict, strict=True):
            return SimpleNamespace(missing_keys=["bias"], unexpected_keys=["extra"])

    original = Loadable.load_state_dict
    module = Loadable()

    with capture_load_state_dict(Loadable) as diagnostics:
        module.load_state_dict({"weight": 0, "extra": 1}, strict=False)

    assert Loadable.load_state_dict is original
    assert diagnostics[0]["missing_keys"] == ["bias"]
    assert diagnostics[0]["unexpected_keys"] == ["extra"]


def test_capture_load_state_dict_diagnostics_restores_method_after_error():
    class Loadable:
        def load_state_dict(self, state_dict, strict=True):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    original = Loadable.load_state_dict

    with pytest.raises(RuntimeError, match="load failed"):
        with capture_load_state_dict(Loadable):
            raise RuntimeError("load failed")

    assert Loadable.load_state_dict is original


def test_assert_action_head_loaded_rejects_missing_action_head_parameters():
    diagnostics = [{"missing_action_keys": ["action_out_proj.weight"]}]

    with pytest.raises(RuntimeError, match="action_out_proj.weight"):
        require_loaded_action_parameters(diagnostics)


def test_assert_action_head_loaded_accepts_other_missing_parameters():
    diagnostics = [{"missing_action_keys": []}]

    require_loaded_action_parameters(diagnostics)
