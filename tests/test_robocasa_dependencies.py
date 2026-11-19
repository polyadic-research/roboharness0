from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robots.robocasa.dependencies import (
    ROBOCASA_COMMIT,
    ROBOSUITE_COMMIT,
    inspect_robocasa_dependencies,
)
from rpent.cli.main import _build_argparser


def _interpreter(tmp_path: Path) -> Path:
    path = tmp_path / "robocasa-python"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _payload(**overrides):
    payload = {
        "schema_version": 1,
        "interpreter": "/opt/robocasa/bin/python",
        "robocasa": {"version": "0.2.0", "commit": ROBOCASA_COMMIT},
        "robosuite": {"version": "1.5.1", "commit": ROBOSUITE_COMMIT},
        "mujoco": {"version": "3.3.2", "importable": True},
        "mujoco_gl": "egl",
        "macros_private": True,
        "assets": {
            "textures": True,
            "generative_textures": True,
            "fixtures_lightwheel": True,
            "objects_lightwheel": True,
            "objects_objaverse": False,
            "objects_aigen": False,
        },
    }
    payload.update(overrides)
    return payload


def _runner(payload, calls):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    return run


def test_preflight_runs_only_in_selected_interpreter_and_reports_identity(
    tmp_path,
) -> None:
    interpreter = _interpreter(tmp_path)
    calls = []

    report = inspect_robocasa_dependencies(
        interpreter,
        runner=_runner(_payload(interpreter=str(interpreter)), calls),
    )

    assert report.interpreter == str(interpreter.resolve())
    assert report.robocasa_commit == ROBOCASA_COMMIT
    assert report.robosuite_commit == ROBOSUITE_COMMIT
    assert report.mujoco_version == "3.3.2"
    assert report.mujoco_gl == "egl"
    assert report.default_assets_ready is True
    assert report.available_registries == frozenset({"lightwheel"})
    assert calls[0][0] == [
        str(interpreter.resolve()),
        "-m",
        "robots.robocasa.dependencies",
        "--inspect-json",
    ]
    assert calls[0][1]["check"] is True
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["timeout"] == 180


def test_preflight_preserves_virtualenv_interpreter_symlink(tmp_path) -> None:
    target = _interpreter(tmp_path)
    virtualenv_python = tmp_path / "venv" / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to(target)
    calls = []

    report = inspect_robocasa_dependencies(
        virtualenv_python,
        runner=_runner(
            _payload(interpreter=str(virtualenv_python)),
            calls,
        ),
    )

    assert report.interpreter == str(virtualenv_python)
    assert calls[0][0][0] == str(virtualenv_python)


@pytest.mark.parametrize(
    ("commit", "accepted"),
    [
        ("456174f62b89b8fca99eaaf33949c29fec9cfc2a", True),
        ("b4684e6ee37d377cc392e98302a6b916d588b415", False),
    ],
)
def test_formal_preflight_requires_reviewed_september_revision(
    tmp_path, commit, accepted
) -> None:
    interpreter = _interpreter(tmp_path)
    payload = _payload(
        interpreter=str(interpreter),
        robocasa={"version": "1.0.1", "commit": commit},
    )
    if accepted:
        report = inspect_robocasa_dependencies(
            interpreter, formal=True, runner=_runner(payload, [])
        )
        assert report.robocasa_commit == commit
    else:
        with pytest.raises(RuntimeError, match="RoboCasa commit.*mismatched"):
            inspect_robocasa_dependencies(
                interpreter, formal=True, runner=_runner(payload, [])
            )


def test_missing_interpreter_fails_before_subprocess(tmp_path) -> None:
    with pytest.raises(ValueError, match="RPENT_ROBOCASA_PYTHON.*does not exist"):
        inspect_robocasa_dependencies(tmp_path / "missing")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"robocasa": {"version": "x", "commit": "0" * 40}}, "RoboCasa commit"),
        ({"robosuite": {"version": "x", "commit": "0" * 40}}, "robosuite commit"),
        ({"mujoco_gl": "glfw"}, "MUJOCO_GL.*egl"),
        ({"macros_private": False}, "setup_macros"),
        (
            {
                "assets": {
                    **_payload()["assets"],
                    "objects_lightwheel": False,
                }
            },
            "lightwheel.*assets",
        ),
    ],
)
def test_formal_preflight_fails_closed_on_identity_or_runtime_drift(
    tmp_path, overrides, message
) -> None:
    interpreter = _interpreter(tmp_path)
    payload = _payload(interpreter=str(interpreter), **overrides)

    with pytest.raises(RuntimeError, match=message):
        inspect_robocasa_dependencies(
            interpreter,
            runner=_runner(payload, []),
        )


def test_optional_registry_must_be_present_only_when_requested(tmp_path) -> None:
    interpreter = _interpreter(tmp_path)
    report = inspect_robocasa_dependencies(
        interpreter,
        runner=_runner(_payload(interpreter=str(interpreter)), []),
    )
    assert report.available_registries == frozenset({"lightwheel"})

    with pytest.raises(RuntimeError, match="objaverse.*assets"):
        inspect_robocasa_dependencies(
            interpreter,
            requested_registries=("objaverse", "lightwheel"),
            runner=_runner(_payload(interpreter=str(interpreter)), []),
        )


def test_subprocess_failure_has_actionable_stderr(tmp_path) -> None:
    interpreter = _interpreter(tmp_path)

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="ModuleNotFoundError: robocasa",
        )

    with pytest.raises(RuntimeError, match="cannot inspect.*ModuleNotFoundError"):
        inspect_robocasa_dependencies(interpreter, runner=fail)


def test_ordinary_help_does_not_import_or_preflight_robocasa(monkeypatch) -> None:
    before = {name for name in sys.modules if name.startswith("robocasa")}
    monkeypatch.delenv("RPENT_ROBOCASA_PYTHON", raising=False)

    help_text = _build_argparser().format_help()

    assert "robocasa" in help_text
    assert {name for name in sys.modules if name.startswith("robocasa")} == before
