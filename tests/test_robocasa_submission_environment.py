"""Installation checks use static metadata and imports, never robot execution."""

from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import setup_robocasa_submission as setup

ROOT = Path(__file__).resolve().parents[1]
TESTED_COORDINATOR = {
    "pydantic-ai-slim": "2.54.0", "pydantic": "2.14.3", "httpx": "0.28.1",
    "openai": "3.24.0", "fastapi": "0.142.2", "uvicorn": "0.54.0",
    "imageio-ffmpeg": "0.6.0",
}


def checker():
    return importlib.import_module("tools.check_robocasa_submission_environment")


def test_coordinator_api_pins_match_the_cpu_compatibility_surface():
    text = (ROOT / setup.ENV_DIR / "coordinator.lock").read_text()
    pins = dict(re.findall(r"^([a-zA-Z0-9_.-]+)==([^\s;\\]+)", text, re.M))
    assert {name: pins.get(name) for name in TESTED_COORDINATOR} == TESTED_COORDINATOR


def test_a_missing_lock_manifest_entry_is_not_silently_accepted(tmp_path):
    target = tmp_path / setup.ENV_DIR
    shutil.copytree(ROOT / setup.ENV_DIR, target)
    path = target / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"].pop("policy.lock")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lock manifest"):
        setup.validate_locks(tmp_path)


def test_setup_plan_includes_import_only_verification_without_execution(tmp_path):
    prefix = tmp_path / "new runtime"
    plan = setup.provision(ROOT, prefix, "uv", apply=False, assets=True, checkpoint=True)
    assert "check_robocasa_submission_environment.py" in " ".join(plan["verification_command"])
    assert plan["verification_scope"] == "imports_and_static_metadata_only"
    assert not prefix.exists()
    assert plan["workload_counts"] == {
        "provider_calls": 0, "model_loads": 0, "environment_constructors": 0,
        "policy_predictions": 0, "control_actions": 0,
    }


def _component(component, prefix):
    versions = setup.read_lock_pins(ROOT, component)
    result = {
        "component": component, "interpreter": str(prefix / component / "bin/python"),
        "python_version": "3.12.0" if component == "policy" else "3.11.0",
        "versions": versions, "imports": dict.fromkeys(checker().IMPORTS[component], True),
        "framework_version": "1.2.3.5", "source_import_path": str(ROOT / "rpent/__init__.py"),
        "workload_counts": dict(checker().ZERO_WORKLOAD),
    }
    if component == "simulator":
        result["simulator_metadata"] = {"robocasa": {"commit": setup.ROBOCASA_COMMIT},
                                         "robosuite": {"commit": setup.ROBOSUITE_COMMIT}, "mujoco_gl": "egl"}
    if component == "policy":
        result.update(cuda_build="12.8", cxx11_abi=True,
                      flash_attention_origin={"url": setup.FLASH_WHEEL})
        result["versions"]["flash-attn"] = "2.8.3"
    return result


def test_version_drift_fails_even_when_the_package_imports(tmp_path):
    report = _component("coordinator", tmp_path)
    report["versions"]["httpx"] = "1.0.dev6"
    findings = checker().validate_component(ROOT, tmp_path, "coordinator", report)
    assert any(row["check"] == "package:httpx" and not row["passed"] for row in findings)


def test_import_failure_and_wrong_interpreter_are_not_readiness(tmp_path):
    report = _component("coordinator", tmp_path)
    report["interpreter"] = "/another/python"
    report["imports"][checker().IMPORTS["coordinator"][0]] = "ImportError"
    findings = checker().validate_component(ROOT, tmp_path, "coordinator", report)
    assert any(row["check"] == "interpreter" and not row["passed"] for row in findings)
    assert any(row["check"].startswith("import:") and not row["passed"] for row in findings)


def test_verification_rejects_reports_that_executed_a_model(tmp_path):
    report = _component("coordinator", tmp_path)
    report["workload_counts"]["model_loads"] = 1
    findings = checker().validate_component(ROOT, tmp_path, "coordinator", report)
    assert any(row["check"] == "zero_workload" and not row["passed"] for row in findings)


def test_verifier_uses_only_dedicated_interpreters_and_offline_probes(tmp_path):
    prefix = tmp_path / "runtime"
    commands = []
    for name in ("coordinator", "simulator", "policy"):
        path = prefix / name / "bin/python"
        path.parent.mkdir(parents=True)
        path.write_text("fixture interpreter, not executable by this test")
        path.chmod(0o755)

    def run(command, **kwargs):
        commands.append((command, kwargs))
        component = command[command.index("--probe") + 1]
        return subprocess.CompletedProcess(command, 0, json.dumps(_component(component, prefix)), "")

    result = checker().verify_installation(ROOT, prefix, runner=run)
    assert result["passed"] is True
    assert len(commands) == 3
    for command, kwargs in commands:
        assert "--probe" in command
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert not any(name in command for name in ("--evaluation-profile", "--task-set"))
    assert result["native_environment_qualified"] is False


def test_failed_component_cannot_generate_a_success_receipt(tmp_path):
    prefix = tmp_path / "runtime"
    for name in ("coordinator", "simulator", "policy"):
        path = prefix / name / "bin/python"
        path.parent.mkdir(parents=True)
        path.write_text("fixture")
        path.chmod(0o755)

    def run(command, **kwargs):
        component = command[command.index("--probe") + 1]
        report = _component(component, prefix)
        report["versions"]["numpy"] = "0.0.0"
        return subprocess.CompletedProcess(command, 0, json.dumps(report), "")

    result = checker().verify_installation(ROOT, prefix, runner=run)
    assert result["passed"] is False
    assert result["native_environment_qualified"] is False


def test_failed_install_preserves_a_failure_receipt_and_no_activation_file(tmp_path, monkeypatch):
    from tools import release
    prefix = tmp_path / "partial installation"
    monkeypatch.setattr(setup, "host_preflight", lambda uv: {"fixture": True})
    monkeypatch.setattr(release, "verify", lambda root: {"passed": True})

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(setup.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        setup.provision(ROOT, prefix, "uv", apply=True, assets=False, checkpoint=False)
    receipt = json.loads((prefix / "setup-receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["returncode"] == 17
    assert receipt["completed_commands"] == 0
    assert receipt["applied"] is False
    assert not (prefix / "environment.sh").exists()


def test_explicit_asset_setup_handles_the_pinned_prompt_and_objaverse(tmp_path, monkeypatch):
    from tools import release
    prefix = tmp_path / "mocked installation"
    calls = []
    monkeypatch.setattr(setup, "host_preflight", lambda uv: {"fixture": True})
    monkeypatch.setattr(release, "verify", lambda root: {"passed": True})
    monkeypatch.setattr(checker(), "verify_installation", lambda *args, **kwargs: {
        "passed": True, "fixture": True, "native_environment_qualified": False,
    })

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(setup.subprocess, "run", run)
    result = setup.provision(ROOT, prefix, "uv", apply=True, assets=True, checkpoint=False)
    asset = [(command, kwargs) for command, kwargs in calls if "robocasa.scripts.download_kitchen_assets" in command]
    assert len(asset) == 1
    assert "objs_objaverse" in asset[0][0]
    assert asset[0][1]["input"] == "y\n"
    assert result["status"] == "installed_and_import_verified"
    assert result["native_environment_qualified"] is False
    assert "RPENT_XIAOMI_MIBOT_SNAPSHOT" not in (prefix / "environment.sh").read_text()


def test_profile_asset_check_rejects_missing_objaverse(tmp_path):
    report = _component("simulator", tmp_path)
    report["simulator_metadata"].update(macros_private=True, assets={
        "textures": True, "generative_textures": True, "fixtures_lightwheel": True,
        "objects_lightwheel": True, "objects_objaverse": False,
    })
    checks = checker().validate_component(ROOT, tmp_path, "simulator", report, require_assets=True)
    assert any(row["check"] == "asset:objects_objaverse" and not row["passed"] for row in checks)
