from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rpent.planning.manifest import build_experiment_manifest
from rpent.utils import config
from rpent.utils.config import (
    RuntimeConfigurationError,
    load_runtime_environment,
    validate_runtime_requirements,
)

_REQUIREMENT_NAMES = (
    "PI05_CHECKPOINT_PATH",
    "SAM3_CHECKPOINT_PATH",
    "SAM3_BPE_PATH",
    "CODEX_BASE_URL",
    "CODEX_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)




@pytest.fixture
def clean_runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    for name in _REQUIREMENT_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def isolated_settings_env(monkeypatch: pytest.MonkeyPatch):
    for name in config._SHELL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("RPENT_ENV_FILE", raising=False)


def test_python_loader_reads_repository_dotenv(
    isolated_settings_env, tmp_path: Path
):
    (tmp_path / ".env").write_text(
        'PI05_CHECKPOINT_PATH="models/pi05"\n'
        'CODEX_BASE_URL="https://example.invalid/v1?a=b" # comment\n'
        "ANTHROPIC_BASE_URL=https://anthropic.example.invalid/v1\n"
        "SAM3_BPE_PATH=\n",
        encoding="utf-8",
    )
    load_runtime_environment(repo_root=tmp_path)
    settings = config._current_settings()

    assert settings.PI05_CHECKPOINT_PATH == tmp_path / "models/pi05"
    assert settings.CODEX_BASE_URL == "https://example.invalid/v1?a=b"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://anthropic.example.invalid/v1"
    assert settings.SAM3_BPE_PATH is None
    assert os.environ["SAM3_BPE_PATH"] == ""


def test_parent_environment_including_empty_wins(
    isolated_settings_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (tmp_path / ".env").write_text(
        "PI05_CHECKPOINT_PATH=from-dotenv\nSAM3_BPE_PATH=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PI05_CHECKPOINT_PATH", "from-parent")
    monkeypatch.setenv("SAM3_BPE_PATH", "")

    load_runtime_environment(repo_root=tmp_path)
    settings = config._current_settings()

    assert settings.PI05_CHECKPOINT_PATH == tmp_path / "from-parent"
    assert settings.SAM3_BPE_PATH is None
    assert os.environ["PI05_CHECKPOINT_PATH"] == str(tmp_path / "from-parent")
    assert os.environ["SAM3_BPE_PATH"] == ""


def test_all_relative_runtime_paths_resolve_from_repository(
    isolated_settings_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path_values = {
        "RPENT_RLINF_ROOT": "repos/rlinf",
        "RPENT_LIBERO_ENV": ".venv",
        "RPENT_SMOLVLA_PYTHON": "envs/smolvla/bin/python",
        "RPENT_ROBOCASA_PYTHON": "envs/robocasa/bin/python",
        "PI05_CHECKPOINT_PATH": "models/pi05",
        "SAM3_CHECKPOINT_PATH": "models/sam3.pt",
        "SAM3_BPE_PATH": "models/bpe.txt.gz",
        "LIBERO_PRO_ASSET_PATH": "assets/liberopro",
        "LIBERO_CONFIG_PATH": "config/libero",
        "OPENPI_DATA_HOME": "cache/openpi",
        "RPENT_HOME": "runtime/home",
        "XDG_CACHE_HOME": "runtime/cache",
        "UV_CACHE_DIR": "runtime/uv",
        "HF_HOME": "runtime/hf",
        "NUMBA_CACHE_DIR": "runtime/numba",
        "MPLCONFIGDIR": "runtime/matplotlib",
        "CODEX_BIN": "bin/codex",
    }
    for name, value in path_values.items():
        monkeypatch.setenv(name, value)

    load_runtime_environment(repo_root=tmp_path)

    for name in path_values:
        resolved = Path(os.environ[name])
        assert resolved.is_absolute(), name
        assert resolved.is_relative_to(tmp_path), name


def test_loader_preserves_isolated_python_interpreter_symlinks(
    isolated_settings_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    system_python = tmp_path / "system-python"
    system_python.touch()
    interpreter = tmp_path / "envs" / "robocasa" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(system_python)
    monkeypatch.setenv("RPENT_ROBOCASA_PYTHON", str(interpreter))

    load_runtime_environment(repo_root=tmp_path)

    assert Path(os.environ["RPENT_ROBOCASA_PYTHON"]) == interpreter


def test_loader_defaults_libero_environment_to_active_prefix(
    isolated_settings_env, tmp_path: Path
):
    load_runtime_environment(repo_root=tmp_path)

    assert config.get_environment_root() == Path(sys.prefix).resolve()


def test_runtime_loader_has_no_public_settings_return(
    isolated_settings_env, tmp_path: Path
):
    assert load_runtime_environment(repo_root=tmp_path) is None


def test_get_repo_root_does_not_reload_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))

    def fail_if_called(**_kwargs):
        raise AssertionError("get_repo_root reloaded runtime configuration")

    monkeypatch.setattr(config, "load_runtime_environment", fail_if_called)

    assert config.get_repo_root() == tmp_path


def test_direct_python_process_loads_dotenv_without_shell_source(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "PI05_CHECKPOINT_PATH=models/from-dotenv\n",
        encoding="utf-8",
    )
    child_env = dict(os.environ)
    child_env["RPENT_REPO_ROOT"] = str(tmp_path)
    child_env.pop("PI05_CHECKPOINT_PATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rpent.utils.config import get_pi05_checkpoint_path; "
            "print(get_pi05_checkpoint_path())",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=child_env,
    )

    assert result.stdout.strip() == str(tmp_path / "models/from-dotenv")


@pytest.mark.parametrize("require_sam3", [False, True])
def test_vla_and_hybrid_startup_name_missing_pi05(
    clean_runtime_env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    require_sam3: bool,
):
    if require_sam3:
        monkeypatch.setenv("SAM3_CHECKPOINT_PATH", str(tmp_path / "sam3.pt"))
        monkeypatch.setenv("SAM3_BPE_PATH", str(tmp_path / "bpe.txt.gz"))
    with pytest.raises(RuntimeConfigurationError, match="PI05_CHECKPOINT_PATH"):
        validate_runtime_requirements(
            require_pi05=True,
            require_sam3=require_sam3,
        )


@pytest.mark.parametrize("missing", ["SAM3_CHECKPOINT_PATH", "SAM3_BPE_PATH"])
def test_hybrid_names_each_missing_sam3_variable(
    clean_runtime_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
):
    monkeypatch.setenv("PI05_CHECKPOINT_PATH", str(tmp_path / "pi05"))
    monkeypatch.setenv("SAM3_CHECKPOINT_PATH", str(tmp_path / "sam3.pt"))
    monkeypatch.setenv("SAM3_BPE_PATH", str(tmp_path / "bpe.txt.gz"))
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeConfigurationError, match=missing):
        validate_runtime_requirements(require_pi05=True, require_sam3=True)


def test_vla_only_does_not_require_sam3(
    clean_runtime_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("PI05_CHECKPOINT_PATH", str(tmp_path / "pi05"))

    assert (
        validate_runtime_requirements(require_pi05=True, require_sam3=False)
        is None
    )


def test_runtime_validator_has_no_unused_base_url_parameter():
    assert "base_url" not in inspect.signature(validate_runtime_requirements).parameters


def test_ordinary_settings_access_does_not_require_checkpoints(clean_runtime_env):
    load_runtime_environment()

    assert config.get_pi05_checkpoint_path() == ""
    assert config.get_sam3_checkpoint_path() is None
    assert config.get_sam3_bpe_path() is None


@pytest.mark.parametrize(
    ("model", "credential"),
    [
        ("anthropic:claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
        ("openai:gpt-5", "OPENAI_API_KEY"),
        ("openai-chat:gpt-5", "OPENAI_API_KEY"),
        ("codexgateway:gpt-5.6-luna", "OPENAI_API_KEY"),
    ],
)
def test_api_planner_names_provider_credential(
    clean_runtime_env, model: str, credential: str
):
    with pytest.raises(RuntimeConfigurationError, match=credential):
        validate_runtime_requirements(planner="api", model=model)


def test_codex_endpoint_requires_key_without_leaking_value(
    clean_runtime_env, monkeypatch: pytest.MonkeyPatch
):
    sentinel = "super-secret-sentinel"
    monkeypatch.setenv("CODEX_BASE_URL", "https://codex.example.invalid/v1")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_runtime_requirements(planner="codex")

    assert "CODEX_API_KEY" in str(caught.value)
    assert sentinel not in str(caught.value)


def test_codex_key_requires_endpoint_without_leaking_secret(
    clean_runtime_env, monkeypatch: pytest.MonkeyPatch
):
    sentinel = "super-secret-sentinel"
    monkeypatch.setenv("CODEX_API_KEY", sentinel)
    monkeypatch.delenv("CODEX_BASE_URL", raising=False)

    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_runtime_requirements(planner="codex")

    assert "CODEX_BASE_URL" in str(caught.value)
    assert sentinel not in str(caught.value)


def test_sensitive_values_are_redacted_from_repr_errors_logs_and_manifest(
    isolated_settings_env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    sentinel = "super-secret-sentinel"
    monkeypatch.setenv("CODEX_API_KEY", sentinel)
    monkeypatch.setenv("CODEX_BASE_URL", "https://codex.example.invalid/v1")

    load_runtime_environment(repo_root=tmp_path)
    settings = config._current_settings()

    assert sentinel not in repr(settings)
    assert sentinel not in caplog.text

    monkeypatch.setenv("RPENT_STREAMLIT_PORT", "not-an-integer")
    with pytest.raises(ValueError) as caught:
        load_runtime_environment(repo_root=tmp_path)
    assert sentinel not in str(caught.value)

    class _ManifestConfig:
        @staticmethod
        def metadata():
            return {"planning_profile": "test"}

    manifest = build_experiment_manifest(
        config=_ManifestConfig(),
        planner_backend="codex",
        model=None,
        max_tokens=1,
        no_images=True,
        system_prompt="safe",
        user_message="safe",
        tool_schemas=[],
        repo_root=tmp_path,
    )
    assert sentinel not in json.dumps(manifest)
