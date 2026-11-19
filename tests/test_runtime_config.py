from pathlib import Path

import pytest

from rpent.utils import config


def _make_environment(root: Path) -> Path:
    python = root / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    return root


def test_environment_root_prefers_valid_explicit_path(monkeypatch, tmp_path):
    environment = _make_environment(tmp_path / "env")
    monkeypatch.setenv("RPENT_LIBERO_ENV", str(environment))

    assert config.get_environment_root() == environment
    assert config.get_python_executable() == environment / "bin" / "python"
    assert config.get_rpent_executable() == environment / "bin" / "rpent"


def test_invalid_explicit_environment_names_variable(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("RPENT_LIBERO_ENV", str(missing))

    with pytest.raises(ValueError, match="RPENT_LIBERO_ENV"):
        config.get_environment_root()


def test_relative_checkpoint_resolves_from_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("PI05_CHECKPOINT_PATH", "models/pi05")

    assert config.get_pi05_checkpoint_path() == str(tmp_path / "models/pi05")


def test_optional_model_path_can_be_missing(monkeypatch):
    monkeypatch.delenv("SAM3_CHECKPOINT_PATH", raising=False)

    assert config.get_sam3_checkpoint_path() is None


def test_relative_sam3_bpe_path_resolves_from_repo(monkeypatch, tmp_path):
    bpe_path = tmp_path / "models" / "bpe_simple_vocab_16e6.txt.gz"
    bpe_path.parent.mkdir()
    bpe_path.touch()
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("SAM3_BPE_PATH", "models/bpe_simple_vocab_16e6.txt.gz")

    assert config.get_sam3_bpe_path(required=True) == bpe_path


def test_required_sam3_bpe_path_names_missing_variable(monkeypatch):
    monkeypatch.delenv("SAM3_BPE_PATH", raising=False)

    with pytest.raises(ValueError, match="SAM3_BPE_PATH is not configured"):
        config.get_sam3_bpe_path(required=True)


@pytest.mark.parametrize(
    ("name", "accessor"),
    [
        ("RPENT_STREAMLIT_PORT", config.get_streamlit_port),
        ("RPENT_DASHBOARD_PORT", config.get_dashboard_port),
    ],
)
def test_invalid_ui_port_names_variable(monkeypatch, name, accessor):
    monkeypatch.setenv(name, "invalid")

    with pytest.raises(ValueError, match=name):
        accessor()


def test_ui_network_defaults(monkeypatch):
    for name in (
        "RPENT_STREAMLIT_HOST",
        "RPENT_STREAMLIT_PORT",
        "RPENT_DASHBOARD_HOST",
        "RPENT_DASHBOARD_PORT",
    ):
        monkeypatch.delenv(name, raising=False)

    assert config.get_streamlit_host() == "127.0.0.1"
    assert config.get_streamlit_port() == 8501
    assert config.get_dashboard_host() == "127.0.0.1"
    assert config.get_dashboard_port() == 8000


def test_dashboard_runtime_dir_is_optional(monkeypatch, tmp_path):
    monkeypatch.delenv("RPENT_DASHBOARD_RUNTIME_DIR", raising=False)
    assert config.get_dashboard_runtime_dir() is None

    runtime_dir = tmp_path / "dashboard-runtime"
    monkeypatch.setenv("RPENT_DASHBOARD_RUNTIME_DIR", str(runtime_dir))
    assert config.get_dashboard_runtime_dir() == runtime_dir.resolve()


def test_shared_service_timeout_defaults_to_300_seconds(monkeypatch):
    monkeypatch.delenv("RPENT_SHARED_SERVICE_TIMEOUT_S", raising=False)

    assert config.get_shared_service_timeout_s() == 300.0


def test_shared_service_timeout_accepts_positive_float(monkeypatch):
    monkeypatch.setenv("RPENT_SHARED_SERVICE_TIMEOUT_S", "42.5")

    assert config.get_shared_service_timeout_s() == 42.5


def test_benchmark_preview_timeout_defaults_to_120_seconds(monkeypatch):
    monkeypatch.delenv("RPENT_BENCHMARK_PREVIEW_TIMEOUT_S", raising=False)

    assert config.get_benchmark_preview_timeout_s() == 120.0


def test_benchmark_preview_timeout_accepts_positive_float(monkeypatch):
    monkeypatch.setenv("RPENT_BENCHMARK_PREVIEW_TIMEOUT_S", "75.5")

    assert config.get_benchmark_preview_timeout_s() == 75.5


def test_shared_service_timeout_treats_blank_as_default(monkeypatch):
    monkeypatch.setenv("RPENT_SHARED_SERVICE_TIMEOUT_S", "")

    assert config.get_shared_service_timeout_s() == 300.0


@pytest.mark.parametrize("value", ["invalid", "0", "-1", "nan", "inf"])
def test_shared_service_timeout_rejects_nonpositive_or_nonfinite_values(
    monkeypatch, value
):
    monkeypatch.setenv("RPENT_SHARED_SERVICE_TIMEOUT_S", value)

    with pytest.raises(ValueError, match="RPENT_SHARED_SERVICE_TIMEOUT_S"):
        config.get_shared_service_timeout_s()
