"""Typed runtime configuration loaded from the repository-root ``.env``."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


def _package_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_PATH_ENV_VARS = (
    "RPENT_RLINF_ROOT",
    "RPENT_LIBERO_ENV",
    "RPENT_SMOLVLA_PYTHON",
    "RPENT_OPENPI_PYTHON",
    "RPENT_ROBOCASA_PI05_CHECKPOINT",
    "RPENT_ROBOCASA_PYTHON",
    "PI05_CHECKPOINT_PATH",
    "SAM3_CHECKPOINT_PATH",
    "SAM3_BPE_PATH",
    "LIBERO_PRO_ASSET_PATH",
    "LIBERO_CONFIG_PATH",
    "OPENPI_DATA_HOME",
    "RPENT_HOME",
    "XDG_CACHE_HOME",
    "UV_CACHE_DIR",
    "HF_HOME",
    "NUMBA_CACHE_DIR",
    "MPLCONFIGDIR",
    "CODEX_BIN",
)

_ISOLATED_PYTHON_ENV_VARS = {
    "RPENT_OPENPI_PYTHON",
    "RPENT_SMOLVLA_PYTHON",
    "RPENT_ROBOCASA_PYTHON",
}

_DEFAULT_ENV = {
    "LIBERO_TYPE": "pro",
    "MUJOCO_GL": "egl",
    "CUDA_VISIBLE_DEVICES": "0",
    "RPENT_STREAMLIT_HOST": "127.0.0.1",
    "RPENT_STREAMLIT_PORT": "8501",
    "RPENT_DASHBOARD_HOST": "127.0.0.1",
    "RPENT_DASHBOARD_PORT": "8000",
    "RPENT_DASHBOARD_LANGUAGE": "zh-cn",
}

_SHELL_ENV_VARS = (
    "RPENT_REPO_ROOT",
    *_PATH_ENV_VARS,
    *_DEFAULT_ENV,
    "RPENT_DASHBOARD_RUNTIME_DIR",
    "RPENT_SHARED_SERVICE_TIMEOUT_S",
    "RPENT_BENCHMARK_PREVIEW_TIMEOUT_S",
    "CODEX_BASE_URL",
    "CODEX_API_KEY",
    "CODEX_MODEL",
    "CODEX_REASONING_EFFORT",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)


class _RuntimeSettings(BaseModel):
    """Internal typed snapshot; credentials remain redacted in repr/errors."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        allow_inf_nan=False,
    )

    RPENT_REPO_ROOT: Path = Field(default_factory=_package_repo_root)
    RPENT_RLINF_ROOT: Path | None = None
    RPENT_LIBERO_ENV: Path | None = Field(default_factory=lambda: Path(sys.prefix))
    PI05_CHECKPOINT_PATH: Path | None = None
    SAM3_CHECKPOINT_PATH: Path | None = None
    SAM3_BPE_PATH: Path | None = None
    LIBERO_PRO_ASSET_PATH: Path | None = None
    LIBERO_TYPE: str = "pro"
    RPENT_STREAMLIT_HOST: str = "127.0.0.1"
    RPENT_STREAMLIT_PORT: int = Field(default=8501, ge=1, le=65535)
    RPENT_DASHBOARD_HOST: str = "127.0.0.1"
    RPENT_DASHBOARD_PORT: int = Field(default=8000, ge=1, le=65535)
    RPENT_DASHBOARD_LANGUAGE: str = "zh-cn"
    RPENT_DASHBOARD_RUNTIME_DIR: Path | None = None
    RPENT_SHARED_SERVICE_TIMEOUT_S: float | None = Field(default=None, gt=0)
    RPENT_BENCHMARK_PREVIEW_TIMEOUT_S: float | None = Field(default=None, gt=0)
    CODEX_BASE_URL: str | None = None
    CODEX_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None

    @field_validator(
        "RPENT_RLINF_ROOT",
        "RPENT_LIBERO_ENV",
        "PI05_CHECKPOINT_PATH",
        "SAM3_CHECKPOINT_PATH",
        "SAM3_BPE_PATH",
        "LIBERO_PRO_ASSET_PATH",
        "RPENT_DASHBOARD_RUNTIME_DIR",
        mode="before",
    )
    @classmethod
    def _blank_path_is_unset(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("CODEX_BASE_URL", mode="before")
    @classmethod
    def _blank_string_is_unset(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator(
        "RPENT_SHARED_SERVICE_TIMEOUT_S",
        "RPENT_BENCHMARK_PREVIEW_TIMEOUT_S",
        mode="before",
    )
    @classmethod
    def _blank_float_is_unset(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value


_MODEL_PATH_FIELDS = (
    "RPENT_REPO_ROOT",
    "RPENT_RLINF_ROOT",
    "RPENT_LIBERO_ENV",
    "PI05_CHECKPOINT_PATH",
    "SAM3_CHECKPOINT_PATH",
    "SAM3_BPE_PATH",
    "LIBERO_PRO_ASSET_PATH",
)


def _resolve_path(
    value: str | Path, root: Path, *, preserve_symlink: bool = False
) -> Path:
    path = Path(value).expanduser()
    absolute = root / path if not path.is_absolute() else path
    return Path(os.path.abspath(absolute)) if preserve_symlink else absolute.resolve()


def _normalize_settings(settings: _RuntimeSettings) -> _RuntimeSettings:
    root = _resolve_path(settings.RPENT_REPO_ROOT, _package_repo_root())
    updates: dict[str, Path] = {"RPENT_REPO_ROOT": root}
    for name in _MODEL_PATH_FIELDS[1:]:
        value = getattr(settings, name)
        if value is not None:
            updates[name] = _resolve_path(value, root)
    return settings.model_copy(update=updates)


def _current_settings() -> _RuntimeSettings:
    """Return a normalized snapshot without reading dotenv or mutating state."""
    return _normalize_settings(_RuntimeSettings.model_validate(dict(os.environ)))


def load_runtime_environment(
    *,
    repo_root: Path | None = None,
    env_file: Path | None = None,
) -> None:
    """Load ``.env`` once, preserving caller values and normalizing paths."""
    package_root = _package_repo_root()
    initial_root = _resolve_path(
        repo_root or os.environ.get("RPENT_REPO_ROOT", package_root),
        package_root,
    )
    selected_env_file = env_file or os.environ.get("RPENT_ENV_FILE")
    dotenv_path = (
        _resolve_path(selected_env_file, initial_root)
        if selected_env_file
        else initial_root / ".env"
    )
    load_dotenv(dotenv_path=dotenv_path, override=False)

    root = _resolve_path(
        repo_root or os.environ.get("RPENT_REPO_ROOT", initial_root),
        package_root,
    )
    os.environ["RPENT_REPO_ROOT"] = str(root)
    os.environ.setdefault("RPENT_LIBERO_ENV", str(Path(sys.prefix).resolve()))
    for name in _PATH_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            os.environ[name] = str(
                _resolve_path(
                    value,
                    root,
                    preserve_symlink=name in _ISOLATED_PYTHON_ENV_VARS,
                )
            )
    for name, value in _DEFAULT_ENV.items():
        os.environ.setdefault(name, value)

    # Parse once here so malformed typed values fail at the loading seam.
    _current_settings()


def _shell_exports() -> str:
    return "\n".join(
        f"export {name}={shlex.quote(os.environ.get(name, ''))}"
        for name in _SHELL_ENV_VARS
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Load RPent runtime settings")
    parser.add_argument(
        "--shell-exports",
        action="store_true",
        help="print safely quoted exports for scripts/load_rpent_env.sh",
    )
    args = parser.parse_args(argv)
    if not args.shell_exports:
        parser.error("--shell-exports is required")
    load_runtime_environment()
    print(_shell_exports())
    return 0


class RuntimeConfigurationError(ValueError):
    """Raised when a selected runtime mode lacks required configuration."""

    def __init__(self, missing: list[str] | tuple[str, ...]) -> None:
        names = tuple(sorted(set(missing)))
        self.missing = names
        super().__init__(
            "Missing required runtime configuration: " + ", ".join(names)
        )


def _secret_is_set(value: SecretStr | None) -> bool:
    return value is not None and bool(value.get_secret_value().strip())


def validate_runtime_requirements(
    *,
    require_pi05: bool = False,
    require_sam3: bool = False,
    require_libero: bool = False,
    planner: str | None = None,
    model: str | None = None,
) -> None:
    """Fail fast for requirements used by the selected runtime only."""
    settings = _current_settings()
    missing: list[str] = []
    if require_pi05 and settings.PI05_CHECKPOINT_PATH is None:
        missing.append("PI05_CHECKPOINT_PATH")
    if require_sam3:
        if settings.SAM3_CHECKPOINT_PATH is None:
            missing.append("SAM3_CHECKPOINT_PATH")
        if settings.SAM3_BPE_PATH is None:
            missing.append("SAM3_BPE_PATH")
    if require_libero:
        environment = settings.RPENT_LIBERO_ENV or Path(sys.prefix)
        if not (environment / "bin" / "python").is_file():
            missing.append("RPENT_LIBERO_ENV")

    if planner == "api" and model:
        provider = model.partition(":")[0]
        if provider == "anthropic" and not _secret_is_set(
            settings.ANTHROPIC_API_KEY
        ):
            missing.append("ANTHROPIC_API_KEY")
        if provider in {"openai", "openai-chat", "codexgateway"} and not _secret_is_set(
            settings.OPENAI_API_KEY
        ):
            missing.append("OPENAI_API_KEY")
    elif planner == "codex":
        has_url = bool((settings.CODEX_BASE_URL or "").strip())
        has_key = _secret_is_set(settings.CODEX_API_KEY)
        if has_url and not has_key:
            missing.append("CODEX_API_KEY")
        if has_key and not has_url:
            missing.append("CODEX_BASE_URL")

    if missing:
        raise RuntimeConfigurationError(missing)


def get_repo_root() -> Path:
    return _current_settings().RPENT_REPO_ROOT


def get_resources_dir(env_name: str) -> Path:
    return get_repo_root() / "resources" / env_name


def get_memory_dir(env_name: str) -> Path:
    return get_resources_dir(env_name) / "memory"


def _checked_existing_path(
    name: str,
    path: Path | None,
    *,
    required: bool = False,
) -> Path | None:
    if path is None:
        if required:
            raise ValueError(f"{name} is not configured")
        return None
    if required and not path.exists():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def get_environment_root() -> Path:
    settings = _current_settings()
    candidate = settings.RPENT_LIBERO_ENV or _resolve_path(
        os.environ.get("VIRTUAL_ENV") or sys.prefix,
        settings.RPENT_REPO_ROOT,
    )
    if not (candidate / "bin" / "python").is_file():
        source = (
            "RPENT_LIBERO_ENV"
            if os.environ.get("RPENT_LIBERO_ENV")
            else "active Python environment"
        )
        raise ValueError(f"{source} has no bin/python: {candidate}")
    return candidate


def get_python_executable() -> Path:
    return get_environment_root() / "bin" / "python"


def get_rpent_executable() -> Path:
    return get_environment_root() / "bin" / "rpent"


def get_pi05_checkpoint_path() -> str:
    path = _current_settings().PI05_CHECKPOINT_PATH
    return str(path) if path is not None else ""


def get_sam3_checkpoint_path(*, required: bool = False) -> Path | None:
    return _checked_existing_path(
        "SAM3_CHECKPOINT_PATH",
        _current_settings().SAM3_CHECKPOINT_PATH,
        required=required,
    )


def get_sam3_bpe_path(*, required: bool = False) -> Path | None:
    return _checked_existing_path(
        "SAM3_BPE_PATH",
        _current_settings().SAM3_BPE_PATH,
        required=required,
    )


def get_libero_pro_asset_path(*, required: bool = False) -> Path | None:
    return _checked_existing_path(
        "LIBERO_PRO_ASSET_PATH",
        _current_settings().LIBERO_PRO_ASSET_PATH,
        required=required,
    )


def get_libero_type() -> str:
    return _current_settings().LIBERO_TYPE


def get_rlinf_repo_path() -> Path | None:
    settings = _current_settings()
    if settings.RPENT_RLINF_ROOT is not None:
        return settings.RPENT_RLINF_ROOT
    legacy = os.environ.get("RLINF_REPO_PATH", "").strip()
    return _resolve_path(legacy, settings.RPENT_REPO_ROOT) if legacy else None


def get_streamlit_host() -> str:
    return _current_settings().RPENT_STREAMLIT_HOST


def get_streamlit_port() -> int:
    return _current_settings().RPENT_STREAMLIT_PORT


def get_dashboard_host() -> str:
    return _current_settings().RPENT_DASHBOARD_HOST


def get_dashboard_port() -> int:
    return _current_settings().RPENT_DASHBOARD_PORT


def get_dashboard_language() -> str:
    return _current_settings().RPENT_DASHBOARD_LANGUAGE


def get_dashboard_runtime_dir() -> Path | None:
    value = _current_settings().RPENT_DASHBOARD_RUNTIME_DIR
    return value.resolve() if value is not None else None


def get_shared_service_timeout_s() -> float:
    return _current_settings().RPENT_SHARED_SERVICE_TIMEOUT_S or 300.0


def get_benchmark_preview_timeout_s() -> float:
    return _current_settings().RPENT_BENCHMARK_PREVIEW_TIMEOUT_S or 120.0


# Importing the shared configuration seam loads dotenv exactly once per process.
load_runtime_environment()


if __name__ == "__main__":
    raise SystemExit(main())
