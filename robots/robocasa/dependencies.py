"""Fail-closed dependency inspection for the isolated RoboCasa runtime."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROBOCASA_COMMIT = "456174f62b89b8fca99eaaf33949c29fec9cfc2a"
ROBOSUITE_COMMIT = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"

_SCHEMA_VERSION = 1
_DEFAULT_ASSETS = (
    "textures",
    "generative_textures",
    "fixtures_lightwheel",
    "objects_lightwheel",
)
_REGISTRY_ASSET = {
    "lightwheel": "objects_lightwheel",
    "objaverse": "objects_objaverse",
    "aigen": "objects_aigen",
}


@dataclass(frozen=True, slots=True)
class RoboCasaDependencyReport:
    """Validated identity and asset report for a RoboCasa interpreter."""

    interpreter: str
    robocasa_version: str
    robocasa_commit: str
    robosuite_version: str
    robosuite_commit: str
    mujoco_version: str
    mujoco_gl: str
    default_assets_ready: bool
    available_registries: frozenset[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "interpreter": self.interpreter,
            "robocasa_version": self.robocasa_version,
            "robocasa_commit": self.robocasa_commit,
            "robosuite_version": self.robosuite_version,
            "robosuite_commit": self.robosuite_commit,
            "mujoco_version": self.mujoco_version,
            "mujoco_gl": self.mujoco_gl,
            "default_assets_ready": self.default_assets_ready,
            "available_registries": sorted(self.available_registries),
        }


def _resolve_interpreter(interpreter: str | Path | None) -> Path:
    configured = interpreter or os.environ.get("RPENT_ROBOCASA_PYTHON")
    if not configured:
        raise ValueError(
            "RPENT_ROBOCASA_PYTHON must name the dedicated RoboCasa Python interpreter"
        )
    path = Path(os.path.abspath(Path(configured).expanduser()))
    if not path.exists():
        raise ValueError(f"RPENT_ROBOCASA_PYTHON does not exist: {path}")
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"RPENT_ROBOCASA_PYTHON is not an executable file: {path}")
    return path


def _extract_json(stdout: str) -> Mapping[str, Any]:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(
        "RoboCasa dependency inspection returned no JSON object; verify that "
        "the selected interpreter can import RPent's robots package"
    )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"RoboCasa dependency report is missing '{key}'")
    return value


def inspect_robocasa_dependencies(
    interpreter: str | Path | None = None,
    *,
    requested_registries: Sequence[str] = ("lightwheel",),
    formal: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> RoboCasaDependencyReport:
    """Inspect and validate RoboCasa exclusively in its selected interpreter."""

    selected = _resolve_interpreter(interpreter)
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )
    command = [
        str(selected),
        "-m",
        "robots.robocasa.dependencies",
        "--inspect-json",
    ]
    try:
        completed = runner(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(
            f"cannot inspect RoboCasa dependencies with {selected}: {detail}"
        ) from exc

    payload = _extract_json(completed.stdout)
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError(
            "unsupported RoboCasa dependency report schema: "
            f"{payload.get('schema_version')!r}"
        )

    reported_interpreter = Path(
        os.path.abspath(Path(str(payload.get("interpreter", ""))).expanduser())
    )
    if reported_interpreter != selected:
        raise RuntimeError(
            "RoboCasa dependency report came from a different interpreter: "
            f"expected {selected}, got {reported_interpreter}"
        )

    robocasa = _mapping(payload, "robocasa")
    robosuite = _mapping(payload, "robosuite")
    mujoco = _mapping(payload, "mujoco")
    assets = _mapping(payload, "assets")

    if robocasa.get("importable") is False:
        raise RuntimeError(
            "RoboCasa cannot be imported in RPENT_ROBOCASA_PYTHON: "
            f"{robocasa.get('import_error', 'unknown import error')}"
        )
    if robosuite.get("importable") is False:
        raise RuntimeError(
            "robosuite cannot be imported in RPENT_ROBOCASA_PYTHON: "
            f"{robosuite.get('import_error', 'unknown import error')}"
        )
    if mujoco.get("importable") is not True:
        raise RuntimeError(
            "MuJoCo cannot be imported in RPENT_ROBOCASA_PYTHON: "
            f"{mujoco.get('import_error', 'unknown import error')}"
        )

    robocasa_commit = str(robocasa.get("commit") or "")
    robosuite_commit = str(robosuite.get("commit") or "")
    if formal and robocasa_commit != ROBOCASA_COMMIT:
        raise RuntimeError(
            "RoboCasa commit is unavailable or mismatched: "
            f"expected {ROBOCASA_COMMIT}, got {robocasa_commit or 'unavailable'}"
        )
    if formal and robosuite_commit != ROBOSUITE_COMMIT:
        raise RuntimeError(
            "robosuite commit is unavailable or mismatched: "
            f"expected {ROBOSUITE_COMMIT}, got {robosuite_commit or 'unavailable'}"
        )

    mujoco_gl = str(payload.get("mujoco_gl") or "")
    if formal and mujoco_gl.lower() != "egl":
        raise RuntimeError(
            "formal RoboCasa evaluation requires MUJOCO_GL=egl; "
            f"got {mujoco_gl or 'unset'}"
        )
    if payload.get("macros_private") is not True:
        raise RuntimeError(
            "RoboCasa macros_private.py is missing; run "
            "'python -m robocasa.scripts.setup_macros' in the simulator env"
        )

    missing_defaults = [
        name for name in _DEFAULT_ASSETS if assets.get(name) is not True
    ]
    if missing_defaults:
        raise RuntimeError(
            "required lightwheel assets are missing: "
            f"{', '.join(missing_defaults)}; run RoboCasa's "
            "download_kitchen_assets command for tex, tex_generative, "
            "fixtures_lw, and objs_lw"
        )

    requested = frozenset(requested_registries)
    unknown = requested.difference(_REGISTRY_ASSET)
    if unknown:
        raise ValueError(f"unknown RoboCasa object registries: {sorted(unknown)}")
    available = frozenset(
        registry
        for registry, asset_key in _REGISTRY_ASSET.items()
        if assets.get(asset_key) is True
    )
    missing_registries = requested.difference(available)
    if missing_registries:
        names = ", ".join(sorted(missing_registries))
        raise RuntimeError(
            f"requested RoboCasa registry {names} has no installed assets"
        )

    return RoboCasaDependencyReport(
        interpreter=str(selected),
        robocasa_version=str(robocasa.get("version") or "unknown"),
        robocasa_commit=robocasa_commit,
        robosuite_version=str(robosuite.get("version") or "unknown"),
        robosuite_commit=robosuite_commit,
        mujoco_version=str(mujoco.get("version") or "unknown"),
        mujoco_gl=mujoco_gl,
        default_assets_ready=True,
        available_registries=available,
    )


def _distribution_metadata(name: str) -> tuple[str, str | None]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable", None
    version = distribution.version
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            vcs_info = json.loads(direct_url).get("vcs_info", {})
        except json.JSONDecodeError:
            vcs_info = {}
        commit = vcs_info.get("commit_id")
        if commit:
            return version, str(commit)
    return version, None


def _git_commit(start: Path | None) -> str | None:
    if start is None:
        return None
    for candidate in (start, *start.parents):
        if not (candidate / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None
    return None


def _import_package(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # isolated preflight must report native import failures
        return None, f"{type(exc).__name__}: {exc}"


def _has_files(path: Path) -> bool:
    return path.is_dir() and any(entry.is_file() for entry in path.rglob("*"))


def _inspect_current_environment() -> dict[str, Any]:
    robocasa, robocasa_error = _import_package("robocasa")
    robosuite, robosuite_error = _import_package("robosuite")
    mujoco, mujoco_error = _import_package("mujoco")

    robocasa_root = (
        Path(robocasa.__file__).resolve().parent
        if robocasa is not None and getattr(robocasa, "__file__", None)
        else None
    )
    robosuite_root = (
        Path(robosuite.__file__).resolve().parent
        if robosuite is not None and getattr(robosuite, "__file__", None)
        else None
    )
    robocasa_version, robocasa_direct_commit = _distribution_metadata("robocasa")
    robosuite_version, robosuite_direct_commit = _distribution_metadata("robosuite")

    asset_root = robocasa_root / "models" / "assets" if robocasa_root else None
    asset_paths = {
        "textures": "textures",
        "generative_textures": "generative_textures",
        "fixtures_lightwheel": "fixtures",
        "objects_lightwheel": "objects/lightwheel",
        "objects_objaverse": "objects/objaverse",
        "objects_aigen": "objects/aigen_objs",
    }
    assets = {
        name: bool(asset_root and _has_files(asset_root / relative))
        for name, relative in asset_paths.items()
    }

    return {
        "schema_version": _SCHEMA_VERSION,
        "interpreter": os.path.abspath(sys.executable),
        "robocasa": {
            "version": robocasa_version,
            "commit": robocasa_direct_commit or _git_commit(robocasa_root),
            "importable": robocasa is not None,
            "import_error": robocasa_error,
        },
        "robosuite": {
            "version": robosuite_version,
            "commit": robosuite_direct_commit or _git_commit(robosuite_root),
            "importable": robosuite is not None,
            "import_error": robosuite_error,
        },
        "mujoco": {
            "version": getattr(mujoco, "__version__", "unavailable"),
            "importable": mujoco is not None,
            "import_error": mujoco_error,
        },
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "macros_private": bool(
            robocasa_root and (robocasa_root / "macros_private.py").is_file()
        ),
        "assets": assets,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-json", action="store_true")
    args = parser.parse_args(argv)
    if not args.inspect_json:
        parser.error("--inspect-json is required")
    print(json.dumps(_inspect_current_environment(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
