"""Check installed dependencies and static artifacts without executing a robot.

No model is loaded, no environment is constructed and no provider is called.
This is an import/metadata check, not a native simulator or rollout qualification.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

try:
    from tools import setup_robocasa_submission as setup
except ModuleNotFoundError:
    import setup_robocasa_submission as setup

ROOT = Path(__file__).resolve().parents[1]
ZERO_WORKLOAD = dict(setup.ZERO_WORKLOAD)
IMPORTS = {
    "coordinator": ("httpx", "pydantic_ai", "fastapi", "uvicorn", "rpent.runtime.api", "rpent.harness.state"),
    "simulator": ("mujoco", "robocasa", "robosuite", "robots.robocasa.env_server"),
    "policy": ("torch", "transformers", "accelerate", "safetensors", "flash_attn", "rpent.policies.xiaomi_mibot_server"),
}


def _deny_network(*args, **kwargs):
    raise RuntimeError("network access is disabled during import-only verification")


def inspect_component(root: Path, component: str, checkpoint: Path | None = None) -> dict:
    """Run only inside a dedicated interpreter; inspect without constructors."""
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network
    socket.create_connection = _deny_network
    pins = setup.read_lock_pins(root, component)
    names = [*pins, *( ["flash-attn"] if component == "policy" else [])]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    imports = {}
    modules = {}
    for name in IMPORTS[component]:
        try:
            modules[name] = importlib.import_module(name)
            imports[name] = True
        except Exception as error:
            # Store types, not provider messages or potentially sensitive stderr.
            imports[name] = type(error).__name__
    framework = importlib.import_module("rpent")
    result = {
        "component": component, "interpreter": os.path.abspath(sys.executable),
        "python_version": ".".join(str(number) for number in sys.version_info[:3]),
        "versions": versions, "imports": imports,
        "framework_version": framework.__version__,
        "source_import_path": str(Path(framework.__file__).resolve()),
        "workload_counts": dict(ZERO_WORKLOAD),
    }
    if component == "simulator":
        try:
            from robots.robocasa.dependencies import _inspect_current_environment
            result["simulator_metadata"] = _inspect_current_environment()
        except Exception as error:
            result["simulator_metadata"] = {"error_type": type(error).__name__}
    if component == "policy":
        torch = modules.get("torch")
        result["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        result["cxx11_abi"] = getattr(getattr(torch, "_C", None), "_GLIBCXX_USE_CXX11_ABI", None)
        try:
            direct = importlib.metadata.distribution("flash-attn").read_text("direct_url.json")
            result["flash_attention_origin"] = json.loads(direct) if direct else {}
        except (importlib.metadata.PackageNotFoundError, ValueError):
            result["flash_attention_origin"] = {}
        if checkpoint is not None:
            try:
                from rpent.policies.xiaomi_mibot_artifacts import (
                    inspect_xiaomi_artifacts,
                )
                from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
                artifact = inspect_xiaomi_artifacts(checkpoint, XIAOMI_MIBOT_CHECKPOINT)
                index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
                shards = set(index["weight_map"].values())
                for name in shards:
                    path = Path(name)
                    if path.is_absolute() or ".." in path.parts or not (checkpoint / path).is_file():
                        raise ValueError("missing or unsafe checkpoint shard")
                    if (checkpoint / path).stat().st_size <= 0:
                        raise ValueError("empty checkpoint shard")
                result["checkpoint_static"] = {
                    "passed": True, "identity_digest": artifact.identity_digest,
                    "metadata_hashes": dict(artifact.file_digests), "weight_shards_present": len(shards),
                    "weights_loaded": False,
                }
            except Exception as error:
                result["checkpoint_static"] = {"passed": False, "error_type": type(error).__name__}
    return result


def validate_component(root: Path, prefix: Path, component: str, report: dict, *,
                       require_assets: bool = False, require_checkpoint: bool = False) -> list[dict]:
    """Reject imported-but-drifted versions and reports from another interpreter."""
    findings = []

    def add(name, passed, expected=None, actual=None):
        findings.append({"component": component, "check": name, "passed": bool(passed),
                         "expected": expected, "actual": actual})

    expected_interpreter = os.path.abspath(prefix / component / "bin/python")
    add("interpreter", report.get("interpreter") == expected_interpreter,
        expected_interpreter, report.get("interpreter"))
    add("component", report.get("component") == component, component, report.get("component"))
    actual_python = report.get("python_version", "")
    add("python_version", ".".join(actual_python.split(".")[:2]) == setup.PYTHON_VERSIONS[component],
        setup.PYTHON_VERSIONS[component], actual_python)
    add("zero_workload", report.get("workload_counts") == ZERO_WORKLOAD)
    add("framework_version", report.get("framework_version") == setup.SOURCE_VERSION,
        setup.SOURCE_VERSION, report.get("framework_version"))
    add("framework_source", report.get("source_import_path") == str((root / "rpent/__init__.py").resolve()))
    for name, expected in setup.read_lock_pins(root, component).items():
        actual = report.get("versions", {}).get(name)
        add("package:" + name, isinstance(actual, str) and actual.split("+", 1)[0] == expected,
            expected, actual)
    for name in IMPORTS[component]:
        actual = report.get("imports", {}).get(name)
        add("import:" + name, actual is True, True, actual)
    if component == "simulator":
        metadata = report.get("simulator_metadata", {})
        for name, revision in (("robocasa", setup.ROBOCASA_COMMIT), ("robosuite", setup.ROBOSUITE_COMMIT)):
            actual = metadata.get(name, {}).get("commit")
            add("upstream:" + name, actual == revision, revision, actual)
        add("egl", metadata.get("mujoco_gl") == "egl", "egl", metadata.get("mujoco_gl"))
        if require_assets:
            profile = json.loads((root / "evaluations/robocasa365/2.0/evaluation-profile.json").read_text())
            registries = profile["execution"]["object_registries"]
            keys = ["textures", "generative_textures", "fixtures_lightwheel"]
            mapping = {"lightwheel": "objects_lightwheel", "objaverse": "objects_objaverse", "aigen": "objects_aigen"}
            keys.extend(mapping[name] for name in registries)
            add("macros_private", metadata.get("macros_private") is True)
            for name in keys:
                add("asset:" + name, metadata.get("assets", {}).get(name) is True)
    if component == "policy":
        add("cuda_build", report.get("cuda_build") == "12.8", "12.8", report.get("cuda_build"))
        add("cxx11_abi", report.get("cxx11_abi") is True)
        add("flash_attention_version", report.get("versions", {}).get("flash-attn") == "2.8.3")
        origin = report.get("flash_attention_origin", {})
        add("flash_attention_origin", unquote(origin.get("url", "")).split("#", 1)[0] == setup.FLASH_WHEEL)
        if require_checkpoint:
            add("checkpoint_static", report.get("checkpoint_static", {}).get("passed") is True)
    return findings


def verify_installation(root: Path, prefix: Path, *, require_assets: bool = False,
                        require_checkpoint: bool = False, runner=subprocess.run) -> dict:
    """Launch three bounded import-only probes; never launch runtime servers."""
    root, prefix = root.resolve(), prefix.resolve()
    setup.validate_source_identity(root)
    setup.validate_locks(root)
    checks, reports = [], {}
    environment = {**os.environ, "MUJOCO_GL": "egl", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "PYTHONDONTWRITEBYTECODE": "1"}
    for component in setup.PYTHON_VERSIONS:
        interpreter = prefix / component / "bin/python"
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            checks.append({"component": component, "check": "interpreter_exists", "passed": False})
            continue
        command = [str(interpreter), "-B", str(root / "tools/check_robocasa_submission_environment.py"),
                   "--root", str(root), "--probe", component]
        if component == "policy" and require_checkpoint:
            command.extend(("--checkpoint", str(prefix / "checkpoint")))
        try:
            completed = runner(command, check=False, capture_output=True, text=True,
                               env=environment, cwd=root, timeout=180)
            report = next(json.loads(line) for line in reversed(completed.stdout.splitlines())
                          if line.strip().startswith("{"))
            reports[component] = report
            checks.append({"component": component, "check": "probe_exit", "passed": completed.returncode == 0})
            checks.extend(validate_component(root, prefix, component, report,
                                             require_assets=require_assets, require_checkpoint=require_checkpoint))
        except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as error:
            checks.append({"component": component, "check": "probe_output", "passed": False,
                           "error_type": type(error).__name__})
    return {
        "schema_version": 1, "passed": bool(checks) and all(row["passed"] for row in checks),
        "scope": "imports_and_static_metadata_only", "source_commit": setup.SOURCE_COMMIT,
        "checks": checks, "components": reports, "workload_counts": dict(ZERO_WORKLOAD),
        "assets_required": require_assets, "checkpoint_required": require_checkpoint,
        "native_environment_qualified": False, "historical_environment_reproduced": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--probe", choices=tuple(setup.PYTHON_VERSIONS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--require-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.probe:
        result = inspect_component(args.root.resolve(), args.probe, args.checkpoint)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.prefix is None:
        parser.error("--prefix is required for complete installation verification")
    result = verify_installation(args.root, args.prefix, require_assets=args.require_assets,
                                 require_checkpoint=args.require_checkpoint)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
