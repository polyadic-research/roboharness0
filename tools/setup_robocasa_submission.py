"""Plan or provision isolated Linux environments for the 2.0 candidate.

Provisioning downloads software/assets, but never starts a planner, policy
inference, simulator episode, or benchmark. Native qualification is separate.
"""

from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = Path("evaluations/robocasa365/2.0/environments")
SOURCE_COMMIT = "c8911c64f24393e8665bd9ecc34f819173d9c3ba"
SOURCE_VERSION = "1.2.3.5"
ROBOCASA_COMMIT = "456174f62b89b8fca99eaaf33949c29fec9cfc2a"
ROBOSUITE_COMMIT = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
FLASH_WHEEL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
UV_VERSION = "0.13.7"
PYTHON_VERSIONS = {"coordinator": "3.11", "simulator": "3.11", "policy": "3.12"}
ZERO_WORKLOAD = {
    "provider_calls": 0, "model_loads": 0, "environment_constructors": 0,
    "policy_predictions": 0, "control_actions": 0,
}


def read_lock_pins(root: Path, component: str) -> dict[str, str]:
    """Require an exact version and artifact hashes for every lock record."""
    if component not in PYTHON_VERSIONS:
        raise ValueError("unknown setup component")
    text = (root / ENV_DIR / f"{component}.lock").read_text()
    records = text.replace("\\\n", " ").splitlines()
    pins = {}
    for record in records:
        record = record.strip()
        if not record or record.startswith("#"):
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)==([^\s;]+)\s+((?:--hash=sha256:[0-9a-f]{64}\s*)+)", record,
        )
        if match is None:
            raise ValueError(f"unhashed or non-exact dependency in {component}.lock")
        name, version = match.group(1).lower().replace("_", "-"), match.group(2)
        if name in pins:
            raise ValueError(f"duplicate dependency in {component}.lock: {name}")
        pins[name] = version
    if not pins:
        raise ValueError(f"empty {component}.lock")
    return pins


def validate_source_identity(root: Path) -> None:
    """Resolve the display alias without rewriting the frozen internal version."""
    provenance = json.loads((root / "SOURCE_PROVENANCE.json").read_text())
    if provenance.get("edition") != "2.0":
        raise ValueError("setup requires the unchanged 2.0 source tree")
    source_commit = provenance.get("source_commits", {}).get("2.0")
    source_version = provenance.get("source_versions", {}).get("2.0")
    release_path = root / "RELEASE.json"
    internal_version = json.loads(release_path.read_text())["harness_version"]
    if (source_commit, source_version, internal_version) != (
        SOURCE_COMMIT, SOURCE_VERSION, SOURCE_VERSION,
    ):
        raise ValueError("setup requires the unchanged historical 2.0 / 1.2.3.5 source")


def validate_locks(root: Path) -> dict:
    manifest = json.loads((root / ENV_DIR / "manifest.json").read_text())
    required = {f"{name}.{suffix}" for name in PYTHON_VERSIONS for suffix in ("in", "lock")}
    if not required.issubset(manifest.get("files", {})) or manifest.get("python") != PYTHON_VERSIONS:
        raise ValueError("incomplete or incompatible environment lock manifest")
    for relative, expected in manifest["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.name != relative:
            raise ValueError("unsafe environment lock path")
        content = (root / ENV_DIR / path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"environment input or lock changed: {relative}")
    for component in PYTHON_VERSIONS:
        read_lock_pins(root, component)
    return manifest


def setup_plan(source: Path, prefix: Path, uv: str, *, assets: bool, checkpoint: bool) -> list[list[str]]:
    """Use argument arrays throughout, including source/prefix paths with spaces."""
    commands = []
    for name, version in PYTHON_VERSIONS.items():
        env = prefix / name
        python = str(env / "bin/python")
        commands.append([uv, "venv", str(env), "--python", version])
        commands.append([uv, "pip", "sync", "--python", python, "--require-hashes",
                         str(source / ENV_DIR / f"{name}.lock"), "--index-url", "https://pypi.org/simple"])
        commands.append([uv, "pip", "install", "--python", python, "--no-deps",
                         "--no-build-isolation", "-e", str(source)])
    simulator_python = str(prefix / "simulator/bin/python")
    for name, url, revision in (
        ("robosuite", "https://github.com/ARISE-Initiative/robosuite.git", ROBOSUITE_COMMIT),
        ("robocasa", "https://github.com/robocasa/robocasa.git", ROBOCASA_COMMIT),
    ):
        target = prefix / "upstream" / name
        commands.extend([
            ["git", "clone", "--no-checkout", url, str(target)],
            ["git", "-C", str(target), "checkout", "--detach", revision],
            [uv, "pip", "install", "--python", simulator_python, "--no-deps",
             "--no-build-isolation", "-e", str(target)],
        ])
    commands.append([uv, "pip", "install", "--python", str(prefix / "policy/bin/python"),
                     "--no-deps", FLASH_WHEEL])
    if assets:
        commands.extend([
            [simulator_python, "-m", "robocasa.scripts.setup_macros"],
            [simulator_python, "-m", "robocasa.scripts.download_kitchen_assets",
             "--type", "tex", "tex_generative", "fixtures_lw", "objs_lw", "objs_objaverse"],
        ])
    if checkpoint:
        candidate = json.loads((source / "evaluations/robocasa365/2.0/candidate.json").read_text())
        policy = candidate["policy"]
        code = (
            "import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])"
        )
        commands.append([str(prefix / "coordinator/bin/python"), "-c", code,
                         policy["repo_id"], policy["revision"], str(prefix / "checkpoint")])
    return commands


def host_preflight(uv: str) -> dict:
    """Read installation prerequisites without allocating a GPU or environment."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise ValueError("--apply requires Linux x86_64; planning is supported on Mac")
    if shutil.which("git") is None or shutil.which(uv) is None:
        raise ValueError(f"install Git and uv=={UV_VERSION} before provisioning")
    version = subprocess.check_output([uv, "--version"], text=True).split()
    if len(version) < 2 or version[1] != UV_VERSION:
        raise ValueError(f"install uv=={UV_VERSION} for this fixed setup")
    libc_name, libc_version = platform.libc_ver()
    if libc_name != "glibc" or tuple(int(part) for part in libc_version.split(".")[:2]) < (2, 28):
        raise ValueError("the pinned Linux wheels require glibc >= 2.28")
    libraries = {name: ctypes.util.find_library(name) for name in ("EGL", "GL")}
    if not all(libraries.values()):
        raise ValueError("install system EGL/OpenGL libraries before provisioning")
    if shutil.which("nvidia-smi") is None:
        raise ValueError("a working NVIDIA driver and nvidia-smi are required")
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader",
    ], text=True, timeout=30).strip().splitlines()
    if not devices:
        raise ValueError("nvidia-smi reported no NVIDIA GPUs")
    return {"system": "Linux", "machine": platform.machine(), "glibc": libc_version,
            "libraries": libraries, "gpu_count": len(devices), "gpu_driver_inventory": devices,
            "cuda_build_required": "12.8", "gpu_allocation_performed": False}


def verification_command(root: Path, prefix: Path, *, assets: bool, checkpoint: bool) -> list[str]:
    command = [str(prefix / "coordinator/bin/python"), "-B",
               str(root / "tools/check_robocasa_submission_environment.py"),
               "--root", str(root), "--prefix", str(prefix)]
    if assets:
        command.append("--require-assets")
    if checkpoint:
        command.append("--require-checkpoint")
    return command


def provision(root: Path, prefix: Path, uv: str, *, apply: bool, assets: bool, checkpoint: bool) -> dict:
    if prefix.exists() or prefix.is_symlink():
        raise ValueError("setup prefix already exists; choose a new directory")
    root, prefix = root.resolve(), prefix.resolve()
    if prefix.is_relative_to(root):
        raise ValueError("setup prefix must be outside the source checkout")
    source = root
    validate_source_identity(root)
    validate_locks(root)
    locks = {}
    for name in ("coordinator", "simulator", "policy"):
        path = root / ENV_DIR / f"{name}.lock"
        content = path.read_bytes()
        if not content or b"--hash=sha256:" not in content:
            raise ValueError(f"missing hashed {name} lock")
        locks[name] = hashlib.sha256(content).hexdigest()
    plan = setup_plan(root, prefix, uv, assets=assets, checkpoint=checkpoint)
    result = {"target": "Linux x86_64, CUDA 12.8-compatible host", "uv_version": "0.13.7",
              "source_edition": "2.0", "source_commit": SOURCE_COMMIT,
              "original_runtime_version": SOURCE_VERSION, "lock_sha256": locks,
              "commands": plan, "applied": False, "native_evaluation_run": False,
              "status": "plan_only", "workload_counts": dict(ZERO_WORKLOAD),
              "verification_scope": "imports_and_static_metadata_only",
              "verification_command": verification_command(root, prefix, assets=assets, checkpoint=checkpoint)}
    result["plan_sha256"] = hashlib.sha256(json.dumps(
        {"source": SOURCE_COMMIT, "locks": locks, "commands": plan,
         "verification_command": result["verification_command"]}, sort_keys=True,
    ).encode()).hexdigest()
    if not apply:
        return result
    result["host_preflight"] = host_preflight(uv)
    try:
        from tools.release import verify
    except ModuleNotFoundError:
        from release import verify
    verify(root)
    prefix.mkdir(parents=True)
    (prefix / "upstream").mkdir()
    environment = {**os.environ, "MUJOCO_GL": "egl"}
    result.update(status="installing", completed_commands=0)
    receipt = prefix / "setup-receipt.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    try:
        for index, command in enumerate(plan):
            options = {"check": True, "env": environment, "cwd": source}
            if "robocasa.scripts.download_kitchen_assets" in command:
                # --download-assets explicitly requests this download. The
                # pinned upstream CLI otherwise blocks on its single y/n prompt.
                options.update(input="y\n", text=True)
            subprocess.run(command, **options)
            result["completed_commands"] = index + 1
            receipt.write_text(json.dumps(result, indent=2) + "\n")
        try:
            from tools.check_robocasa_submission_environment import verify_installation
        except ModuleNotFoundError:
            from check_robocasa_submission_environment import verify_installation
        report = verify_installation(root, prefix, require_assets=assets, require_checkpoint=checkpoint)
        (prefix / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        result["verification_passed"] = report["passed"]
        if not report["passed"]:
            raise RuntimeError("import/static verification failed; inspect verification.json")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__)
        if isinstance(error, subprocess.CalledProcessError):
            result["returncode"] = error.returncode
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    configured = {
        "RPENT_ROBOCASA_PYTHON": str(prefix / "simulator/bin/python"),
        "RPENT_XIAOMI_MIBOT_PYTHON": str(prefix / "policy/bin/python"),
        "RPENT_MAIN_PYTHON": str(prefix / "coordinator/bin/python"),
        "MUJOCO_GL": "egl",
    }
    if checkpoint:
        configured["RPENT_XIAOMI_MIBOT_SNAPSHOT"] = str(prefix / "checkpoint")
    shell = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in configured.items()) + "\n"
    shell += f'export PATH={shlex.quote(str(prefix / "coordinator/bin"))}:"$PATH"\n'
    (prefix / "environment.sh").write_text(shell)
    result.update(applied=True, status="installed_and_import_verified", native_environment_qualified=False)
    (prefix / "setup-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--check-locks", action="store_true")
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--download-assets", action="store_true")
    parser.add_argument("--download-checkpoint", action="store_true")
    args = parser.parse_args()
    try:
        if args.check_locks:
            print(json.dumps(validate_locks(args.root), indent=2))
            return 0
        if args.prefix is None:
            parser.error("setup requires --prefix")
        result = provision(args.root.resolve(), args.prefix.expanduser().resolve(), args.uv,
                           apply=args.apply, assets=args.download_assets, checkpoint=args.download_checkpoint)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
