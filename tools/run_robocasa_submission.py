"""Plan an unchanged-source reviewer run, or archive existing result evidence.

The default path reads configuration and prints an argument array only. Starting
a new evaluation requires --execute and a reviewer declaration about gateway
settings. Historical selected200 replay is deliberately unsupported.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "evaluations/robocasa365/2.0/evaluation-profile.json"
PLANNER = "evaluations/robocasa365/2.0/planner.json"
SOURCE_COMMIT = "c8911c64f24393e8665bd9ecc34f819173d9c3ba"
SOURCE_DIGEST = "751d205d8b996b8ae36aa9992ac539eef5a7e993752a0381f022237e9c9085fc"
HISTORICAL_RECORDS = (
    "README.zh-CN.md", "version-identity.json", "verification.json",
    "selected200-summary-20261107-r002.json", "selected200-index-20261107-r002.json",
)
_SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
_PLANNER_CLI = {
    "planner": "api", "model": "codexgateway:gpt-6-astra", "max_tokens": 0,
    "plan_generation_timeout_s": 180, "planner_timeout_s": None,
}


def _json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    return data


def _source_identity(data: dict) -> None:
    if data.get("commit") != SOURCE_COMMIT or data.get("digest") != SOURCE_DIGEST:
        raise ValueError("configuration must identify the unchanged historical 2.0 source")


def load_profile(root: Path, path: Path) -> dict:
    """Check the future workload against the actual frozen task registry."""
    profile = _json(path)
    if profile.get("schema_version") != 1 or profile.get("kind") != "future_reviewer_evaluation":
        raise ValueError("only the future reviewer profile is executable; historical replay is unsupported")
    if profile.get("profile_id") != "robocasa365-leaderboard-v1" or profile.get("historical_replay") is not False:
        raise ValueError("unsupported evaluation profile")
    _source_identity(profile.get("scientific_source", {}))
    if profile.get("simulator") != {
        "robocasa_version": "1.0.1", "robocasa_commit": "456174f62b89b8fca99eaaf33949c29fec9cfc2a",
        "robosuite_commit": "5ce6643f3092639d08f7b0f90ed1c6a84f50552c",
    }:
        raise ValueError("simulator identities differ from the frozen evaluation dependencies")
    registry = _json(root / "robots/robocasa/task_groups.json")
    workload = profile["workload"]
    expected = {
        "task_sets": list(_SPLITS), "scene_split": "pretrain", "scenario_seed": 7,
        "scenario_sampling": "seeded_task_environment_sequential_resets",
        "rollouts_per_task": 50, "task_count": 50, "scheduled_rollouts": 2500,
        "registry_digest": registry["registry_digest"],
        "horizon_source": "robocasa_1_0_1_upstream",
        "tasks": [
            {"task_name": name, "task_set": split, "horizon": registry["tasks"][name]["upstream_horizon"]}
            for split in _SPLITS for name in registry["groups"][split]["tasks"]
        ],
    }
    if workload != expected:
        raise ValueError("workload differs from the frozen 50-task / 2500-rollout profile")
    execution = profile["execution"]
    if execution != {
        "planning_profile": "robocasa-horizon-gated-v1", "jobs": 1,
        "recording_mode": "none", "require_complete_artifacts": True,
        "object_registries": ["lightwheel", "objaverse"],
        "protocol": "unchanged_historical_2.0",
    }:
        raise ValueError("execution settings differ from the unchanged-source reviewer profile")
    policy = profile["policy"]
    if (
        policy.get("family") != "xiaomi-mibot"
        or policy.get("repo_id") != "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
        or policy.get("revision") != "0d1aa76d0d82debc9b611e4d1e231096434d5be4"
        or policy.get("weight_edits") is not False
        or policy.get("dtype") != "bfloat16"
        or re.fullmatch(r"cuda:[0-9]+", str(policy.get("device", ""))) is None
    ):
        raise ValueError("policy differs from the unchanged Xiaomi checkpoint")
    return profile


def _planner_config(path: Path) -> dict:
    try:
        from tools.configure_robocasa_planner import load_config
    except ModuleNotFoundError:
        from configure_robocasa_planner import load_config
    config = load_config(path)
    if config.get("cli") != _PLANNER_CLI:
        raise ValueError("planner CLI settings differ from the declared historical configuration")
    private = config.get("private_environment", {})
    if private.get("base_url_env") != "OPENAI_BASE_URL" or private.get("api_key_env") != "OPENAI_API_KEY":
        raise ValueError("planner credentials must be private OPENAI_* environment values")
    gateway = config.get("gateway", {})
    if gateway.get("reasoning_effort") != "xhigh" or gateway.get("http_request_timeout_s") != 330:
        raise ValueError("planner gateway settings differ from the configured historical values")
    return config


def build_plan(
    root: Path,
    runtime_prefix: Path,
    output_dir: Path,
    *,
    profile_path: Path | None = None,
    planner_path: Path | None = None,
) -> dict:
    """Read public JSON only; do not import or launch the runtime."""
    root, runtime_prefix = root.resolve(), runtime_prefix.expanduser().resolve()
    output_dir = output_dir.expanduser().absolute()
    profile_path = profile_path or root / PROFILE
    planner_path = planner_path or root / PLANNER
    profile = load_profile(root, profile_path)
    _planner_config(planner_path)
    execution, policy = profile["execution"], profile["policy"]
    argv = [
        str(runtime_prefix / "coordinator/bin/python"), "-m", "rpent.cli.main",
        "--env", "robocasa", "--evaluation-profile", profile["profile_id"],
        "--vla", policy["family"], "--planning-profile", execution["planning_profile"],
        "--planner", _PLANNER_CLI["planner"], "--model", _PLANNER_CLI["model"],
        "--max-tokens", str(_PLANNER_CLI["max_tokens"]),
        "--plan-generation-timeout-s", str(_PLANNER_CLI["plan_generation_timeout_s"]),
        "--jobs", "1", "--recording-mode", execution["recording_mode"],
        "--require-complete-artifacts", "--object-registries", ",".join(execution["object_registries"]),
        "--policy-device", policy["device"], "--policy-dtype", policy["dtype"],
        "--policy.path", policy["repo_id"], "--output-dir", str(output_dir),
    ]
    return {
        "dry_run": True, "executed": False, "source_root": str(root),
        "runtime_prefix": str(runtime_prefix), "output_dir": str(output_dir), "argv": argv,
        "source_commit": SOURCE_COMMIT, "source_digest": SOURCE_DIGEST,
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "planner_sha256": hashlib.sha256(planner_path.read_bytes()).hexdigest(),
        "profile_id": profile["profile_id"], "scheduled_tasks": 50, "scheduled_rollouts": 2500,
        "historical_replay": False, "historical_result_reference": "results/2.0-selected200/REPORT.md",
        "required_private_environment": ["OPENAI_BASE_URL", "OPENAI_API_KEY"],
        "runtime_environment": {
            "RPENT_ROBOCASA_PYTHON": str(runtime_prefix / "simulator/bin/python"),
            "RPENT_XIAOMI_MIBOT_PYTHON": str(runtime_prefix / "policy/bin/python"),
            "RPENT_XIAOMI_MIBOT_SNAPSHOT": str(runtime_prefix / "checkpoint"),
            "MUJOCO_GL": "egl",
        },
        "gateway_settings": {
            "reasoning_effort": "xhigh", "http_request_timeout_s": 330,
            "confirmation_required": True,
            "confirmation_kind": "reviewer declaration, not an automated upstream identity/settings audit",
            "client_sends_reasoning_effort": False,
            "whole_solve_timeout_override_s": None,
        },
        "scope": "Future reviewer workload using the original protocol; not a completed evaluation, historical replay or eligibility certification.",
    }


def execute_plan(plan: dict, *, gateway_settings_confirmed: bool, environment: dict | None = None) -> dict:
    """Start one future reviewer evaluation only after explicit local checks."""
    if not gateway_settings_confirmed:
        raise ValueError("--api-settings-confirmed is required to acknowledge the supplied API settings")
    output = Path(plan["output_dir"])
    if output.exists() or output.is_symlink():
        raise ValueError("evaluation output directory already exists; no overwrite, retry or resume is performed")
    if not output.parent.is_dir() or output.resolve().is_relative_to(Path(plan["source_root"]).resolve()):
        raise ValueError("choose a new evaluation output outside the source with an existing parent")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise ValueError("reviewer execution requires Linux x86_64; planning is supported on Mac")
    private = dict(os.environ if environment is None else environment)
    missing = [name for name in plan["required_private_environment"] if not private.get(name)]
    if missing:
        raise ValueError("missing private environment: " + ", ".join(missing))
    url = urlsplit(private["OPENAI_BASE_URL"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("OPENAI_BASE_URL must be an HTTP(S) endpoint without embedded credentials or query data")
    for relative in ("coordinator/bin/python", "simulator/bin/python", "policy/bin/python"):
        interpreter = Path(plan["runtime_prefix"]) / relative
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise ValueError("runtime interpreter is missing or not executable: " + relative)
    if not (Path(plan["runtime_prefix"]) / "checkpoint/config.json").is_file():
        raise ValueError("the setup prefix does not contain the downloaded Xiaomi checkpoint")
    try:
        from tools.release import verify
    except ModuleNotFoundError:
        from release import verify
    verify(Path(plan["source_root"]))
    process = subprocess.run(plan["argv"], cwd=plan["source_root"], env={**private, **plan["runtime_environment"]}, check=False)
    return {
        **plan, "dry_run": False, "executed": True,
        "gateway_settings_confirmed_by_reviewer": True,
        "gateway_settings_independently_audited": False,
        "process_returncode": process.returncode,
        "evaluation_qualification": "Inspect native terminal/qualification evidence; process exit is not a new published result.",
    }


def _historical_verifier(root: Path) -> dict:
    try:
        from tools.verify_selected200 import verify
    except ModuleNotFoundError:
        from verify_selected200 import verify
    return verify(root)


def verify_historical_configuration(root: Path) -> dict:
    """Cross-check maintained selection/parameter manifests against fixed records."""
    report = _historical_verifier(root)
    folder = root / "evaluations/robocasa365/2.0"
    selection = _json(folder / "historical-selection.json")
    parameters = _json(folder / "historical-parameters.json")
    index = _json(root / "results/2.0-selected200/selected200-index-20261107-r002.json")
    summary = _json(root / "results/2.0-selected200/selected200-summary-20261107-r002.json")
    for manifest in (selection, parameters):
        _source_identity(manifest.get("scientific_source", {}))
    if (
        selection.get("kind") != "historical_selected200_reference"
        or selection.get("replay_supported") is not False
        or selection.get("official_2500_result") is not False
        or selection.get("all200_workflow_archive_complete") is not False
        or (selection.get("unique_scenes"), selection.get("successes"), selection.get("failures")) != (200, 141, 59)
        or selection.get("selected_batches") != summary["batches"]
        or selection.get("excluded_batch_ids") != index["excluded_batch_ids"]
    ):
        raise ValueError("historical selection manifest differs from the fixed record scope")
    expected_counts = {
        key: {field: row[field] for field in ("cases", "successes", "failures")}
        for key, row in report["split_counts"].items()
    }
    if selection.get("split_counts") != expected_counts:
        raise ValueError("historical split counts differ from the original accepted outcomes")
    for key, filename in (
        ("result_index", "selected200-index-20261107-r002.json"),
        ("result_summary", "selected200-summary-20261107-r002.json"),
    ):
        if selection.get(key) != {"path": "results/2.0-selected200/" + filename, "sha256": report["record_sha256"][filename]}:
            raise ValueError("historical selection references a replaced record")
    by_scene = {row["scene_id"]: row for row in index["selected_results"]}
    expected_overrides = [
        {"scene_id": scene, "nested_planning_timeout_s": 600,
         "selected_attempt_id": by_scene[scene]["selected_attempt_id"]}
        for scene in summary["parameter_override_disclosure"]["user_requested_600second_nested_planning_cases"]
    ]
    if (
        parameters.get("kind") != "historical_parameter_disclosure"
        or parameters.get("selected_overrides") != expected_overrides
        or parameters.get("override_count") != 10
        or parameters.get("default_nested_planning_timeout_s") != 180
        or parameters.get("configured_model") != "gpt-6-astra"
        or parameters.get("configured_reasoning_effort") != "xhigh"
        or parameters.get("configured_max_tokens") != 0
        or parameters.get("configured_http_timeout_s") != 330
        or parameters.get("environment_seed") != 7
        or parameters.get("selected_native_episode_indices") != [1, 2, 3, 4]
        or parameters.get("selection_rule") != summary["supplemental_selection_rule"]
        or parameters.get("mixed_parameter_summary") is not True
        or parameters.get("uniform_180_second_protocol_claimed") is not False
        or parameters.get("policy_rng_reconstruction_permitted") is not False
        or parameters.get("protocol_changed_for_preparation") is not False
    ):
        raise ValueError("historical parameter manifest differs from the original selection/overrides")
    return report


def archive_historical(root: Path, output: Path) -> dict:
    """Archive only public small records with deterministic headers; never overwrite."""
    report = verify_historical_configuration(root)
    relative_files = [
        "SOURCE_PROVENANCE.json", "tools/verify_selected200.py", "tools/verify_results.py",
        "evaluations/robocasa365/2.0/historical-selection.json",
        "evaluations/robocasa365/2.0/historical-parameters.json",
        "evaluations/robocasa365/2.0/planner.json",
        "evaluations/robocasa365/2.0/planner-evidence.json",
        "evaluations/robocasa365/2.0/PLANNER.md",
        "results/2.0-selected200/REPORT.md",
        *("results/2.0-selected200/" + name for name in HISTORICAL_RECORDS),
    ]
    for name in relative_files:
        if not (root / name).is_file() or (root / name).is_symlink():
            raise ValueError("historical evidence member is missing or a symlink: " + name)
    if sum((root / name).stat().st_size for name in relative_files) > 1_000_000:
        raise ValueError("historical evidence packet exceeds its small-report boundary")

    def header(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        info.pax_headers = {}
        return info

    with output.open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in sorted(relative_files):
                    archive.add(root / name, arcname=name, recursive=False, filter=header)
    return {
        "archive": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size, "cases": report["cases"], "raw_archives_included": False,
        "file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in relative_files},
        "historical_replay": False, "new_model_calls": 0, "new_policy_predictions": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", type=Path, default=ROOT / PROFILE)
    parser.add_argument("--planner-config", type=Path, default=ROOT / PLANNER)
    parser.add_argument("--env-file", type=Path, help="Explicit private OPENAI_* file, read only with --execute; values are never printed.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--api-settings-confirmed", "--gateway-settings-confirmed", dest="gateway_settings_confirmed", action="store_true", help="Acknowledge the privately supplied API model/settings before execution")
    parser.add_argument("--archive-historical-results", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.archive_historical_results is not None:
            if args.execute or args.runtime_prefix is not None or args.output_dir is not None or args.env_file is not None:
                raise ValueError("historical evidence archival is separate from reviewer execution")
            result = archive_historical(ROOT, args.archive_historical_results)
        else:
            if args.runtime_prefix is None or args.output_dir is None:
                parser.error("reviewer planning requires --runtime-prefix and --output-dir")
            result = build_plan(ROOT, args.runtime_prefix, args.output_dir, profile_path=args.profile, planner_path=args.planner_config)
            result["private_environment_file_supplied"] = args.env_file is not None
            if args.execute:
                try:
                    from tools.configure_robocasa_planner import (
                        load_private_environment,
                    )
                except ModuleNotFoundError:
                    from configure_robocasa_planner import load_private_environment
                private = load_private_environment(_planner_config(args.planner_config), env_file=args.env_file)
                result = execute_plan(result, gateway_settings_confirmed=args.gateway_settings_confirmed,
                                      environment={**os.environ, **private})
        print(json.dumps(result, indent=2))
        return int(result.get("process_returncode", 0))
    except (OSError, ValueError, KeyError) as exc:
        print("Submission entry failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
