"""Prepare the selected 2.0 source and inspect current RoboCasa Agentic gaps.

This tool never launches an evaluation or submits a PR. Schema validation and
trace scans do not certify scientific eligibility or organizer verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

CANDIDATE = "evaluations/robocasa365/2.0/candidate.json"
SOURCE_COMMIT = "c8911c64f24393e8665bd9ecc34f819173d9c3ba"
POLICY_REPO = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
POLICY_REVISION = "0d1aa76d0d82debc9b611e4d1e231096434d5be4"
SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
FIELDS = {
    "model_name", "submitter", "model_type", "date", "robocasa_version",
    "base_policy", "code_url", "commit_hash", "checkpoint_url", "paper_link",
    "open_source", "notes", *(f"{split}_success" for split in SPLITS),
}
ROOT = Path(__file__).resolve().parents[1]
SELECTED200_DISCLOSURE = (
    "Unchanged Harness 2.0 is original 1.2.3.5 at "
    "c8911c64f24393e8665bd9ecc34f819173d9c3ba; Xiaomi base weights unchanged, Apache-2.0. "
    "Historical author-run selected200 r002: 50 tasks x4 episodes, 141/200 successes; "
    "Atomic-Seen 62/72, Composite-Seen 45/64, Composite-Unseen 34/64. "
    "Selected batches fresh50/followup01/followup02/followup04; followup03 excluded. "
    "Ten selected attempts used a 600-second nested-planning override, so this is a "
    "mixed-parameter supplemental panel, not an official 2500-rollout score. "
    "Original success feedback, success-dependent verification and RGB-D back_project "
    "are retained and disclosed for organizer review. Configured planner: OpenAI Codex SDK via codexgateway, "
    "gpt-6-astra, reasoning_effort=xhigh, max_tokens=0; upstream provider/model identity "
    "not independently verified. Fixed offline capability priors retained; dynamic "
    "cross-episode memory not independently audited. All200 workflow/archive closure "
    "is not claimed. No organizer eligibility approval or waiver is claimed. "
    "The source distribution is a curated RoboCasa/Xiaomi subset; retained runtime "
    "and model-bound inputs keep their original bytes, with exact exclusions in "
    "publication/source-manifest.json."
)


def load_candidate(path: Path) -> dict:
    candidate = json.loads(path.read_text())
    validate_candidate(candidate)
    return candidate


def validate_candidate(candidate: dict) -> None:
    if candidate.get("edition") != "2.0" or candidate.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("candidate must select the frozen 2.0 source")
    policy = candidate.get("policy", {})
    if policy.get("repo_id") != POLICY_REPO or policy.get("revision") != POLICY_REVISION:
        raise ValueError("candidate checkpoint differs from the frozen 2.0 checkpoint")
    if policy.get("weight_edits") is not False:
        raise ValueError("Agentic base policy must have no weight edits")
    validate_agentic(candidate["submission"], draft=True)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value != "N/A" and "<" not in value


def _url(value: object) -> bool:
    if not _text(value):
        return False
    parsed = urlsplit(value)
    return parsed.scheme == "https" and bool(parsed.hostname) and not (
        parsed.username or parsed.password or parsed.query or parsed.fragment
    )


def validate_agentic(payload: dict, *, draft: bool = False) -> None:
    """Validate the current official Agentic fields, separate from the old exporter."""
    if not isinstance(payload, dict) or set(payload) != FIELDS:
        raise ValueError("expected current Agentic fields; legacy fields are not accepted")
    for field in FIELDS - {f"{split}_success" for split in SPLITS}:
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"invalid Agentic string field: {field}")
        if not draft and not _text(payload[field]):
            raise ValueError(f"unresolved Agentic field: {field}")
    if not draft and payload["notes"].lstrip().startswith("PREPARATION DRAFT"):
        raise ValueError("Remove the preparation draft label from the final submission notes")
    if payload["model_type"] != "VLA" or payload["open_source"] != "yes":
        raise ValueError("this candidate is an open-source Agentic entry with VLA base type")
    if payload["base_policy"] != "Xiaomi-Robotics-1-RoboCasa365":
        raise ValueError("base_policy must identify the selected unchanged Xiaomi policy")
    if payload["robocasa_version"] != "1.0.1":
        raise ValueError("candidate simulator version differs from its pinned contract")
    expected_url = f"https://huggingface.co/{POLICY_REPO}/tree/{POLICY_REVISION}"
    if payload["checkpoint_url"] != expected_url:
        raise ValueError("checkpoint_url must contain the selected immutable HF revision")
    for split in SPLITS:
        value = payload[f"{split}_success"]
        if value == "N/A" and draft:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"invalid {split} percentage")
    if draft:
        return
    if re.fullmatch(r"[0-9a-f]{40}", payload["commit_hash"]) is None:
        raise ValueError("commit_hash must be a full immutable public commit")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", payload["date"]) is None:
        raise ValueError("date must be YYYY-MM-DD")
    date.fromisoformat(payload["date"])
    for field in ("code_url", "paper_link"):
        if not _url(payload[field]):
            raise ValueError(f"{field} must be a public HTTPS URL")
    paper = urlsplit(payload["paper_link"])
    if paper.hostname != "arxiv.org" or not paper.path.startswith("/abs/"):
        raise ValueError("paper_link must identify an arXiv abstract")


def build_agentic(metadata: dict, metrics: dict) -> dict:
    """Convert complete metrics to the current shape; this is not eligibility approval."""
    if metrics.get("is_complete") is not True or any(
        type(metrics.get(key)) is not int or metrics[key] != expected
        for key, expected in (
            ("scheduled_tasks", 50), ("completed_tasks", 50),
            ("scheduled_rollouts", 2500), ("completed_rollouts", 2500),
        )
    ):
        raise ValueError("complete 50-task / 2500-rollout metrics are required")
    rates = metrics.get("split_success_rates", {})
    if set(rates) != set(SPLITS):
        raise ValueError("all three official split rates are required")
    payload = dict(metadata)
    for split, value in rates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("metrics must contain finite rates between zero and one")
        payload[f"{split}_success"] = value * 100
    validate_agentic(payload)
    return payload


def _runtime_file(root: Path, relative: str) -> Path:
    provenance = json.loads((root / "SOURCE_PROVENANCE.json").read_text())
    if provenance.get("edition") != "2.0" or provenance.get("source_commits", {}).get("2.0") != SOURCE_COMMIT:
        raise ValueError("run this command against the unchanged 2.0 source")
    return root / relative


def audit_runtime(root: Path) -> dict:
    """Locate known protocol conflicts in the selected source, with reproducible receipts."""
    findings = []
    for code, relative, needle in (
        ("success_feedback", "rpent/context/projectors.py", '"success": observation.environment_success'),
        ("success_verification", "rpent/harness/hooks.py", "if feedback.environment_success:"),
        ("depth_grounding", "robots/robocasa/toolkit.py", '"back_project",'),
    ):
        path = _runtime_file(root, relative)
        content = path.read_bytes()
        lines = content.decode().splitlines()
        matching = next((index for index, line in enumerate(lines, 1) if needle in line), None)
        if matching is not None:
            findings.append({
                "code": code, "file": relative, "line": matching,
                "sha256": hashlib.sha256(content).hexdigest(),
                "status": "unchanged_protocol_disclosed_for_organizer_review",
            })
    return {"edition": "2.0", "findings": findings,
            "scope": "Known source-path checks; absence is not a complete information-flow audit."}


def _review_record(root: Path, reference: object) -> bool:
    """Require a real, hashed review artifact, rather than a filled checkbox."""
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        return False
    try:
        content = (root / reference["path"]).read_bytes()
        return hashlib.sha256(content).hexdigest() == reference["sha256"]
    except (OSError, TypeError):
        return False


def check_candidate(root: Path, candidate: dict) -> dict:
    validate_candidate(candidate)
    audit = audit_runtime(root)
    submission = candidate["submission"]
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"id": name, "passed": passed, "detail": detail})

    add("source_identity", True, f"2.0 at {SOURCE_COMMIT}; unchanged Xiaomi checkpoint")
    add("project_license", _text(candidate.get("project_license")) and (root / "LICENSE").is_file(), "Project license metadata and root LICENSE are required; inherited/base licenses remain separate.")
    add("observation_protocol", not audit["findings"] and _review_record(root, candidate.get("protocol_review")),
        "Original success feedback and RGB-D remain unchanged by owner instruction. Disclose the differences for organizer eligibility review; no waiver is claimed.")
    add("paper", _url(submission.get("paper_link")), "Public arXiv/preprint URL is required.")
    add("public_identity", _url(submission.get("code_url")) and re.fullmatch(r"[0-9a-f]{40}", submission["commit_hash"]) is not None,
        "Use the new public repository and its full commit, not internal ancestry.")
    planner = candidate.get("planner", {})
    add("planner", _text(planner.get("provider")) and _text(planner.get("model_id")) and isinstance(planner.get("settings"), dict),
        "Declare the actual planner provider/model/settings; do not guess from example commands.")
    memory = candidate.get("memory", {})
    add("memory", isinstance(memory.get("cross_episode"), bool) and _text(memory.get("calibration_provenance")),
        "Disclose cross-episode state and the origin of fixed policy capability priors.")
    scores_declared = any(submission[f"{split}_success"] != "N/A" for split in SPLITS)
    add("full_evaluation", not scores_declared or _review_record(root, candidate.get("full_evaluation")),
        "The included selected200 is a historical author-run panel, not a completed official 2500-rollout evaluation.")
    return {"edition": "2.0", "ready_for_submission": all(item["passed"] for item in checks),
            "checks": checks, "runtime_audit": audit,
            "scope": "Preparation checks only. Hashed review references require human source/protocol/evidence review; organizer eligibility is not certified."}


def build_selected200(root: Path, metadata: dict, *, draft: bool = False) -> dict:
    """Package the exact historical panel without presenting it as official metrics."""
    try:
        from tools.verify_selected200 import verify
    except ModuleNotFoundError:
        from verify_selected200 import verify
    result = verify(root)
    payload = dict(metadata)
    for split in SPLITS:
        payload[f"{split}_success"] = result["split_success_rates"][split] * 100
    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("selected200 notes must be text")
    if SELECTED200_DISCLOSURE not in notes:
        payload["notes"] = f"{notes.strip()}\n\n{SELECTED200_DISCLOSURE}".strip()
    validate_agentic(payload, draft=draft)
    return payload


def recompute_metrics(evaluation: Path, *, expected_commit: str) -> dict:
    """Read the original batch ledgers; never trust a hand-edited score summary."""
    from robots.robocasa.competition_worker import _validate_official_manifest
    from robots.robocasa.tasks import packaged_registry_snapshot
    from rpent.evaluation.coordinator import EvaluationCoordinator
    from rpent.evaluation.manifest import EvaluationManifest
    from rpent.evaluation.official_metrics import aggregate_official_metrics
    from rpent.evaluation.results import ResultStore

    evaluation = evaluation.resolve()
    manifest = EvaluationManifest.load(evaluation / "evaluation_manifest.json")
    if manifest.rpent_dirty or manifest.rpent_commit != expected_commit:
        raise ValueError("evaluation must use the declared clean public source commit")
    checkpoint = manifest.policy.get("checkpoint")
    if not hasattr(checkpoint, "get") or (
        checkpoint.get("repo_id"), checkpoint.get("revision")
    ) != (POLICY_REPO, POLICY_REVISION):
        raise ValueError("evaluation checkpoint differs from the selected 2.0 policy")
    if any((case.checkpoint_repo_id, case.checkpoint_revision) != (POLICY_REPO, POLICY_REVISION)
           for case in manifest.cases):
        raise ValueError("evaluation Cases contain another checkpoint")
    _validate_official_manifest(manifest)
    snapshot = packaged_registry_snapshot()
    if any(case.horizon != snapshot["tasks"][case.task_name]["upstream_horizon"]
           for case in manifest.cases):
        raise ValueError("Cases must use the packaged upstream horizon")
    if any(case.environment != "robocasa" or case.binding_name != "robocasa-xiaomi-mibot-v1"
           or case.preset_name != "robocasa365-pandaomron-xiaomi-mibot-v1" for case in manifest.cases):
        raise ValueError("Cases must use the selected 2.0 Xiaomi binding and preset")
    # Raw accepted results also bind canonical absolute artifact paths. Do not
    # rewrite those rows or silently reinterpret a moved evidence archive.
    if Path(str(manifest.runtime.get("evaluation_root", ""))).resolve() != evaluation:
        raise ValueError("read or mount evidence at the manifest's original evaluation_root")
    coordinator = EvaluationCoordinator(
        manifest=manifest, manifest_path=evaluation / "evaluation_manifest.json",
        results_path=evaluation / "results.jsonl", summary_path=evaluation / "summary.json",
        policy_daemon=None,
    )
    retained = ResultStore(evaluation / "results.jsonl", manifest).load()
    batches = tuple(evidence for task in dict.fromkeys(case.task_name for case in manifest.cases)
                    for evidence in coordinator._load_task_evidences(task))
    metrics = aggregate_official_metrics(manifest, retained, batches).to_payload()
    if not metrics["is_complete"]:
        raise ValueError("complete 50-task batch ledgers and accepted results are required")
    return metrics


def scan_model_request(value: object, path: str = "request") -> list[str]:
    """Conservatively find prohibited values in normalized model-bound requests."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = f"{path}.{key}"
            if key in {"environment_success", "base_task_success", "simulator_success", "reward", "depth_m", "world_map", "segmentation", "ground_truth"} or (key == "success" and path.endswith(".environment")):
                found.append(location)
            if key in {"name", "tool", "tool_name"} and item == "back_project":
                found.append(location)
            found.extend(scan_model_request(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(scan_model_request(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            found.extend(scan_model_request(decoded, path))
        elif re.search(r"environment[_. ]success\s*[:=]|environment\.success|back_project|\bdepth_m\b", value):
            found.append(path)
    return found


def audit_trace(path: Path) -> dict:
    requests = 0
    findings = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "model_request" or "request" not in row:
                raise ValueError("trace must contain normalized model_request records only")
            requests += 1
            for location in scan_model_request(row["request"]):
                findings.append({"line": line_number, "path": location})
    if not requests:
        raise ValueError("trace contains no model_request records")
    return {"requests_scanned": requests, "findings": findings,
            "prohibited_fields_found": bool(findings),
            "scope": "Leak scan only; does not prove request coverage, runtime isolation or benchmark eligibility."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export", "selected200-json", "convert-results", "validate-json", "audit-trace"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--draft", action="store_true", help="Allow unresolved publication metadata; never claim ready to submit.")
    args = parser.parse_args()
    try:
        if args.command == "check":
            result = check_candidate(args.root, load_candidate(args.root / CANDIDATE))
            code = 0 if result["ready_for_submission"] else 2
        elif args.command == "export":
            try:
                from tools.release import export
            except ModuleNotFoundError:
                from release import export
            if args.output is None:
                parser.error("export requires --output")
            result = export(args.root, args.output, "2.0")
            result["ready_for_submission"] = False
            result["scope"] = "Original 2.0 source plus preparation materials; protocol gaps remain."
            code = 0
        elif args.command == "selected200-json":
            if args.output is None:
                parser.error("selected200-json requires --output")
            metadata = (json.loads(args.metadata.read_text()) if args.metadata else
                        load_candidate(args.root / CANDIDATE)["submission"])
            payload = build_selected200(args.root, metadata, draft=args.draft)
            with args.output.open("x") as stream:
                stream.write(json.dumps(payload, indent=2) + "\n")
            result, code = {"output": str(args.output), "historical_selected200": True,
                            "official_evaluation_complete": False, "draft": args.draft,
                            "eligibility_verified": False}, 0
        elif args.command == "convert-results":
            if args.input is None or args.metadata is None or args.output is None:
                parser.error("convert-results requires --input, --metadata and --output")
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("refusing to overwrite an existing submission file")
            sys.path.insert(0, str(args.root))
            metadata = json.loads(args.metadata.read_text())
            metrics = recompute_metrics(args.input, expected_commit=metadata["commit_hash"])
            payload = build_agentic(metadata, metrics)
            with args.output.open("x") as stream:
                stream.write(json.dumps(payload, indent=2) + "\n")
            result, code = {"schema_valid": True, "metrics_recomputed": True,
                            "eligibility_verified": False, "output": str(args.output)}, 0
        elif args.command == "validate-json":
            if args.input is None:
                parser.error("validate-json requires --input")
            validate_agentic(json.loads(args.input.read_text()))
            result, code = {"schema_valid": True, "eligibility_verified": False}, 0
        else:
            if args.input is None:
                parser.error("audit-trace requires --input")
            result = audit_trace(args.input)
            code = 2 if result["prohibited_fields_found"] else 0
        print(json.dumps(result, indent=2))
        return code
    except (OSError, ValueError, KeyError) as exc:
        print(f"Submission preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
