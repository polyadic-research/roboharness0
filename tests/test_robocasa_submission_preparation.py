"""Submission preparation cannot turn a source export into a benchmark claim."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools import prepare_robocasa_submission as prep
from tools import setup_robocasa_submission as setup

ROOT = Path(__file__).resolve().parents[1]


def _valid_submission():
    candidate = prep.load_candidate(ROOT / prep.CANDIDATE)
    return {
        **candidate["submission"],
        "submitter": "Example research team",
        "date": "2026-11-12",
        "code_url": "https://github.com/example/harness",
        "commit_hash": "a" * 40,
        "paper_link": "https://arxiv.org/abs/2601.01234",
        "notes": prep.SELECTED200_DISCLOSURE,
        "atomic_seen_success": 50.0,
        "composite_seen_success": 25.0,
        "composite_unseen_success": 0.0,
    }


def test_new_agentic_format_uses_submitter_type_and_base_policy():
    payload = _valid_submission()
    prep.validate_agentic(payload)
    assert {"submitter", "model_type", "base_policy"} <= payload.keys()
    assert not {"policy_family", "submission_source", "wandb"} & payload.keys()
    assert "training_config" not in payload


@pytest.mark.parametrize("update", [
    {"policy_family": "Xiaomi"}, {"submission_source": "external"},
    {"submitter": "N/A"}, {"submitter": "<TEAM_NAME>"},
    {"paper_link": "<ARXIV_URL>"}, {"commit_hash": "main"},
    {"atomic_seen_success": True}, {"atomic_seen_success": float("nan")},
    {"atomic_seen_success": 101},
    {"checkpoint_url": "https://huggingface.co/example/model/tree/main"},
    {"model_type": "Agent + VLA"}, {"date": "2026-02-30"},
    {"notes": "PREPARATION DRAFT, NOT SUBMITTED. " + prep.SELECTED200_DISCLOSURE},
])
def test_submission_rejects_stale_fields_or_unverifiable_claims(update):
    with pytest.raises(ValueError):
        prep.validate_agentic({**_valid_submission(), **update})


def test_c16_is_not_accepted_as_three_split_metrics():
    with pytest.raises(ValueError, match="complete 50-task"):
        prep.build_agentic(_valid_submission(), {
            "is_complete": True, "scheduled_tasks": 16, "completed_tasks": 16,
            "scheduled_rollouts": 16, "completed_rollouts": 16,
            "split_success_rates": {"composite_seen": 0.375},
        })


def test_incomplete_full_schedule_cannot_be_exported():
    metrics = _full_metrics()
    metrics["completed_rollouts"] -= 1
    with pytest.raises(ValueError, match="complete 50-task"):
        prep.build_agentic(_valid_submission(), metrics)


def _full_metrics():
    return {
        "is_complete": True, "scheduled_tasks": 50, "completed_tasks": 50,
        "scheduled_rollouts": 2500, "completed_rollouts": 2500,
        "split_success_rates": {
            "atomic_seen": 0.5, "composite_seen": 0.25, "composite_unseen": 0.0,
        },
    }


def test_export_preserves_zero_score_and_uses_percentages():
    payload = prep.build_agentic(_valid_submission(), _full_metrics())
    assert [payload[f"{split}_success"] for split in prep.SPLITS] == [50, 25, 0]


def test_runtime_audit_checks_selected_edition_and_keeps_evidence():
    report = prep.audit_runtime(ROOT)
    assert report["edition"] == "2.0"
    codes = {finding["code"] for finding in report["findings"]}
    assert {"success_feedback", "success_verification", "depth_grounding"} <= codes
    for finding in report["findings"]:
        assert finding["line"] > 0
        assert len(finding["sha256"]) == 64


def test_confirmed_license_does_not_clear_remaining_submission_gaps():
    candidate = prep.load_candidate(ROOT / prep.CANDIDATE)
    result = prep.check_candidate(ROOT, candidate)
    assert result["ready_for_submission"] is False
    failures = {check["id"] for check in result["checks"] if not check["passed"]}
    assert {"observation_protocol", "paper"} <= failures
    assert "project_license" not in failures
    assert not {"native_reproduction", "reviewer_quota"}.intersection(
        check["id"] for check in result["checks"]
    )


def test_candidate_cannot_silently_change_source_or_checkpoint():
    candidate = prep.load_candidate(ROOT / prep.CANDIDATE)
    altered = copy.deepcopy(candidate)
    altered["policy"]["revision"] = "a" * 40
    with pytest.raises(ValueError, match="checkpoint"):
        prep.validate_candidate(altered)


def test_only_drafts_can_keep_unmeasured_scores_as_na():
    payload = _valid_submission()
    payload["composite_unseen_success"] = "N/A"
    prep.validate_agentic(payload, draft=True)
    with pytest.raises(ValueError, match="percentage"):
        prep.validate_agentic(payload)


def test_final_submission_requires_an_arxiv_abstract():
    payload = {**_valid_submission(), "paper_link": "https://example.org/writeup"}
    with pytest.raises(ValueError, match="arXiv"):
        prep.validate_agentic(payload)


def test_unmeasured_scores_do_not_require_an_invented_author_run():
    candidate = prep.load_candidate(ROOT / prep.CANDIDATE)
    for split in prep.SPLITS:
        candidate["submission"][f"{split}_success"] = "N/A"
    result = prep.check_candidate(ROOT, candidate)
    check = next(item for item in result["checks"] if item["id"] == "full_evaluation")
    assert check["passed"] is True


def test_historical_selected200_draft_uses_its_own_denominators():
    candidate = prep.load_candidate(ROOT / prep.CANDIDATE)
    payload = prep.build_selected200(ROOT, candidate["submission"], draft=True)
    assert payload["atomic_seen_success"] == pytest.approx(62 / 72 * 100)
    assert payload["composite_seen_success"] == pytest.approx(45 / 64 * 100)
    assert payload["composite_unseen_success"] == pytest.approx(34 / 64 * 100)
    assert "not an official 2500-rollout score" in payload["notes"]


def test_historical_selected200_cannot_silently_drop_protocol_disclosure():
    metadata = {**_valid_submission(), "notes": "selected200 200 600 RGB-D success"}
    payload = prep.build_selected200(ROOT, metadata, draft=True)
    assert payload["notes"].startswith(metadata["notes"])
    assert prep.SELECTED200_DISCLOSURE in payload["notes"]
    assert "followup03 excluded" in payload["notes"]
    assert "not an official 2500-rollout score" in payload["notes"]
    assert "All200 workflow/archive closure is not claimed" in payload["notes"]


def test_claimed_numbers_require_a_review_reference():
    candidate = prep.load_candidate(ROOT / prep.CANDIDATE)
    candidate["submission"]["composite_seen_success"] = 37.5
    result = prep.check_candidate(ROOT, candidate)
    check = next(item for item in result["checks"] if item["id"] == "full_evaluation")
    assert check["passed"] is False


def test_trace_scan_rejects_nested_ground_truth_and_depth():
    findings = prep.scan_model_request({
        "messages": [{"tool_result": {
            "environment": {"success": False}, "depth_m": [[0.2]],
        }}],
    })
    assert len(findings) == 2


def test_trace_scan_does_not_treat_visual_verdict_as_native_success():
    assert prep.scan_model_request({
        "messages": [{"verdict": "pass", "evidence": "drawer looks closed"}],
    }) == []


def test_empty_or_unlabelled_trace_cannot_pass(tmp_path):
    trace = tmp_path / "requests.jsonl"
    trace.write_text("")
    with pytest.raises(ValueError, match="model_request"):
        prep.audit_trace(trace)
    trace.write_text(json.dumps({"success": False}) + "\n")
    with pytest.raises(ValueError, match="model_request"):
        prep.audit_trace(trace)


def test_strings_in_tool_response_cannot_hide_prohibited_feedback():
    findings = prep.scan_model_request({
        "messages": [{"content": '{"environment":{"success":false}}'}],
    })
    assert findings


def test_setup_keeps_three_interpreters_and_pins_both_simulator_sources(tmp_path):
    prefix = tmp_path / "isolated setup"
    result = setup.provision(ROOT, prefix, "uv", apply=False, assets=True, checkpoint=True)
    assert result["source_edition"] == "2.0"
    assert result["native_evaluation_run"] is False
    assert not prefix.exists()
    commands = result["commands"]
    sync = [command for command in commands if "sync" in command]
    assert len(sync) == 3
    assert all("--require-hashes" in command for command in sync)
    checkouts = [command[-1] for command in commands if "checkout" in command]
    assert checkouts == [setup.ROBOSUITE_COMMIT, setup.ROBOCASA_COMMIT]
    assert not any("--evaluation-profile" in command for command in commands)
    download = commands[-1]
    assert prep.POLICY_REPO in download and prep.POLICY_REVISION in download


def test_setup_preserves_existing_prefix(tmp_path):
    prefix = tmp_path / "existing"
    prefix.mkdir()
    (prefix / "keep").write_text("user data")
    with pytest.raises(ValueError, match="already exists"):
        setup.provision(ROOT, prefix, "uv", apply=True, assets=False, checkpoint=False)
    assert (prefix / "keep").read_text() == "user data"


def test_fabricated_metrics_file_without_manifest_or_ledger_is_rejected(tmp_path):
    (tmp_path / "official_metrics.json").write_text(json.dumps(_full_metrics()))
    with pytest.raises((ValueError, FileNotFoundError), match="manifest"):
        prep.recompute_metrics(tmp_path, expected_commit="a" * 40)


def test_free_text_cannot_hide_negative_success_feedback():
    assert prep.scan_model_request({"content": "environment_success=false"})


def test_prior_paths_exist_in_the_selected_edition():
    candidate = prep.load_candidate(ROOT / prep.CANDIDATE)
    for path in candidate["memory"]["fixed_priors"]:
        assert prep._runtime_file(ROOT, path).is_file()


def _synthetic_complete_ledgers(root):
    """Create labelled synthetic CPU fixtures, never benchmark evidence."""
    from rpent.evaluation.batches import TaskBatchAttempt, TaskBatchProcessOutcome
    from tests.test_robocasa_competition_worker import _manifest, _result

    original = _manifest(root)
    cases = tuple(replace(case, checkpoint_repo_id=prep.POLICY_REPO,
                          checkpoint_revision=prep.POLICY_REVISION,
                          binding_name="robocasa-xiaomi-mibot-v1",
                          preset_name="robocasa365-pandaomron-xiaomi-mibot-v1")
                  for case in original.cases)
    manifest = replace(original, cases=cases, policy={"checkpoint": {
        "repo_id": prep.POLICY_REPO, "revision": prep.POLICY_REVISION,
    }})
    manifest.write_new(root / "evaluation_manifest.json")
    all_results, outcomes = [], []
    for task in dict.fromkeys(case.task_name for case in cases):
        task_cases = tuple(case for case in cases if case.task_name == task)
        batch_id = f"synthetic-{task}"
        batch_root = root / "task_batches" / task / batch_id
        batch_root.mkdir(parents=True)
        results = [_result(case, manifest, f"{batch_id}-{case.episode_index:03d}-{case.case_id[:12]}", batch_id)
                   for case in task_cases]
        running = TaskBatchAttempt(manifest.manifest_id, task, batch_id, "running",
                                   tuple(case.case_id for case in task_cases), (),
                                   "2026-11-12T00:00:00+00:00", None)
        complete = replace(running, status="complete", completed_case_ids=running.expected_case_ids,
                           finished_at="2026-11-12T00:01:00+00:00")
        (batch_root / "batch.jsonl").write_text("\n".join(json.dumps(x.to_payload()) for x in (running, complete)) + "\n")
        rows = [json.dumps(result.to_payload()) for result in results]
        (batch_root / "results.jsonl").write_text("\n".join(rows) + "\n")
        all_results.extend(rows)
        outcomes.append(TaskBatchProcessOutcome(manifest.manifest_id, task, batch_id, 0))
    (root / "results.jsonl").write_text("\n".join(all_results) + "\n")
    (root / "coordinator_batch_outcomes.jsonl").write_text("\n".join(json.dumps(x.to_payload()) for x in outcomes) + "\n")
    return manifest


def test_results_are_recomputed_from_synthetic_complete_ledgers(tmp_path):
    manifest = _synthetic_complete_ledgers(tmp_path)
    (tmp_path / "official_metrics.json").write_text('{"is_complete":false}')
    metrics = prep.recompute_metrics(tmp_path, expected_commit=manifest.rpent_commit)
    assert metrics["completed_rollouts"] == 2500
    assert metrics["split_success_rates"] == dict.fromkeys(prep.SPLITS, 1.0)


def test_missing_batch_process_outcome_blocks_even_a_complete_summary(tmp_path):
    manifest = _synthetic_complete_ledgers(tmp_path)
    path = tmp_path / "coordinator_batch_outcomes.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="complete 50-task"):
        prep.recompute_metrics(tmp_path, expected_commit=manifest.rpent_commit)


def test_self_consistent_but_shortened_horizons_cannot_be_submitted(tmp_path):
    manifest = _synthetic_complete_ledgers(tmp_path)
    first_task = manifest.cases[0].task_name
    shortened = replace(manifest, cases=tuple(
        replace(case, horizon=case.horizon - 1) if case.task_name == first_task else case
        for case in manifest.cases
    ), simulator={**manifest.simulator, "tasks": tuple(
        {**fact, "upstream_horizon": fact["upstream_horizon"] - 1}
        if fact["name"] == first_task else dict(fact) for fact in manifest.simulator["tasks"]
    )})
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps(shortened.to_payload()))
    with pytest.raises(ValueError, match="packaged upstream horizon"):
        prep.recompute_metrics(tmp_path, expected_commit=manifest.rpent_commit)
