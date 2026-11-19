import pytest

from rpent.runtime.termination import classify_termination


@pytest.mark.parametrize(
    ("finish", "harness", "classification", "error_code", "exit_code"),
    [
        (
            {"status": "success"},
            {"contract_success": True},
            "SUCCESS",
            None,
            0,
        ),
        (
            {"status": " FAILURE "},
            {"contract_success": False},
            "FAILED",
            "explicit_failure",
            1,
        ),
        (
            None,
            {"state": "terminal_failure"},
            "FAILED",
            "terminal_failure",
            1,
        ),
        (
            {"status": "success"},
            {"contract_success": False},
            "INCOMPLETE",
            "incomplete_termination",
            1,
        ),
        (
            None,
            {"status": "ready"},
            "INCOMPLETE",
            "incomplete_termination",
            1,
        ),
    ],
)
def test_terminal_classification_requires_explicit_contract_completion(
    finish, harness, classification, error_code, exit_code
):
    result = classify_termination(
        finish_result=finish,
        error=None,
        harness_snapshot=harness,
        stop_reason="natural_end",
    )

    assert result.classification == classification
    assert result.error_code == error_code
    assert result.exit_code == exit_code


def test_missing_finish_has_stable_error_and_reports_stop_reason():
    result = classify_termination(
        finish_result=None,
        error=None,
        harness_snapshot={"status": "ready"},
        stop_reason="turn_limit",
    )

    assert result.classification == "INCOMPLETE"
    assert result.summary.endswith("(stop_reason=turn_limit)")
    assert result.error == "incomplete_termination: planner ended without finish"


def test_terminal_failure_accepts_status_snapshot_key():
    result = classify_termination(
        finish_result=None,
        error="recovery exhausted",
        harness_snapshot={"status": "terminal_failure"},
    )

    assert result.classification == "FAILED"
    assert result.error == "recovery exhausted"


@pytest.mark.parametrize("finish", [None, {"status": "success"}])
def test_runtime_error_overrides_apparent_or_missing_completion(finish):
    result = classify_termination(
        finish_result=finish,
        error="response stream disconnected",
        harness_snapshot={"contract_success": True},
        stop_reason="error",
    )

    assert result.classification == "FAILED"
    assert result.error_code == "runtime_error"
    assert result.exit_code == 1
    assert result.error == "response stream disconnected"


@pytest.mark.parametrize("environment_success", [False, None])
def test_formal_benchmark_success_requires_simulator_predicate(
    environment_success,
):
    result = classify_termination(
        finish_result={"status": "success"},
        error=None,
        harness_snapshot={
            "contract_success": True,
            "environment_success": environment_success,
        },
        require_environment_success=True,
    )

    assert result.classification == "INCOMPLETE"
    assert result.error_code == "environment_success_required"
    assert result.exit_code == 1


def test_formal_benchmark_success_accepts_both_authorities():
    result = classify_termination(
        finish_result={"status": "success"},
        error=None,
        harness_snapshot={
            "contract_success": True,
            "environment_success": True,
        },
        require_environment_success=True,
    )

    assert result.classification == "SUCCESS"
