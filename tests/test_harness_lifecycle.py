"""Tests for the authoritative external Harness lifecycle contract."""

from __future__ import annotations

from itertools import product

import pytest

import rpent.harness.lifecycle as lifecycle
from rpent.harness.lifecycle import (
    ALLOWED_OPERATIONS,
    TERMINAL_STATES,
    TRANSITIONS,
    HarnessLifecycleState,
    LifecycleEvent,
    transition,
)

EXPECTED_TRANSITIONS = {
    ("action_required", "action_progressed"): "verification_required",
    ("action_required", "action_failed"): "recovery_required",
    ("action_required", "exception_requested"): "recovery_required",
    ("action_required", "budget_exhausted"): "terminal_failure",
    ("verification_required", "verification_uncertain"): "verification_required",
    (
        "verification_required",
        "outcome_satisfied_awaiting_final_verification",
    ): "verification_required",
    ("verification_required", "outcome_satisfied"): "action_required",
    ("verification_required", "stage_completed"): "observation_required",
    ("verification_required", "contract_completed"): "complete",
    ("verification_required", "verification_failed"): "recovery_required",
    ("verification_required", "exception_requested"): "recovery_required",
    ("verification_required", "budget_exhausted"): "terminal_failure",
    ("observation_required", "stage_advanced"): "action_required",
    ("recovery_required", "recovery_applied"): "action_required",
    ("recovery_required", "budget_exhausted"): "terminal_failure",
}


@pytest.mark.parametrize("paused,cancelled,charged", [(6, 0, False), (0, 6, False), (5, 0, True), (-1, 7, True), (True, 5, True)])
def test_review_pause_accounting_requires_a_complete_nonexecuted_suffix(paused, cancelled, charged):
    result = {
        "requested_actions": 6, "executed_actions": 0,
        "paused_actions": paused, "cancelled_actions": cancelled,
        "transitions_used": 0, "review_required": True,
        "stop_reason": "preclose_review_required",
        "environment": {"success": False, "terminated": False, "truncated": False},
    }
    assert lifecycle.consumes_physical_attempt(
        tool_name="smolvla_pick", execution_status="completed", result=result,
    ) is charged


def test_lifecycle_declares_stable_external_states() -> None:
    assert {state.value for state in HarnessLifecycleState} == {
        "action_required",
        "verification_required",
        "observation_required",
        "recovery_required",
        "complete",
        "terminal_failure",
    }


def test_transition_table_exactly_matches_external_contract() -> None:
    actual = {
        (source.value, event.value): target.value
        for source, events in TRANSITIONS.items()
        for event, target in events.items()
    }
    assert actual == EXPECTED_TRANSITIONS


@pytest.mark.parametrize(
    ("source", "event"),
    list(product(HarnessLifecycleState, LifecycleEvent)),
)
def test_every_state_event_pair_is_explicitly_legal_or_rejected(
    source: HarnessLifecycleState,
    event: LifecycleEvent,
) -> None:
    expected = EXPECTED_TRANSITIONS.get((source.value, event.value))
    if expected is None:
        with pytest.raises(ValueError, match="illegal Harness lifecycle transition"):
            transition(source, event)
    else:
        assert transition(source, event) is HarnessLifecycleState(expected)


def test_transition_accepts_stable_string_values() -> None:
    assert (
        transition("action_required", "action_progressed")
        is HarnessLifecycleState.VERIFICATION_REQUIRED
    )


def test_unknown_states_and_events_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown Harness lifecycle state"):
        transition("invented", LifecycleEvent.ACTION_FAILED)
    with pytest.raises(ValueError, match="unknown Harness lifecycle event"):
        transition(HarnessLifecycleState.ACTION_REQUIRED, "invented")


def test_every_nonterminal_state_has_an_allowed_operation() -> None:
    assert set(ALLOWED_OPERATIONS) == set(HarnessLifecycleState)
    for state in HarnessLifecycleState:
        if state not in TERMINAL_STATES:
            assert ALLOWED_OPERATIONS[state]


def test_every_event_is_reachable_from_a_declared_state() -> None:
    reachable = {event for events in TRANSITIONS.values() for event in events}
    assert reachable == set(LifecycleEvent)


def test_terminal_states_have_no_outgoing_transitions() -> None:
    assert TERMINAL_STATES == {
        HarnessLifecycleState.COMPLETE,
        HarnessLifecycleState.TERMINAL_FAILURE,
    }
    for state in TERMINAL_STATES:
        assert not TRANSITIONS[state]


def test_recovery_operations_do_not_expose_direct_scope_selection() -> None:
    assert ALLOWED_OPERATIONS[HarnessLifecycleState.RECOVERY_REQUIRED] == (
        "apply_recovery",
        "execution_status",
    )
    advertised = {
        operation
        for operations in ALLOWED_OPERATIONS.values()
        for operation in operations
    }
    assert "replan" not in advertised
    assert "request_replan" not in advertised
    assert "select_outcome" not in advertised


def test_request_exception_is_available_before_but_not_during_recovery() -> None:
    assert "request_exception" in ALLOWED_OPERATIONS[
        HarnessLifecycleState.ACTION_REQUIRED
    ]
    assert "request_exception" not in ALLOWED_OPERATIONS[
        HarnessLifecycleState.VERIFICATION_REQUIRED
    ]
    assert (
        "request_exception"
        not in ALLOWED_OPERATIONS[HarnessLifecycleState.RECOVERY_REQUIRED]
    )


def test_request_exception_has_transition_from_every_advertised_state() -> None:
    advertised_states = {
        state
        for state, operations in ALLOWED_OPERATIONS.items()
        if "request_exception" in operations
    }
    assert advertised_states == {HarnessLifecycleState.ACTION_REQUIRED}
    for state in advertised_states:
        assert (
            transition(state, LifecycleEvent.EXCEPTION_REQUESTED)
            is HarnessLifecycleState.RECOVERY_REQUIRED
        )


def test_pending_recovery_action_requires_physical_tool_before_exception() -> None:
    guarded = lifecycle.allowed_operations(
        HarnessLifecycleState.ACTION_REQUIRED,
        recovery_action_pending=True,
    )

    assert "configured_physical_tool" in guarded
    assert "request_exception" not in guarded
    assert lifecycle.allowed_operations(
        HarnessLifecycleState.ACTION_REQUIRED,
        recovery_action_pending=False,
    ) == ALLOWED_OPERATIONS[HarnessLifecycleState.ACTION_REQUIRED]


def test_atomic_satisfaction_can_wait_for_authoritative_final_verification() -> None:
    assert (
        transition(
            HarnessLifecycleState.VERIFICATION_REQUIRED,
            LifecycleEvent.OUTCOME_SATISFIED_AWAITING_FINAL_VERIFICATION,
        )
        is HarnessLifecycleState.VERIFICATION_REQUIRED
    )


def test_start_instruction_is_not_an_in_lifecycle_operation() -> None:
    advertised = {
        operation
        for operations in ALLOWED_OPERATIONS.values()
        for operation in operations
    }
    assert "start_instruction" not in advertised


def test_lifecycle_tables_are_immutable() -> None:
    with pytest.raises(TypeError):
        TRANSITIONS[HarnessLifecycleState.ACTION_REQUIRED] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        TRANSITIONS[HarnessLifecycleState.ACTION_REQUIRED][
            LifecycleEvent.ACTION_PROGRESSED
        ] = HarnessLifecycleState.COMPLETE  # type: ignore[index]
    with pytest.raises(TypeError):
        ALLOWED_OPERATIONS[HarnessLifecycleState.ACTION_REQUIRED] = ()  # type: ignore[index]
