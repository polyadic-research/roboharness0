"""Tests for immutable outcome definitions and evidence-backed graph state."""

import dataclasses

import pytest

from rpent.harness.outcomes import OutcomeEvidence, OutcomeGraph, ReorderRequest


def _evidence(
    summary: str,
    *,
    authoritative: bool = True,
    predicate_result: str | None = "satisfied",
    execution_id: str = "exec-1",
) -> OutcomeEvidence:
    return OutcomeEvidence(
        execution_id=execution_id,
        source="verifier",
        summary=summary,
        confidence=1.0,
        observation_id=f"obs-{execution_id}",
        captured_at="2026-08-04T00:00:00+00:00",
        verifier="vision-predicate",
        verifier_version="1",
        predicate_result=predicate_result,
        authoritative=authoritative,
    )


def _payload() -> list[dict[str, object]]:
    return [
        {
            "id": "outcome-1",
            "instruction_index": 1,
            "description": "identify target",
            "success_predicate": "target is identified",
            "depends_on": [],
            "lifecycle": "event",
        },
        {
            "id": "outcome-2",
            "instruction_index": 2,
            "description": "place target",
            "success_predicate": "target remains in drawer",
            "depends_on": ["outcome-1"],
            "lifecycle": "terminal",
        },
    ]


def test_builds_acyclic_graph_in_instruction_order() -> None:
    graph = OutcomeGraph.from_payload(_payload())

    assert graph.outcome_ids == ("outcome-1", "outcome-2")
    assert graph.ready_ids == ("outcome-1",)
    assert graph.snapshot()["counts"]["pending"] == 1


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            [{**_payload()[0], "depends_on": ["outcome-9"]}],
            "unknown dependency",
        ),
        (
            [
                {**_payload()[0], "depends_on": ["outcome-2"]},
                {**_payload()[1], "depends_on": ["outcome-1"]},
            ],
            "cycle",
        ),
        ([{**_payload()[0], "id": "  "}], "id"),
        ([{**_payload()[0], "lifecycle": "temporary"}], "lifecycle"),
        ([{**_payload()[0], "description": " "}], "description"),
        ([{**_payload()[0], "success_predicate": "\t"}], "success_predicate"),
        ([_payload()[0], {**_payload()[1], "id": "outcome-1"}], "duplicate outcome id"),
        (
            [_payload()[0], {**_payload()[1], "instruction_index": 1}],
            "instruction_index",
        ),
    ],
)
def test_rejects_invalid_outcome_contract(
    payload: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        OutcomeGraph.from_payload(payload)


def test_definitions_are_immutable_and_input_is_not_aliased() -> None:
    payload = _payload()
    graph = OutcomeGraph.from_payload(payload)
    payload[0]["description"] = "fabricated replacement"

    assert graph.snapshot()["outcomes"][0]["description"] == "identify target"
    with pytest.raises(dataclasses.FrozenInstanceError):
        graph.definitions[0].description = "fabricated replacement"  # type: ignore[misc]
    assert not hasattr(graph, "delete")


def test_unknown_outcome_cannot_be_selected_or_silently_created() -> None:
    graph = OutcomeGraph.from_payload(_payload())

    with pytest.raises(ValueError, match="unknown outcome"):
        graph.select("outcome-9")
    assert graph.outcome_ids == ("outcome-1", "outcome-2")


def test_satisfaction_requires_active_outcome_and_evidence() -> None:
    graph = OutcomeGraph.from_payload(_payload())

    with pytest.raises(ValueError, match="in progress"):
        graph.satisfy("outcome-1", _evidence("not selected"))
    graph.select("outcome-1")
    with pytest.raises((TypeError, ValueError), match="evidence"):
        graph.satisfy("outcome-1", None)  # type: ignore[arg-type]


def test_satisfaction_rejects_advisory_or_conflicting_evidence() -> None:
    graph = OutcomeGraph.from_payload(_payload())
    graph.select("outcome-1")

    with pytest.raises(ValueError, match="authoritative"):
        graph.satisfy(
            "outcome-1",
            _evidence("looks identified", authoritative=False),
        )
    with pytest.raises(ValueError, match="satisfied"):
        graph.satisfy(
            "outcome-1",
            _evidence("not identified", predicate_result="unsatisfied"),
        )


def test_dependencies_gate_selection_and_satisfaction_unlocks_next() -> None:
    graph = OutcomeGraph.from_payload(_payload())
    with pytest.raises(ValueError, match="not ready"):
        graph.select("outcome-2")

    graph.select("outcome-1")
    graph.satisfy("outcome-1", _evidence("identified"))

    assert graph.ready_ids == ("outcome-2",)


def test_only_one_outcome_can_be_in_progress() -> None:
    graph = OutcomeGraph.from_payload(
        [
            _payload()[0],
            {**_payload()[1], "depends_on": [], "lifecycle": "event"},
        ]
    )
    graph.select("outcome-1")

    with pytest.raises(ValueError, match="already in progress"):
        graph.select("outcome-2", ReorderRequest("outcome-2", (), "safe", ("visible",)))


def test_out_of_order_selection_requires_supported_reorder_request() -> None:
    graph = OutcomeGraph.from_payload(
        [
            _payload()[0],
            {**_payload()[1], "depends_on": [], "lifecycle": "event"},
        ]
    )
    with pytest.raises(ValueError, match="reorder request"):
        graph.select("outcome-2")
    with pytest.raises(ValueError, match="evidence"):
        graph.select(
            "outcome-2", ReorderRequest("outcome-2", ("outcome-1",), "safe", ())
        )

    graph.select(
        "outcome-2",
        ReorderRequest(
            selected_outcome_id="outcome-2",
            deferred_outcome_ids=("outcome-1",),
            reason="temporarily_unobservable",
            evidence=("drawer handle is occluded",),
        ),
    )
    assert graph.active_outcome_id == "outcome-2"


def test_progress_and_failure_are_explicit_evidence_backed_states() -> None:
    graph = OutcomeGraph.from_payload(_payload())
    graph.select("outcome-1")
    graph.progress("outcome-1", _evidence("approached"))
    assert graph.snapshot()["outcomes"][0]["status"] == "progressing"
    assert graph.active_outcome_id == "outcome-1"

    graph.fail("outcome-1", _evidence("missed"))
    assert graph.snapshot()["outcomes"][0]["status"] == "failed"
    assert "outcome-1" not in graph.ready_ids
    graph.prepare_recovery(("outcome-1",), reason="repair registered")
    assert graph.ready_ids == ("outcome-1",)


def test_terminal_outcome_requires_fresh_authoritative_final_evidence() -> None:
    graph = OutcomeGraph.from_payload(_payload())
    graph.select("outcome-1")
    graph.satisfy("outcome-1", _evidence("identified"))
    graph.select("outcome-2")
    graph.satisfy("outcome-2", _evidence("placed", execution_id="place"))

    assert graph.final_audit()["contract_success"] is False
    with pytest.raises(ValueError, match="authoritative"):
        graph.mark_final_verified(
            "outcome-2",
            _evidence(
                "looks placed", authoritative=False, predicate_result="satisfied"
            ),
        )
    with pytest.raises(ValueError, match="satisfied"):
        graph.mark_final_verified(
            "outcome-2",
            _evidence("not placed", authoritative=True, predicate_result="unsatisfied"),
        )

    graph.mark_final_verified(
        "outcome-2",
        _evidence(
            "still placed",
            authoritative=True,
            predicate_result="satisfied",
            execution_id="final",
        ),
    )
    assert graph.final_audit()["contract_success"] is True


def test_invalidation_reopens_only_durable_affected_outcomes() -> None:
    graph = OutcomeGraph.from_payload(
        [
            _payload()[0],
            {
                "id": "outcome-2",
                "instruction_index": 2,
                "description": "drawer stays open",
                "success_predicate": "drawer is open",
                "depends_on": ["outcome-1"],
                "lifecycle": "persistent",
            },
            {
                "id": "outcome-3",
                "instruction_index": 3,
                "description": "cup stays inside",
                "success_predicate": "cup is inside drawer",
                "depends_on": ["outcome-2"],
                "lifecycle": "terminal",
            },
        ]
    )
    graph.select("outcome-1")
    graph.satisfy("outcome-1", _evidence("identified"))
    graph.select("outcome-2")
    graph.satisfy("outcome-2", _evidence("opened"))
    graph.mark_final_verified(
        "outcome-2",
        _evidence(
            "still open",
            authoritative=True,
            predicate_result="satisfied",
            execution_id="final-open",
        ),
    )
    graph.select("outcome-3")

    invalidated = graph.invalidate_affected(
        ("outcome-1", "outcome-2"), reason="drawer moved during transport"
    )

    assert invalidated == ("outcome-2",)
    items = {item["id"]: item for item in graph.snapshot()["outcomes"]}
    assert items["outcome-1"]["status"] == "satisfied"
    assert items["outcome-2"]["status"] == "invalidated"
    assert items["outcome-2"]["final_verified"] is False
    assert graph.recovery_ids_for(("outcome-3",)) == ("outcome-2", "outcome-3")


def test_invalidated_outcome_requires_explicit_recovery_preparation() -> None:
    graph = OutcomeGraph.from_payload([{**_payload()[0], "lifecycle": "persistent"}])
    graph.select("outcome-1")
    graph.satisfy("outcome-1", _evidence("identified"))
    graph.invalidate("outcome-1", reason="changed by later action")

    with pytest.raises(ValueError, match="not ready"):
        graph.select("outcome-1")
    graph.prepare_recovery(("outcome-1",), reason="repair registered")
    graph.select("outcome-1")
    assert graph.active_outcome_id == "outcome-1"


def test_recovery_includes_broken_prerequisite_of_satisfied_stage_outcome() -> None:
    graph = OutcomeGraph.from_payload(
        [
            {**_payload()[0], "lifecycle": "persistent"},
            _payload()[1],
        ]
    )
    graph.select("outcome-1")
    graph.satisfy("outcome-1", _evidence("identified"))
    graph.select("outcome-2")
    graph.satisfy("outcome-2", _evidence("placed"))
    graph.invalidate("outcome-1", reason="later action disturbed the target")

    assert graph.recovery_ids_for(("outcome-2",)) == ("outcome-1",)
