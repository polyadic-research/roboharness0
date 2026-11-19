from __future__ import annotations

import asyncio
import concurrent.futures
import math
import re
from dataclasses import FrozenInstanceError

import pytest

from rpent.recording.contracts import EventEnvelope
from rpent.ui.case_runtime import RuntimeUpdate
from rpent.ui.event_hub import AttemptSequence, EvaluationEventHub


def _event(
    case_id: str,
    sequence: int,
    *,
    evaluation_id: str = "evaluation-1",
    attempt_id: str = "attempt-1",
    recorded_at: str = "2026-08-07T00:00:00+00:00",
) -> EventEnvelope:
    return EventEnvelope(
        schema_version=1,
        evaluation_id=evaluation_id,
        case_id=case_id,
        attempt_id=attempt_id,
        sequence=sequence,
        recorded_at=recorded_at,
        event_type="test_event",
        step_id=None,
        operation_id=None,
        execution_id=None,
        payload={},
    )


def test_read_preserves_arrival_order_across_cases_and_filters_evaluation() -> None:
    hub = EvaluationEventHub(capacity=3, secret=b"test-secret")
    cursor = hub.cursor()
    hub.publish(_event("case-a", 20))
    hub.publish(_event("other", 1, evaluation_id="evaluation-2"))
    hub.publish(_event("case-b", 1))

    batch = hub.read(cursor, evaluation_id="evaluation-1")

    assert [event.case_id for event in batch.events] == ["case-a", "case-b"]
    assert batch.next_cursor != cursor
    assert batch.expired is False


def test_runtime_update_discards_delayed_revision_and_wakes_lifecycle() -> None:
    hub = EvaluationEventHub(capacity=3, secret=b"test-secret")
    cursor = hub.cursor("evaluation-1")

    assert hub.publish_runtime_update(
        RuntimeUpdate("evaluation-1", "case-a", "attempt-1", 2, _event("case-a", 1))
    )
    assert not hub.publish_runtime_update(
        RuntimeUpdate("evaluation-1", "case-a", "attempt-1", 1, _event("case-a", 2))
    )
    assert hub.publish_runtime_update(
        RuntimeUpdate("evaluation-1", "case-a", "attempt-1", 3, None)
    )

    batch = hub.read(cursor, evaluation_id="evaluation-1")
    assert [event.sequence for event in batch.events] == [1]


def test_hub_rejects_media_bearing_notifications_even_for_direct_publish() -> None:
    hub = EvaluationEventHub(capacity=3, secret=b"test-secret")
    event = _event("case-a", 1)
    event.payload["nested"] = {"image_bytes": "abc"}

    with pytest.raises(ValueError, match="media field"):
        hub.publish(event)


def test_cursor_is_opaque_url_safe_and_rejects_tampering() -> None:
    hub = EvaluationEventHub(capacity=3, secret=b"test-secret")
    cursor = hub.cursor()

    assert re.fullmatch(r"[A-Za-z0-9_-]+", cursor)
    assert cursor != "0"
    assert not cursor.isdecimal()
    replacement = "A" if cursor[-1] != "A" else "B"
    tampered = cursor[:-1] + replacement

    with pytest.raises(ValueError, match="cursor"):
        hub.read(tampered, evaluation_id="evaluation-1")


def test_cursor_does_not_reveal_reusable_encoding_for_an_offset() -> None:
    hub = EvaluationEventHub(capacity=3, secret=b"test-secret")

    assert hub.cursor() != hub.cursor()


def test_case_cursor_binds_case_attempt_and_sequence_and_rejects_tampering() -> None:
    hub = EvaluationEventHub(capacity=3, secret=b"test-secret")
    cursor = hub.case_cursor("evaluation-1", "case-a", "attempt-1", 7)

    assert re.fullmatch(r"[A-Za-z0-9_-]+", cursor)
    assert hub.case_position(cursor, "evaluation-1", "case-a") == AttemptSequence(
        "attempt-1", 7
    )
    with pytest.raises(ValueError, match="scope"):
        hub.case_position(cursor, "evaluation-1", "case-b")
    replacement = "A" if cursor[-1] != "A" else "B"
    with pytest.raises(ValueError, match="cursor"):
        hub.case_position(cursor[:-1] + replacement, "evaluation-1", "case-a")


def test_case_cursor_can_bind_the_pre_attempt_checkpoint() -> None:
    hub = EvaluationEventHub(capacity=3, secret=b"test-secret")
    cursor = hub.case_cursor("evaluation-1", "case-a", None, 0)

    assert hub.case_position(cursor, "evaluation-1", "case-a") == AttemptSequence(
        None, 0
    )


def test_duplicate_identity_includes_evaluation_case_attempt_and_sequence() -> None:
    hub = EvaluationEventHub(capacity=8, secret=b"test-secret")
    cursor = hub.cursor()
    duplicate = _event("case-a", 1)

    assert hub.publish(duplicate) is True
    assert hub.publish(duplicate) is False
    assert hub.publish(_event("case-a", 1, evaluation_id="evaluation-2")) is True
    assert hub.publish(_event("case-a", 1, attempt_id="attempt-2")) is True
    assert hub.publish(_event("case-b", 1)) is True

    assert len(hub.read(cursor, evaluation_id="evaluation-1").events) == 3


def test_expired_cursor_returns_case_sequences_and_continuable_cursor() -> None:
    hub = EvaluationEventHub(capacity=2, secret=b"test-secret")
    expired_cursor = hub.cursor()
    hub.publish(_event("case-a", 1))
    hub.publish(_event("case-b", 4))
    hub.publish(_event("case-a", 2))

    expired = hub.read(expired_cursor, evaluation_id="evaluation-1")

    assert expired.expired is True
    assert expired.events == ()
    assert expired.last_sequences == {
        "case-a": AttemptSequence("attempt-1", 2),
        "case-b": AttemptSequence("attempt-1", 4),
    }
    hub.publish(_event("case-b", 5))
    resumed = hub.read(expired.next_cursor, evaluation_id="evaluation-1")
    assert [(event.case_id, event.sequence) for event in resumed.events] == [
        ("case-b", 5)
    ]


def test_capacity_boundary_keeps_exactly_capacity_notifications() -> None:
    hub = EvaluationEventHub(capacity=3, secret=b"test-secret")
    cursor = hub.cursor()
    for sequence in range(1, 4):
        hub.publish(_event("case-a", sequence))

    batch = hub.read(cursor, evaluation_id="evaluation-1")

    assert batch.expired is False
    assert [event.sequence for event in batch.events] == [1, 2, 3]


def test_dedupe_high_water_survives_entry_eviction_within_bounded_window() -> None:
    hub = EvaluationEventHub(capacity=1, dedupe_capacity=2, secret=b"test-secret")
    hub.publish(_event("case-a", 2))
    hub.publish(_event("case-b", 1))

    assert hub.publish(_event("case-a", 2)) is False
    assert hub.publish(_event("case-a", 1)) is False


def test_dedupe_metadata_evicts_at_its_explicit_capacity() -> None:
    hub = EvaluationEventHub(capacity=1, dedupe_capacity=1, secret=b"test-secret")
    hub.publish(_event("case-a", 1))
    hub.publish(_event("case-b", 1))

    assert hub.publish(_event("case-a", 1)) is True


def test_expired_snapshot_metadata_is_bounded_with_dedupe_window() -> None:
    hub = EvaluationEventHub(capacity=1, dedupe_capacity=1, secret=b"test-secret")
    cursor = hub.cursor("evaluation-1")
    hub.publish(_event("case-a", 1))
    hub.publish(_event("case-b", 1, evaluation_id="evaluation-2"))

    batch = hub.read(cursor, evaluation_id="evaluation-1")

    assert batch.expired is True
    assert batch.last_sequences == {}


def test_old_attempt_cannot_replace_current_attempt_snapshot() -> None:
    hub = EvaluationEventHub(capacity=2, dedupe_capacity=2, secret=b"test-secret")
    expired_cursor = hub.cursor("evaluation-1")
    hub.publish(
        _event(
            "case-a",
            3,
            attempt_id="attempt-old",
            recorded_at="2026-08-07T00:00:03+00:00",
        )
    )
    hub.publish(
        _event(
            "case-a",
            1,
            attempt_id="attempt-new",
            recorded_at="2026-08-07T00:00:10+00:00",
        )
    )
    hub.publish(_event("case-b", 1, recorded_at="2026-08-07T00:00:11+00:00"))
    hub.publish(
        _event(
            "case-a",
            4,
            attempt_id="attempt-old",
            recorded_at="2026-08-07T00:00:04+00:00",
        )
    )

    batch = hub.read(expired_cursor, evaluation_id="evaluation-1")
    assert batch.expired is True
    assert batch.last_sequences["case-a"] == AttemptSequence("attempt-new", 1)

    hub.publish(
        _event(
            "case-a",
            1,
            attempt_id="attempt-newer",
            recorded_at="2026-08-07T00:00:12+00:00",
        )
    )
    current = hub.read(batch.next_cursor, evaluation_id="evaluation-1")
    assert current.last_sequences["case-a"] == AttemptSequence("attempt-newer", 1)


def test_snapshot_and_hub_event_are_deeply_immutable() -> None:
    hub = EvaluationEventHub(capacity=1, secret=b"test-secret")
    event = _event("case-a", 1)
    event.payload["nested"] = {"value": 1}
    cursor = hub.cursor("evaluation-1")
    hub.publish(event)
    event.payload["nested"]["value"] = 2
    event.payload["late"] = True

    batch = hub.read(cursor, evaluation_id="evaluation-1")
    assert batch.events[0].payload == {"nested": {"value": 1}}
    with pytest.raises(TypeError):
        batch.last_sequences["case-b"] = AttemptSequence("attempt-1", 1)
    with pytest.raises(FrozenInstanceError):
        batch.last_sequences["case-a"].sequence = 9


def test_read_returns_fresh_event_payload_copies() -> None:
    hub = EvaluationEventHub(capacity=2, secret=b"test-secret")
    cursor = hub.cursor("evaluation-1")
    event = _event("case-a", 1)
    event.payload["nested"] = {"value": 1}
    hub.publish(event)

    first = hub.read(cursor, evaluation_id="evaluation-1")
    first.events[0].payload["nested"]["value"] = 99
    first.events[0].payload["caller"] = True
    second = hub.read(cursor, evaluation_id="evaluation-1")

    assert second.events[0].payload == {"nested": {"value": 1}}


def test_paginated_read_does_not_skip_matching_events_across_evaluations() -> None:
    hub = EvaluationEventHub(capacity=12, secret=b"test-secret")
    cursor = hub.cursor("evaluation-1")
    hub.publish(_event("other-a", 1, evaluation_id="evaluation-2"))
    hub.publish(_event("case-a", 1))
    hub.publish(_event("other-b", 1, evaluation_id="evaluation-2"))
    hub.publish(_event("case-a", 2))
    hub.publish(_event("case-a", 3))

    first = hub.read(cursor, evaluation_id="evaluation-1", max_events=2)
    second = hub.read(first.next_cursor, evaluation_id="evaluation-1", max_events=2)
    final = hub.read(second.next_cursor, evaluation_id="evaluation-1", max_events=2)

    assert [event.sequence for event in first.events] == [1, 2]
    assert [event.sequence for event in second.events] == [3]
    assert final.events == ()


def test_paginated_read_respects_serialized_byte_limit() -> None:
    hub = EvaluationEventHub(capacity=4, secret=b"test-secret")
    cursor = hub.cursor("evaluation-1")
    first_event = _event("case-a", 1)
    second_event = _event("case-a", 2)
    hub.publish(first_event)
    hub.publish(second_event)
    one_event_bytes = len(
        __import__("json")
        .dumps(
            first_event.to_payload(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        .encode()
    )

    first = hub.read(
        cursor,
        evaluation_id="evaluation-1",
        max_events=8,
        max_bytes=one_event_bytes,
    )
    second = hub.read(
        first.next_cursor,
        evaluation_id="evaluation-1",
        max_events=8,
        max_bytes=one_event_bytes,
    )

    assert [event.sequence for event in first.events] == [1]
    assert [event.sequence for event in second.events] == [2]


def test_async_subscription_coalesces_wakes_and_unregisters() -> None:
    async def run() -> None:
        hub = EvaluationEventHub(capacity=4, secret=b"test-secret")
        subscription = hub.open_async_subscription("evaluation-1")
        hub.publish(_event("case-a", 1))
        hub.publish(_event("case-a", 2))
        await asyncio.wait_for(subscription.wait(), 1)
        assert subscription.pending_wakes == 0
        assert hub.async_subscriber_count == 1
        subscription.close()
        assert hub.async_subscriber_count == 0

    asyncio.run(run())


def test_async_subscription_schedules_only_one_callback_until_acknowledged() -> None:
    class Loop:
        def __init__(self) -> None:
            self.callbacks = []

        def call_soon_threadsafe(self, callback, *args) -> None:
            self.callbacks.append((callback, args))

    hub = EvaluationEventHub(capacity=1_100, secret=b"test-secret")
    loop = Loop()
    subscriptions = [
        hub.open_async_subscription("evaluation-1", loop=loop) for _ in range(64)
    ]

    for sequence in range(1, 1_001):
        hub.publish(_event("case-a", sequence))

    assert len(loop.callbacks) == 64
    for subscription in subscriptions:
        subscription.close()


def test_async_subscription_unregisters_when_event_loop_is_closed() -> None:
    class ClosedLoop:
        def call_soon_threadsafe(self, callback, *args) -> None:
            raise RuntimeError("Event loop is closed")

    hub = EvaluationEventHub(capacity=4, secret=b"test-secret")
    hub.open_async_subscription("evaluation-1", loop=ClosedLoop())

    hub.publish(_event("case-a", 1))

    assert hub.async_subscriber_count == 0


def test_coalesced_publish_before_ack_is_observed_by_cursor_read() -> None:
    async def run() -> None:
        hub = EvaluationEventHub(capacity=4, secret=b"test-secret")
        cursor = hub.cursor("evaluation-1")
        subscription = hub.open_async_subscription("evaluation-1")
        hub.publish(_event("case-a", 1))
        hub.publish(_event("case-a", 2))

        assert await subscription.wait(1)
        batch = hub.read(cursor, evaluation_id="evaluation-1")
        assert [event.sequence for event in batch.events] == [1, 2]
        subscription.close()

    asyncio.run(run())


def test_evaluation_scoped_cursor_rejects_cross_evaluation_use() -> None:
    hub = EvaluationEventHub(capacity=2, secret=b"test-secret")
    cursor = hub.cursor("evaluation-1")

    with pytest.raises(ValueError, match="scope"):
        hub.read(cursor, evaluation_id="evaluation-2")


def test_same_secret_restart_expires_cursor_instead_of_treating_offset_as_current() -> (
    None
):
    first = EvaluationEventHub(capacity=2, secret=b"test-secret")
    first.publish(_event("case-a", 1))
    cursor = first.cursor("evaluation-1")
    restarted = EvaluationEventHub(capacity=2, secret=b"test-secret")

    batch = restarted.read(cursor, evaluation_id="evaluation-1")

    assert batch.expired is True
    assert batch.expired_reason == "hub_restarted"
    assert batch.last_sequences == {}
    assert (
        restarted.read(batch.next_cursor, evaluation_id="evaluation-1").expired is False
    )


def test_different_secret_rejects_cursor() -> None:
    cursor = EvaluationEventHub(capacity=2, secret=b"first-secret").cursor()

    with pytest.raises(ValueError, match="cursor"):
        EvaluationEventHub(capacity=2, secret=b"second-secret").read(
            cursor, evaluation_id="evaluation-1"
        )


def test_concurrent_publish_and_read_are_safe() -> None:
    hub = EvaluationEventHub(capacity=200, secret=b"test-secret")
    cursor = hub.cursor()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        published = list(
            executor.map(
                hub.publish,
                (_event(f"case-{index}", 1) for index in range(100)),
            )
        )

    batch = hub.read(cursor, evaluation_id="evaluation-1")
    assert all(published)
    assert len(batch.events) == 100
    assert len({event.case_id for event in batch.events}) == 100


def test_subscribe_waits_for_a_new_notification() -> None:
    hub = EvaluationEventHub(capacity=3, secret=b"test-secret")
    cursor = hub.cursor()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(
            hub.subscribe, cursor, evaluation_id="evaluation-1", timeout=1.0
        )
        hub.publish(_event("case-a", 1))
        batch = waiting.result(timeout=1.0)

    assert [event.case_id for event in batch.events] == ["case-a"]


@pytest.mark.parametrize("capacity", [0, -1, True])
def test_capacity_must_be_a_positive_integer(capacity: object) -> None:
    with pytest.raises(ValueError, match="capacity"):
        EvaluationEventHub(capacity=capacity)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "timeout", [None, True, -1, math.nan, math.inf, 3600.1, 1e308, "1"]
)
def test_subscribe_rejects_invalid_timeout(timeout: object) -> None:
    hub = EvaluationEventHub(capacity=1)

    with pytest.raises(ValueError, match="timeout"):
        hub.subscribe(
            hub.cursor(),
            evaluation_id="evaluation-1",
            timeout=timeout,  # type: ignore[arg-type]
        )
