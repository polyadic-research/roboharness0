from __future__ import annotations

import numpy as np
import pytest

from rpent.episode.contracts import StepResult
from rpent.episode.controller import EpisodeController
from rpent.episode.runtime import EpisodeRuntime
from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
from rpent.policies.xiaomi_mibot_client import XiaomiMiBoTPolicySession


def _observation(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "state": np.full(16, value, dtype=np.float32),
        "robot0_agentview_left": np.full((8, 8, 3), value, dtype=np.uint8),
        "robot0_agentview_right": np.full((8, 8, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand": np.full((8, 8, 3), value, dtype=np.uint8),
    }


class _Rpc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, float | None]] = []
        self.close_calls = 0
        self.overrides: dict[str, object] = {}

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        del args
        payload = dict(kwargs or {})
        self.calls.append((method, payload, timeout_s))
        if method == "load":
            return {
                **payload,
                "loaded": True,
                "artifact_identity": "a" * 64,
                "remote_code_revision": XIAOMI_MIBOT_CHECKPOINT.revision,
                "server_version": "rpent-xiaomi-mibot/1",
                **self.overrides,
            }
        if method == "reset":
            return {**payload, **self.overrides}
        if method == "predict":
            return {
                "protocol_version": 1,
                "checkpoint": payload["checkpoint"],
                "episode_id": payload["episode_id"],
                "request_id": payload["request_id"],
                "actions": np.zeros((16, 12), dtype=np.float32),
                **self.overrides,
            }
        if method == "close":
            return {"ok": True}
        raise AssertionError(method)

    def close(self) -> None:
        self.close_calls += 1


def _session(rpc: _Rpc | None = None, **kwargs):
    rpc = rpc or _Rpc()
    return rpc, XiaomiMiBoTPolicySession(
        rpc, checkpoint=XIAOMI_MIBOT_CHECKPOINT, **kwargs
    )


def _transition(step_id: int, *, terminal: bool = False) -> StepResult:
    return StepResult(
        episode_id="episode-1",
        step_id=step_id,
        observation=_observation(step_id),
        reward=0.0,
        success=terminal,
        terminated=False,
        truncated=False,
    )


def test_policy_package_exports_xiaomi_session() -> None:
    from rpent.policies import XiaomiMiBoTPolicySession as exported

    assert exported is XiaomiMiBoTPolicySession


def test_load_validates_complete_server_identity_and_timeout() -> None:
    rpc, session = _session()

    response = session.load(device="cuda:0", dtype="bfloat16", timeout_s=123.0)

    assert response.loaded
    assert rpc.calls[0][0] == "load"
    assert rpc.calls[0][2] == 123.0
    assert rpc.calls[0][1]["checkpoint_identity"] == (
        XIAOMI_MIBOT_CHECKPOINT.identity_digest
    )


def test_predict_requires_reset_and_bounds_executed_prefix() -> None:
    _rpc, session = _session()
    with pytest.raises(RuntimeError, match="reset"):
        session.predict(_observation(), "KettleBoiling")

    session.reset("episode-1")
    for value in (17, 0, True):
        with pytest.raises(ValueError, match="max_actions"):
            session.predict(_observation(), "KettleBoiling", max_actions=value)
    assert session.predict(_observation(), "Slide the dishwasher rack.", max_actions=5).shape == (5, 12)


def test_request_ids_increment_and_reset_restarts_episode_sequence() -> None:
    rpc, session = _session()
    session.reset("episode-1")

    session.predict(_observation(), "KettleBoiling")
    session.predict(_observation(), "KettleBoiling", max_actions=16)
    session.reset("episode-2")
    session.predict(_observation(), "KettleBoiling")

    payloads = [payload for method, payload, _ in rpc.calls if method == "predict"]
    assert [payload["request_id"] for payload in payloads] == [0, 1, 0]
    assert [payload["episode_id"] for payload in payloads] == [
        "episode-1",
        "episode-1",
        "episode-2",
    ]


def test_confirmed_transitions_are_sent_once_in_chronological_order() -> None:
    rpc, session = _session()
    session.reset("episode-1")
    session.predict(_observation(), "KettleBoiling")
    for step_id in range(1, 7):
        session.confirmed(np.zeros(12, dtype=np.float32), _transition(step_id))

    session.predict(_observation(6), "KettleBoiling")
    session.predict(_observation(6), "KettleBoiling")

    payloads = [payload for method, payload, _ in rpc.calls if method == "predict"]
    assert [item["transition_id"] for item in payloads[1]["history_updates"]] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert payloads[2]["history_updates"] == []


def test_runtime_only_cameras_are_removed_at_the_policy_boundary() -> None:
    rpc, session = _session()
    session.reset("episode-1")
    observation = _observation()
    observation["robot0_agentview_center"] = np.zeros((8, 8, 3), dtype=np.uint8)

    session.predict(observation, "KettleBoiling")
    result = _transition(1)
    result.observation["robot0_agentview_center"] = np.zeros(
        (8, 8, 3), dtype=np.uint8
    )
    session.confirmed(np.zeros(12, dtype=np.float32), result)
    session.predict(observation, "KettleBoiling")

    payloads = [payload for method, payload, _ in rpc.calls if method == "predict"]
    expected = {
        "state",
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    }
    assert set(payloads[0]["observation"]) == expected
    assert set(payloads[1]["history_updates"][0]["observation"]) == expected


def test_confirmed_transition_rejects_episode_or_step_drift_and_terminal_predict() -> None:
    _rpc, session = _session()
    session.reset("episode-1")
    session.confirmed(np.zeros(12, dtype=np.float32), _transition(1))
    with pytest.raises(ValueError, match="step_id"):
        session.confirmed(np.zeros(12, dtype=np.float32), _transition(1))

    wrong = _transition(2)
    object.__setattr__(wrong, "episode_id", "episode-2")
    with pytest.raises(ValueError, match="episode_id"):
        session.confirmed(np.zeros(12, dtype=np.float32), wrong)

    session.confirmed(np.zeros(12, dtype=np.float32), _transition(2, terminal=True))
    with pytest.raises(RuntimeError, match="terminal"):
        session.predict(_observation(2), "KettleBoiling")


def test_failed_transition_is_not_added_to_history() -> None:
    rpc, session = _session()
    session.reset("episode-1")
    session.failed(np.zeros(12, dtype=np.float32), 0, RuntimeError("boom"))
    session.predict(_observation(), "KettleBoiling")
    payload = next(payload for method, payload, _ in rpc.calls if method == "predict")
    assert payload["history_updates"] == []


def test_stale_response_does_not_advance_request_or_discard_updates() -> None:
    rpc, session = _session()
    session.reset("episode-1")
    session.confirmed(np.zeros(12, dtype=np.float32), _transition(1))
    rpc.overrides = {"episode_id": "stale"}

    with pytest.raises(ValueError, match="episode_id"):
        session.predict(_observation(1), "KettleBoiling")

    rpc.overrides = {}
    session.predict(_observation(1), "KettleBoiling")
    payloads = [payload for method, payload, _ in rpc.calls if method == "predict"]
    assert [payload["request_id"] for payload in payloads] == [0, 0]
    assert [item["transition_id"] for item in payloads[1]["history_updates"]] == [1]


def test_close_is_idempotent_and_rejects_further_use() -> None:
    rpc, session = _session()
    session.reset("episode-1")

    session.close()
    session.close()

    assert [method for method, _, _ in rpc.calls].count("close") == 1
    assert rpc.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.predict(_observation(), "KettleBoiling")


def test_episode_runtime_composes_recording_and_policy_transition_observers() -> None:
    rpc, session = _session()
    session.reset("episode-1")

    class Simulator:
        def step(self, action):
            return _transition(1)

    class Recorder:
        def __init__(self):
            self.results = []

        def confirmed(self, action, result):
            self.results.append(result)

        def failed(self, action, action_index, error):
            raise AssertionError("unexpected failure")

    class Toolkit:
        def attach_transition_observer(self, observer):
            self.observer = observer

    controller = EpisodeController(Simulator(), session, None)  # type: ignore[arg-type]
    toolkit = Toolkit()
    runtime = EpisodeRuntime(
        episode_id="episode-1",
        task="KettleBoiling",
        seed=1001,
        reset_monotonic=0.0,
        observation=_observation(),
        simulator=Simulator(),  # type: ignore[arg-type]
        policy=session,
        controller=controller,
        toolkit=toolkit,
    )
    recorder = Recorder()
    runtime.set_transition_observer(recorder)

    controller.execute_action(np.zeros(12, dtype=np.float32))
    session.predict(_observation(1), "KettleBoiling")

    assert [result.step_id for result in recorder.results] == [1]
    payload = next(payload for method, payload, _ in rpc.calls if method == "predict")
    assert [item["transition_id"] for item in payload["history_updates"]] == [1]
    assert toolkit.observer is controller.transition_observer


def test_predict_resends_the_same_request_after_an_rpc_error(monkeypatch) -> None:
    import rpent.policies.xiaomi_mibot_client as client_module
    from rpent.utils.socket_rpc import RpcError

    monkeypatch.setattr(client_module, "PREDICT_RETRY_DELAYS_S", (0.0, 0.0))
    rpc, session = _session()
    session.reset("episode-1")
    session.confirmed(np.zeros(12, dtype=np.float32), _transition(1))
    original = rpc.call
    failures = []

    def flaky(method, args=(), kwargs=None, *, timeout_s=None):
        if method == "predict" and not failures:
            failures.append(method)
            rpc.calls.append((method, dict(kwargs or {}), timeout_s))
            raise RpcError("predict", "HTTP request failed: timed out")
        return original(method, args, kwargs, timeout_s=timeout_s)

    rpc.call = flaky
    actions = session.predict(_observation(1), "KettleBoiling")

    assert actions.shape == (16, 12) and session.predict_retries == 1
    sent = [(payload, timeout) for method, payload, timeout in rpc.calls if method == "predict"]
    assert [payload["request_id"] for payload, _ in sent] == [0, 0]
    assert all(payload["history_updates"] == sent[0][0]["history_updates"] for payload, _ in sent)
    assert all(timeout == client_module.PREDICT_TIMEOUT_S for _, timeout in sent)
    session.predict(_observation(1), "KettleBoiling")
    assert [p["request_id"] for m, p, _ in rpc.calls if m == "predict"][-1] == 1


def test_predict_gives_up_after_three_attempts_without_advancing(monkeypatch) -> None:
    import rpent.policies.xiaomi_mibot_client as client_module
    from rpent.utils.socket_rpc import RpcError

    monkeypatch.setattr(client_module, "PREDICT_RETRY_DELAYS_S", (0.0, 0.0))
    rpc, session = _session()
    session.reset("episode-1")
    session.confirmed(np.zeros(12, dtype=np.float32), _transition(1))
    original = rpc.call

    def down(method, args=(), kwargs=None, *, timeout_s=None):
        if method == "predict":
            rpc.calls.append((method, dict(kwargs or {}), timeout_s))
            raise RpcError("predict", "server unavailable")
        return original(method, args, kwargs, timeout_s=timeout_s)

    rpc.call = down
    with pytest.raises(RpcError, match="unavailable"):
        session.predict(_observation(1), "KettleBoiling")
    assert len([m for m, _p, _t in rpc.calls if m == "predict"]) == 3
    rpc.call = original
    session.predict(_observation(1), "KettleBoiling")
    payload = [p for m, p, _ in rpc.calls if m == "predict"][-1]
    assert payload["request_id"] == 0 and [u["transition_id"] for u in payload["history_updates"]] == [1]
