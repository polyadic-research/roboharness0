"""The first VLA prediction may follow a long direct-control segment."""
import base64
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa.xiaomi_binding import make_xiaomi_policy_binding
from rpent.episode.contracts import StepResult
from rpent.episode.runtime import EpisodeBootstrap, open_episode
from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
from rpent.policies.xiaomi_mibot_client import XiaomiMiBoTPolicySession
from rpent.policies.xiaomi_mibot_protocol import (
    XiaomiMiBoTPredictRequest,
    XiaomiMiBoTPredictResponse,
    XiaomiMiBoTResetRequest,
    XiaomiMiBoTResetResponse,
)
from rpent.policies.xiaomi_mibot_server import XiaomiMiBoTServer


def observation(step=0):
    return {"state": np.full(16, step, dtype=np.float32), **{
        key: np.full((256, 256, 3), step, dtype=np.uint8)
        for key in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
    }}


def rng_evidence(seed):
    value = b"test-only-rng-state"
    blob = {"base64": base64.b64encode(value).decode(), "sha256": hashlib.sha256(value).hexdigest()}
    return {"schema_version": 1, "seed": seed, "torch_initial_seed": seed,
            "python_state_sha256": "a" * 64, "numpy_state_sha256": "b" * 64,
            "torch_cpu_state": blob, "torch_cuda_states": [blob]}


class ValidationTransport:
    def __init__(self):
        self.server = XiaomiMiBoTServer.__new__(XiaomiMiBoTServer)
        self.server._history = []

    def call(self, method, *, kwargs=None, **unused):
        if method == "reset":
            self.server._history = []
            return dict(kwargs)
        request = XiaomiMiBoTPredictRequest.from_payload(kwargs)
        self.server._history.extend(self.server._validated_additions(request))
        return XiaomiMiBoTPredictResponse(request.checkpoint, request.episode_id,
                                        request.request_id, np.zeros((16, 12), np.float32)).to_payload()

    def close(self):
        pass


def test_runtime_primes_initial_history_before_any_direct_control():
    rpc = ValidationTransport()
    policy = XiaomiMiBoTPolicySession(rpc, checkpoint=XIAOMI_MIBOT_CHECKPOINT, close_remote=False)

    class Simulator:
        episode_id = "deferred"
        step_id = 0

        def reset(self, task, seed):
            return observation()

        def step(self, action):
            self.step_id += 1
            return StepResult(self.episode_id, self.step_id, observation(self.step_id), 0.0, False, False, False)

        def close(self):
            pass

    sim = Simulator()
    bootstrap = EpisodeBootstrap("test", "CoffeeSetupMug", 7, "same task",
                                 lambda: sim, lambda: policy, make_xiaomi_policy_binding(),
                                 lambda controller, obs: SimpleNamespace())
    with open_episode(bootstrap) as runtime:
        runtime.set_transition_observer(SimpleNamespace(confirmed=lambda *args: None, failed=lambda *args: None))
        for _ in range(9):
            runtime.controller.execute_action(np.zeros(12, np.float32))
        policy.predict(observation(9), "same task")
        assert rpc.server.history_transition_ids == tuple(range(10))
        np.testing.assert_array_equal(rpc.server._history[0]["observation"]["state"], observation()["state"])


def test_seeded_reset_roundtrip_requires_matching_actual_rng_evidence():
    request = XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "seeded", seed=3251203365)
    assert XiaomiMiBoTResetRequest.from_payload(request.to_payload()) == request
    response = XiaomiMiBoTResetResponse.from_payload({**request.to_payload(), "rng_state": rng_evidence(request.seed)})
    response.validate_for(request)
    assert XiaomiMiBoTResetResponse.from_payload(response.to_payload()).rng_state == response.rng_state
    with pytest.raises(ValueError, match="RNG"):
        XiaomiMiBoTResetResponse.from_payload(request.to_payload()).validate_for(request)
    bad = {**request.to_payload(), "rng_state": rng_evidence(7)}
    with pytest.raises(ValueError, match="RNG"):
        XiaomiMiBoTResetResponse.from_payload(bad).validate_for(request)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_reset_rejects_invalid_policy_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "seeded", seed=seed)


def test_server_applies_seed_after_load_on_every_real_episode_reset():
    from test_xiaomi_mibot_server import _server

    server, _, _ = _server()
    calls = []
    server._rng_initializer = lambda seed: calls.append(seed) or rng_evidence(seed)
    for episode, seed in [("one", 123), ("two", 456), ("retry-one", 123)]:
        request = XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, episode, seed=seed)
        reply = server.reset(request)
        reply.validate_for(request)
    assert calls == [123, 456, 123]


def test_seeded_client_archives_actual_state_and_initial_frame_is_immutable(tmp_path):
    import json

    rpc = ValidationTransport()
    original = rpc.call

    def seeded_reply(method, *, kwargs=None, **unused):
        reply = original(method, kwargs=kwargs)
        if method == "reset":
            reply["rng_state"] = rng_evidence(kwargs["seed"])
        return reply

    rpc.call = seeded_reply
    journal = tmp_path / "rng.jsonl"
    policy = XiaomiMiBoTPolicySession(rpc, checkpoint=XIAOMI_MIBOT_CHECKPOINT,
                                     policy_seed=123, rng_journal=journal)
    with pytest.raises(RuntimeError, match="initial observation"):
        policy.initialize_observation(observation())
    policy.reset("seeded")
    initial = observation()
    policy.initialize_observation(initial)
    initial["state"][:] = 99
    with pytest.raises(RuntimeError, match="once"):
        policy.initialize_observation(initial)
    policy.confirmed(np.zeros(12, np.float32), StepResult("seeded", 1, observation(1), 0, False, False, False))
    policy.predict(observation(1), "task")
    assert rpc.server._history[0]["observation"]["state"][0] == 0
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert records[0]["policy_seed"] == 123 and records[0]["episode_id"] == "seeded"
    assert records[0]["rng_state"] == rng_evidence(123)
    policy.reset("new-episode")
    policy.initialize_observation(observation(3))
    policy.predict(observation(3), "task")
    assert rpc.server.history_transition_ids == (0,)


def test_rng_evidence_rejects_tampered_state_bytes():
    from rpent.policies.xiaomi_mibot_rng import validate_rng_evidence

    evidence = rng_evidence(123)
    evidence["torch_cpu_state"]["base64"] = "not base64"
    with pytest.raises(ValueError, match="RNG"):
        validate_rng_evidence(evidence, 123)
