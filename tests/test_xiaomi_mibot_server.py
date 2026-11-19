from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
from rpent.policies.xiaomi_mibot_protocol import (
    XiaomiMiBoTLoadRequest,
    XiaomiMiBoTPredictRequest,
    XiaomiMiBoTResetRequest,
)
from rpent.policies.xiaomi_mibot_server import XiaomiMiBoTServer


class _Tensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)
        self.to_calls = []

    @property
    def shape(self):
        return self.value.shape

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def cpu(self):
        return self

    def float(self):
        self.value = self.value.astype(np.float32)
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, index):
        return _Tensor(self.value[index])


class _Processor:
    def __init__(self) -> None:
        self.apply_calls = []
        self.decode_inputs = []

    def apply_chat_template(self, messages, **kwargs):
        self.apply_calls.append((messages, kwargs))
        return {"floating": _Tensor([1.0]), "tokens": _Tensor([1])}

    def decode_action(self, actions, *, robot_type):
        self.decode_inputs.append((actions, robot_type))
        assert actions.shape == (1, 16, 60)
        return _Tensor(np.arange(16 * 60, dtype=np.float32).reshape(1, 16, 60))


class _Model:
    device = "cuda:0"
    dtype = "bfloat16"

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(actions=_Tensor(np.zeros((1, 16, 60), dtype=np.float32)))


def _observation(value: int) -> dict[str, np.ndarray]:
    state = np.zeros(16, dtype=np.float32)
    state[0:3] = value
    state[3:7] = [0, 0, 0, 1]
    state[7:10] = value
    state[10:14] = [0, 0, 0, 1]
    return {
        "state": state,
        "robot0_agentview_left": np.full((8, 8, 3), value, dtype=np.uint8),
        "robot0_agentview_right": np.full((8, 8, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand": np.full((8, 8, 3), value, dtype=np.uint8),
    }


def _server():
    processor = _Processor()
    model = _Model()
    artifact = SimpleNamespace(
        checkpoint=XIAOMI_MIBOT_CHECKPOINT,
        identity_digest="a" * 64,
    )
    server = XiaomiMiBoTServer(
        snapshot_path="/checkpoint",
        binding_identity="b" * 64,
        artifact_loader=lambda path, checkpoint: artifact,
        processor_loader=lambda path: processor,
        model_loader=lambda path, device, dtype: model,
    )
    server.load(
        XiaomiMiBoTLoadRequest(
            XIAOMI_MIBOT_CHECKPOINT, "b" * 64, "cuda:0", "bfloat16"
        )
    )
    return server, processor, model


def _request(
    request_id: int,
    transition_id: int,
    *,
    episode_id: str = "episode-1",
    terminal: bool = False,
    history_updates=(),
):
    return XiaomiMiBoTPredictRequest(
        checkpoint=XIAOMI_MIBOT_CHECKPOINT,
        episode_id=episode_id,
        request_id=request_id,
        transition_id=transition_id,
        instruction="KettleBoiling",
        observation=_observation(transition_id),
        terminal=terminal,
        history_updates=history_updates,
    )


def test_load_uses_injected_official_runtime_and_reports_health_identity() -> None:
    server, _processor, _model = _server()

    health = server.health()

    assert health.loaded
    assert health.checkpoint == XIAOMI_MIBOT_CHECKPOINT
    assert health.checkpoint_identity == XIAOMI_MIBOT_CHECKPOINT.identity_digest
    assert health.binding_identity == "b" * 64
    assert health.artifact_identity == "a" * 64
    assert health.remote_code_revision == XIAOMI_MIBOT_CHECKPOINT.revision


def test_initial_request_repeats_earliest_observation_four_times() -> None:
    server, processor, _model = _server()
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1"))

    server.predict(_request(0, 0))

    messages, kwargs = processor.apply_calls[-1]
    videos = [item for item in messages[0]["content"] if item["type"] == "video"]
    for video in videos:
        assert [int(np.asarray(frame)[0, 0, 0]) for frame in video["video"]] == [
            0,
            0,
            0,
            0,
        ]
    assert kwargs["state"].shape == (1, 4, 60)
    np.testing.assert_array_equal(kwargs["state"][0, :, 0], [0, 0, 0, 0])


def test_backlog_preserves_every_transition_then_samples_interval_two() -> None:
    server, processor, _model = _server()
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1"))
    server.predict(_request(0, 0))
    updates = tuple(
        {
            "transition_id": value,
            "observation": _observation(value),
            "terminal": False,
        }
        for value in range(1, 7)
    )

    server.predict(_request(1, 6, history_updates=updates))

    messages, kwargs = processor.apply_calls[-1]
    videos = [item for item in messages[0]["content"] if item["type"] == "video"]
    assert [int(np.asarray(frame)[0, 0, 0]) for frame in videos[0]["video"]] == [
        0,
        2,
        4,
        6,
    ]
    np.testing.assert_array_equal(kwargs["state"][0, :, 0], [0, 2, 4, 6])


def test_processor_arguments_and_decode_before_slice_match_official_path() -> None:
    server, processor, model = _server()
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1"))

    response = server.predict(_request(0, 0))

    _messages, kwargs = processor.apply_calls[-1]
    assert kwargs["tokenize"] is True
    assert kwargs["return_dict"] is True
    assert kwargs["return_tensors"] == "pt"
    assert kwargs["do_resize"] is False
    assert kwargs["robot_type"] == "robocasa365"
    assert processor.decode_inputs[0][0].shape == (1, 16, 60)
    assert processor.decode_inputs[0][1] == "robocasa365"
    assert response.actions.shape == (16, 12)
    np.testing.assert_array_equal(
        response.actions,
        np.arange(16 * 60, dtype=np.float32).reshape(16, 60)[
            :, [7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6]
        ],
    )
    assert model.calls


def test_reset_clears_history_and_prevents_cross_episode_reuse() -> None:
    server, processor, _model = _server()
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1"))
    server.predict(_request(0, 0))
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-2"))

    server.predict(_request(0, 0, episode_id="episode-2"))

    messages, _kwargs = processor.apply_calls[-1]
    video = next(item for item in messages[0]["content"] if item["type"] == "video")
    assert [int(np.asarray(frame)[0, 0, 0]) for frame in video["video"]] == [0] * 4


def test_rejects_stale_request_transition_episode_and_terminal_prediction() -> None:
    server, _processor, _model = _server()
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1"))
    server.predict(_request(0, 0))
    server.predict(_request(1, 0))

    with pytest.raises(ValueError, match="request_id"):
        server.predict(_request(0, 0))  # older than the last reply: not a resend
    with pytest.raises(ValueError, match="episode_id"):
        server.predict(_request(2, 0, episode_id="episode-2"))
    with pytest.raises(ValueError, match="transition"):
        server.predict(_request(2, 2))
    with pytest.raises(RuntimeError, match="terminal"):
        server.predict(_request(2, 1, terminal=True))


def test_retries_reuse_latest_confirmed_history_without_duplicate_append() -> None:
    server, processor, _model = _server()
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1"))
    server.predict(_request(0, 0))

    server.predict(_request(1, 0))

    assert len(processor.apply_calls) == 2
    assert server.history_transition_ids == (0,)


def test_close_is_idempotent_and_rejects_further_prediction() -> None:
    server, _processor, _model = _server()
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1"))
    server.close()
    server.close()

    with pytest.raises(RuntimeError, match="closed"):
        server.predict(_request(0, 0))


def test_failed_inference_keeps_the_session_so_the_same_request_succeeds() -> None:
    server, processor, model = _server()
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1"))
    server.predict(_request(0, 0))
    update = {"transition_id": 1, "observation": _observation(1), "terminal": False}
    request = _request(1, 2, history_updates=(update,))
    original = model.__class__.__call__

    def fail_once(self, **kwargs):
        model.__class__.__call__ = original
        raise RuntimeError("CUDA out of memory")

    model.__class__.__call__ = fail_once
    try:
        with pytest.raises(RuntimeError, match="out of memory"):
            server.predict(request)
    finally:
        model.__class__.__call__ = original
    assert server.history_transition_ids == (0,)

    response = server.predict(request)  # the client resends it unchanged
    assert response.request_id == 1
    assert server.history_transition_ids == (0, 1, 2)
    server.predict(_request(2, 2))


def test_a_resent_request_gets_the_same_reply_without_inference() -> None:
    server, processor, model = _server()
    server.reset(XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1"))
    first = server.predict(_request(0, 0))
    calls = len(model.calls)

    again = server.predict(_request(0, 0))  # the reply was lost on the way back

    assert again is first and len(model.calls) == calls
    assert server.history_transition_ids == (0,)
    assert server.predict(_request(1, 1)).request_id == 1
