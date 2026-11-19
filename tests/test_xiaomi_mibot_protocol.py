from __future__ import annotations

import numpy as np
import pytest

from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
from rpent.policies.xiaomi_mibot_protocol import (
    XIAOMI_MIBOT_PROTOCOL_VERSION,
    XIAOMI_MIBOT_WIRE_OPERATIONS,
    XiaomiMiBoTCloseRequest,
    XiaomiMiBoTHealthResponse,
    XiaomiMiBoTLoadRequest,
    XiaomiMiBoTPredictRequest,
    XiaomiMiBoTPredictResponse,
    XiaomiMiBoTResetRequest,
    XiaomiMiBoTResetResponse,
)


def _observation(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "state": np.full(16, value, dtype=np.float32),
        "robot0_agentview_left": np.full((8, 8, 3), value, dtype=np.uint8),
        "robot0_agentview_right": np.full((8, 8, 3), value, dtype=np.uint8),
        "robot0_eye_in_hand": np.full((8, 8, 3), value, dtype=np.uint8),
    }


def _predict_request(**overrides) -> XiaomiMiBoTPredictRequest:
    values = {
        "checkpoint": XIAOMI_MIBOT_CHECKPOINT,
        "episode_id": "episode-1",
        "request_id": 0,
        "transition_id": 6,
        "instruction": "KettleBoiling",
        "observation": _observation(),
        "terminal": False,
    }
    values.update(overrides)
    return XiaomiMiBoTPredictRequest(**values)


def test_protocol_version_and_operations_are_stable() -> None:
    assert XIAOMI_MIBOT_PROTOCOL_VERSION == 1
    assert XIAOMI_MIBOT_WIRE_OPERATIONS == (
        "health",
        "load",
        "reset",
        "predict",
        "close",
    )


def test_load_and_health_round_trip_complete_runtime_identity() -> None:
    request = XiaomiMiBoTLoadRequest(
        checkpoint=XIAOMI_MIBOT_CHECKPOINT,
        binding_identity="b" * 64,
        device="cuda:0",
        dtype="bfloat16",
    )
    restored = XiaomiMiBoTLoadRequest.from_payload(request.to_payload())
    assert restored == request

    response = XiaomiMiBoTHealthResponse.from_payload(
        {
            **request.to_payload(),
            "loaded": True,
            "artifact_identity": "a" * 64,
            "remote_code_revision": XIAOMI_MIBOT_CHECKPOINT.revision,
            "server_version": "rpent-xiaomi-mibot/1",
        }
    )
    response.validate_for(request)
    assert response.loaded is True


def test_load_rejects_protocol_checkpoint_and_binding_drift() -> None:
    request = XiaomiMiBoTLoadRequest(
        checkpoint=XIAOMI_MIBOT_CHECKPOINT,
        binding_identity="b" * 64,
        device="cuda:0",
        dtype="bfloat16",
    )
    protocol = request.to_payload()
    protocol["protocol_version"] = 2
    with pytest.raises(ValueError, match="protocol_version"):
        XiaomiMiBoTLoadRequest.from_payload(protocol)
    checkpoint = request.to_payload()
    checkpoint["checkpoint_identity"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint_identity"):
        XiaomiMiBoTLoadRequest.from_payload(checkpoint)

    health = XiaomiMiBoTHealthResponse(
        checkpoint=request.checkpoint,
        checkpoint_identity=request.checkpoint.identity_digest,
        binding_identity="c" * 64,
        artifact_identity="a" * 64,
        remote_code_revision=request.checkpoint.revision,
        device=request.device,
        dtype=request.dtype,
        server_version="rpent-xiaomi-mibot/1",
        loaded=True,
    )
    with pytest.raises(ValueError, match="binding_identity"):
        health.validate_for(request)


def test_reset_response_rejects_episode_mismatch() -> None:
    request = XiaomiMiBoTResetRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1")
    response = XiaomiMiBoTResetResponse(XIAOMI_MIBOT_CHECKPOINT, "episode-2")

    with pytest.raises(ValueError, match="episode_id"):
        response.validate_for(request)


def test_predict_payload_preserves_confirmed_native_observation() -> None:
    request = _predict_request()

    restored = XiaomiMiBoTPredictRequest.from_payload(request.to_payload())

    assert restored.episode_id == "episode-1"
    assert restored.transition_id == 6
    assert restored.instruction == "KettleBoiling"
    assert all(
        np.array_equal(restored.observation[key], request.observation[key])
        for key in request.observation
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(state=np.zeros(15, dtype=np.float32)), "state"),
        (lambda value: value.pop("robot0_eye_in_hand"), "robot0_eye_in_hand"),
        (
            lambda value: value.update(
                robot0_agentview_left=np.zeros((3, 8, 8), dtype=np.uint8)
            ),
            "HWC",
        ),
    ],
)
def test_predict_rejects_malformed_native_observation(mutate, message: str) -> None:
    observation = _observation()
    mutate(observation)
    with pytest.raises(ValueError, match=message):
        _predict_request(observation=observation)


def test_predict_response_is_exact_official_chunk_shape_and_dtype() -> None:
    response = XiaomiMiBoTPredictResponse(
        checkpoint=XIAOMI_MIBOT_CHECKPOINT,
        episode_id="episode-1",
        request_id=0,
        actions=np.zeros((16, 12), dtype=np.float32),
    )

    restored = XiaomiMiBoTPredictResponse.from_payload(response.to_payload())

    assert restored.actions.shape == (16, 12)
    assert restored.actions.dtype == np.float32


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((15, 12), dtype=np.float32),
        np.zeros((16, 60), dtype=np.float32),
        np.zeros((16, 12), dtype=np.float64),
    ],
)
def test_predict_response_rejects_wrong_shape_or_dtype(actions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="actions"):
        XiaomiMiBoTPredictResponse(
            XIAOMI_MIBOT_CHECKPOINT, "episode-1", 0, actions
        )


def test_predict_response_rejects_episode_or_request_mismatch() -> None:
    request = _predict_request()
    response = XiaomiMiBoTPredictResponse(
        XIAOMI_MIBOT_CHECKPOINT,
        "episode-2",
        1,
        np.zeros((16, 12), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="episode_id"):
        response.validate_for(request)


def test_close_request_round_trip_preserves_optional_episode() -> None:
    request = XiaomiMiBoTCloseRequest(XIAOMI_MIBOT_CHECKPOINT, "episode-1")
    assert XiaomiMiBoTCloseRequest.from_payload(request.to_payload()) == request
