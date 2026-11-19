"""Episode-scoped client for the isolated Xiaomi MiBoT service."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from rpent.episode.contracts import StepResult
from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_POLICY_SPEC,
    XiaomiMiBoTCheckpoint,
)
from rpent.policies.xiaomi_mibot_protocol import (
    XiaomiMiBoTCloseRequest,
    XiaomiMiBoTHealthResponse,
    XiaomiMiBoTLoadRequest,
    XiaomiMiBoTPredictRequest,
    XiaomiMiBoTPredictResponse,
    XiaomiMiBoTResetRequest,
    XiaomiMiBoTResetResponse,
)
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcClient
from rpent.utils.socket_rpc import RpcError

logger = get_logger("xiaomi_mibot_client")

# A predict can fail transiently (GPU memory, a slow reply after a long LLM
# segment carries hundreds of buffered frames). The server commits history only
# after inference and replays its last reply for a resent request, so the same
# request is resent instead of losing the policy for the rest of the episode.
PREDICT_TIMEOUT_S = 120.0
PREDICT_RETRY_DELAYS_S = (1.0, 5.0)

_POLICY_OBSERVATION_KEYS = frozenset(
    {
        "state",
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    }
)


def _policy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in observation.items()
        if key in _POLICY_OBSERVATION_KEYS
    }


class _XiaomiTransitionObserver:
    def __init__(self, session: XiaomiMiBoTPolicySession) -> None:
        self._session = session

    def confirmed(self, action: np.ndarray, result: StepResult) -> None:
        self._session.confirmed(action, result)

    def failed(self, action: np.ndarray, action_index: int, error: Exception) -> None:
        self._session.failed(action, action_index, error)


class XiaomiMiBoTPolicySession:
    """Buffer confirmed transitions and request official 16-action chunks."""

    def __init__(
        self,
        client: RpcClient,
        *,
        checkpoint: XiaomiMiBoTCheckpoint,
        binding_identity: str | None = None,
        close_remote: bool = True,
        policy_seed: int | None = None,
        rng_journal: str | Path | None = None,
    ) -> None:
        self._client = client
        self.checkpoint = checkpoint
        self.binding_identity = binding_identity or hashlib.sha256(
            checkpoint.binding_name.encode("utf-8")
        ).hexdigest()
        self.close_remote = bool(close_remote)
        self.policy_seed = policy_seed
        self.rng_journal = Path(rng_journal) if rng_journal is not None else None
        self.rng_state = None
        self._episode_id: str | None = None
        self._next_request_id = 0
        self._latest_transition_id = 0
        self._history_updates: list[dict[str, Any]] = []
        self._terminal = False
        self._closed = False
        self.predict_retries = 0
        self.transition_observer = _XiaomiTransitionObserver(self)

    def load(
        self, *, device: str, dtype: str, timeout_s: float | None = None
    ) -> XiaomiMiBoTHealthResponse:
        """Load and verify the pinned service identity."""
        self._require_open()
        request = XiaomiMiBoTLoadRequest(
            self.checkpoint, self.binding_identity, device, dtype
        )
        payload = self._client.call(
            "load", kwargs=request.to_payload(), timeout_s=timeout_s
        )
        if not isinstance(payload, Mapping):
            raise ValueError("load response must be a mapping")
        response = XiaomiMiBoTHealthResponse.from_payload(payload)
        response.validate_for(request)
        return response

    def reset(self, episode_id: str) -> None:
        """Clear all local and remote episode history."""
        self._require_open()
        request = XiaomiMiBoTResetRequest(self.checkpoint, episode_id,
                                         seed=self.policy_seed if episode_id != "__rpent_idle__" else None)
        payload = self._client.call("reset", kwargs=request.to_payload())
        if not isinstance(payload, Mapping):
            raise ValueError("reset response must be a mapping")
        response = XiaomiMiBoTResetResponse.from_payload(payload)
        response.validate_for(request)
        self.rng_state = response.rng_state
        if request.seed is not None and self.rng_journal is not None:
            self.rng_journal.parent.mkdir(parents=True, exist_ok=True)
            record = {"episode_id": episode_id, "policy_seed": request.seed,
                      "checkpoint": self.checkpoint.to_payload(), "rng_state": response.rng_state}
            descriptor = os.open(self.rng_journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
                if os.write(descriptor, data) != len(data):
                    raise OSError("incomplete Xiaomi RNG journal write")
            finally:
                os.close(descriptor)
        self._episode_id = episode_id
        self._next_request_id = 0
        self._latest_transition_id = 0
        self._history_updates.clear()
        self._terminal = False

    def initialize_observation(self, observation: Mapping[str, Any]) -> None:
        """Retain transition zero before direct control can precede prediction."""
        self._require_open()
        if self._episode_id is None or self._latest_transition_id or self._next_request_id or self._history_updates:
            raise RuntimeError("initial observation must be registered once immediately after reset")
        self._history_updates.append({
            "transition_id": 0, "terminal": False,
            "observation": {key: np.array(value, copy=True)
                            for key, value in _policy_observation(observation).items()},
        })

    def predict(
        self,
        observation: Mapping[str, Any],
        instruction: str,
        *,
        max_actions: int | None = None,
    ) -> np.ndarray:
        """Send all unseen confirmed transitions and return exactly 16 actions."""
        self._require_open()
        if self._episode_id is None:
            raise RuntimeError("Xiaomi MiBoT policy session must be reset before prediction")
        if self._terminal:
            raise RuntimeError("cannot predict after a terminal transition")
        if max_actions is not None and (
            isinstance(max_actions, bool) or not isinstance(max_actions, int)
            or not 1 <= max_actions <= XIAOMI_MIBOT_POLICY_SPEC.generated_chunk_size
        ):
            raise ValueError("max_actions must be absent or an integer in 1..16")
        request = XiaomiMiBoTPredictRequest(
            checkpoint=self.checkpoint,
            episode_id=self._episode_id,
            request_id=self._next_request_id,
            transition_id=self._latest_transition_id,
            instruction=instruction,
            observation=_policy_observation(observation),
            terminal=False,
            history_updates=tuple(self._history_updates),
        )
        payload = self._predict_with_retry(request)
        if not isinstance(payload, Mapping):
            raise ValueError("predict response must be a mapping")
        response = XiaomiMiBoTPredictResponse.from_payload(payload)
        response.validate_for(request)
        self._history_updates.clear()
        self._next_request_id += 1
        return response.actions[:max_actions].copy()

    def _predict_with_retry(self, request: XiaomiMiBoTPredictRequest) -> Any:
        """Send one request, resending it unchanged after a transport or server error."""
        payload = request.to_payload()
        for attempt, delay in enumerate((*PREDICT_RETRY_DELAYS_S, None)):
            try:
                return self._client.call("predict", kwargs=payload, timeout_s=PREDICT_TIMEOUT_S)
            except RpcError as exc:
                if delay is None:
                    raise
                self.predict_retries += 1
                logger.warning(
                    "Xiaomi predict request %s failed (attempt %d): %s; resending in %.0f s",
                    request.request_id, attempt + 1, exc, delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def confirmed(self, action: np.ndarray, result: StepResult) -> None:
        """Buffer one simulator-confirmed transition for the next request."""
        del action
        self._require_open()
        if self._episode_id is None:
            raise RuntimeError("cannot confirm a transition before reset")
        if result.episode_id != self._episode_id:
            raise ValueError("confirmed transition episode_id does not match session")
        expected = self._latest_transition_id + 1
        if result.step_id != expected:
            raise ValueError(
                f"confirmed transition step_id mismatch: expected {expected}, "
                f"actual {result.step_id}"
            )
        self._history_updates.append(
            {
                "transition_id": result.step_id,
                "observation": _policy_observation(result.observation),
                "terminal": result.terminal,
            }
        )
        self._latest_transition_id = result.step_id
        self._terminal = result.terminal

    def failed(self, action: np.ndarray, action_index: int, error: Exception) -> None:
        """Ignore unconfirmed action attempts; they cannot enter model history."""
        del action, action_index, error

    def close(self) -> None:
        """Close remote ownership and transport exactly once."""
        if self._closed:
            return
        self._closed = True
        error = None
        if self.close_remote:
            try:
                request = XiaomiMiBoTCloseRequest(self.checkpoint, self._episode_id)
                self._client.call("close", kwargs=request.to_payload())
            except Exception as exc:
                error = exc
        try:
            self._client.close()
        except Exception as exc:
            error = error or exc
        if error is not None:
            raise error

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Xiaomi MiBoT policy session is closed")


__all__ = ["XiaomiMiBoTPolicySession"]
