"""Isolated official-processor runtime for Xiaomi MiBoT."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from robots.robocasa.xiaomi_binding import (
    XIAOMI_CAMERA_KEYS,
    build_xiaomi_messages,
    native_state_to_xiaomi14,
    pack_xiaomi_state,
    sample_history,
    xiaomi_action_chunk_to_rpent,
)
from rpent.policies.xiaomi_mibot_artifacts import inspect_xiaomi_artifacts
from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_POLICY_SPEC,
    XiaomiMiBoTCheckpoint,
)
from rpent.policies.xiaomi_mibot_protocol import (
    XIAOMI_MIBOT_PROTOCOL_VERSION,
    XiaomiMiBoTCloseRequest,
    XiaomiMiBoTHealthResponse,
    XiaomiMiBoTLoadRequest,
    XiaomiMiBoTPredictRequest,
    XiaomiMiBoTPredictResponse,
    XiaomiMiBoTResetRequest,
    XiaomiMiBoTResetResponse,
)
from rpent.policies.xiaomi_mibot_rng import initialize_episode_rng
from rpent.utils.rpc import RpcFacade

SERVER_VERSION = "rpent-xiaomi-mibot/1"


def _load_processor(path: str | Path) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=False)


def _load_model(path: str | Path, device: str, dtype: str) -> Any:
    import torch
    from transformers import AutoModel

    if dtype != "bfloat16":
        raise ValueError("Xiaomi MiBoT version 1 requires dtype bfloat16")
    return AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    ).to(device=device, dtype=torch.bfloat16)


def _inference_context():
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.inference_mode()


def _request_key(request: XiaomiMiBoTPredictRequest) -> tuple:
    """Identify a resend: same episode, request, transition and buffered updates."""
    return (
        request.episode_id,
        request.request_id,
        request.transition_id,
        request.terminal,
        tuple(int(item["transition_id"]) for item in request.history_updates),
    )


def _same_observation(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return set(left) == set(right) and all(
        np.array_equal(np.asarray(left[key]), np.asarray(right[key])) for key in left
    )


class XiaomiMiBoTServer:
    """Own the pinned processor/model and episode-scoped confirmed history."""

    def __init__(
        self,
        *,
        snapshot_path: str | Path,
        binding_identity: str,
        artifact_loader: Callable[..., Any] = inspect_xiaomi_artifacts,
        processor_loader: Callable[[str | Path], Any] = _load_processor,
        model_loader: Callable[[str | Path, str, str], Any] = _load_model,
        expected_checkpoint: XiaomiMiBoTCheckpoint | None = None,
        expected_device: str | None = None,
        expected_dtype: str | None = None,
    ) -> None:
        self._snapshot_path = Path(snapshot_path)
        self._binding_identity = binding_identity
        self._artifact_loader = artifact_loader
        self._processor_loader = processor_loader
        self._model_loader = model_loader
        self._expected_checkpoint = expected_checkpoint
        self._expected_device = expected_device
        self._expected_dtype = expected_dtype
        self._checkpoint: XiaomiMiBoTCheckpoint | None = None
        self._artifact = None
        self._processor = None
        self._model = None
        self._device: str | None = None
        self._dtype: str | None = None
        self._episode_id: str | None = None
        self._next_request_id = 0
        self._history: list[dict[str, Any]] = []
        self._terminal = False
        self._closed = False
        # The last successful reply, returned again when the client resends the
        # same request after losing the reply (a timeout or broken connection).
        self._last_reply: tuple[tuple, XiaomiMiBoTPredictResponse] | None = None
        self._rng_initializer = initialize_episode_rng

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def history_transition_ids(self) -> tuple[int, ...]:
        return tuple(item["transition_id"] for item in self._history)

    def load(self, request: XiaomiMiBoTLoadRequest) -> XiaomiMiBoTHealthResponse:
        self._require_open()
        if self.is_loaded:
            raise RuntimeError("Xiaomi MiBoT policy is already loaded")
        if request.binding_identity != self._binding_identity:
            raise ValueError("binding_identity does not match server binding")
        if self._expected_checkpoint is not None and request.checkpoint != self._expected_checkpoint:
            raise ValueError("checkpoint does not match configured server identity")
        if self._expected_device is not None and request.device != self._expected_device:
            raise ValueError("device does not match configured server identity")
        if self._expected_dtype is not None and request.dtype != self._expected_dtype:
            raise ValueError("dtype does not match configured server identity")
        artifact = self._artifact_loader(self._snapshot_path, request.checkpoint)
        if artifact.checkpoint != request.checkpoint:
            raise ValueError("artifact checkpoint identity does not match load request")
        processor = self._processor_loader(self._snapshot_path)
        model = self._model_loader(
            self._snapshot_path, request.device, request.dtype
        )
        self._checkpoint = request.checkpoint
        self._artifact = artifact
        self._processor = processor
        self._model = model
        self._device = request.device
        self._dtype = request.dtype
        return self.health()

    def health(self) -> XiaomiMiBoTHealthResponse:
        self._require_open()
        if not self.is_loaded or self._checkpoint is None or self._artifact is None:
            raise RuntimeError("Xiaomi MiBoT policy is not loaded")
        return XiaomiMiBoTHealthResponse(
            checkpoint=self._checkpoint,
            checkpoint_identity=self._checkpoint.identity_digest,
            binding_identity=self._binding_identity,
            artifact_identity=self._artifact.identity_digest,
            remote_code_revision=self._checkpoint.revision,
            device=str(self._device),
            dtype=str(self._dtype),
            server_version=SERVER_VERSION,
            loaded=True,
        )

    def reset(self, request: XiaomiMiBoTResetRequest) -> XiaomiMiBoTResetResponse:
        self._require_loaded_checkpoint(request.checkpoint)
        rng_state = self._rng_initializer(request.seed) if request.seed is not None else None
        self._episode_id = request.episode_id
        self._next_request_id = 0
        self._history = []
        self._terminal = False
        self._last_reply = None
        return XiaomiMiBoTResetResponse(request.checkpoint, request.episode_id,
                                       seed=request.seed, rng_state=rng_state)

    def predict(
        self, request: XiaomiMiBoTPredictRequest
    ) -> XiaomiMiBoTPredictResponse:
        self._require_loaded_checkpoint(request.checkpoint)
        if request.episode_id != self._episode_id:
            raise ValueError("episode_id does not match active policy episode")
        key = _request_key(request)
        if self._last_reply is not None and self._last_reply[0] == key:
            return self._last_reply[1]  # a resend of a request that already succeeded
        if request.request_id != self._next_request_id:
            raise ValueError(
                "request_id does not match next policy request: "
                f"expected {self._next_request_id}, actual {request.request_id}"
            )
        if self._terminal:
            raise RuntimeError("cannot predict after a terminal transition")
        additions = self._validated_additions(request)
        # Commit the new transitions and the request counter only after
        # inference succeeds: a failed attempt (out of memory, a processor
        # error) leaves the session as it was, so the client can resend the
        # same request instead of being refused for the rest of the episode.
        history = [*self._history, *additions]
        if request.terminal or (history and history[-1]["terminal"]):
            self._history = history
            self._terminal = True
            raise RuntimeError("cannot predict from a terminal transition")

        states = sample_history(
            [native_state_to_xiaomi14(item["observation"]["state"]) for item in history],
            length=XIAOMI_MIBOT_POLICY_SPEC.observation_history,
            interval=XIAOMI_MIBOT_POLICY_SPEC.observation_interval,
        )
        images = {
            key: sample_history(
                [item["observation"][key] for item in history],
                length=XIAOMI_MIBOT_POLICY_SPEC.observation_history,
                interval=XIAOMI_MIBOT_POLICY_SPEC.observation_interval,
            )
            for key in XIAOMI_CAMERA_KEYS
        }
        messages = build_xiaomi_messages(images, request.instruction)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            state=pack_xiaomi_state(states),
            robot_type=request.checkpoint.robot_type,
        )
        moved = self._move_inputs(inputs)
        with _inference_context():
            outputs = self._model(**moved)
        raw_actions = outputs.actions.cpu()
        decoded = self._processor.decode_action(
            raw_actions, robot_type=request.checkpoint.robot_type
        )
        selected = decoded[0, :, :12].float().cpu().numpy()
        actions = xiaomi_action_chunk_to_rpent(selected)
        response = XiaomiMiBoTPredictResponse(
            request.checkpoint, request.episode_id, request.request_id, actions
        )
        self._history = history
        self._next_request_id += 1
        self._last_reply = (key, response)
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._processor = None
        self._model = None
        self._artifact = None
        self._checkpoint = None
        self._history = []
        self._last_reply = None
        self._episode_id = None

    def _validated_additions(
        self, request: XiaomiMiBoTPredictRequest
    ) -> list[dict[str, Any]]:
        latest = self._history[-1]["transition_id"] if self._history else -1
        additions = [dict(item) for item in request.history_updates]
        for item in additions:
            expected = latest + 1
            if item["transition_id"] != expected:
                raise ValueError(
                    f"transition history must be consecutive: expected {expected}, "
                    f"actual {item['transition_id']}"
                )
            latest = item["transition_id"]
        if not self._history and not additions:
            if request.transition_id != 0:
                raise ValueError("initial transition_id must be 0")
            additions.append(
                {
                    "transition_id": 0,
                    "observation": dict(request.observation),
                    "terminal": request.terminal,
                }
            )
            latest = 0
        elif request.transition_id == latest + 1:
            additions.append(
                {
                    "transition_id": request.transition_id,
                    "observation": dict(request.observation),
                    "terminal": request.terminal,
                }
            )
            latest = request.transition_id
        elif request.transition_id != latest:
            raise ValueError(
                f"transition_id mismatch: latest {latest}, actual {request.transition_id}"
            )
        if additions and additions[-1]["transition_id"] == request.transition_id:
            if not _same_observation(additions[-1]["observation"], request.observation):
                raise ValueError("current observation does not match latest transition")
            if additions[-1]["terminal"] != request.terminal:
                raise ValueError("current terminal flag does not match latest transition")
        return additions

    def _move_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
                continue
            is_floating = getattr(value, "is_floating_point", lambda: False)()
            kwargs = {"device": self._model.device}
            if is_floating:
                kwargs["dtype"] = self._model.dtype
            moved[key] = value.to(**kwargs)
        return moved

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Xiaomi MiBoT server is closed")

    def _require_loaded_checkpoint(self, checkpoint: Any) -> None:
        self._require_open()
        if not self.is_loaded or self._checkpoint is None:
            raise RuntimeError("Xiaomi MiBoT policy is not loaded")
        if checkpoint != self._checkpoint:
            raise ValueError("checkpoint identity does not match loaded policy")


class _XiaomiMiBoTRpcFacade(RpcFacade):
    def __init__(self, server: XiaomiMiBoTServer) -> None:
        super().__init__()
        self._server = server

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if args:
            raise ValueError("Xiaomi MiBoT RPC accepts keyword arguments only")
        if method == "load":
            return self._server.load(
                XiaomiMiBoTLoadRequest.from_payload(kwargs)
            ).to_payload()
        if method == "health":
            if kwargs:
                raise ValueError("health accepts no arguments")
            return self._server.health().to_payload()
        if method == "reset":
            return self._server.reset(
                XiaomiMiBoTResetRequest.from_payload(kwargs)
            ).to_payload()
        if method == "predict":
            return self._server.predict(
                XiaomiMiBoTPredictRequest.from_payload(kwargs)
            ).to_payload()
        if method == "close":
            XiaomiMiBoTCloseRequest.from_payload(kwargs)
            self._server.close()
            return {"ok": True, "protocol_version": XIAOMI_MIBOT_PROTOCOL_VERSION}
        raise ValueError(f"unknown Xiaomi MiBoT RPC method: {method!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("http", "socket"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--binding-identity", required=True)
    parser.add_argument("--checkpoint-repo-id", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--binding-name", required=True)
    parser.add_argument("--robot-type", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    args = parser.parse_args()
    server = XiaomiMiBoTServer(
        snapshot_path=args.snapshot_path,
        binding_identity=args.binding_identity,
        expected_checkpoint=XiaomiMiBoTCheckpoint(
            args.checkpoint_repo_id,
            args.checkpoint_revision,
            args.binding_name,
            args.robot_type,
        ),
        expected_device=args.device,
        expected_dtype=args.dtype,
    )
    try:
        _XiaomiMiBoTRpcFacade(server).serve(
            transport=args.transport, host=args.host, port=args.port
        )
    finally:
        server.close()


__all__ = ["SERVER_VERSION", "XiaomiMiBoTServer"]


if __name__ == "__main__":
    main()
