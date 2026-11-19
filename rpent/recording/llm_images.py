"""Retain exact model-bound images, with an immutable record for each request.

Images are content-addressed and are never resized. Gateway capture runs on
the final HTTP body, after its native-thread history selection. Other API
models capture the selected pydantic model input after image pruning.
"""

from __future__ import annotations

import base64
import dataclasses
import fcntl
import hashlib
import io
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image
from pydantic_ai import BinaryContent

from rpent.recording.media import atomic_write_bytes

logger = logging.getLogger(__name__)
_CAMERA = re.compile(r"Image\s+\d+\s+camera:\s*([\w.-]+)")
_STEP = re.compile(r'"(?:step|step_id)"\s*:\s*(\d+)')


def _context(text: str, previous_step: int | None) -> tuple[list[str], int | None]:
    steps = _STEP.findall(text)
    return _CAMERA.findall(text), int(steps[-1]) if steps else previous_step


class LLMImageArchive:
    """Write before sending; storage errors prevent an unrecorded request."""

    def __init__(self, root: str | Path, *, recorder: Any = None):
        self.root = Path(root)
        self.recorder = recorder
        self._contexts: dict[
            tuple[str, ...], tuple[list[str | None], int | None, list[dict[str, Any] | None]]
        ] = {}
        self._context_lock = threading.Lock()

    def note_tool_result(self, result: Any) -> None:
        """Remember source camera metadata without storing an unseen image.

        Older projectors omit camera labels from model-visible text. Join the
        exact ordered image hashes to their environment-owned camera fields;
        do not add text or images to the model's request.
        """
        source = getattr(result, "result", None)
        blocks = getattr(result, "content_blocks", ())
        if not isinstance(source, dict) or not isinstance(source.get("_image_camera_ids"), dict):
            return
        fields = ["_image_bytes", "_image_cam_bytes", "_image_right_bytes", "_image_wrist_bytes"]
        derivations = source.get("_image_derivation")
        derivations = derivations if isinstance(derivations, dict) else {}
        by_hash: dict[str, list[tuple[str, dict[str, Any] | None]]] = {}
        assignments = sorted(source["_image_camera_ids"].items(),
                             key=lambda pair: fields.index(pair[1]) if pair[1] in fields else len(fields))
        for camera, field in assignments:
            raw = source.get(field) if isinstance(field, str) else None
            if isinstance(camera, str) and isinstance(raw, bytes):
                derivation = derivations.get(field)
                by_hash.setdefault(hashlib.sha256(raw).hexdigest(), []).append(
                    (camera, dict(derivation) if isinstance(derivation, dict) else None)
                )
        hashes, cameras, derived = [], [], []
        for block in blocks:
            if block.get("type") != "image":
                continue
            raw = base64.b64decode(block["source"]["data"], validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            matches = by_hash.get(digest, [])
            camera, derivation = matches.pop(0) if matches else (None, None)
            hashes.append(digest)
            cameras.append(camera)
            derived.append(derivation)
        if hashes:
            text = "\n".join(p.get("text", "") for p in blocks if p.get("type") == "text")
            _, step = _context(text, None)
            with self._context_lock:
                self._contexts[tuple(hashes)] = (cameras, step, derived)

    def _restore_context(self, images: list[dict[str, Any]]) -> None:
        with self._context_lock:
            context = self._contexts.get(tuple(image["sha256"] for image in images))
        if context is None:
            return
        cameras, step, derived = context
        for image, camera, derivation in zip(images, cameras, derived, strict=True):
            if image.get("camera_id") is None:
                image["camera_id"] = camera
            if derivation is not None:
                # A crop is not a camera frame: audits check it against its source.
                image["derivation"] = derivation
                if image.get("observation_step") is None:
                    image["observation_step"] = derivation.get("source_step")
            if image.get("observation_step") is None:
                image["observation_step"] = step

    def _write_image(self, raw: bytes, media_type: str) -> dict[str, Any]:
        suffix = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(media_type)
        if suffix is None:
            raise ValueError(f"unsupported archived image type: {media_type}")
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            image.verify()
        sha256 = hashlib.sha256(raw).hexdigest()
        relative = f"llm_inputs/images/{sha256}.{suffix}"
        path = self.root / relative
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
                raise OSError("archived image hash mismatch")
        else:
            atomic_write_bytes(path, raw)
        return {"relative_path": relative, "sha256": sha256, "size_bytes": len(raw),
                "media_type": media_type, "width": width, "height": height}

    def _append(self, name: str, record: dict[str, Any]) -> None:
        path = self.root / "llm_inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Nested runtimes can share an artifact directory and use different
        # archive instances. Lock the journal itself across threads/processes.
        with path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _commit(self, *, model: str, stage: str, snapshot: Any,
                images: list[dict[str, Any]], session_id: str | None = None) -> str:
        request_id = uuid4().hex
        request_file = f"llm_inputs/requests/{request_id}.json"
        record = {"schema_version": 1, "request_id": request_id,
                  "recorded_at": datetime.now(timezone.utc).isoformat(),
                  "model": model, "capture_stage": stage,
                  "status": "prepared_for_send", "session_id": session_id,
                  "images": images, "request_file": request_file}
        atomic_write_bytes(self.root / request_file,
                           (json.dumps({**record, "input": snapshot}, ensure_ascii=False,
                                       indent=2, allow_nan=False) + "\n").encode())
        self._append("requests.jsonl", record)
        if self.recorder is not None:
            try:
                self.recorder.record("llm_input_archived", archive_root=str(self.root), **record)
            except Exception as exc:
                # The dedicated fsynced journal remains authoritative.
                logger.warning("Image archive event unavailable (%s)", type(exc).__name__)
        return request_id

    async def on_request(self, request: Any) -> None:
        if request.method != "POST" or not request.url.path.rstrip("/").endswith("/chat/completions"):
            return
        body = json.loads(await request.aread())
        images: list[dict[str, Any]] = []
        step = None
        pending_cameras: list[str] = []
        for message_index, message in enumerate(body.get("messages", [])):
            content = message.get("content")
            if isinstance(content, str):
                found, step = _context(content, step)
                if found:
                    pending_cameras = found
                continue
            if not isinstance(content, list):
                continue
            text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
            cameras, step = _context(text, step)
            cameras = cameras or pending_cameras
            image_index = 0
            first_image = len(images)
            for part_index, part in enumerate(content):
                if part.get("type") != "image_url":
                    continue
                url = part["image_url"]["url"]
                if not url.startswith("data:") or ";base64," not in url:
                    raise ValueError("model image must contain inline bytes to retain exact input")
                prefix, encoded = url.split(";base64,", 1)
                image = self._write_image(base64.b64decode(encoded, validate=True), prefix.removeprefix("data:"))
                image.update(camera_id=cameras[image_index] if image_index < len(cameras) else None,
                             observation_step=step, message_index=message_index, part_index=part_index,
                             image_index=image_index)
                images.append(image)
                image_index += 1
                # Snapshot preserves every text/tool field; replace only image
                # bytes with a resolvable lossless reference. Never store headers.
                part["image_url"]["url"] = {"archive_reference": image["relative_path"], "sha256": image["sha256"]}
            if image_index:
                self._restore_context(images[first_image:])
                pending_cameras = []
        request.extensions["rpent_llm_request_id"] = self._commit(
            model=str(body.get("model", "")), stage="http_request_body", snapshot=body,
            images=images, session_id=request.headers.get("X-Codex-Session-ID"))

    async def on_response(self, response: Any) -> None:
        request_id = response.request.extensions.get("rpent_llm_request_id")
        if request_id is not None:
            self._append("responses.jsonl", {"schema_version": 1, "request_id": request_id,
                                             "recorded_at": datetime.now(timezone.utc).isoformat(),
                                             "status": "http_response_received", "status_code": response.status_code})

    def capture_tool_content(self, content: list[dict[str, Any]], *, model: str,
                             tool_name: str | None, tool_call_id: str | None) -> None:
        """Retain exact image blocks actually returned to a native SDK client.

        SDK-owned request histories are opaque; this stage is explicitly a
        tool response, rather than a claim about the provider's final context.
        """
        if not any(part.get("type") == "image" for part in content):
            return
        snapshot = json.loads(json.dumps(content))
        text = "\n".join(part.get("text", "") for part in snapshot if part.get("type") == "text")
        cameras, step = _context(text, None)
        images = []
        for part_index, part in enumerate(snapshot):
            if part.get("type") != "image":
                continue
            image = self._write_image(base64.b64decode(part.pop("data"), validate=True), part["mimeType"])
            image.update(camera_id=cameras[len(images)] if len(images) < len(cameras) else None,
                         observation_step=step, part_index=part_index, image_index=len(images))
            images.append(image)
            part["archive_reference"] = image["relative_path"]
            part["sha256"] = image["sha256"]
        self._restore_context(images)
        self._commit(model=model, stage="mcp_tool_response", images=images,
                     snapshot={"tool_name": tool_name, "tool_call_id": tool_call_id, "content": snapshot})

    def capture_model_request(self, messages: Any, *, model: str) -> None:
        images: list[dict[str, Any]] = []
        step = None
        cameras: list[str] = []
        image_index = 0

        def convert(value: Any, location: str):
            nonlocal step, cameras, image_index
            if isinstance(value, BinaryContent):
                image = self._write_image(value.data, value.media_type)
                image.update(camera_id=cameras[image_index] if image_index < len(cameras) else None,
                             observation_step=step, image_index=image_index, location=location)
                images.append(image)
                image_index += 1
                return {"archive_reference": image["relative_path"], "sha256": image["sha256"]}
            if isinstance(value, (list, tuple)):
                text = "\n".join(v for v in value if isinstance(v, str))
                found, step = _context(text, step)
                if found or any(isinstance(v, BinaryContent) for v in value):
                    cameras, image_index = found or cameras, 0
                first_image = len(images)
                converted = [convert(child, f"{location}/{i}") for i, child in enumerate(value)]
                self._restore_context(images[first_image:])
                return converted
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                return {field.name: convert(getattr(value, field.name), f"{location}/{field.name}")
                        for field in dataclasses.fields(value)}
            if isinstance(value, dict):
                return {str(k): convert(v, f"{location}/{k}") for k, v in value.items()}
            if isinstance(value, str):
                if value == "[earlier camera image omitted to bound request size]":
                    image_index += 1
                found, step = _context(value, step)
                if found:
                    cameras, image_index = found, 0
                return value
            if isinstance(value, datetime):
                return value.isoformat()
            if value is None or isinstance(value, (int, float, bool)):
                return value
            return str(value)

        snapshot = convert(messages, "messages")
        self._commit(model=model, stage="provider_model_input", snapshot=snapshot, images=images)
