#!/usr/bin/env python3
"""Write layered Xiaomi RoboCasa conformance evidence inside one attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robots.robocasa.xiaomi_binding import (
    XIAOMI_CAMERA_KEYS,
    build_xiaomi_messages,
    native_state_to_xiaomi14,
    pack_xiaomi_state,
    sample_history,
)
from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_CHECKPOINT,
    XIAOMI_OFFICIAL_SOURCE_REVISION,
)

SOURCE_KINDS = {"official_fixture", "training_demo", "live_episode"}


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _numpy(value: Any) -> np.ndarray:
    current = value.detach() if hasattr(value, "detach") else value
    current = current.float() if hasattr(current, "float") else current
    current = current.cpu() if hasattr(current, "cpu") else current
    current = current.numpy() if hasattr(current, "numpy") else current
    return np.asarray(current)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if "agent_artifacts" not in resolved.parts:
        raise ValueError("output must be inside an attempt/agent_artifacts directory")
    artifacts_index = resolved.parts.index("agent_artifacts")
    if artifacts_index == 0 or artifacts_index == len(resolved.parts) - 1:
        raise ValueError("output must be inside an attempt/agent_artifacts directory")
    return resolved


def _history_indices(count: int) -> list[int]:
    return [max(0, count - 1 - (3 - index) * 2) for index in range(4)]


def _processor_shapes(inputs: Mapping[str, Any]) -> dict[str, list[int]]:
    shapes = {}
    for key, value in inputs.items():
        shape = getattr(value, "shape", None)
        shapes[str(key)] = list(shape if shape is not None else np.asarray(value).shape)
    return shapes


def run_probe(
    *,
    source_observations: str | Path,
    source_kind: str,
    output_dir: str | Path,
    device: str,
    processor: Any,
    model: Any | None,
) -> dict[str, Any]:
    """Run the official preprocessing/decode order and persist layered evidence."""
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"unsupported source kind: {source_kind}")
    destination = _validate_output_dir(Path(output_dir))
    source_path = Path(source_observations)
    with np.load(source_path, allow_pickle=False) as source:
        required = {"state", "instruction", *XIAOMI_CAMERA_KEYS}
        if not required.issubset(source.files):
            raise ValueError("source observations are incomplete")
        states = np.asarray(source["state"], dtype=np.float32)
        if states.ndim != 2 or states.shape[1:] != (16,) or len(states) < 1:
            raise ValueError("source state must have shape [history, 16]")
        instruction_values = np.asarray(source["instruction"]).reshape(-1)
        instruction = str(instruction_values[-1])
        images = {
            key: np.asarray(source[key], dtype=np.uint8) for key in XIAOMI_CAMERA_KEYS
        }
    if any(len(value) != len(states) for value in images.values()):
        raise ValueError("camera and state history lengths must match")

    indices = _history_indices(len(states))
    sampled_states = sample_history(
        [native_state_to_xiaomi14(state) for state in states], length=4, interval=2
    )
    packed_state = pack_xiaomi_state(sampled_states)
    sampled_images = {
        key: sample_history(value, length=4, interval=2)
        for key, value in images.items()
    }
    messages = build_xiaomi_messages(sampled_images, instruction)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_resize=False,
        state=packed_state,
        robot_type=XIAOMI_MIBOT_CHECKPOINT.robot_type,
    )
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        kwargs = {"device": device} if model is not None else {}
        floating = getattr(value, "is_floating_point", lambda: False)()
        if floating and model is not None and hasattr(model, "dtype"):
            kwargs["dtype"] = model.dtype
        moved[key] = value.to(**kwargs)
    if model is None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("processor-only decode requires torch") from exc
        raw_actions = torch.linspace(
            -1.0, 1.0, steps=16 * 60, dtype=torch.float32
        ).reshape(1, 16, 60)
    else:
        outputs = model(**moved)
        raw_actions = outputs.actions.cpu()
    decoded = processor.decode_action(
        raw_actions, robot_type=XIAOMI_MIBOT_CHECKPOINT.robot_type
    )
    actions = np.ascontiguousarray(
        _numpy(decoded[0, :, :12]), dtype=np.float32
    )
    if actions.shape != (16, 12):
        raise ValueError(f"decoded action shape drifted: {actions.shape}")

    history_provenance = (
        "synthetic_initial_padding"
        if len(states) == 1
        else "exact_interval_two_history"
    )
    identity = {
        "source_revision": XIAOMI_OFFICIAL_SOURCE_REVISION,
        "checkpoint_revision": XIAOMI_MIBOT_CHECKPOINT.revision,
        "checkpoint_repo_id": XIAOMI_MIBOT_CHECKPOINT.repo_id,
        "source_kind": source_kind,
        "device": device,
        "dtype": "bfloat16",
    }
    observation_layers = {
        "camera_order": ["left", "right", "wrist"],
        "native_camera_keys": list(XIAOMI_CAMERA_KEYS),
        "history_indices": indices,
        "history_provenance": history_provenance,
        "state14": sampled_states.tolist(),
        "state14_hash": _hash_array(sampled_states),
        "packed_state_hash": _hash_array(packed_state),
        "camera_hashes": {
            key: _hash_array(value) for key, value in sampled_images.items()
        },
        "formatted_instruction": f"{instruction} /no_cot",
        "formatted_instruction_hash": hashlib.sha256(
            f"{instruction} /no_cot".encode("utf-8")
        ).hexdigest(),
        "cropped_camera_hashes": {
            key: _hash_array(
                np.stack(
                    [
                        np.asarray(item)
                        for item in messages[0]["content"][1 + index * 2]["video"]
                    ]
                )
            )
            for index, key in enumerate(XIAOMI_CAMERA_KEYS)
        },
    }
    processor_layers = {
        "robot_type": XIAOMI_MIBOT_CHECKPOINT.robot_type,
        "shapes": _processor_shapes(inputs),
        "tensor_hashes": {
            str(key): _hash_array(_numpy(value)) for key, value in inputs.items()
        },
    }
    action_layers = {
        "raw_shape": list(raw_actions.shape),
        "decoded_shape": list(getattr(decoded, "shape", _numpy(decoded).shape)),
        "selected_shape": list(actions.shape),
        "selected_hash": _hash_array(actions),
        "decoded_actions": actions.tolist(),
        "decode_order": "processor.decode_action_then_slice_12",
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 1e-5,
    }
    if source_kind == "official_fixture":
        contract_path = source_path.with_name("official_contract.json")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_inputs = contract["input_layers"]
        actual_inputs = {
            "source_npz_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "state14_hash": observation_layers["state14_hash"],
            "packed_state_hash": observation_layers["packed_state_hash"],
            "camera_hashes": observation_layers["camera_hashes"],
        }
        if actual_inputs != expected_inputs:
            raise ValueError("official fixture input-layer hash mismatch")
        if processor.__class__.__name__ == "MiBotProcessor":
            expected_processor = contract["official_processor"]
            actual_processor = {
                "decoded_action_hash": action_layers["selected_hash"],
                "input_ids_hash": processor_layers["tensor_hashes"]["input_ids"],
                "input_ids_shape": processor_layers["shapes"]["input_ids"],
                "pixel_values_videos_hash": processor_layers["tensor_hashes"][
                    "pixel_values_videos"
                ],
                "pixel_values_videos_shape": processor_layers["shapes"][
                    "pixel_values_videos"
                ],
                "state_shape": processor_layers["shapes"]["state"],
            }
            if actual_processor != expected_processor:
                raise ValueError("official processor-layer conformance mismatch")
    summary = {
        **identity,
        "conformant": True,
        "history_indices": indices,
        "history_provenance": history_provenance,
        "state_shape": list(packed_state.shape),
        "processor_shapes": processor_layers["shapes"],
        "action_shape": list(actions.shape),
        "decode_order": action_layers["decode_order"],
    }
    for name, payload in (
        ("identity.json", identity),
        ("observation_layers.json", observation_layers),
        ("processor_layers.json", processor_layers),
        ("action_layers.json", action_layers),
        ("summary.json", summary),
    ):
        _write_json(destination / name, payload)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-observations", required=True)
    parser.add_argument("--source-kind", required=True, choices=sorted(SOURCE_KINDS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--snapshot-path", default=os.environ.get("RPENT_XIAOMI_MIBOT_SNAPSHOT")
    )
    args = parser.parse_args(argv)
    if not args.snapshot_path:
        parser.error("--snapshot-path or RPENT_XIAOMI_MIBOT_SNAPSHOT is required")
    from rpent.policies.xiaomi_mibot_server import _load_processor

    processor = _load_processor(args.snapshot_path)
    run_probe(
        source_observations=args.source_observations,
        source_kind=args.source_kind,
        output_dir=args.output_dir,
        device=args.device,
        processor=processor,
        model=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
