"""Explicit episode noise seeding and verifiable initial Torch RNG snapshots."""
from __future__ import annotations

import base64
import hashlib
import json
import random
from collections.abc import Mapping
from typing import Any

import numpy as np


def _blob(value: bytes) -> dict[str, str]:
    return {"base64": base64.b64encode(value).decode("ascii"),
            "sha256": hashlib.sha256(value).hexdigest()}


def initialize_episode_rng(seed: int) -> dict[str, Any]:
    """Match the historical evaluator's reset, after model loading has finished."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy_state = np.random.get_state()
    return {
        "schema_version": 1, "seed": seed, "torch_initial_seed": torch.initial_seed(),
        "python_state_sha256": hashlib.sha256(json.dumps(random.getstate()).encode()).hexdigest(),
        "numpy_state_sha256": hashlib.sha256(json.dumps([
            numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]
        ]).encode()).hexdigest(),
        "torch_cpu_state": _blob(torch.get_rng_state().cpu().numpy().tobytes()),
        "torch_cuda_states": [_blob(state.cpu().numpy().tobytes())
                              for state in torch.cuda.get_rng_state_all()],
    }


def validate_rng_evidence(evidence: Any, seed: int) -> None:
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != 1:
        raise ValueError("RNG evidence is missing or unsupported")
    if evidence.get("seed") != seed or evidence.get("torch_initial_seed") != seed:
        raise ValueError("RNG seed does not match the requested episode seed")
    states = evidence.get("torch_cuda_states")
    if not isinstance(states, list):
        raise ValueError("RNG CUDA states must be a list")
    for blob in [evidence.get("torch_cpu_state"), *states]:
        if not isinstance(blob, Mapping):
            raise ValueError("RNG state bytes are missing")
        try:
            raw = base64.b64decode(blob["base64"], validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("RNG state bytes are invalid") from exc
        if not raw or hashlib.sha256(raw).hexdigest() != blob.get("sha256"):
            raise ValueError("RNG state checksum differs")
