"""Measure one owned Xiaomi service and fixture prediction without an LLM."""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from robots.robocasa.xiaomi_binding import make_xiaomi_policy_binding
from rpent.policies.lerobot_smolvla_artifacts import binding_identity
from rpent.policies.xiaomi_mibot_catalog import (
    XIAOMI_MIBOT_CHECKPOINT,
    XIAOMI_MIBOT_POLICY_SPEC,
)
from rpent.policies.xiaomi_mibot_daemon import XiaomiMiBoTDaemon
from rpent.recording.media import atomic_write_json


def gpu_memory_used(gpu: int) -> int:
    return int(subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10).strip())


class MemorySampler:
    def __init__(self, gpu, query):
        self.gpu, self.query = gpu, query
        self.values, self.errors = [], []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        try:
            self.values.append(self.query(self.gpu))
        except Exception as error:
            self.errors.append(str(error))

    def _run(self):
        while not self.done.wait(0.1):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def stop(self):
        self.sample()
        self.done.set()
        self.thread.join(timeout=15)


def run_service_probe(*, snapshot, output, physical_device=0, device="cuda:0",
                      daemon_factory=XiaomiMiBoTDaemon, memory_query=gpu_memory_used):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "service-probe.json"
    if destination.exists():
        raise ValueError("preserve existing service probe; choose a new output")
    fixture = Path(__file__).parents[1] / "tests/fixtures/xiaomi_mibot/input_observations.npz"
    with np.load(fixture, allow_pickle=False) as source:
        observation = {key: source[key][-1].copy() for key in ("state", *XIAOMI_MIBOT_POLICY_SPEC.input_camera_ids)}
        instruction = str(source["instruction"].reshape(-1)[-1])
    daemon = daemon_factory(
        checkpoint=XIAOMI_MIBOT_CHECKPOINT, snapshot_path=snapshot,
        binding_identity=binding_identity(make_xiaomi_policy_binding()),
        device=device, physical_device=str(physical_device),
        dtype=XIAOMI_MIBOT_POLICY_SPEC.default_dtype,
        log_path=output / "xiaomi_policy_server.log",
    )
    memory = MemorySampler(physical_device, memory_query)
    result = {"completed": False, "checkpoint": XIAOMI_MIBOT_CHECKPOINT.to_payload(),
              "input_kind": "official_fixture_initial_history_padding", "physical_gpu": physical_device}
    memory.start()
    original = None
    try:
        started = time.monotonic()
        session = daemon.start()
        result["load_time_s"] = time.monotonic() - started
        result["health"] = dict(daemon.check_health())
        session.reset("xiaomi-setup-fixture")
        memory.sample()
        result["loaded_gpu_used_mib"] = memory.values[-1] if memory.values else None
        started = time.monotonic()
        actions = session.predict(observation, instruction,
                                  max_actions=XIAOMI_MIBOT_POLICY_SPEC.default_actions_per_request)
        result["predict_latency_s"] = time.monotonic() - started
        if actions.shape != (XIAOMI_MIBOT_POLICY_SPEC.generated_chunk_size, *XIAOMI_MIBOT_POLICY_SPEC.native_action_shape) or not np.isfinite(actions).all():
            raise ValueError("fixture prediction violated the native action contract")
        result.update(completed=True, action_shape=list(actions.shape), actions=actions.tolist())
        memory.sample()
    except BaseException as error:
        original = error
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        memory.stop()
        result.update(gpu_peak_used_mib=max(memory.values) if memory.values else None,
                      gpu_samples=len(memory.values), gpu_sampling_errors=memory.errors)
        try:
            daemon.stop()
        except Exception as error:
            result["cleanup_error"] = str(error)
            if original is None:
                result["completed"] = False
                atomic_write_json(destination, result)
                raise
            original.add_note(f"probe cleanup failed: {error}")
        atomic_write_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-device", type=int, default=0)
    args = parser.parse_args()
    run_service_probe(snapshot=args.snapshot, output=args.output, physical_device=args.physical_device)


if __name__ == "__main__":
    main()
