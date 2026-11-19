"""GPU/host admission and isolated episode lanes sharing Pi0.5 weights."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from rpent.policies.xiaomi_mibot_catalog import XIAOMI_REPLICA_MEMORY_MIB


@dataclass(frozen=True)
class PolicyLane:
    endpoint: str
    environment: Mapping[str, str]
    session_capacity: int = 1


def host_available_memory_mib():
    """Linux MemAvailable already accounts for reclaimable cache."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("host available memory is unknown")


def allocate_replicas(
    free_memory_mib: Mapping[int, int],
    *,
    jobs: int,
    replica_memory_mib: int = 11264,
    reserve_mib: int = 1536,
) -> tuple[int, ...]:
    """Reserve model + simulator memory, then round-robin within GPU capacity.

    The default 11 GiB per replica includes the simulator, in addition to a
    per-GPU reserve. This is admission control, not an allocator hard limit.
    """
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if replica_memory_mib <= 0 or reserve_mib < 0:
        raise ValueError("invalid GPU memory budget")
    capacity = {
        gpu: max(0, (free - reserve_mib) // replica_memory_mib)
        for gpu, free in free_memory_mib.items()
    }
    if sum(capacity.values()) < jobs:
        raise ValueError(f"GPU capacity {capacity} cannot admit {jobs} replicas")
    selected = []
    while len(selected) < jobs:
        for gpu in capacity:
            if capacity[gpu] and len(selected) < jobs:
                selected.append(gpu)
                capacity[gpu] -= 1
    return tuple(selected)


def gpu_free_memory(devices: tuple[int, ...]) -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    available = {
        int(index.strip()): int(free.strip())
        for index, free in (line.split(",") for line in output.splitlines())
    }
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("GPU device list must be nonempty and unique")
    if any(device not in available for device in devices):
        raise ValueError("requested physical GPU is unavailable")
    return {device: available[device] for device in devices}


class Pi05ReplicaPool:
    """Persistent replicas with bounded independently seeded sessions."""

    def __init__(
        self,
        *,
        jobs,
        devices,
        output_dir,
        seed=0,
        replica_memory_mib=11264,
        reserve_mib=1536,
        memory_query=gpu_free_memory,
        daemon_factory=None,
        sessions_per_replica=1,
        host_memory_query=host_available_memory_mib,
    ):
        self.jobs, self.devices = jobs, tuple(devices)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.replica_memory_mib = replica_memory_mib
        self.reserve_mib = reserve_mib
        self.memory_query = memory_query
        self.daemon_factory = daemon_factory
        if (isinstance(sessions_per_replica, bool)
                or not isinstance(sessions_per_replica, int) or sessions_per_replica < 1):
            raise ValueError("sessions_per_replica must be a positive integer")
        self.sessions_per_replica = sessions_per_replica
        self.host_memory_query = host_memory_query
        self.daemons = []
        self.lanes = ()
        self.endpoint = ""

    def start(self):
        if self.lanes:
            return
        from rpent.policies.robocasa_pi05_daemon import OpenPiRoboCasaDaemon
        from rpent.recording.media import atomic_write_json

        factory = self.daemon_factory or OpenPiRoboCasaDaemon
        free = self.memory_query(self.devices)
        replicas = (self.jobs + self.sessions_per_replica - 1) // self.sessions_per_replica
        host_free = self.host_memory_query()
        # Measured SDS footprints: 4 GiB per model and per simulator,
        # plus 2 GiB for coordinator, planning workers and transient buffers.
        host_required = (replicas + self.jobs) * 4096 + 2048
        if host_free < host_required:
            raise ValueError(f"host memory: need {host_required} MiB, available {host_free} MiB")
        placement = allocate_replicas(
            free,
            jobs=replicas,
            replica_memory_mib=self.replica_memory_mib + (self.sessions_per_replica - 1) * 3072,
            reserve_mib=self.reserve_mib,
        )
        try:
            # Stagger checkpoint loading to bound host RAM and CUDA startup peaks.
            for index, gpu in enumerate(placement):
                print(f"Pi0.5 lane {index}: loading on physical GPU {gpu}", flush=True)
                daemon = factory(
                    device="cuda:0",
                    physical_device=str(gpu),
                    seed=self.seed,
                    log_path=self.output_dir / f"pi05_policy_lane_{index}.log",
                    max_episodes=self.sessions_per_replica,
                )
                self.daemons.append(daemon)
                daemon.start()
                warmup = getattr(daemon, "warmup", None)
                if callable(warmup):
                    warmup()
                print(f"Pi0.5 lane {index}: inference warmup ready", flush=True)
            lanes = []
            for index in range(self.jobs):
                replica = index % replicas
                gpu = placement[replica]
                lanes.append(
                    PolicyLane(
                        self.daemons[replica].endpoint,
                        {
                            "CUDA_VISIBLE_DEVICES": str(gpu),
                            "MUJOCO_EGL_DEVICE_ID": str(gpu),
                            "RPENT_EVALUATION_LANE": str(index),
                            "RPENT_PI05_SESSION_SLOT": str(index // replicas),
                        },
                        session_capacity=self.sessions_per_replica,
                    )
                )
            self.lanes = tuple(lanes)
            self.endpoint = self.lanes[0].endpoint
            atomic_write_json(
                self.output_dir / "parallel_runtime.json",
                {
                    "jobs": self.jobs,
                    "replicas": replicas,
                    "sessions_per_replica": self.sessions_per_replica,
                    "host_available_before_mib": host_free,
                    "host_required_mib": host_required,
                    "free_memory_before_mib": free,
                    "replica_memory_mib": self.replica_memory_mib,
                    "gpu_reserve_mib": self.reserve_mib,
                    "lanes": [
                        {"gpu": int(lane.environment["CUDA_VISIBLE_DEVICES"]),
                         "endpoint": lane.endpoint,
                         "slot": lane.environment["RPENT_PI05_SESSION_SLOT"]}
                        for lane in self.lanes
                    ],
                },
            )
        except BaseException as exc:
            try:
                self.stop()
            except Exception as cleanup_error:
                exc.add_note(f"pool cleanup also failed: {type(cleanup_error).__name__}")
            raise

    def stop(self):
        errors = []
        for daemon in reversed(self.daemons):
            try:
                daemon.stop()
            except Exception as exc:
                errors.append(exc)
        self.daemons.clear()
        self.lanes = ()
        if errors:
            raise RuntimeError("Pi0.5 replica cleanup failed") from errors[0]


class XiaomiReplicaPool:
    """One stateful Xiaomi service and one isolated episode per physical GPU."""

    sessions_per_replica = 1

    def __init__(self, *, jobs, devices, output_dir,
                 replica_memory_mib=XIAOMI_REPLICA_MEMORY_MIB, reserve_mib=1536,
                 dtype="bfloat16", memory_query=gpu_free_memory, daemon_factory=None,
                 host_memory_query=host_available_memory_mib):
        self.jobs, self.devices = jobs, tuple(devices)
        self.output_dir = Path(output_dir)
        self.replica_memory_mib, self.reserve_mib = replica_memory_mib, reserve_mib
        self.dtype = dtype
        self.memory_query, self.host_memory_query = memory_query, host_memory_query
        self.daemon_factory = daemon_factory
        self.daemons, self.lanes, self.endpoint = [], (), ""

    def start(self):
        if self.lanes:
            return
        import os

        from robots.robocasa.xiaomi_binding import make_xiaomi_policy_binding
        from rpent.policies.lerobot_smolvla_artifacts import binding_identity
        from rpent.policies.xiaomi_mibot_catalog import XIAOMI_MIBOT_CHECKPOINT
        from rpent.policies.xiaomi_mibot_daemon import XiaomiMiBoTDaemon
        from rpent.recording.media import atomic_write_json

        snapshot = os.environ.get("RPENT_XIAOMI_MIBOT_SNAPSHOT")
        if not snapshot:
            raise RuntimeError("RPENT_XIAOMI_MIBOT_SNAPSHOT must point to the pinned snapshot")
        free = self.memory_query(self.devices)
        # Cap each GPU at one replica, even if its memory could fit two.
        placement = allocate_replicas(
            {gpu: min(value, self.replica_memory_mib + self.reserve_mib)
             for gpu, value in free.items()},
            jobs=self.jobs, replica_memory_mib=self.replica_memory_mib,
            reserve_mib=self.reserve_mib,
        )
        host_free = self.host_memory_query()
        host_required = self.jobs * (6144 + 4096) + 2048
        if host_free < host_required:
            raise ValueError(f"host memory: need {host_required} MiB, available {host_free} MiB")
        factory = self.daemon_factory or XiaomiMiBoTDaemon
        try:
            for index, gpu in enumerate(placement):
                daemon = factory(
                    checkpoint=XIAOMI_MIBOT_CHECKPOINT, snapshot_path=snapshot,
                    binding_identity=binding_identity(make_xiaomi_policy_binding()),
                    device="cuda:0", physical_device=str(gpu), dtype=self.dtype,
                    log_path=self.output_dir / f"xiaomi_policy_lane_{index}.log",
                )
                self.daemons.append(daemon)
                daemon.start()
            self.lanes = tuple(PolicyLane(daemon.endpoint, {
                "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_EGL_DEVICE_ID": str(gpu),
                "RPENT_EVALUATION_LANE": str(index),
                "RPENT_SHARED_POLICY_LOADED": "1",
            }) for index, (daemon, gpu) in enumerate(zip(self.daemons, placement)))
            self.endpoint = self.lanes[0].endpoint
            atomic_write_json(self.output_dir / "parallel_runtime.json", {
                "policy_family": "xiaomi-mibot", "jobs": self.jobs,
                "replicas": self.jobs, "sessions_per_replica": 1,
                "replica_memory_mib": self.replica_memory_mib,
                "gpu_reserve_mib": self.reserve_mib,
                "host_available_before_mib": host_free, "host_required_mib": host_required,
                "free_memory_before_mib": free,
                "lanes": [{"gpu": gpu, "endpoint": daemon.endpoint}
                          for daemon, gpu in zip(self.daemons, placement)],
            })
        except BaseException as exc:
            try:
                self.stop()
            except Exception as cleanup_error:
                exc.add_note(f"pool cleanup also failed: {type(cleanup_error).__name__}")
            raise

    def stop(self):
        errors = []
        for daemon in reversed(self.daemons):
            try:
                daemon.stop()
            except Exception as exc:
                errors.append(exc)
        self.daemons.clear()
        self.lanes = ()
        if errors:
            raise RuntimeError("Xiaomi replica cleanup failed") from errors[0]
