"""Bounded episode concurrency with exclusive policy replicas and GPU budgets."""

import json
import threading
from types import SimpleNamespace

import pytest
from test_episode_coordinator import _coordinator, _manifest, _Process


def test_xiaomi_replica_budget_stays_with_coordinator():
    from rpent.evaluation.front_controller import _worker_base_argv

    assert _worker_base_argv(["--xiaomi-replica-memory-mib", "14336", "--vla", "xiaomi-mibot"]) == ["--vla", "xiaomi-mibot"]


def test_xiaomi_pool_uses_two_exclusive_replicas(tmp_path, monkeypatch):
    from rpent.evaluation.parallel import XiaomiReplicaPool

    created = []
    class Daemon:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.endpoint = f"http://127.0.0.1:{19000 + len(created)}"
            created.append(self)
        def start(self):
            return None
        def stop(self):
            return None

    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_SNAPSHOT", "/pinned/snapshot")
    pool = XiaomiReplicaPool(jobs=2, devices=(0, 1), output_dir=tmp_path,
                             memory_query=lambda devices: {0: 24564, 1: 24564},
                             host_memory_query=lambda: 65536, daemon_factory=Daemon)
    pool.start()
    try:
        assert len(created) == len(pool.lanes) == 2
        assert {d.options["physical_device"] for d in created} == {"0", "1"}
        assert len({lane.endpoint for lane in pool.lanes}) == 2
        assert all(lane.environment["RPENT_SHARED_POLICY_LOADED"] == "1" for lane in pool.lanes)
        assert all(lane.session_capacity == 1 for lane in pool.lanes)
        assert all("RPENT_PI05_SESSION_SLOT" not in lane.environment for lane in pool.lanes)
    finally:
        pool.stop()


@pytest.mark.parametrize("family,accepted", [("xiaomi-mibot", True), ("smolvla", False)])
def test_parallel_family_admission(tmp_path, family, accepted):
    from rpent.cli.main import _build_argparser
    from rpent.envs import get_env_spec
    from rpent.evaluation.front_controller import build_evaluation_manifest

    spec = get_env_spec("robocasa")
    parser = _build_argparser()
    spec.add_cli_args(parser, use_dashboard=False)
    argv = ["--env", "robocasa", "--task", "CloseDrawer", "--vla", family,
            "--jobs", "2", "--policy-devices", "0,1", "--dry-run", "--output-dir", str(tmp_path)]
    args = parser.parse_args(argv)
    if not accepted:
        with pytest.raises(ValueError, match="parallel"):
            build_evaluation_manifest(args, spec, argv)
    else:
        manifest, _ = build_evaluation_manifest(args, spec, argv)
        assert manifest.policy["policy_family"] == family


@pytest.mark.parametrize("shared", [False, True])
def test_four_workers_overlap_and_keep_exclusive_lanes(tmp_path, shared):
    from rpent.evaluation.parallel import PolicyLane

    manifest = _manifest(tmp_path, task_count=8)
    barrier = threading.Barrier(4, timeout=5)
    lock = threading.Lock()
    active = set()
    seen = []
    peak = 0

    class Process(_Process):
        def wait(self, timeout=None):
            nonlocal peak
            lane = self.lane
            with lock:
                assert lane not in active
                active.add(lane)
                peak = max(peak, len(active))
            barrier.wait()
            result = super().wait(timeout)
            with lock:
                active.remove(lane)
            return result

    def launch(*, command, case, result_path, env):
        endpoint = env["RPENT_SHARED_POLICY_ENDPOINT"]
        lane = int(env["RPENT_EVALUATION_LANE"])
        assert endpoint == f"http://127.0.0.1:{19000 + (lane % 2 if shared else lane)}"
        assert env["CUDA_VISIBLE_DEVICES"] == str(lane % 2)
        assert env["MUJOCO_EGL_DEVICE_ID"] == str(lane % 2)
        with lock:
            seen.append(case.case_id)
        process = Process(
            pid=2000 + case.ordinal,
            events=[],
            case=case,
            result_path=result_path,
            manifest=manifest,
            behavior="success",
            attempt_id=command[command.index("--attempt-id") + 1],
        )
        process.lane = lane
        return process

    daemon = SimpleNamespace(
        endpoint="http://127.0.0.1:19000",
        start=lambda: None,
        stop=lambda: None,
        lanes=tuple(
            PolicyLane(
                f"http://127.0.0.1:{19000 + (i % 2 if shared else i)}",
                {
                    "CUDA_VISIBLE_DEVICES": str(i % 2),
                    "MUJOCO_EGL_DEVICE_ID": str(i % 2),
                    "RPENT_EVALUATION_LANE": str(i),
                    "RPENT_PI05_SESSION_SLOT": str(i // 2),
                },
                session_capacity=2 if shared else 1,
            )
            for i in range(4)
        ),
    )
    outcome = _coordinator(tmp_path, manifest, daemon, launch, jobs=4).run()
    assert outcome.completed == 8 and outcome.exit_code == 0
    assert peak == 4 and len(set(seen)) == len(seen) == 8
    assert json.loads((tmp_path / "summary.json").read_text())["completed"] == 8
    # Resume consumes retained results without another worker or model request.
    outcome = _coordinator(tmp_path, manifest, daemon, launch, jobs=4).run(resume=True)
    assert outcome.completed == 8 and len(seen) == 8


def test_replica_capacity_accounts_for_busy_memory_and_balances_devices():
    from rpent.evaluation.parallel import allocate_replicas

    assert allocate_replicas({0: 24564, 1: 24564}, jobs=4) == (0, 1, 0, 1)
    with pytest.raises(ValueError, match="capacity"):
        allocate_replicas({0: 24564, 1: 24564}, jobs=5)
    assert allocate_replicas({0: 13000, 1: 24564}, jobs=3) == (0, 1, 1)
    with pytest.raises(ValueError, match="capacity"):
        allocate_replicas({0: 13000, 1: 24564}, jobs=4)


def test_replica_start_failure_stops_every_created_service(tmp_path, monkeypatch):
    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_SNAPSHOT", "/pinned/snapshot")
    from rpent.evaluation.parallel import XiaomiReplicaPool

    created = []

    class Daemon:
        def __init__(self, **kwargs):
            self.device = kwargs["physical_device"]
            self.endpoint = f"http://127.0.0.1:{19000 + len(created)}"
            self.stopped = False
            created.append(self)

        def start(self):
            if len(created) == 3:
                raise RuntimeError("load failed")

        def stop(self):
            self.stopped = True

    pool = XiaomiReplicaPool(
        jobs=4,
        devices=(0, 1, 2, 3),
        output_dir=tmp_path,
        memory_query=lambda devices: {0: 65536, 1: 65536, 2: 65536, 3: 65536},
        host_memory_query=lambda: 131072,
        daemon_factory=Daemon,
    )
    with pytest.raises(RuntimeError, match="load failed"):
        pool.start()
    assert len(created) == 3 and all(d.stopped for d in created)


def test_parallel_requires_distinct_policy_endpoints_before_worker_launch(tmp_path):
    from rpent.evaluation.parallel import PolicyLane

    manifest = _manifest(tmp_path)
    daemon = SimpleNamespace(
        endpoint="http://same",
        start=lambda: None,
        stop=lambda: None,
        lanes=(PolicyLane("http://same", {}),) * 2,
    )
    calls = []
    outcome = _coordinator(
        tmp_path, manifest, daemon, lambda **kw: calls.append(kw), jobs=2
    ).run()
    assert outcome.infrastructure_error and not calls


@pytest.mark.parametrize("jobs", [0, -1, True, 1.5])
def test_invalid_concurrency_is_rejected_before_launch(tmp_path, jobs):
    with pytest.raises(ValueError, match="positive integer"):
        _coordinator(tmp_path, _manifest(tmp_path), None, None, jobs=jobs)


def test_parallel_timeout_reaps_before_reusing_replica(tmp_path):
    from rpent.evaluation.parallel import PolicyLane

    manifest = _manifest(tmp_path, task_count=5)
    processes = {}
    visited = []

    class ReapedProcess(_Process):
        def wait(self, timeout=None):
            code = super().wait(timeout)
            self.reaped = True
            return code

    def launch(*, command, case, result_path, env):
        endpoint = env["RPENT_SHARED_POLICY_ENDPOINT"]
        if endpoint in processes:
            assert processes[endpoint].reaped
        process = ReapedProcess(
            pid=2000 + case.ordinal,
            events=[],
            case=case,
            result_path=result_path,
            manifest=manifest,
            behavior="timeout" if case.ordinal == 0 else "success",
            attempt_id=command[command.index("--attempt-id") + 1],
        )
        processes[endpoint] = process
        visited.append(case.case_id)
        return process

    daemon = SimpleNamespace(
        endpoint="http://one",
        start=lambda: None,
        stop=lambda: None,
        lanes=(PolicyLane("http://one", {}), PolicyLane("http://two", {})),
    )
    outcome = _coordinator(
        tmp_path, manifest, daemon, launch, jobs=2, worker_timeout_s=0.03
    ).run()
    assert outcome.completed == 5 and len(set(visited)) == 5
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status_counts"]["timeout"] == 1
    assert summary["successful"] == 4


def test_cli_parallel_configuration_is_not_forwarded_to_episode_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("RPENT_XIAOMI_MIBOT_SNAPSHOT", "/pinned/snapshot")
    from rpent.cli.main import _build_argparser
    from rpent.envs import get_env_spec
    from rpent.evaluation.front_controller import build_evaluation_manifest

    spec = get_env_spec("robocasa")
    parser = _build_argparser()
    spec.add_cli_args(parser, use_dashboard=False)
    argv = [
        "--env",
        "robocasa",
        "--task",
        "CloseDrawer",
        "--vla",
        "xiaomi-mibot",
        "--jobs",
        "4",
        "--policy-devices",
        "0,1",
        "--dry-run",
        "--output-dir",
        str(tmp_path),
    ]
    args = parser.parse_args(argv)
    manifest, _ = build_evaluation_manifest(
        args, spec, argv, git_identity=lambda: ("a" * 40, False)
    )
    assert "--jobs" not in manifest.cli_arguments["argv"]
    assert "--policy-devices" not in manifest.cli_arguments["argv"]
    daemon = spec.build_policy_daemon(args, tmp_path)
    assert daemon.jobs == 4 and daemon.devices == (0, 1)


def test_parallel_cancellation_reaps_all_active_workers_without_dispatching_more(
    tmp_path,
):
    import subprocess

    from rpent.evaluation.control import CancellationToken
    from rpent.evaluation.parallel import PolicyLane

    manifest = _manifest(tmp_path, task_count=8)
    cancellation = CancellationToken()
    lock = threading.Lock()
    processes = []

    class Process:
        pid = 2345
        stopped = False
        reaped = False

        def wait(self, timeout=None):
            if self.stopped:
                self.reaped = True
                return -15
            threading.Event().wait(0.005)
            raise subprocess.TimeoutExpired("worker", timeout)

        def terminate(self):
            self.stopped = True

        def kill(self):
            self.stopped = True

    def launch(**kwargs):
        process = Process()
        with lock:
            processes.append(process)
            if len(processes) == 4:
                cancellation.cancel("test cancellation")
        return process

    daemon = SimpleNamespace(
        endpoint="http://one",
        start=lambda: None,
        stop=lambda: None,
        lanes=tuple(PolicyLane(f"http://lane{i}", {}) for i in range(4)),
    )
    outcome = _coordinator(
        tmp_path, manifest, daemon, launch, jobs=4, cancellation=cancellation
    ).run()
    assert outcome.interrupted
    assert len(processes) == 4 and all(p.reaped for p in processes)
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 and all(row["status"] == "cancelled" for row in rows)




@pytest.mark.skipif(
    __import__("sys").platform != "linux", reason="Linux worker tree ownership"
)
@pytest.mark.parametrize("terminate", [False, True])
def test_real_worker_reap_cleans_simulator_descendants(tmp_path, terminate):
    import os
    import signal
    import sys
    import time
    from pathlib import Path

    from rpent.evaluation.coordinator import _default_worker_launcher

    child_file = tmp_path / "simulator.pid"
    code = (
        "import subprocess,sys,time,pathlib; "
        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
        f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); "
        f"time.sleep({60 if terminate else 0.1})"
    )
    process = _default_worker_launcher(
        command=[sys.executable, "-c", code],
        case=None,
        result_path=tmp_path / "unused.json",
        env={**os.environ, "RPENT_WORKER_LOG_PATH": str(tmp_path / "console.log")},
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(child_file.read_text())
        if terminate:
            process.terminate()
        process.wait(timeout=5)
        child_stat = Path(f"/proc/{child_pid}/stat")
        deadline = time.monotonic() + 1
        while (
            child_stat.exists()
            and child_stat.read_text().split()[2] != "Z"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert not child_stat.exists() or child_stat.read_text().split()[2] == "Z"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass




@pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX process groups")
def test_repeated_wait_does_not_signal_an_already_reaped_group(tmp_path, monkeypatch):
    import os
    import sys

    from rpent.evaluation.coordinator import _default_worker_launcher

    process = _default_worker_launcher(
        command=[sys.executable, "-c", "pass"],
        case=None,
        result_path=tmp_path / "unused.json",
        env={**os.environ, "RPENT_WORKER_LOG_PATH": str(tmp_path / "console.log")},
    )
    assert process.wait(timeout=5) == 0

    def no_signal(*args):
        raise AssertionError("process group could now be reused")

    monkeypatch.setattr(os, "killpg", no_signal)
    assert process.wait(timeout=5) == 0


@pytest.mark.skipif(
    __import__("os").name != "posix", reason="POSIX graceful cancellation"
)
def test_worker_termination_allows_api_session_finally_cleanup(tmp_path):
    import os
    import sys
    import time

    from rpent.evaluation.coordinator import _default_worker_launcher

    ready = tmp_path / "ready"
    cleaned = tmp_path / "api-session-released"
    code = f"""import pathlib,time
try:
    pathlib.Path({str(ready)!r}).touch()
    time.sleep(60)
finally:
    pathlib.Path({str(cleaned)!r}).touch()
"""
    process = _default_worker_launcher(
        command=[sys.executable, "-c", code],
        case=None,
        result_path=tmp_path / "unused.json",
        env={**os.environ, "RPENT_WORKER_LOG_PATH": str(tmp_path / "console.log")},
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        process.terminate()
        process.wait(timeout=5)
        assert cleaned.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
