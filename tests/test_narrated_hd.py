"""Native camera detail and wall-time synchronization are export requirements."""

import json
from types import SimpleNamespace

import numpy as np


def test_state_capture_observes_substeps_without_extra_physics(tmp_path):
    from rpent.recording.narrated.hd import StateCapture

    class Physics:
        mjtState = SimpleNamespace(mjSTATE_INTEGRATION=1)

        @staticmethod
        def mj_stateSize(model, spec):
            return 2

        @staticmethod
        def mj_getState(model, data, target, spec):
            target[:] = [data.time, data.position]

    class Sim:
        def __init__(self):
            self.model = SimpleNamespace(_model=object(), get_xml=lambda: "<mujoco/>")
            self.data = SimpleNamespace(_data=SimpleNamespace(time=0.0, position=0.0))
            self.steps = 0

        def step(self):
            self.steps += 1
            self.data._data.time += 0.01
            self.data._data.position += 1
            return "original-result"

    sim = Sim()
    original = sim.step
    capture = StateCapture(tmp_path, physics=Physics, wall_clock=lambda: 100.0)
    capture.observe(sim)
    for _ in range(5):
        assert sim.step() == "original-result"
    capture.observe(sim)
    capture.close()
    assert sim.steps == 5
    assert sim.step == original
    rows = [json.loads(s) for s in (tmp_path / "states.jsonl").read_text().splitlines()]
    values = np.fromfile(tmp_path / "states.bin", dtype="<f8").reshape(-1, 2)
    assert len(rows) == len(values)
    assert values[0].tolist() == [0, 0]
    assert values[-1].tolist() == [0.05, 5]
    assert all(row["wall_s"] == 100 for row in rows)








def test_capture_write_failure_does_not_hide_physics_result(tmp_path):
    from rpent.recording.narrated.hd import StateCapture

    class Physics:
        mjtState = SimpleNamespace(mjSTATE_INTEGRATION=1)
        mj_stateSize = staticmethod(lambda model, spec: 1)
        mj_getState = staticmethod(lambda model, data, state, spec: state.fill(0))

    sim = SimpleNamespace(
        model=SimpleNamespace(_model=object(), get_xml=lambda: "<mujoco/>"),
        data=SimpleNamespace(_data=SimpleNamespace(time=0.0)),
        step=lambda: "executed",
    )
    capture = StateCapture(tmp_path, physics=Physics)
    capture.observe(sim)
    capture.values.close()  # Simulate a failing recording sink, after initial capture.
    sim.data._data.time = 0.1
    assert sim.step() == "executed"
    assert (tmp_path / "error.json").is_file()
    capture.close()


def test_lite_physics_step2_is_sampled_and_restored(tmp_path):
    from rpent.recording.narrated.hd import StateCapture

    class Physics:
        mjtState = SimpleNamespace(mjSTATE_INTEGRATION=1)
        mj_stateSize = staticmethod(lambda model, spec: 1)
        mj_getState = staticmethod(
            lambda model, data, state, spec: state.fill(data.time)
        )

    data = SimpleNamespace(time=0.0)

    def step2():
        data.time += 0.01
        return "step2"

    sim = SimpleNamespace(
        model=SimpleNamespace(_model=object(), get_xml=lambda: "<mujoco/>"),
        data=SimpleNamespace(_data=data),
        step=lambda: None,
        step2=step2,
    )
    capture = StateCapture(tmp_path, physics=Physics)
    capture.observe(sim)
    for _ in range(6):
        assert sim.step2() == "step2"
    capture.observe(sim)
    capture.close()
    assert sim.step2 is step2
    assert len((tmp_path / "states.jsonl").read_text().splitlines()) > 2




def test_failed_initial_state_setup_disables_capture_without_affecting_robot(
    tmp_path, monkeypatch
):
    import sys

    from rpent.recording.narrated.hd import capture_environment_state

    def broken_xml():
        raise OSError("unavailable recording sink")

    physics = SimpleNamespace(
        mjtState=SimpleNamespace(mjSTATE_INTEGRATION=1), mj_stateSize=lambda m, s: 1
    )
    monkeypatch.setitem(sys.modules, "mujoco", physics)
    monkeypatch.setenv("RPENT_NARRATION_STATES_DIR", str(tmp_path))
    sim = SimpleNamespace(
        model=SimpleNamespace(_model=object(), get_xml=broken_xml),
        data=SimpleNamespace(_data=object()),
        step=lambda: "robot",
    )
    adapter = SimpleNamespace(env=SimpleNamespace(sim=sim))
    capture_environment_state(adapter)
    assert adapter._narration_capture_disabled is True
    assert sim.step() == "robot"
