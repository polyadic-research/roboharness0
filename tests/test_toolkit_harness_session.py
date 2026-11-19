"""Production boundary tests for the strict HarnessSession toolkit path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from robots.robocasa.display import RoboCasaDisplayAdapter
from rpent.harness.lifecycle import ALLOWED_OPERATIONS
from rpent.recording.contracts import RecordingProfile
from rpent.recording.display import png_bytes
from rpent.recording.media import camera_storage_component
from rpent.recording.recorder import CaseRecorder
from rpent.tools.toolkit import Toolkit
from tests.synthetic_tasks import SyntheticDisplayAdapter

LEGACY_CONTROL_TOOLS = {
    "create_plan",
    "register_plan",
    "view_plan",
    "replan",
    "request_replan",
    "select_outcome",
    "advance_stage",
}


class RecordingSession:
    def __init__(self) -> None:
        class Controller:
            def __init__(self) -> None:
                self.physical_tools = set()
                self.started = False

            def set_physical_tools(self, tools):
                if self.started:
                    raise ValueError("physical tools are frozen")
                self.physical_tools = set(tools)

        self.controller = Controller()
        self.preflight = {"allowed": True}
        self.preflight_calls: list[tuple[str, str]] = []
        self.feedback_calls = []
        self.abort_calls = []
        self.operation_calls: list[tuple[str, dict]] = []
        self.finish_result = {"ok": True, "status": "success"}
        self.status = {
            "state": "verification_required",
            "required_operation": "submit_observation",
        }

    def physical_action_allowed(self, name, execution_id):
        self.preflight_calls.append((name, execution_id))
        return dict(self.preflight)

    def record_action_feedback(self, feedback):
        self.feedback_calls.append(feedback)
        return {"ok": True, "state": "verification_required"}

    def abort_physical_action(self, execution_id, reason, evidence):
        self.abort_calls.append((execution_id, reason, evidence))
        return {"ok": True, "state": "recovery_required"}

    def execution_status(self):
        return dict(self.status)

    def finish(self, status, summary):
        self.operation_calls.append(("finish", {"status": status, "summary": summary}))
        return dict(self.finish_result)

    def __getattr__(self, name):
        def operation(**kwargs):
            self.operation_calls.append((name, kwargs))
            return {"ok": True, "operation": name}

        return operation


def _strict_toolkit(
    tmp_path: Path,
    session=None,
    move_handler=None,
    observe_handler=None,
):
    toolkit = Toolkit()
    toolkit.configure_harness_physical_tools({"move", "unregistered_move"})
    toolkit.configure_harness_evidence_tools({"move", "observe_env"})
    toolkit.add_tool(
        "move",
        {"name": "move", "description": "Move.", "input_schema": {"type": "object"}},
        move_handler or (lambda **_: {"moved": True}),
    )
    toolkit.add_tool(
        "observe_env",
        {
            "name": "observe_env",
            "description": "Observe without mutation.",
            "input_schema": {"type": "object"},
        },
        observe_handler or (lambda: {"visible": True}),
    )
    for name in LEGACY_CONTROL_TOOLS:
        toolkit.add_tool(
            name, {"name": name, "input_schema": {}}, lambda: {"bad": True}
        )
    recording = session or RecordingSession()
    toolkit.attach_harness_session(recording, tmp_path)
    return toolkit, recording


def test_strict_schema_is_static_union_without_placeholder_or_legacy_tools(tmp_path):
    toolkit, _ = _strict_toolkit(tmp_path)

    names = {spec["name"] for spec in toolkit.get_tools_spec()}
    lifecycle_names = {
        name
        for operations in ALLOWED_OPERATIONS.values()
        for name in operations
        if name != "configured_physical_tool"
    }

    assert names == (
        lifecycle_names
        | {
            "start_instruction",
            "move",
            "observe_env",
            "read_text_file",
            "write_text_file",
            "list_dir",
        }
    )
    assert not (names & LEGACY_CONTROL_TOOLS)
    assert "configured_physical_tool" not in names
    schemas = {spec["name"]: spec["input_schema"] for spec in toolkit.get_tools_spec()}
    assert "observed_at" not in schemas["submit_observation"]["properties"]
    assert "observed_at" not in schemas["verify_outcomes"]["properties"]
    descriptions = {spec["name"]: spec["description"] for spec in toolkit.get_tools_spec()}
    assert "new read-only observation" in descriptions["verify_outcomes"]
    assert "later capture time" in descriptions["verify_outcomes"]


def test_legacy_and_strict_harness_attachments_are_mutually_exclusive(tmp_path):
    class LegacyHarness:
        def set_physical_tools(self, tools):
            pass

        def tool_definitions(self):
            return []

    legacy = Toolkit()
    legacy.attach_harness(LegacyHarness())
    with pytest.raises(RuntimeError, match="already attached"):
        legacy.attach_harness_session(RecordingSession(), tmp_path)

    strict = Toolkit()
    strict.attach_harness_session(RecordingSession(), tmp_path)
    with pytest.raises(RuntimeError, match="already attached"):
        strict.attach_harness(LegacyHarness())


def test_strict_attach_synchronizes_physical_tools_before_session_start(tmp_path):
    toolkit = Toolkit()
    toolkit.configure_harness_physical_tools({"move", "release"})
    session = RecordingSession()

    toolkit.attach_harness_session(session, tmp_path)

    assert session.controller.physical_tools == {"move", "release"}


def test_strict_attach_failure_does_not_partially_attach_session(tmp_path):
    toolkit = Toolkit()
    toolkit.configure_harness_physical_tools({"move"})
    session = RecordingSession()
    session.controller.started = True

    with pytest.raises(ValueError, match="frozen"):
        toolkit.attach_harness_session(session, tmp_path)

    assert toolkit._harness_session is None


def test_strict_attach_rejects_unregistered_evidence_source_tool(tmp_path):
    toolkit = Toolkit()
    toolkit.configure_harness_evidence_tools({"missing_observer"})

    with pytest.raises(ValueError, match="evidence tools are not registered"):
        toolkit.attach_harness_session(RecordingSession(), tmp_path)

    assert toolkit._harness_session is None


def test_strict_attach_freezes_physical_tool_classification(tmp_path):
    toolkit, session = _strict_toolkit(tmp_path)
    before = set(toolkit._harness_physical_tools)

    with pytest.raises(RuntimeError, match="frozen"):
        toolkit.configure_harness_physical_tools({"different_move"})

    assert toolkit._harness_physical_tools == before
    assert session.controller.physical_tools == before

    with pytest.raises(RuntimeError, match="frozen"):
        toolkit.configure_harness_evidence_tools({"different_observer"})


def test_strict_attach_freezes_tool_registry_against_handler_override(tmp_path):
    toolkit, _ = _strict_toolkit(tmp_path)

    with pytest.raises(RuntimeError, match="registry is frozen"):
        toolkit.add_tool(
            "finish",
            {"name": "finish", "input_schema": {"type": "object"}},
            lambda: {"_finish": True, "status": "success"},
        )


def test_physical_preflight_rejection_never_executes_or_posts_feedback(tmp_path):
    calls = []
    toolkit, session = _strict_toolkit(
        tmp_path,
        move_handler=lambda: calls.append("executed") or {"moved": True},
    )
    session.preflight = {
        "allowed": False,
        "error": "blocked",
        "state": "recovery_required",
    }

    result = toolkit.execute_tool("move", {})

    assert result.result["error"] == "blocked"
    assert calls == []
    assert len(session.preflight_calls) == 1
    assert session.feedback_calls == []
    assert session.abort_calls == []


@pytest.mark.parametrize("handler_result", [{"moved": True}, {"error": "jammed"}])
def test_physical_handler_feedback_is_recorded_exactly_once(tmp_path, handler_result):
    toolkit, session = _strict_toolkit(
        tmp_path, move_handler=lambda: handler_result
    )

    result = toolkit.execute_tool("move", {})

    assert result.result.get("harness", {})["ok"] is True
    assert len(session.preflight_calls) == 1
    assert len(session.feedback_calls) == 1
    assert session.feedback_calls[0].raw_result == handler_result
    assert session.abort_calls == []


def test_unregistered_physical_tool_is_rejected_before_preflight(tmp_path):
    toolkit, session = _strict_toolkit(tmp_path)

    result = toolkit.execute_tool("unregistered_move", {})

    assert "unknown or unexposed tool" in result.result["error"]
    assert session.preflight_calls == []


def test_handler_exception_aborts_dispatched_action_fail_closed(tmp_path):
    def explode():
        raise RuntimeError("motor transport vanished")

    toolkit, session = _strict_toolkit(tmp_path, move_handler=explode)

    result = toolkit.execute_tool("move", {})

    assert "motor transport vanished" in result.result["error"]
    assert session.feedback_calls == []
    assert len(session.abort_calls) == 1
    assert session.abort_calls[0][0] == result.call_id


def test_feedback_normalization_exception_aborts_fail_closed(tmp_path):
    toolkit, session = _strict_toolkit(tmp_path)

    with patch(
        "rpent.planning.feedback.action_feedback_from_result",
        side_effect=RuntimeError("normalizer broke"),
    ):
        result = toolkit.execute_tool("move", {})

    assert "normalizer broke" in result.result["error"]
    assert session.feedback_calls == []
    assert len(session.abort_calls) == 1


def test_finish_sentinel_is_only_added_after_session_accepts_finish(tmp_path):
    toolkit, session = _strict_toolkit(tmp_path)
    session.finish_result = {"ok": False, "error": "contract incomplete"}

    blocked = toolkit.execute_tool("finish", {"status": "success", "summary": "done"})
    assert blocked.is_finish is False
    assert "_finish" not in blocked.result

    session.finish_result = {"ok": True, "status": "success", "summary": "done"}
    accepted = toolkit.execute_tool("finish", {"status": "success", "summary": "done"})
    assert accepted.is_finish is True
    assert accepted.result["_finish"] is True


def test_evidence_timestamps_are_derived_and_model_value_is_ignored(tmp_path):
    evidence = tmp_path / "state.json"
    evidence.write_text("{}")
    os.utime(evidence, (1_800_000_000, 1_800_000_000))
    toolkit, session = _strict_toolkit(
        tmp_path,
        observe_handler=lambda: {"image_path": str(evidence)},
    )
    toolkit.execute_tool("observe_env", {})

    result = toolkit.execute_tool(
        "verify_outcomes",
        {
            "outcome_ids": ["outcome-1"],
            "observation_refs": ["state.json"],
            "observed_at": ["1900-01-01T00:00:00Z"],
        },
    )

    assert result.result["ok"] is True
    name, kwargs = session.operation_calls[-1]
    assert name == "verify_outcomes"
    assert kwargs["observed_at"] == ["2027-01-15T08:00:00+00:00"]


def test_reobserved_content_path_gets_a_new_trusted_capture_time(tmp_path):
    evidence = tmp_path / "unchanged.png"
    evidence.write_bytes(b"same content")
    os.utime(evidence, (1_700_000_000, 1_700_000_000))
    toolkit, session = _strict_toolkit(
        tmp_path,
        observe_handler=lambda: {"image_path": str(evidence)},
    )

    toolkit.execute_tool("observe_env", {})
    toolkit.execute_tool(
        "submit_observation", {"observation_refs": [evidence.name]}
    )
    first_capture = session.operation_calls[-1][1]["observed_at"][0]

    toolkit.execute_tool("observe_env", {})
    toolkit.execute_tool(
        "verify_outcomes",
        {"outcome_ids": ["outcome-1"], "observation_refs": [evidence.name]},
    )
    final_capture = session.operation_calls[-1][1]["observed_at"][0]

    assert final_capture > first_capture


def test_evidence_path_escape_is_rejected_without_calling_session(tmp_path):
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}")
    toolkit, session = _strict_toolkit(tmp_path)

    result = toolkit.execute_tool(
        "submit_observation",
        {"observation_refs": [str(outside)], "observed_at": ["2099-01-01T00:00:00Z"]},
    )

    assert "outside run output" in result.result["error"]
    assert session.operation_calls == []


def test_stage_boundary_observations_are_grounded_text_not_evidence_paths(tmp_path):
    toolkit, session = _strict_toolkit(tmp_path)
    session.status = {"state": "observation_required"}

    result = toolkit.execute_tool(
        "submit_observation",
        {"observation_refs": ["camera now shows the bowl inside the tray"]},
    )

    assert result.result["ok"] is True
    name, kwargs = session.operation_calls[-1]
    assert name == "submit_observation"
    assert kwargs == {"observation_refs": ["camera now shows the bowl inside the tray"]}


def test_strict_audit_write_is_confined_to_run_output(tmp_path):
    toolkit, _ = _strict_toolkit(tmp_path)

    written = toolkit.execute_tool(
        "write_text_file", {"path": "audit.json", "content": '{"ok": true}'}
    )
    escaped = toolkit.execute_tool(
        "write_text_file", {"path": "../escaped.json", "content": "unsafe"}
    )

    audit_path = tmp_path / "agent_artifacts" / "audit.json"
    assert written.result["path"] == str(audit_path)
    assert audit_path.read_text() == '{"ok": true}'
    assert "outside run output" in escaped.result["error"]
    assert not (tmp_path.parent / "escaped.json").exists()


def test_agent_written_artifact_cannot_be_used_as_verification_evidence(tmp_path):
    toolkit, session = _strict_toolkit(tmp_path)
    toolkit.execute_tool(
        "write_text_file",
        {"path": "fabricated.json", "content": '{"predicate": "satisfied"}'},
    )

    result = toolkit.execute_tool(
        "verify_outcomes",
        {
            "outcome_ids": ["outcome-1"],
            "observation_refs": ["agent_artifacts/fabricated.json"],
        },
    )

    assert "only artifacts registered from environment tool output" in result.result[
        "error"
    ]
    assert session.operation_calls == []


@pytest.mark.parametrize(
    "filename", ["experiment_manifest.json", "events.jsonl", "planner.txt"]
)
def test_unregistered_run_metadata_cannot_be_verification_evidence(
    tmp_path, filename
):
    artifact = tmp_path / filename
    artifact.write_text('{"claim": "satisfied"}')
    toolkit, session = _strict_toolkit(tmp_path)

    result = toolkit.execute_tool(
        "verify_outcomes",
        {"outcome_ids": ["outcome-1"], "observation_refs": [filename]},
    )

    assert "only artifacts registered from environment tool output" in result.result[
        "error"
    ]
    assert session.operation_calls == []


def test_failed_evidence_tool_cannot_register_paths_echoed_from_arguments(tmp_path):
    manifest = tmp_path / "experiment_manifest.json"
    manifest.write_text('{"claim": "satisfied"}')
    toolkit, session = _strict_toolkit(tmp_path)

    failed = toolkit.execute_tool(
        "observe_env", {"forged_path": str(manifest)}
    )
    verified = toolkit.execute_tool(
        "verify_outcomes",
        {
            "outcome_ids": ["outcome-1"],
            "observation_refs": ["experiment_manifest.json"],
        },
    )

    assert "bad arguments" in failed.result["error"]
    assert "only artifacts registered from environment tool output" in verified.result[
        "error"
    ]
    assert session.operation_calls == []


def test_empty_error_marker_still_prevents_evidence_registration(tmp_path):
    artifact = tmp_path / "untrusted.json"
    artifact.write_text("{}")
    toolkit, session = _strict_toolkit(
        tmp_path,
        observe_handler=lambda: {"error": "", "image_path": str(artifact)},
    )
    toolkit.execute_tool("observe_env", {})

    verified = toolkit.execute_tool(
        "verify_outcomes",
        {"outcome_ids": ["outcome-1"], "observation_refs": ["untrusted.json"]},
    )

    assert "only artifacts registered from environment tool output" in verified.result[
        "error"
    ]
    assert session.operation_calls == []


@pytest.mark.parametrize(
    ("display", "image_fields", "image_camera_ids"),
    (
        (
            RoboCasaDisplayAdapter(
                width=2,
                height=2,
                overview_width=2,
                overview_height=2,
            ),
            (
                "_image_center_bytes",
                "_image_bytes",
                "_image_right_bytes",
                "_image_wrist_bytes",
            ),
            {
                "robot0_agentview_center": "_image_center_bytes",
                "robot0_agentview_left": "_image_bytes",
                "robot0_agentview_right": "_image_right_bytes",
                "robot0_eye_in_hand": "_image_wrist_bytes",
            },
        ),
        (
            SyntheticDisplayAdapter(width=2, height=2, camera_ids=("image", "wrist_image")),
            ("_image_bytes", "_image_wrist_bytes"),
            None,
        ),
    ),
    ids=("robocasa", "synthetic"),
)
def test_recorder_retained_observe_images_are_trusted_but_listed_files_are_not(
    tmp_path, monkeypatch, display, image_fields, image_camera_ids
):
    monkeypatch.chdir(tmp_path.parent)
    relative_root = Path(tmp_path.name)
    image = png_bytes(np.zeros((2, 2, 3), dtype=np.uint8))

    def observe():
        result = dict.fromkeys(image_fields, image)
        if image_camera_ids is not None:
            result["_image_camera_ids"] = image_camera_ids
        return result

    toolkit, session = _strict_toolkit(
        relative_root,
        observe_handler=observe,
    )
    recorder = CaseRecorder(
        root=relative_root,
        evaluation_id="evaluation-1",
        case_id="case-1",
        attempt_id="attempt-1",
        episode_id="episode-1",
        display=display,
        profile=RecordingProfile("all"),
    )
    toolkit.attach_event_recorder(recorder)
    forged = relative_root / "forged.png"
    forged.write_bytes(image)

    observed = toolkit.execute_tool("observe_env", {})
    toolkit.execute_tool("list_dir", {"path": str(tmp_path)})
    accepted = toolkit.execute_tool(
        "submit_observation",
        {"observation_refs": observed.result["observation_refs"]},
    )
    rejected = toolkit.execute_tool(
        "submit_observation", {"observation_refs": [forged.name]}
    )

    expected = [
        f"live/{camera_storage_component(camera.camera_id)}.png"
        for camera in recorder.cameras
    ]
    assert observed.result["observation_refs"] == expected
    assert all((relative_root / ref).is_file() for ref in expected)
    projected_refs = json.loads(observed.projection.text)["observation_refs"]
    if isinstance(display, RoboCasaDisplayAdapter):
        assert projected_refs == expected[1:]
        assert len(observed.projection.images) == 3
        assert "robot0_agentview_center" not in json.dumps(observed.content_blocks)
    else:
        assert projected_refs == expected
    assert accepted.result["ok"] is True
    assert "only artifacts registered from environment tool output" in rejected.result[
        "error"
    ]
    assert [name for name, _ in session.operation_calls] == ["submit_observation"]
    recorder.close()
