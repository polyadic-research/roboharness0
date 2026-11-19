"""Tests for versioned model-facing context projection."""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import pytest

from rpent.context import resolve_context_policy
from rpent.context.projectors import (
    EpisodeCompactContextV1,
    LiberoCompactContextV1,
    RawToolContextV1,
)
from rpent.planning import (
    JsonlEventRecorder,
    load_events,
    observation_from_tool_result,
    resolve_planning_stack,
)
from rpent.tools.toolkit import Toolkit, ToolResult


def _state_result() -> dict:
    return {
        "step": 2,
        "task_language": "put the bowl on the plate",
        "state": {
            "robot0_eef_pos": [0.1, 0.2, 0.3],
            "robot0_gripper_qpos": [0.02, -0.02],
            "object_names": ["bowl", "plate"],
        },
        "image_cam_hi_path": "images_cam_hi/image_cam_hi_02.png",
        "world_map_hi": "world_hi/world_hi_02.npy",
        "environment": {
            "success": False,
            "terminated": True,
            "truncated": False,
        },
        "libero_terminated": False,
        "base_task_success": True,
        "episode_done": False,
        "episode_truncated": False,
        "environment_termination_ignored": True,
        "log": {
            "command": {"action": "move_to", "xyz": [0.1, 0.2, 0.3]},
            "result": {"final_dist_m": 0.01, "steps_used": 12},
            "elapsed_s": 1.2,
        },
        "_image_bytes": b"agent",
        "_image_cam_bytes": b"camera",
    }


class ContextProjectorTest(unittest.TestCase):
    def test_raw_v1_preserves_original_text_and_image_order(self) -> None:
        result = _state_result()
        projected = RawToolContextV1().project_tool_result(
            tool_name="move_to",
            result=result,
            max_text_chars=60000,
        )
        expected = dict(result)
        expected.pop("_image_bytes")
        expected.pop("_image_cam_bytes")
        self.assertEqual(projected.text, json.dumps(expected, indent=2, default=str))
        self.assertEqual(
            [image.role for image in projected.images],
            ["agent", "camera"],
        )
        blocks = projected.content_blocks()
        self.assertEqual(
            blocks[1]["source"]["data"],
            base64.b64encode(b"agent").decode(),
        )

    def test_raw_v1_keeps_robocasa_refs_visible_after_projecting_right_image(self) -> None:
        projected = RawToolContextV1().project_tool_result(
            tool_name="observe",
            result={
                "_image_right_bytes": b"right" * 20_000,
                "observation_refs": ["live/right.png"],
            },
            max_text_chars=60_000,
        )

        self.assertEqual(json.loads(projected.text)["observation_refs"], ["live/right.png"])
        self.assertEqual(
            [(image.role, image.data) for image in projected.images],
            [("camera", b"right" * 20_000)],
        )

    def test_raw_v1_hides_display_only_overview_from_planner_context(self) -> None:
        projected = RawToolContextV1().project_tool_result(
            tool_name="observe",
            result={
                "_image_center_bytes": b"overview",
                "_image_camera_ids": {
                    "robot0_agentview_center": "_image_center_bytes",
                },
                "_image_bytes": b"left",
            },
            max_text_chars=60_000,
        )

        self.assertEqual(json.loads(projected.text), {})
        self.assertEqual(
            [(image.role, image.data) for image in projected.images],
            [("agent", b"left")],
        )

    def test_compact_v1_normalizes_stateful_results(self) -> None:
        projected = LiberoCompactContextV1().project_tool_result(
            tool_name="move_to",
            result=_state_result(),
            max_text_chars=60000,
        )
        payload = json.loads(projected.text)
        self.assertEqual(payload["tool"], "move_to")
        self.assertEqual(payload["observation"]["step"], 2)
        self.assertEqual(payload["observation"]["eef_position"], [0.1, 0.2, 0.3])
        self.assertEqual(payload["action_result"]["steps_used"], 12)
        self.assertTrue(payload["environment"]["base_task_success"])
        self.assertTrue(payload["environment"]["termination_ignored"])
        self.assertFalse(payload["environment"]["success"])
        self.assertTrue(payload["environment"]["terminated"])
        self.assertFalse(payload["environment"]["truncated"])
        self.assertFalse(payload["libero_terminated"])
        self.assertNotIn("log", payload)
        self.assertNotIn("holding_object", payload["observation"])
        self.assertNotIn("last_action", payload["observation"])
        self.assertNotIn("last_action_result", payload["observation"])
        self.assertNotIn("image_paths", payload["observation"])
        self.assertEqual(payload["artifacts"]["world_map_hi"], "world_hi/world_hi_02.npy")

    def test_compact_v1_projects_observation_refs_without_image_bytes(self) -> None:
        result = _state_result()
        result["observation_refs"] = ["live/right.png", "live/wrist.png"]
        result["_image_right_bytes"] = b"right" * 20_000
        projected = EpisodeCompactContextV1().project_tool_result(
            tool_name="observe",
            result=result,
            max_text_chars=60_000,
        )

        payload = json.loads(projected.text)
        self.assertEqual(
            payload["observation_refs"],
            ["live/right.png", "live/wrist.png"],
        )
        self.assertNotIn("_image_right_bytes", projected.text)
        self.assertEqual(
            [(image.role, len(image.data)) for image in projected.images],
            [("agent", 5), ("camera", 6), ("camera", 100_000)],
        )

    def test_generic_compact_context_does_not_invent_libero_alias(self) -> None:
        result = _state_result()
        result.pop("libero_terminated", None)
        projected = EpisodeCompactContextV1().project_tool_result(
            tool_name="move_to",
            result=result,
            max_text_chars=60000,
        )

        self.assertNotIn("libero_terminated", json.loads(projected.text))

    def test_compact_v1_keeps_full_inspection_payload(self) -> None:
        result = {"world_xyz": [0.1, 0.2, 0.3], "resolution": "high"}
        projected = LiberoCompactContextV1().project_tool_result(
            tool_name="back_project",
            result=result,
            max_text_chars=60000,
        )
        self.assertEqual(json.loads(projected.text), result)

    def test_registry_rejects_unknown_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown context policy"):
            resolve_context_policy("missing")

    def test_generic_and_legacy_compact_context_ids_share_behavior(self) -> None:
        generic = resolve_context_policy("episode-compact-v1")
        legacy = resolve_context_policy("libero-compact-v1")

        self.assertEqual(generic.id, "episode-compact-v1")
        self.assertEqual(legacy.id, "libero-compact-v1")
        self.assertIsInstance(legacy, EpisodeCompactContextV1)
        self.assertEqual(LiberoCompactContextV1.id, "libero-compact-v1")

    def test_generic_compact_context_preserves_robocasa_proprioception(
        self,
    ) -> None:
        state = [float(index) / 10 for index in range(16)]
        projected = resolve_context_policy(
            "episode-compact-v1"
        ).project_tool_result(
            tool_name="observe",
            result={
                "step": 3,
                "state": {"proprioception": state},
                "task_language": "close the fridge",
                "_image_center_bytes": b"overview",
                "_image_camera_ids": {
                    "robot0_agentview_center": "_image_center_bytes",
                },
                "_image_bytes": b"left",
                "_image_right_bytes": b"right",
                "_image_wrist_bytes": b"wrist",
            },
            max_text_chars=60_000,
        )

        payload = json.loads(projected.text)
        self.assertEqual(payload["observation"]["proprioception"], state)
        self.assertNotIn("_image_center_bytes", projected.text)
        self.assertNotIn("_image_camera_ids", projected.text)
        self.assertEqual(
            [image.role for image in projected.images],
            ["agent", "camera", "wrist"],
        )


@pytest.mark.parametrize("policy_id", ("raw-tool-result-v1", "episode-compact-v1"))
def test_camera_labels_stay_with_visible_images_when_overview_is_filtered(policy_id):
    from rpent.recording.media import camera_storage_component

    result = {
        "step": 3,
        "state": {"proprioception": [0.0] * 16},
        "_image_wrist_bytes": b"wrist",
        "_image_center_bytes": b"overview",
        "_image_right_bytes": b"right",
        "_image_bytes": b"left",
        "_image_camera_ids": {
            "robot0_eye_in_hand": "_image_wrist_bytes",
            "robot0_agentview_center": "_image_center_bytes",
            "robot0_agentview_right": "_image_right_bytes",
            "robot0_agentview_left": "_image_bytes",
        },
    }
    result["observation_refs"] = [
        f"live/{camera_storage_component(camera)}.png"
        for camera in (
            "robot0_agentview_center", "robot0_agentview_left",
            "robot0_agentview_right", "robot0_eye_in_hand",
        )
    ]
    projection = resolve_context_policy(policy_id).project_tool_result(
        tool_name="observe", result=result, max_text_chars=60_000,
    )
    blocks = projection.content_blocks()

    assert len(blocks) == 7
    assert json.loads(projection.text)["observation_refs"] == result["observation_refs"][1:]
    assert "_image_camera_ids" not in projection.text
    assert "robot0_agentview_center" not in projection.text
    assert len(projection.images) == 3
    for index, (camera, data) in enumerate((
        ("robot0_agentview_left", b"left"),
        ("robot0_agentview_right", b"right"),
        ("robot0_eye_in_hand", b"wrist"),
    )):
        label, image = blocks[1 + 2 * index : 3 + 2 * index]
        assert label == {
            "type": "text", "text": f"Image {index + 1} camera: {camera}",
        }
        assert image["type"] == "image"
        assert base64.b64decode(image["source"]["data"]) == data
    assert result["_image_center_bytes"] == b"overview"
    assert "robot0_agentview_center" in result["_image_camera_ids"]

    # API/Gateway runtimes split text from images; numbered labels must still
    # identify each image in the same order after this conversion.
    from rpent.runtime.api import _content_blocks_to_pydantic

    text, images = _content_blocks_to_pydantic(blocks)
    assert [image.data for image in images] == [b"left", b"right", b"wrist"]
    assert "robot0_agentview_center" not in text
    for index, camera in enumerate((
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand",
    )):
        assert f"Image {index + 1} camera: {camera}" in text



@pytest.mark.parametrize(
    "value",
    [[0.0, float("nan")], [0.0, "bad"], [True, 0.0], "bad"],
)
def test_observation_summary_rejects_malformed_proprioception(value) -> None:
    observation = observation_from_tool_result(
        {"step": 1, "state": {"proprioception": value}}
    )

    assert observation is not None
    assert observation.proprioception is None


class ContextIntegrationTest(unittest.TestCase):
    def test_profile_resolves_context_and_harness(self) -> None:
        stack = resolve_planning_stack("libero-compact-trace-v1")
        self.assertEqual(stack.context_policy.id, "libero-compact-v1")
        with tempfile.TemporaryDirectory() as tmp:
            harness = stack.build_harness(trace_path=Path(tmp) / "trace.jsonl")
            self.assertIsNotNone(harness)
            assert harness is not None
            self.assertEqual(harness.mode, "trace")

    def test_toolkit_records_selected_projection_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "events.jsonl"
            recorder = JsonlEventRecorder(event_path, episode_id="context-test")
            toolkit = Toolkit()
            toolkit.attach_event_recorder(recorder)
            toolkit.set_context_policy(LiberoCompactContextV1())
            toolkit.add_tool(
                "observe",
                {"name": "observe", "input_schema": {}},
                _state_result,
            )
            result: ToolResult = toolkit.execute_tool("observe", {})
            self.assertEqual(json.loads(result.projection.text)["observation"]["step"], 2)
            projected = [
                event
                for event in load_events(event_path)
                if event["event"] == "context_projected"
            ]
            self.assertEqual(len(projected), 1)
            self.assertEqual(projected[0]["context_policy"], "libero-compact-v1")


if __name__ == "__main__":
    unittest.main()
