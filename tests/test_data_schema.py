from robots.robocasa.data_schema import robocasa_data_schema
from rpent.recording.contracts import CameraDescriptor
from rpent.recording.data_schema import build_data_schema


def camera(camera_id: str, label: str, role: str | None) -> CameraDescriptor:
    return CameraDescriptor(camera_id, label, role, 256, 256, 3, 20.0)


def test_generic_schema_preserves_camera_order_and_falls_back_dimension_names() -> None:
    schema = build_data_schema(
        cameras=(
            camera("wrist", "Wrist", "unknown-role"),
            camera("main", "Main", "primary"),
        ),
        action_shape=(3,),
        state_shape=(2,),
    )

    assert schema.schema_version == 1
    assert [item.id for item in schema.cameras] == ["wrist", "main"]
    assert [item.name for item in schema.action.dimensions] == [
        "action[0]", "action[1]", "action[2]",
    ]
    assert [item.name for item in schema.state.dimensions] == ["state[0]", "state[1]"]
    assert [item.id for item in schema.signals] == [
        "environment.success", "contract.success", "reward",
        "environment.terminated", "environment.truncated",
    ]


def test_robocasa_schema_uses_only_checked_in_adapter_semantics() -> None:
    cameras = (
        camera("robot0_agentview_left", "Agent Left", "primary"),
        camera("robot0_agentview_right", "Agent Right", None),
        camera("robot0_eye_in_hand", "Wrist", "wrist"),
    )

    schema = robocasa_data_schema(cameras)

    assert schema.action.shape == (12,)
    assert [item.group for item in schema.action.dimensions] == [
        "base", "base", "base", "base", "control", "arm", "arm", "arm",
        "arm", "arm", "arm", "gripper",
    ]
    assert schema.action.dimensions[0].name == "action.base_motion[0]"
    assert schema.action.dimensions[11].name == "action.gripper_close"
    assert schema.state.shape == (16,)
    assert schema.state.dimensions[0].name == "state.base_position[0]"
    assert schema.state.dimensions[14].name == "state.gripper_qpos[0]"
