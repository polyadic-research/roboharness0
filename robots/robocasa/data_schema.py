"""RoboCasa data semantics verified by the policy and Gym adapters."""

from rpent.recording.contracts import CameraDescriptor
from rpent.recording.data_schema import CaseDataSchema, build_data_schema


def _expanded(name: str, label: str, group: str, count: int):
    return tuple(
        (f"{name}[{index}]", f"{label} {index}", group, None)
        for index in range(count)
    )


ACTION_DIMENSIONS = (
    *_expanded("action.base_motion", "Base motion", "base", 4),
    ("action.control_mode", "Control mode", "control", None),
    *_expanded("action.end_effector_position", "End-effector position", "arm", 3),
    *_expanded("action.end_effector_rotation", "End-effector rotation", "arm", 3),
    ("action.gripper_close", "Gripper close", "gripper", None),
)

STATE_DIMENSIONS = (
    *_expanded("state.base_position", "Base position", "base", 3),
    *_expanded("state.base_quaternion", "Base quaternion", "base", 4),
    *_expanded(
        "state.relative_end_effector_position",
        "Relative end-effector position",
        "arm",
        3,
    ),
    *_expanded(
        "state.relative_end_effector_quaternion",
        "Relative end-effector quaternion",
        "arm",
        4,
    ),
    *_expanded("state.gripper_qpos", "Gripper position", "gripper", 2),
)


def robocasa_data_schema(
    cameras: tuple[CameraDescriptor, ...],
) -> CaseDataSchema:
    return build_data_schema(
        cameras=cameras,
        action_shape=(12,),
        state_shape=(16,),
        action_dimensions=ACTION_DIMENSIONS,
        state_dimensions=STATE_DIMENSIONS,
    )


__all__ = ["robocasa_data_schema"]
