from __future__ import annotations

import numpy as np
import pytest

from rpent.episode.binding import FeatureBinding, FeatureValidationError, PolicyBinding


def robocerebra_binding() -> PolicyBinding:
    return PolicyBinding(
        name="robocerebra-smolvla-v1",
        inputs=(
            FeatureBinding("state", "observation.state", (8,), np.dtype("float32")),
            FeatureBinding(
                "image",
                "observation.images.camera1",
                (256, 256, 3),
                np.dtype("uint8"),
            ),
            FeatureBinding(
                "wrist_image",
                "observation.images.camera2",
                (256, 256, 3),
                np.dtype("uint8"),
            ),
        ),
        action_shape=(7,),
        action_dtype=np.dtype("float32"),
        action_low=-1.0,
        action_high=1.0,
        allowed_missing_policy_inputs=frozenset({"observation.images.camera3"}),
    )


def valid_native_observation() -> dict[str, np.ndarray]:
    return {
        "state": np.zeros(8, dtype=np.float32),
        "image": np.zeros((256, 256, 3), dtype=np.uint8),
        "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
    }


def test_binding_maps_native_names_and_preserves_arrays() -> None:
    native = valid_native_observation()

    mapped = robocerebra_binding().bind_observation(native)

    assert set(mapped) == {
        "observation.state",
        "observation.images.camera1",
        "observation.images.camera2",
    }
    assert mapped["observation.state"] is native["state"]


def test_binding_rejects_wrong_state_shape() -> None:
    native = valid_native_observation()
    native["state"] = np.zeros(6, dtype=np.float32)

    with pytest.raises(FeatureValidationError, match=r"state.*expected \(8,\).*(6,)"):
        robocerebra_binding().bind_observation(native)


def test_binding_rejects_wrong_image_dtype() -> None:
    native = valid_native_observation()
    native["image"] = np.zeros((256, 256, 3), dtype=np.float32)

    with pytest.raises(FeatureValidationError, match=r"image.*uint8.*float32"):
        robocerebra_binding().bind_observation(native)


def test_binding_rejects_missing_native_feature() -> None:
    native = valid_native_observation()
    del native["wrist_image"]

    with pytest.raises(FeatureValidationError, match=r"wrist_image.*missing"):
        robocerebra_binding().bind_observation(native)


def test_binding_rejects_non_finite_state() -> None:
    native = valid_native_observation()
    native["state"][0] = np.nan

    with pytest.raises(FeatureValidationError, match=r"state.*finite.*non-finite"):
        robocerebra_binding().bind_observation(native)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((4, 12), dtype=np.float32),
        np.zeros((4, 7), dtype=np.float64),
        np.full((4, 7), np.nan, dtype=np.float32),
        np.full((4, 7), 1.1, dtype=np.float32),
    ],
)
def test_binding_rejects_invalid_action_chunk(invalid: np.ndarray) -> None:
    with pytest.raises(FeatureValidationError):
        robocerebra_binding().validate_action_chunk(invalid)


def test_binding_validates_action_chunk_and_preserves_array() -> None:
    actions = np.zeros((4, 7), dtype=np.float32)

    accepted = robocerebra_binding().validate_action_chunk(actions)

    assert accepted.shape == (4, 7)
    assert accepted is actions


def test_binding_accepts_action_values_at_inclusive_bounds() -> None:
    actions = np.array([[-1.0, 1.0, 0.0, 0.5, -0.5, 1.0, -1.0]], dtype=np.float32)

    assert robocerebra_binding().validate_action_chunk(actions) is actions


def test_binding_can_explicitly_clip_actions_to_controller_bounds() -> None:
    binding = PolicyBinding(
        name="clipped",
        inputs=(),
        action_shape=(2,),
        action_dtype=np.dtype("float32"),
        action_low=-1.0,
        action_high=1.0,
        clip_actions_to_bounds=True,
    )
    actions = np.array([[-5.0, 3.0]], dtype=np.float32)

    adapted = binding.validate_action_chunk(actions)

    np.testing.assert_array_equal(adapted, np.array([[-1.0, 1.0]], dtype=np.float32))


@pytest.mark.parametrize("invalid", [0, 1, "yes", None])
def test_policy_binding_requires_boolean_clip_mode(invalid: object) -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'invalid'.*clip_actions_to_bounds.*boolean.*actual",
    ):
        PolicyBinding(
            name="invalid",
            inputs=(),
            action_shape=(2,),
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
            clip_actions_to_bounds=invalid,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_value",
    [-1.0001, 1.0001, -np.inf, np.inf],
)
def test_binding_rejects_actions_outside_finite_range(invalid_value: float) -> None:
    actions = np.zeros((1, 7), dtype=np.float32)
    actions[0, 0] = invalid_value

    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'robocerebra-smolvla-v1'.*actions.*expected.*actual",
    ):
        robocerebra_binding().validate_action_chunk(actions)


def test_binding_rejects_empty_action_chunk() -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'robocerebra-smolvla-v1'.*actions.*non-empty.*actual",
    ):
        robocerebra_binding().validate_action_chunk(np.empty((0, 7), dtype=np.float32))


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(7, dtype=np.float32),
        np.zeros((1, 2, 7), dtype=np.float32),
    ],
)
def test_binding_requires_two_dimensional_action_chunk(invalid: np.ndarray) -> None:
    with pytest.raises(
        FeatureValidationError, match=r"actions.*expected.*chunk_length, 7"
    ):
        robocerebra_binding().validate_action_chunk(invalid)


def test_binding_retains_allowed_missing_policy_inputs_as_metadata() -> None:
    binding = robocerebra_binding()

    assert binding.allowed_missing_policy_inputs == frozenset(
        {"observation.images.camera3"}
    )


def test_policy_binding_normalizes_mutable_schema_containers() -> None:
    state = FeatureBinding("state", "observation.state", (8,), np.dtype("float32"))
    inputs = [state]
    action_shape = [7]
    allowed_missing = {"observation.images.camera3"}

    binding = PolicyBinding(
        name="immutable",
        inputs=inputs,  # type: ignore[arg-type]
        action_shape=action_shape,  # type: ignore[arg-type]
        action_dtype=np.dtype("float32"),
        action_low=-1.0,
        action_high=1.0,
        allowed_missing_policy_inputs=allowed_missing,  # type: ignore[arg-type]
    )
    inputs.append(
        FeatureBinding("extra", "observation.extra", (1,), np.dtype("float32"))
    )
    action_shape[0] = 12
    allowed_missing.add("observation.images.camera4")

    assert binding.inputs == (state,)
    assert binding.action_shape == (7,)
    assert binding.allowed_missing_policy_inputs == frozenset(
        {"observation.images.camera3"}
    )


def test_policy_binding_rejects_non_feature_input() -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'invalid'.*inputs.*FeatureBinding.*actual object",
    ):
        PolicyBinding(
            name="invalid",
            inputs=[object()],  # type: ignore[list-item]
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
        )


def test_policy_binding_rejects_non_string_allowed_missing_key() -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'invalid'.*allowed-missing.*string.*actual int",
    ):
        PolicyBinding(
            name="invalid",
            inputs=(),
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
            allowed_missing_policy_inputs={1},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_allowed_missing", ["camera3", b"camera3"])
def test_policy_binding_rejects_scalar_allowed_missing_container(
    invalid_allowed_missing: str | bytes,
) -> None:
    with pytest.raises(
        FeatureValidationError,
        match=rf"binding 'invalid'.*allowed-missing.*collection.*actual {type(invalid_allowed_missing).__name__}",
    ):
        PolicyBinding(
            name="invalid",
            inputs=(),
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
            allowed_missing_policy_inputs=invalid_allowed_missing,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("native_key", "policy_key"),
    [
        ("", "observation.state"),
        ("state", ""),
    ],
)
def test_feature_binding_rejects_empty_keys(native_key: str, policy_key: str) -> None:
    with pytest.raises(FeatureValidationError, match="non-empty"):
        FeatureBinding(native_key, policy_key, (8,), np.dtype("float32"))


@pytest.mark.parametrize("key_field", ["native", "policy"])
def test_feature_binding_rejects_ndarray_key_with_consistent_error(
    key_field: str,
) -> None:
    invalid_key = np.array(["one", "two"])
    native_key = invalid_key if key_field == "native" else "state"
    policy_key = invalid_key if key_field == "policy" else "observation.state"

    with pytest.raises(
        FeatureValidationError,
        match=r"binding '<unbound>'.*expected.*string.*actual ndarray",
    ):
        FeatureBinding(
            native_key,  # type: ignore[arg-type]
            policy_key,  # type: ignore[arg-type]
            (8,),
            np.dtype("float32"),
        )


@pytest.mark.parametrize("invalid_shape", [(0,), (-1,), (True,), (1.5,)])
def test_feature_binding_rejects_non_positive_integer_dimensions(
    invalid_shape: tuple[object, ...],
) -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding '<unbound>'.*state.*positive integer dimensions.*actual",
    ):
        FeatureBinding(
            "state",
            "observation.state",
            invalid_shape,  # type: ignore[arg-type]
            np.dtype("float32"),
        )


def test_feature_binding_wraps_invalid_dtype() -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding '<unbound>'.*state.*valid numeric dtype.*actual",
    ):
        FeatureBinding("state", "observation.state", (8,), "not-a-dtype")  # type: ignore[arg-type]


def test_feature_binding_rejects_non_numeric_dtype() -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding '<unbound>'.*state.*numeric dtype.*actual.*str",
    ):
        FeatureBinding("state", "observation.state", (8,), np.dtype("str"))


def test_policy_binding_rejects_empty_name() -> None:
    with pytest.raises(FeatureValidationError, match=r"binding.*name.*non-empty"):
        PolicyBinding(
            name="",
            inputs=(),
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
        )


def test_policy_binding_rejects_ndarray_name_with_consistent_error() -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding '<invalid>'.*name.*expected.*string.*actual ndarray",
    ):
        PolicyBinding(
            name=np.array(["one", "two"]),  # type: ignore[arg-type]
            inputs=(),
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
        )


@pytest.mark.parametrize(
    "invalid_shape",
    [(), (7, 1), (0,), (-1,), (True,), (1.5,)],
)
def test_policy_binding_requires_one_positive_integer_action_dimension(
    invalid_shape: tuple[object, ...],
) -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'invalid'.*action_shape.*one positive integer.*actual",
    ):
        PolicyBinding(
            name="invalid",
            inputs=(),
            action_shape=invalid_shape,  # type: ignore[arg-type]
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
        )


def test_policy_binding_wraps_invalid_action_dtype() -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'invalid'.*action_dtype.*valid floating dtype.*actual",
    ):
        PolicyBinding(
            name="invalid",
            inputs=(),
            action_shape=(7,),
            action_dtype="not-a-dtype",  # type: ignore[arg-type]
            action_low=-1.0,
            action_high=1.0,
        )


def test_policy_binding_rejects_non_floating_action_dtype() -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'invalid'.*action_dtype.*floating dtype.*actual int32",
    ):
        PolicyBinding(
            name="invalid",
            inputs=(),
            action_shape=(7,),
            action_dtype=np.dtype("int32"),
            action_low=-1.0,
            action_high=1.0,
        )


@pytest.mark.parametrize(
    ("action_low", "action_high"),
    [
        (np.nan, 1.0),
        (-1.0, np.nan),
        (-np.inf, 1.0),
        (-1.0, np.inf),
        (1.0, -1.0),
        (False, 1.0),
        (-1.0, True),
        (0j, 1.0),
    ],
)
def test_policy_binding_rejects_invalid_action_bounds(
    action_low: object, action_high: object
) -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'invalid'.*action_bounds.*finite real.*low <= high.*actual",
    ):
        PolicyBinding(
            name="invalid",
            inputs=(),
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=action_low,  # type: ignore[arg-type]
            action_high=action_high,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("action_low", "action_high"),
    [(-(10**1000), 1.0), (-1.0, 10**1000)],
)
def test_policy_binding_wraps_overflowing_integer_action_bounds(
    action_low: int | float, action_high: int | float
) -> None:
    with pytest.raises(
        FeatureValidationError,
        match=r"binding 'invalid'.*action_bounds.*expected.*finite real.*actual",
    ):
        PolicyBinding(
            name="invalid",
            inputs=(),
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=action_low,
            action_high=action_high,
        )


def test_policy_binding_rejects_empty_allowed_missing_key() -> None:
    with pytest.raises(FeatureValidationError, match=r"allowed-missing.*non-empty"):
        PolicyBinding(
            name="invalid",
            inputs=(),
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
            allowed_missing_policy_inputs=frozenset({""}),
        )


@pytest.mark.parametrize("duplicate_kind", ["native", "policy"])
def test_policy_binding_rejects_duplicate_feature_keys(duplicate_kind: str) -> None:
    second = (
        FeatureBinding("state", "observation.extra", (8,), np.dtype("float32"))
        if duplicate_kind == "native"
        else FeatureBinding("extra", "observation.state", (8,), np.dtype("float32"))
    )

    with pytest.raises(FeatureValidationError, match=rf"duplicate {duplicate_kind}"):
        PolicyBinding(
            name="invalid",
            inputs=(
                FeatureBinding("state", "observation.state", (8,), np.dtype("float32")),
                second,
            ),
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
        )


def test_policy_binding_rejects_mapped_allowed_missing_overlap() -> None:
    with pytest.raises(FeatureValidationError, match=r"observation.state.*both"):
        PolicyBinding(
            name="invalid",
            inputs=(
                FeatureBinding("state", "observation.state", (8,), np.dtype("float32")),
            ),
            action_shape=(7,),
            action_dtype=np.dtype("float32"),
            action_low=-1.0,
            action_high=1.0,
            allowed_missing_policy_inputs=frozenset({"observation.state"}),
        )


def test_allowed_missing_does_not_require_complete_checkpoint_input_set() -> None:
    binding = robocerebra_binding()

    assert binding.bind_observation(valid_native_observation())
