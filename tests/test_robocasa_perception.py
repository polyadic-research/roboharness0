from __future__ import annotations

import numpy as np
import pytest

from robots.robocasa.perception import localize_rgbd_points


def test_back_projects_pixels_and_converts_world_to_rotated_base() -> None:
    frame = {
        "step_id": 7,
        "camera_name": "robot0_agentview_center",
        "rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "depth_m": np.full((4, 6), 2.0, dtype=np.float32),
        "intrinsic_K": np.array(
            [[2.0, 0.0, 3.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "extrinsic_cam2world": np.eye(4, dtype=np.float64),
    }
    state = np.zeros(16, dtype=np.float32)
    state[:3] = [0.5, -0.5, 0.0]
    state[3:7] = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]

    result = localize_rgbd_points(
        frame,
        state,
        expected_step=7,
        points=[[2, 4]],
    )

    np.testing.assert_allclose(result["world_xyz"], [1.0, 0.0, 2.0])
    np.testing.assert_allclose(result["relative_base_xyz"], [0.5, -0.5, 2.0])
    assert result["camera"] == "robot0_agentview_center"
    assert result["step"] == 7
    assert result["valid_samples"] == 1
    assert result["max_reprojection_error_px"] < 1e-6


def test_rejects_more_than_32_selected_pixels() -> None:
    frame = {
        "step_id": 0,
        "camera_name": "robot0_agentview_left",
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth_m": np.ones((8, 8), dtype=np.float32),
        "intrinsic_K": np.eye(3),
        "extrinsic_cam2world": np.eye(4),
    }
    state = np.zeros(16, dtype=np.float32)
    state[6] = 1.0

    with pytest.raises(ValueError, match="between 1 and 32"):
        localize_rgbd_points(
            frame,
            state,
            expected_step=0,
            points=[[0, 0]] * 33,
        )


def test_rejects_non_uint8_rgb() -> None:
    frame = {
        "step_id": 0,
        "camera_name": "robot0_agentview_left",
        "rgb": np.zeros((8, 8, 3), dtype=np.float32),
        "depth_m": np.ones((8, 8), dtype=np.float32),
        "intrinsic_K": np.eye(3),
        "extrinsic_cam2world": np.eye(4),
    }
    state = np.zeros(16, dtype=np.float32)
    state[6] = 1.0

    with pytest.raises(ValueError, match="uint8"):
        localize_rgbd_points(
            frame,
            state,
            expected_step=0,
            points=[[0, 0]],
        )


def test_rejects_nonfinite_camera_calibration() -> None:
    frame = {
        "step_id": 0,
        "camera_name": "robot0_agentview_left",
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth_m": np.ones((8, 8), dtype=np.float32),
        "intrinsic_K": np.array(
            [[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "extrinsic_cam2world": np.eye(4),
    }
    state = np.zeros(16, dtype=np.float32)
    state[6] = 1.0

    with pytest.raises(ValueError, match="finite"):
        localize_rgbd_points(
            frame,
            state,
            expected_step=0,
            points=[[0, 0]],
        )


def test_rejects_noninteger_pixel_coordinates() -> None:
    frame = {
        "step_id": 0,
        "camera_name": "robot0_agentview_left",
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth_m": np.ones((8, 8), dtype=np.float32),
        "intrinsic_K": np.eye(3),
        "extrinsic_cam2world": np.eye(4),
    }
    state = np.zeros(16, dtype=np.float32)
    state[6] = 1.0

    with pytest.raises(ValueError, match="integer.*row.*col"):
        localize_rgbd_points(
            frame,
            state,
            expected_step=0,
            points=[[0.5, 0]],
        )


def test_rejects_nonfinite_public_state() -> None:
    frame = {
        "step_id": 0,
        "camera_name": "robot0_agentview_left",
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth_m": np.ones((8, 8), dtype=np.float32),
        "intrinsic_K": np.eye(3),
        "extrinsic_cam2world": np.eye(4),
    }
    state = np.zeros(16, dtype=np.float32)
    state[0] = np.nan
    state[6] = 1.0

    with pytest.raises(ValueError, match="state.*finite"):
        localize_rgbd_points(
            frame,
            state,
            expected_step=0,
            points=[[0, 0]],
        )


def test_rejects_stale_frame_without_geometry_result() -> None:
    frame = {
        "step_id": 3,
        "camera_name": "robot0_agentview_left",
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "depth_m": np.ones((2, 2), dtype=np.float32),
        "intrinsic_K": np.eye(3),
        "extrinsic_cam2world": np.eye(4),
    }
    state = np.zeros(16, dtype=np.float32)
    state[6] = 1.0

    with pytest.raises(ValueError, match="step.*expected_step"):
        localize_rgbd_points(frame, state, expected_step=4, points=[[0, 0]])


def test_rejects_zero_norm_base_quaternion() -> None:
    frame = {
        "step_id": 0,
        "camera_name": "robot0_agentview_left",
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "depth_m": np.ones((2, 2), dtype=np.float32),
        "intrinsic_K": np.eye(3),
        "extrinsic_cam2world": np.eye(4),
    }

    with pytest.raises(ValueError, match="quaternion.*non-zero"):
        localize_rgbd_points(
            frame,
            np.zeros(16, dtype=np.float32),
            expected_step=0,
            points=[[0, 0]],
        )


def test_rejects_out_of_bounds_pixel() -> None:
    frame = {
        "step_id": 0,
        "camera_name": "robot0_agentview_left",
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "depth_m": np.ones((2, 2), dtype=np.float32),
        "intrinsic_K": np.eye(3),
        "extrinsic_cam2world": np.eye(4),
    }
    state = np.zeros(16, dtype=np.float32)
    state[6] = 1.0

    with pytest.raises(ValueError, match="outside"):
        localize_rgbd_points(frame, state, expected_step=0, points=[[2, 1]])


def test_rejects_selection_without_valid_metric_depth() -> None:
    frame = {
        "step_id": 0,
        "camera_name": "robot0_agentview_left",
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "depth_m": np.zeros((2, 2), dtype=np.float32),
        "intrinsic_K": np.eye(3),
        "extrinsic_cam2world": np.eye(4),
    }
    state = np.zeros(16, dtype=np.float32)
    state[6] = 1.0

    with pytest.raises(ValueError, match="no valid metric depth"):
        localize_rgbd_points(frame, state, expected_step=0, points=[[0, 0]])


def test_coordinate_median_resists_one_background_depth_outlier() -> None:
    depth = np.zeros((10, 10), dtype=np.float32)
    target_points = [[5, 5], [5, 6], [6, 5], [6, 6], [4, 5]]
    for row, col in target_points:
        depth[row, col] = 1.0
    depth[9, 9] = 5.0
    frame = {
        "step_id": 2,
        "camera_name": "robot0_agentview_center",
        "rgb": np.zeros((10, 10, 3), dtype=np.uint8),
        "depth_m": depth,
        "intrinsic_K": np.array(
            [[1000.0, 0.0, 5.0], [0.0, 1000.0, 5.0], [0.0, 0.0, 1.0]]
        ),
        "extrinsic_cam2world": np.eye(4),
    }
    state = np.zeros(16, dtype=np.float32)
    state[6] = 1.0

    result = localize_rgbd_points(
        frame,
        state,
        expected_step=2,
        points=[*target_points, [9, 9]],
    )

    assert result["valid_samples"] == 6
    assert result["world_xyz"][2] == pytest.approx(1.0)
    assert result["sample_depth_range_m"] == [1.0, 5.0]
    assert result["spread_m"] > 3.9
