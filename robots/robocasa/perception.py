"""Non-privileged RGB-D geometry for RoboCasa camera observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _quaternion_xyzw_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def localize_rgbd_points(
    frame: Mapping[str, Any],
    state: Any,
    *,
    expected_step: int,
    points: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Back-project selected pixels and express their median in the base frame."""

    if frame.get("step_id") != expected_step:
        raise ValueError("RGB-D frame step does not match expected_step")
    camera = str(frame["camera_name"])
    rgb = np.asarray(frame["rgb"])
    depth = np.asarray(frame["depth_m"], dtype=np.float64)
    intrinsic = np.asarray(frame["intrinsic_K"], dtype=np.float64)
    extrinsic = np.asarray(frame["extrinsic_cam2world"], dtype=np.float64)
    state_array = np.asarray(state, dtype=np.float64)
    if rgb.dtype != np.uint8:
        raise ValueError("RGB-D frame RGB must use uint8 pixels")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError("RGB-D frame shapes are inconsistent")
    if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError("RGB-D calibration shapes are invalid")
    if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
        raise ValueError("RGB-D calibration must contain finite values")
    if state_array.shape != (16,):
        raise ValueError("RoboCasa localization state must have shape (16,)")
    if not np.isfinite(state_array).all():
        raise ValueError("RoboCasa localization state must contain finite values")
    if not 1 <= len(points) <= 32:
        raise ValueError("RGB-D points must contain between 1 and 32 pixels")
    quaternion = state_array[3:7]
    if np.linalg.norm(quaternion) <= 1e-8:
        raise ValueError("RoboCasa base quaternion must have non-zero norm")

    inverse_intrinsic = np.linalg.inv(intrinsic)
    inverse_extrinsic = np.linalg.inv(extrinsic)
    world_samples: list[np.ndarray] = []
    depths: list[float] = []
    reprojection_errors: list[float] = []
    height, width = depth.shape
    for point in points:
        if (
            not isinstance(point, Sequence)
            or len(point) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in point)
        ):
            raise ValueError("RGB-D point must contain integer row and col")
        row, col = point
        if not 0 <= row < height or not 0 <= col < width:
            raise ValueError("RGB-D point is outside the camera frame")
        depth_m = float(depth[row, col])
        if not np.isfinite(depth_m) or depth_m <= 0.0:
            continue
        camera_xyz = inverse_intrinsic @ np.array(
            [float(col), float(row), 1.0], dtype=np.float64
        )
        camera_xyz *= depth_m
        world_h = extrinsic @ np.append(camera_xyz, 1.0)
        world_xyz = world_h[:3]
        world_samples.append(world_xyz)
        depths.append(depth_m)

        projected_camera = (inverse_extrinsic @ np.append(world_xyz, 1.0))[:3]
        projected_pixel = intrinsic @ projected_camera
        projected_col = projected_pixel[0] / projected_pixel[2]
        projected_row = projected_pixel[1] / projected_pixel[2]
        reprojection_errors.append(
            float(np.hypot(projected_row - row, projected_col - col))
        )

    if not world_samples:
        raise ValueError("RGB-D points contain no valid metric depth")
    stacked = np.stack(world_samples)
    world_xyz = np.median(stacked, axis=0)
    base_rotation = _quaternion_xyzw_matrix(quaternion)
    relative_base_xyz = base_rotation.T @ (world_xyz - state_array[:3])
    spread_m = float(np.max(np.linalg.norm(stacked - world_xyz, axis=1)))
    return {
        "camera": camera,
        "step": int(expected_step),
        "valid_samples": len(world_samples),
        "world_xyz": [float(value) for value in world_xyz],
        "relative_base_xyz": [float(value) for value in relative_base_xyz],
        "spread_m": spread_m,
        "sample_depth_range_m": [min(depths), max(depths)],
        "reprojection_errors_px": reprojection_errors,
        "max_reprojection_error_px": max(reprojection_errors),
    }


__all__ = ["localize_rgbd_points"]
