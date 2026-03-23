from __future__ import annotations

from app.local_world import build_initial_local_world, project_local_marker, render_local_world_frame
from app.runtime_loops import _build_local_partial_detection
from vision.aruco_detector import ArucoDetector


def test_build_initial_local_world_loads_multiple_obstacles() -> None:
    settings = {
        "local_world": {
            "width_m": 18.0,
            "height_m": 12.0,
            "marker_x_m": 13.5,
            "marker_y_m": 0.0,
            "camera_fov_deg": 70.0,
            "marker_scale_px_m": 260.0,
            "desired_altitude_m": 1.2,
            "min_altitude_m": 0.15,
            "max_altitude_m": 3.0,
            "obstacles": [
                {"x_m": 3.0, "y_m": 0.0, "radius_m": 0.7},
                {"x_m": 6.0, "y_m": 1.5, "radius_m": 0.9},
            ],
        }
    }

    world = build_initial_local_world(settings)

    assert world.width_m == 18.0
    assert len(world.obstacles) == 2
    assert world.obstacles[0].x_m == 3.0
    assert world.obstacles[1].radius_m == 0.9


def test_render_local_world_frame_marks_nearest_visible_obstacle() -> None:
    settings = {
        "aruco": {"dictionary": "DICT_4X4_50", "marker_id": 0},
        "camera": {"image_width": 320, "image_height": 240},
        "local_world": {
            "width_m": 18.0,
            "height_m": 12.0,
            "marker_x_m": 13.5,
            "marker_y_m": 0.0,
            "camera_fov_deg": 70.0,
            "marker_scale_px_m": 260.0,
            "desired_altitude_m": 1.2,
            "min_altitude_m": 0.15,
            "max_altitude_m": 3.0,
            "obstacles": [
                {"x_m": 3.0, "y_m": 0.2, "radius_m": 0.8},
                {"x_m": 7.0, "y_m": 2.2, "radius_m": 1.0},
            ],
        },
    }
    world = build_initial_local_world(settings)

    frame, depth, marker_visible, marker_distance_m, obstacle_distance_m, obstacle_side = (
        render_local_world_frame(settings, world)
    )

    assert frame.shape == (240, 320, 3)
    assert depth.shape == (240, 320)
    assert marker_visible is True
    assert marker_distance_m > 0.0
    assert obstacle_distance_m is not None
    assert obstacle_distance_m < 4.0
    assert obstacle_side == "right"
    assert float(depth.min()) < 8.0


def test_rendered_local_marker_remains_detectable_with_obstacles_in_scene() -> None:
    settings = {
        "aruco": {"dictionary": "DICT_4X4_50", "marker_id": 0},
        "camera": {"image_width": 640, "image_height": 480},
        "local_world": {
            "width_m": 18.0,
            "height_m": 12.0,
            "marker_x_m": 8.5,
            "marker_y_m": 0.0,
            "camera_fov_deg": 70.0,
            "marker_scale_px_m": 340.0,
            "desired_altitude_m": 1.2,
            "min_altitude_m": 0.15,
            "max_altitude_m": 3.0,
            "obstacles": [
                {"x_m": 4.2, "y_m": -1.2, "radius_m": 0.75},
                {"x_m": 4.2, "y_m": 1.2, "radius_m": 0.75},
            ],
        },
    }
    world = build_initial_local_world(settings)
    detector = ArucoDetector(dictionary_name="DICT_4X4_50")

    frame, _, marker_visible, _, _, _ = render_local_world_frame(settings, world)
    detection = detector.detect(frame, target_marker_id=0)

    assert marker_visible is True
    assert detection.detected is True
    assert detection.marker_id == 0


def test_project_local_marker_reports_partial_visibility_near_frame_edge() -> None:
    settings = {
        "camera": {"image_width": 320, "image_height": 240},
        "local_world": {
            "width_m": 18.0,
            "height_m": 12.0,
            "marker_x_m": 5.0,
            "marker_y_m": 3.8,
            "camera_fov_deg": 70.0,
            "marker_scale_px_m": 340.0,
            "desired_altitude_m": 1.2,
            "min_altitude_m": 0.15,
            "max_altitude_m": 3.0,
            "obstacles": [],
            "obstacle_x_m": 3.2,
            "obstacle_y_m": 0.6,
            "obstacle_radius_m": 0.55,
        },
    }
    world = build_initial_local_world(settings)

    projection = project_local_marker(settings, world)

    assert projection.visible_in_front is True
    assert projection.visible_in_frame is True
    assert projection.clipped_right_px <= 320
    assert projection.clipped_left_px >= 0
    assert projection.clipped_right_px == 320
    assert projection.clipped_left_px < 320


def test_local_partial_detection_hint_is_available_when_marker_is_partially_visible() -> None:
    settings = {
        "camera": {"image_width": 320, "image_height": 240},
        "local_world": {
            "width_m": 18.0,
            "height_m": 12.0,
            "marker_x_m": 5.0,
            "marker_y_m": 3.8,
            "camera_fov_deg": 70.0,
            "marker_scale_px_m": 340.0,
            "desired_altitude_m": 1.2,
            "min_altitude_m": 0.15,
            "max_altitude_m": 3.0,
            "obstacles": [],
            "obstacle_x_m": 3.2,
            "obstacle_y_m": 0.6,
            "obstacle_radius_m": 0.55,
        },
    }
    world = build_initial_local_world(settings)

    detection = _build_local_partial_detection(settings, world, target_marker_id=0)

    assert detection is not None
    assert detection.detected is True
    assert detection.marker_id == 0
    assert detection.area > 0.0
