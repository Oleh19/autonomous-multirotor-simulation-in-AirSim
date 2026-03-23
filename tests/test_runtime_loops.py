from __future__ import annotations

from app.runtime_loops import build_target_tracking_command, should_keep_recent_target_lock
from control.visual_servo import VisualServoCommand
from vision.aruco_detector import ArucoDetection


def test_build_target_tracking_command_keeps_forward_motion_when_marker_is_visible_off_center() -> None:
    servo_command = VisualServoCommand(
        vy=0.12,
        vz=-0.05,
        yaw_rate=3.0,
        duration_s=0.2,
        marker_detected=True,
    )

    command = build_target_tracking_command(
        servo_command,
        aligned=False,
        approach_forward_speed_m_s=0.2,
        visible_tracking_forward_speed_m_s=0.08,
    )

    assert command.vx == 0.08
    assert command.vy == 0.12
    assert command.vz == -0.05
    assert command.yaw_rate == 3.0
    assert "track and re-center" in command.reason


def test_build_target_tracking_command_uses_full_approach_speed_when_aligned() -> None:
    servo_command = VisualServoCommand(
        vy=0.01,
        vz=0.0,
        yaw_rate=0.2,
        duration_s=0.2,
        marker_detected=True,
    )

    command = build_target_tracking_command(
        servo_command,
        aligned=True,
        approach_forward_speed_m_s=0.2,
        visible_tracking_forward_speed_m_s=0.08,
    )

    assert command.vx == 0.2
    assert command.vy == 0.01
    assert command.yaw_rate == 0.2
    assert "continuous approach" in command.reason


def test_build_target_tracking_command_scales_forward_speed_with_error() -> None:
    command = build_target_tracking_command(
        VisualServoCommand(
            error_x_px=220.0,
            error_y_px=0.0,
            vy=0.1,
            vz=0.0,
            yaw_rate=4.0,
            duration_s=0.2,
            marker_detected=True,
        ),
        aligned=False,
        approach_forward_speed_m_s=0.2,
        visible_tracking_forward_speed_m_s=0.08,
        frame_width=640,
        frame_height=480,
    )

    assert 0.08 <= command.vx < 0.2


def test_should_keep_recent_target_lock_uses_recent_detection_memory() -> None:
    recent_detection = ArucoDetection(
        detected=True,
        marker_id=0,
        center_x=300.0,
        center_y=220.0,
        corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        area=9000.0,
    )

    detection = should_keep_recent_target_lock(
        target_locked=True,
        detection=ArucoDetection(False, None, None, None, (), 0.0),
        last_target_detection=recent_detection,
        last_target_seen_at_s=10.0,
        now_s=10.4,
        target_memory_timeout_s=0.75,
    )

    assert detection is recent_detection
