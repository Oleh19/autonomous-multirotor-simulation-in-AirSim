from control.visual_servo import VisualServoConfig, VisualServoController
from vision.aruco_detector import ArucoDetection


def _controller() -> VisualServoController:
    return VisualServoController(
        config=VisualServoConfig(
            command_duration_s=0.2,
            max_lateral_velocity_m_s=0.5,
            max_vertical_velocity_m_s=0.4,
            max_yaw_rate_deg_s=10.0,
            yaw_error_deadband_px=10.0,
            vertical_error_deadband_px=12.0,
            lateral_kp=0.4,
            lateral_ki=0.0,
            lateral_kd=0.05,
            vertical_kp=0.35,
            vertical_ki=0.0,
            vertical_kd=0.04,
            yaw_kp=5.0,
            yaw_ki=0.0,
            yaw_kd=0.2,
        )
    )


def test_horizontal_error_prefers_yaw_before_lateral_strafe() -> None:
    controller = _controller()
    detection = ArucoDetection(
        detected=True,
        marker_id=7,
        center_x=420.0,
        center_y=240.0,
        corners=((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)),
        area=9000.0,
    )

    command = controller.compute_command(detection=detection, frame_width=640, frame_height=480)

    assert command.yaw_rate != 0.0
    assert command.vy == 0.0


def test_small_vertical_error_stays_stable() -> None:
    controller = _controller()
    detection = ArucoDetection(
        detected=True,
        marker_id=7,
        center_x=320.0,
        center_y=246.0,
        corners=((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)),
        area=9000.0,
    )

    command = controller.compute_command(detection=detection, frame_width=640, frame_height=480)

    assert command.vz == 0.0
