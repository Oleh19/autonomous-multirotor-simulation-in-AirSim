from __future__ import annotations

import logging

from control.landing_controller import PrecisionLandingConfig, PrecisionLandingController
from control.obstacle_avoidance import ObstacleAvoidanceCommand
from control.visual_servo import VisualServoCommand
from vision.aruco_detector import ArucoDetection
from vision.depth_analyzer import DepthAnalysis


class _FakeVisualServo:
    def __init__(self, commands: list[VisualServoCommand]) -> None:
        self._commands = commands
        self.reset_calls = 0

    def compute_command(
        self,
        detection: ArucoDetection,
        frame_width: int,
        frame_height: int,
    ) -> VisualServoCommand:
        return self._commands.pop(0)

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeObstacleAvoidance:
    def __init__(self, command: ObstacleAvoidanceCommand) -> None:
        self.command = command

    def compute_command(self, analysis: DepthAnalysis) -> ObstacleAvoidanceCommand:
        return self.command


def _controller(
    visual_servo: _FakeVisualServo,
    obstacle_command: ObstacleAvoidanceCommand | None = None,
) -> PrecisionLandingController:
    return PrecisionLandingController(
        config=PrecisionLandingConfig(
            alignment_tolerance_px=24.0,
            final_marker_area=22000.0,
            descend_rate_m_s=0.3,
            descend_step_duration_s=0.25,
            touchdown_altitude_m=0.15,
            lost_marker_limit=2,
            loop_pause_s=0.1,
            max_steps=80,
        ),
        visual_servo=visual_servo,
        obstacle_avoidance=_FakeObstacleAvoidance(
            obstacle_command
            or ObstacleAvoidanceCommand(vy=0.5, yaw_rate=12.0, duration_s=0.25, chosen_side="right")
        ),
        logger=logging.getLogger("test.precision_landing"),
    )


def _detection(
    *,
    detected: bool,
    area: float = 12000.0,
    center_x: float | None = 320.0,
    center_y: float | None = 240.0,
) -> ArucoDetection:
    return ArucoDetection(
        detected=detected,
        marker_id=7 if detected else None,
        center_x=center_x if detected else None,
        center_y=center_y if detected else None,
        corners=((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)) if detected else (),
        area=area,
    )


def _clear_depth() -> DepthAnalysis:
    return DepthAnalysis(
        obstacle_detected=False,
        nearest_distance_m=5.0,
        left_clearance_m=5.0,
        center_clearance_m=5.0,
        right_clearance_m=5.0,
        safer_side="left",
    )


def test_precision_landing_safe_stops_after_repeated_marker_loss() -> None:
    controller = _controller(_FakeVisualServo(commands=[VisualServoCommand(), VisualServoCommand()]))

    first_command = controller.compute_command(
        detection=_detection(detected=False),
        frame_width=640,
        frame_height=480,
        altitude_m=0.8,
        depth_analysis=_clear_depth(),
    )
    second_command = controller.compute_command(
        detection=_detection(detected=False),
        frame_width=640,
        frame_height=480,
        altitude_m=0.8,
        depth_analysis=_clear_depth(),
    )

    assert first_command.safe_stop is False
    assert second_command.safe_stop is True
    assert "hover and recover" in second_command.reason


def test_precision_landing_uses_avoidance_when_obstacle_detected() -> None:
    controller = _controller(
        _FakeVisualServo(commands=[VisualServoCommand(vy=0.0, vz=0.0, yaw_rate=0.0, duration_s=0.25, marker_detected=True)]),
        obstacle_command=ObstacleAvoidanceCommand(vy=0.5, yaw_rate=12.0, duration_s=0.25, chosen_side="right"),
    )

    command = controller.compute_command(
        detection=_detection(detected=True, area=15000.0),
        frame_width=640,
        frame_height=480,
        altitude_m=0.8,
        depth_analysis=DepthAnalysis(
            obstacle_detected=True,
            nearest_distance_m=1.0,
            left_clearance_m=1.5,
            center_clearance_m=1.0,
            right_clearance_m=2.0,
            safer_side="right",
        ),
    )

    assert command.marker_visible is True
    assert command.aligned is True
    assert command.vy == 0.5
    assert command.yaw_rate == 12.0
    assert "avoidance" in command.reason
