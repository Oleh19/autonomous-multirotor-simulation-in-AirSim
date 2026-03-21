from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import sys
import time

import cv2
import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.airsim_client import AirSimClientAdapter, AirSimConnectionConfig
from app.bootstrap import PROJECT_ROOT, bootstrap_app
from control.obstacle_avoidance import ObstacleAvoidanceController
from control.visual_servo import VisualServoConfig, VisualServoController
from vision.aruco_detector import ArucoDetection, ArucoDetector
from vision.depth_analyzer import DepthAnalysis, DepthAnalyzer
from vision.frame_fetcher import FrameFetcher


@dataclass
class LandingCommand:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    duration_s: float = 0.0
    trigger_land: bool = False
    safe_stop: bool = False
    marker_visible: bool = False
    aligned: bool = False
    reason: str = "idle"


@dataclass
class PrecisionLandingConfig:
    alignment_tolerance_px: float
    final_marker_area: float
    descend_rate_m_s: float
    descend_step_duration_s: float
    touchdown_altitude_m: float
    lost_marker_limit: int
    loop_pause_s: float
    max_steps: int


class PrecisionLandingController:
    def __init__(
        self,
        config: PrecisionLandingConfig,
        visual_servo: VisualServoController,
        obstacle_avoidance: ObstacleAvoidanceController,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.visual_servo = visual_servo
        self.obstacle_avoidance = obstacle_avoidance
        self.logger = logger or logging.getLogger("drone_cv.precision_landing")
        self.lost_marker_count = 0

    def compute_command(
        self,
        detection: ArucoDetection,
        frame_width: int,
        frame_height: int,
        altitude_m: float,
        depth_analysis: DepthAnalysis,
    ) -> LandingCommand:
        servo_command = self.visual_servo.compute_command(
            detection=detection,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if not detection.detected:
            self.lost_marker_count += 1
            self.visual_servo.reset()
            safe_stop = self.lost_marker_count >= self.config.lost_marker_limit
            return LandingCommand(
                duration_s=self.config.descend_step_duration_s,
                safe_stop=safe_stop,
                reason=(
                    "marker lost during descend; hover and recover"
                    if safe_stop
                    else f"marker temporarily lost ({self.lost_marker_count}/{self.config.lost_marker_limit})"
                ),
            )

        self.lost_marker_count = 0
        aligned = (
            abs(servo_command.error_x_px) <= self.config.alignment_tolerance_px
            and abs(servo_command.error_y_px) <= self.config.alignment_tolerance_px
        )

        if altitude_m <= self.config.touchdown_altitude_m or detection.area >= self.config.final_marker_area:
            return LandingCommand(
                duration_s=self.config.descend_step_duration_s,
                trigger_land=True,
                marker_visible=True,
                aligned=aligned,
                reason="close enough to trigger AirSim landing",
            )

        if not aligned:
            return LandingCommand(
                vx=0.0,
                vy=servo_command.vy,
                vz=servo_command.vz,
                yaw_rate=servo_command.yaw_rate,
                duration_s=self.config.descend_step_duration_s,
                marker_visible=True,
                aligned=False,
                reason="marker not aligned; correcting before descent",
            )

        if depth_analysis.obstacle_detected:
            avoidance_command = self.obstacle_avoidance.compute_command(depth_analysis)
            return LandingCommand(
                vx=0.0,
                vy=avoidance_command.vy,
                vz=0.0,
                yaw_rate=avoidance_command.yaw_rate,
                duration_s=avoidance_command.duration_s,
                marker_visible=True,
                aligned=True,
                reason="obstacle detected during descent; applying avoidance",
            )

        return LandingCommand(
            vx=0.0,
            vy=servo_command.vy,
            vz=self.config.descend_rate_m_s,
            yaw_rate=servo_command.yaw_rate,
            duration_s=self.config.descend_step_duration_s,
            marker_visible=True,
            aligned=True,
            reason="marker aligned; descending one step",
        )


def run_precision_landing() -> int:
    context = bootstrap_app()
    logger = context["logger"]
    settings = context["settings"]
    airsim_settings = settings.get("airsim", {})
    camera_settings = settings.get("camera", {})
    aruco_settings = settings.get("aruco", {})
    control_settings = settings.get("control", {})
    depth_settings = settings.get("depth", {})
    landing_settings = settings.get("landing", {})

    adapter = AirSimClientAdapter(
        config=AirSimConnectionConfig(
            host=str(airsim_settings.get("host", "127.0.0.1")),
            port=int(airsim_settings.get("port", 41451)),
            timeout_seconds=float(airsim_settings.get("timeout_seconds", 10.0)),
            vehicle_name=str(airsim_settings.get("vehicle_name", "")),
        ),
        logger=logger,
    )
    adapter.connect()
    adapter.confirm_connection()
    adapter.enable_api_control(True)

    frame_fetcher = FrameFetcher(
        adapter=adapter,
        rgb_camera_name=str(camera_settings.get("rgb_camera_name", "front_center")),
        depth_camera_name=str(camera_settings.get("depth_camera_name", "front_center")),
        logger=logger,
    )
    detector = ArucoDetector(
        dictionary_name=str(aruco_settings.get("dictionary", "DICT_4X4_50"))
    )
    visual_servo = VisualServoController(
        config=VisualServoConfig(
            command_duration_s=float(control_settings.get("servo_command_duration_s", 0.2)),
            max_lateral_velocity_m_s=float(control_settings.get("servo_max_lateral_velocity_m_s", 0.5)),
            max_vertical_velocity_m_s=float(control_settings.get("servo_max_vertical_velocity_m_s", 0.4)),
            max_yaw_rate_deg_s=float(control_settings.get("servo_max_yaw_rate_deg_s", 10.0)),
            yaw_error_deadband_px=float(control_settings.get("servo_yaw_error_deadband_px", 10.0)),
            lateral_kp=float(control_settings.get("servo_lateral_kp", 0.4)),
            lateral_ki=float(control_settings.get("servo_lateral_ki", 0.0)),
            lateral_kd=float(control_settings.get("servo_lateral_kd", 0.05)),
            vertical_kp=float(control_settings.get("servo_vertical_kp", 0.35)),
            vertical_ki=float(control_settings.get("servo_vertical_ki", 0.0)),
            vertical_kd=float(control_settings.get("servo_vertical_kd", 0.04)),
            yaw_kp=float(control_settings.get("servo_yaw_kp", 5.0)),
            yaw_ki=float(control_settings.get("servo_yaw_ki", 0.0)),
            yaw_kd=float(control_settings.get("servo_yaw_kd", 0.2)),
        ),
        logger=logger,
    )
    obstacle_avoidance = ObstacleAvoidanceController(
        avoidance_speed_m_s=float(depth_settings.get("avoidance_speed_m_s", 0.5)),
        yaw_rate_deg_s=float(depth_settings.get("avoidance_yaw_rate_deg_s", 12.0)),
        command_duration_s=float(depth_settings.get("avoidance_command_duration_s", 0.25)),
        logger=logger,
    )
    depth_analyzer = DepthAnalyzer(
        obstacle_distance_m=float(depth_settings.get("obstacle_distance_m", 2.0)),
        min_valid_depth_m=float(depth_settings.get("min_valid_depth_m", 0.2)),
        max_valid_depth_m=float(depth_settings.get("max_valid_depth_m", 20.0)),
    )
    landing_controller = PrecisionLandingController(
        config=PrecisionLandingConfig(
            alignment_tolerance_px=float(landing_settings.get("alignment_tolerance_px", 24.0)),
            final_marker_area=float(landing_settings.get("final_marker_area", 22000.0)),
            descend_rate_m_s=float(landing_settings.get("final_descent_rate_m_s", 0.3)),
            descend_step_duration_s=float(landing_settings.get("descend_step_duration_s", 0.25)),
            touchdown_altitude_m=float(landing_settings.get("touchdown_altitude_m", 0.15)),
            lost_marker_limit=int(landing_settings.get("lost_marker_limit", 3)),
            loop_pause_s=float(landing_settings.get("loop_pause_s", 0.1)),
            max_steps=int(landing_settings.get("max_steps", 80)),
        ),
        visual_servo=visual_servo,
        obstacle_avoidance=obstacle_avoidance,
        logger=logger,
    )

    target_marker_id = int(aruco_settings.get("marker_id", 0))
    for step_index in range(landing_controller.config.max_steps):
        frame_bundle = frame_fetcher.fetch()
        detection = detector.detect(frame_bundle.rgb_bgr, target_marker_id=target_marker_id)
        depth_analysis = depth_analyzer.analyze(frame_bundle.depth_m)
        telemetry = adapter.get_telemetry()
        command = landing_controller.compute_command(
            detection=detection,
            frame_width=frame_bundle.rgb_bgr.shape[1],
            frame_height=frame_bundle.rgb_bgr.shape[0],
            altitude_m=telemetry.altitude_m,
            depth_analysis=depth_analysis,
        )

        debug_frame = detector.draw_overlay(frame_bundle.rgb_bgr, detection)
        _draw_landing_status(debug_frame, step_index + 1, command, telemetry.altitude_m)
        output_path = _save_landing_debug_frame(debug_frame)

        print(
            f"step={step_index + 1} altitude={telemetry.altitude_m:.2f} "
            f"marker_detected={detection.detected} reason={command.reason} debug={output_path}"
        )

        if command.safe_stop:
            adapter.hover()
            print("Precision landing aborted safely after marker loss")
            return 1

        if command.trigger_land:
            adapter.land()
            adapter.arm(False)
            adapter.enable_api_control(False)
            print("Precision landing triggered through AirSim adapter")
            print(f"Saved precision-landing debug frame to {output_path}")
            return 0

        if command.marker_visible:
            adapter.move_by_velocity_body(
                vx_m_s=command.vx,
                vy_m_s=command.vy,
                vz_m_s=command.vz,
                duration_s=command.duration_s,
                yaw_rate_deg_s=command.yaw_rate,
            )
        else:
            adapter.hover()

        time.sleep(landing_controller.config.loop_pause_s)

    adapter.hover()
    print("Precision landing step limit exceeded; hovering for safety")
    return 1


def _draw_landing_status(
    frame_bgr: np.ndarray,
    step_index: int,
    command: LandingCommand,
    altitude_m: float,
) -> None:
    lines = [
        f"Landing Step: {step_index}",
        f"Altitude: {altitude_m:.2f} m",
        f"Reason: {command.reason}",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28


def _save_landing_debug_frame(frame_bgr: np.ndarray) -> Path:
    output_dir = PROJECT_ROOT / "debug_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "precision_landing_debug.png"
    cv2.imwrite(str(output_path), frame_bgr)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_precision_landing())
