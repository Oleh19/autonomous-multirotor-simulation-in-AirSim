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

from app.bootstrap import (
    PROJECT_ROOT,
    bootstrap_app,
    build_airsim_adapter,
    build_aruco_detector,
    build_depth_analyzer,
    build_frame_fetcher,
    build_obstacle_avoidance_controller,
)
from control.obstacle_avoidance import ObstacleAvoidanceController
from control.visual_servo import VisualServoController, build_visual_servo_controller
from vision.aruco_detector import ArucoDetection
from vision.depth_analyzer import DepthAnalysis


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
    control_settings = settings.get("control", {})
    landing_settings = settings.get("landing", {})

    adapter = build_airsim_adapter(settings, logger)
    adapter.connect()
    adapter.confirm_connection()
    adapter.enable_api_control(True)

    frame_fetcher = build_frame_fetcher(settings, adapter, logger)
    detector = build_aruco_detector(settings)
    visual_servo = build_visual_servo_controller(control_settings, logger=logger)
    obstacle_avoidance = build_obstacle_avoidance_controller(settings, logger)
    depth_analyzer = build_depth_analyzer(settings)
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

    target_marker_id = int(settings.get("aruco", {}).get("marker_id", 0))
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
