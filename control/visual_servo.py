from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import sys

import cv2
import numpy as np


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.airsim_client import AirSimClientAdapter, AirSimConnectionConfig
from app.bootstrap import PROJECT_ROOT, bootstrap_app
from control.pid import PIDController
from vision.aruco_detector import ArucoDetection, ArucoDetector
from vision.frame_fetcher import FrameFetcher


@dataclass
class VisualServoCommand:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    duration_s: float = 0.0
    error_x_px: float = 0.0
    error_y_px: float = 0.0
    marker_detected: bool = False


@dataclass
class VisualServoConfig:
    command_duration_s: float
    max_lateral_velocity_m_s: float
    max_vertical_velocity_m_s: float
    max_yaw_rate_deg_s: float
    yaw_error_deadband_px: float
    vertical_error_deadband_px: float
    lateral_kp: float
    lateral_ki: float
    lateral_kd: float
    vertical_kp: float
    vertical_ki: float
    vertical_kd: float
    yaw_kp: float
    yaw_ki: float
    yaw_kd: float


class VisualServoController:
    def __init__(
        self,
        config: VisualServoConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("drone_cv.visual_servo")
        self.lateral_pid = PIDController(
            kp=config.lateral_kp,
            ki=config.lateral_ki,
            kd=config.lateral_kd,
            output_min=-config.max_lateral_velocity_m_s,
            output_max=config.max_lateral_velocity_m_s,
        )
        self.vertical_pid = PIDController(
            kp=config.vertical_kp,
            ki=config.vertical_ki,
            kd=config.vertical_kd,
            output_min=-config.max_vertical_velocity_m_s,
            output_max=config.max_vertical_velocity_m_s,
        )
        self.yaw_pid = PIDController(
            kp=config.yaw_kp,
            ki=config.yaw_ki,
            kd=config.yaw_kd,
            output_min=-config.max_yaw_rate_deg_s,
            output_max=config.max_yaw_rate_deg_s,
        )

    def compute_command(
        self,
        detection: ArucoDetection,
        frame_width: int,
        frame_height: int,
    ) -> VisualServoCommand:
        if not detection.detected or detection.center_x is None or detection.center_y is None:
            self.reset()
            return VisualServoCommand(duration_s=self.config.command_duration_s)

        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0
        error_x_px = detection.center_x - frame_center_x
        error_y_px = detection.center_y - frame_center_y
        dt = self.config.command_duration_s

        yaw_error = 0.0
        if abs(error_x_px) >= self.config.yaw_error_deadband_px:
            yaw_error = -(error_x_px / frame_center_x)
        yaw_rate = self.yaw_pid.update(error=yaw_error, dt=dt)

        lateral_error = 0.0
        if yaw_error == 0.0:
            lateral_error = -(error_x_px / frame_center_x)
        vy = self.lateral_pid.update(error=lateral_error, dt=dt)

        vertical_error = 0.0
        if abs(error_y_px) >= self.config.vertical_error_deadband_px:
            vertical_error = error_y_px / frame_center_y
        vz = self.vertical_pid.update(error=vertical_error, dt=dt)

        command = VisualServoCommand(
            vx=0.0,
            vy=vy,
            vz=vz,
            yaw_rate=yaw_rate,
            duration_s=self.config.command_duration_s,
            error_x_px=error_x_px,
            error_y_px=error_y_px,
            marker_detected=True,
        )
        self.logger.info("Computed visual-servo command: %s", command)
        return command

    def reset(self) -> None:
        self.lateral_pid.reset()
        self.vertical_pid.reset()
        self.yaw_pid.reset()


def run_visual_servo_step() -> int:
    context = bootstrap_app()
    logger = context["logger"]
    settings = context["settings"]
    airsim_settings = settings.get("airsim", {})
    camera_settings = settings.get("camera", {})
    aruco_settings = settings.get("aruco", {})
    control_settings = settings.get("control", {})

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
    detector = ArucoDetector(dictionary_name=str(aruco_settings.get("dictionary", "DICT_4X4_50")))
    servo = VisualServoController(
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

    frame_bundle = frame_fetcher.fetch()
    target_marker_id = int(aruco_settings.get("marker_id", 0))
    detection = detector.detect(frame_bundle.rgb_bgr, target_marker_id=target_marker_id)
    command = servo.compute_command(
        detection=detection,
        frame_width=frame_bundle.rgb_bgr.shape[1],
        frame_height=frame_bundle.rgb_bgr.shape[0],
    )

    debug_frame = detector.draw_overlay(frame_bundle.rgb_bgr, detection)
    output_path = _save_visual_servo_debug_frame(debug_frame)

    if command.marker_detected:
        adapter.move_by_velocity_body(
            vx_m_s=command.vx,
            vy_m_s=command.vy,
            vz_m_s=command.vz,
            duration_s=command.duration_s,
            yaw_rate_deg_s=command.yaw_rate,
        )
        print(
            "Visual servo command sent: "
            f"vy={command.vy:.3f} vz={command.vz:.3f} yaw_rate={command.yaw_rate:.3f}"
        )
    else:
        logger.warning("Target ArUco marker %s was not detected; no movement command sent.", target_marker_id)
        print("Marker not detected; no movement command sent")

    print(f"Saved visual-servo debug frame to {output_path}")
    return 0


def _save_visual_servo_debug_frame(frame_bgr: np.ndarray) -> Path:
    output_dir = PROJECT_ROOT / "debug_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "visual_servo_debug.png"
    cv2.imwrite(str(output_path), frame_bgr)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_visual_servo_step())
