from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for bare environments
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as settings_file:
        content = settings_file.read()

    if yaml is not None:
        return yaml.safe_load(content) or {}

    return _load_simple_yaml(content)


def configure_logging(level_name: str) -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("drone_cv")


def bootstrap_app() -> dict[str, Any]:
    settings = load_settings()
    logger = configure_logging(settings.get("app", {}).get("log_level", "INFO"))
    logger.debug("Settings loaded from %s", SETTINGS_PATH)
    return {"settings": settings, "logger": logger}


def validate_settings(settings: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_paths = [
        ("app", "name"),
        ("app", "log_level"),
        ("airsim", "host"),
        ("airsim", "port"),
        ("camera", "rgb_camera_name"),
        ("camera", "depth_camera_name"),
        ("aruco", "dictionary"),
        ("aruco", "marker_id"),
    ]
    for section, key in required_paths:
        section_data = settings.get(section)
        if not isinstance(section_data, dict):
            errors.append(f"Missing config section: {section}")
            continue
        if key not in section_data:
            errors.append(f"Missing config value: {section}.{key}")

    port = settings.get("airsim", {}).get("port")
    if isinstance(port, int):
        if port <= 0:
            errors.append("Config value airsim.port must be greater than 0")
    else:
        errors.append("Config value airsim.port must be an integer")

    runtime_duration = settings.get("runtime", {}).get("run_duration_s")
    if runtime_duration is not None and float(runtime_duration) <= 0:
        errors.append("Config value runtime.run_duration_s must be greater than 0")

    return errors


def build_runtime_components(settings: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    from adapters.airsim_client import AirSimClientAdapter, AirSimConnectionConfig
    from control.obstacle_avoidance import ObstacleAvoidanceController
    from control.visual_servo import VisualServoConfig, VisualServoController
    from telemetry.recorder import TelemetryRecorder
    from telemetry.models import RuntimeSharedState
    from vision.aruco_detector import ArucoDetector
    from vision.depth_analyzer import DepthAnalyzer
    from vision.frame_fetcher import FrameFetcher

    airsim_settings = settings.get("airsim", {})
    camera_settings = settings.get("camera", {})
    aruco_settings = settings.get("aruco", {})
    control_settings = settings.get("control", {})
    depth_settings = settings.get("depth", {})
    recording_settings = settings.get("recording", {})

    adapter = AirSimClientAdapter(
        config=AirSimConnectionConfig(
            host=str(airsim_settings.get("host", "127.0.0.1")),
            port=int(airsim_settings.get("port", 41451)),
            timeout_seconds=float(airsim_settings.get("timeout_seconds", 10.0)),
            vehicle_name=str(airsim_settings.get("vehicle_name", "")),
        ),
        logger=logger,
    )
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
    depth_analyzer = DepthAnalyzer(
        obstacle_distance_m=float(depth_settings.get("obstacle_distance_m", 2.0)),
        min_valid_depth_m=float(depth_settings.get("min_valid_depth_m", 0.2)),
        max_valid_depth_m=float(depth_settings.get("max_valid_depth_m", 20.0)),
    )
    obstacle_avoidance = ObstacleAvoidanceController(
        avoidance_speed_m_s=float(depth_settings.get("avoidance_speed_m_s", 0.5)),
        yaw_rate_deg_s=float(depth_settings.get("avoidance_yaw_rate_deg_s", 12.0)),
        command_duration_s=float(depth_settings.get("avoidance_command_duration_s", 0.25)),
        logger=logger,
    )
    recording_dir = PROJECT_ROOT / str(recording_settings.get("output_dir", "artifacts"))
    return {
        "adapter": adapter,
        "frame_fetcher": frame_fetcher,
        "detector": detector,
        "visual_servo": visual_servo,
        "depth_analyzer": depth_analyzer,
        "obstacle_avoidance": obstacle_avoidance,
        "recorder": TelemetryRecorder(
            output_dir=recording_dir,
            save_debug_frames=bool(recording_settings.get("save_debug_frames", True)),
            debug_frame_interval_s=float(
                recording_settings.get("debug_frame_interval_s", 1.0)
            ),
        ),
        "shared_state": RuntimeSharedState(),
        "state_lock": asyncio.Lock(),
    }


def _load_simple_yaml(content: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in content.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, raw_value = raw_line.strip().partition(":")
        value = raw_value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if not value:
            nested: dict[str, Any] = {}
            current[key] = nested
            stack.append((indent, nested))
            continue

        current[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith(("\"", "'")) and value.endswith(("\"", "'")):
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
