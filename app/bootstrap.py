from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for bare environments
    yaml = None

from pydantic import ValidationError

from app.settings import AppSettings, format_settings_validation_error
from telemetry.logger import JsonLogFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    return load_settings_model(path).model_dump()


def load_settings_model(path: Path = SETTINGS_PATH) -> AppSettings:
    with path.open("r", encoding="utf-8") as settings_file:
        content = settings_file.read()

    if yaml is not None:
        raw_settings = yaml.safe_load(content) or {}
    else:
        raw_settings = _load_simple_yaml(content)

    return AppSettings.model_validate(raw_settings)


def configure_logging(level_name: str, log_format: str = "text") -> logging.Logger:
    level = getattr(logging, level_name.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    root_logger.addHandler(handler)
    return logging.getLogger("drone_cv")


def bootstrap_app() -> dict[str, Any]:
    try:
        settings_model = load_settings_model()
    except ValidationError as exc:
        errors = "; ".join(format_settings_validation_error(exc))
        raise ValueError(f"Invalid config/settings.yaml: {errors}") from exc
    settings = settings_model.model_dump()
    logger = configure_logging(
        settings.get("app", {}).get("log_level", "INFO"),
        settings.get("app", {}).get("log_format", "text"),
    )
    logger.debug("Settings loaded from %s", SETTINGS_PATH)
    return {"settings": settings, "settings_model": settings_model, "logger": logger}


def validate_settings(settings: dict[str, Any]) -> list[str]:
    try:
        AppSettings.model_validate(settings)
    except ValidationError as exc:
        return format_settings_validation_error(exc)
    return []


def build_airsim_adapter(settings: dict[str, Any], logger: logging.Logger):
    from adapters.airsim_client import AirSimClientAdapter, AirSimConnectionConfig

    airsim_settings = settings.get("airsim", {})
    return AirSimClientAdapter(
        config=AirSimConnectionConfig(
            host=str(airsim_settings.get("host", "127.0.0.1")),
            port=int(airsim_settings.get("port", 41451)),
            timeout_seconds=float(airsim_settings.get("timeout_seconds", 10.0)),
            vehicle_name=str(airsim_settings.get("vehicle_name", "")),
        ),
        logger=logger,
    )


def build_frame_fetcher(settings: dict[str, Any], adapter, logger: logging.Logger):
    from vision.frame_fetcher import FrameFetcher

    camera_settings = settings.get("camera", {})
    return FrameFetcher(
        adapter=adapter,
        rgb_camera_name=str(camera_settings.get("rgb_camera_name", "front_center")),
        depth_camera_name=str(camera_settings.get("depth_camera_name", "front_center")),
        logger=logger,
    )


def build_aruco_detector(settings: dict[str, Any]):
    from vision.aruco_detector import ArucoDetector

    aruco_settings = settings.get("aruco", {})
    return ArucoDetector(
        dictionary_name=str(aruco_settings.get("dictionary", "DICT_4X4_50"))
    )


def build_depth_analyzer(settings: dict[str, Any]):
    from vision.depth_analyzer import DepthAnalyzer

    depth_settings = settings.get("depth", {})
    return DepthAnalyzer(
        obstacle_distance_m=float(depth_settings.get("obstacle_distance_m", 2.0)),
        min_valid_depth_m=float(depth_settings.get("min_valid_depth_m", 0.2)),
        max_valid_depth_m=float(depth_settings.get("max_valid_depth_m", 20.0)),
    )


def build_obstacle_avoidance_controller(
    settings: dict[str, Any],
    logger: logging.Logger,
):
    from control.obstacle_avoidance import ObstacleAvoidanceController

    depth_settings = settings.get("depth", {})
    return ObstacleAvoidanceController(
        avoidance_speed_m_s=float(depth_settings.get("avoidance_speed_m_s", 0.5)),
        yaw_rate_deg_s=float(depth_settings.get("avoidance_yaw_rate_deg_s", 12.0)),
        command_duration_s=float(depth_settings.get("avoidance_command_duration_s", 0.25)),
        logger=logger,
    )


def build_runtime_components(settings: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    from control.safety import CommandSafetyLimits, CommandSafetyLimiter
    from control.visual_servo import build_visual_servo_controller
    from telemetry.recorder import TelemetryRecorder
    from telemetry.models import RuntimeSharedState

    control_settings = settings.get("control", {})
    depth_settings = settings.get("depth", {})
    mission_settings = settings.get("mission", {})
    recording_settings = settings.get("recording", {})
    runtime = settings.get("runtime", {})

    adapter = build_airsim_adapter(settings, logger)
    frame_fetcher = build_frame_fetcher(settings, adapter, logger)
    detector = build_aruco_detector(settings)
    visual_servo = build_visual_servo_controller(control_settings, logger=logger)
    depth_analyzer = build_depth_analyzer(settings)
    obstacle_avoidance = build_obstacle_avoidance_controller(settings, logger)
    safety_limiter = CommandSafetyLimiter(
        CommandSafetyLimits(
            max_velocity_xy_m_s=float(control_settings.get("max_velocity_xy", 1.0)),
            max_velocity_z_m_s=float(control_settings.get("max_velocity_z", 0.5)),
            max_yaw_rate_deg_s=float(control_settings.get("yaw_rate_deg_s", 15.0)),
            min_command_duration_s=min(
                float(runtime.get("control_interval_s", 0.2)),
                float(control_settings.get("approach_command_duration_s", 0.25)),
            ),
            max_command_duration_s=max(
                float(mission_settings.get("search_step_duration_s", 0.35)),
                float(mission_settings.get("descend_step_duration_s", 0.25)),
                float(control_settings.get("servo_command_duration_s", 0.2)),
                float(depth_settings.get("avoidance_command_duration_s", 0.25)),
                float(control_settings.get("approach_command_duration_s", 0.25)),
            ),
        )
    )
    recording_dir = PROJECT_ROOT / str(recording_settings.get("output_dir", "artifacts"))
    return {
        "adapter": adapter,
        "frame_fetcher": frame_fetcher,
        "detector": detector,
        "visual_servo": visual_servo,
        "depth_analyzer": depth_analyzer,
        "obstacle_avoidance": obstacle_avoidance,
        "safety_limiter": safety_limiter,
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
