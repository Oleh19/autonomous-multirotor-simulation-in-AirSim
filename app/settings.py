from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AppSection(StrictBaseModel):
    name: str = Field(min_length=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_format: Literal["text", "json"] = "text"


class RuntimeSection(StrictBaseModel):
    telemetry_interval_s: float = Field(gt=0.0)
    frame_interval_s: float = Field(gt=0.0)
    vision_interval_s: float = Field(gt=0.0)
    mission_interval_s: float = Field(gt=0.0)
    control_interval_s: float = Field(gt=0.0)
    run_duration_s: float = Field(gt=0.0)


class WatchdogSection(StrictBaseModel):
    enabled: bool
    loop_interval_s: float = Field(gt=0.0)
    stale_after_s: float = Field(gt=0.0)


class FreshnessSection(StrictBaseModel):
    telemetry_max_age_s: float = Field(gt=0.0)
    frames_max_age_s: float = Field(gt=0.0)
    detection_max_age_s: float = Field(gt=0.0)
    depth_analysis_max_age_s: float = Field(gt=0.0)


class UiSection(StrictBaseModel):
    enabled: bool
    interval_s: float = Field(gt=0.0)
    window_name: str = Field(min_length=1)


class LocalWorldSection(StrictBaseModel):
    width_m: float = Field(gt=0.0)
    height_m: float = Field(gt=0.0)
    marker_x_m: float
    marker_y_m: float
    obstacle_x_m: float
    obstacle_y_m: float
    obstacle_radius_m: float = Field(gt=0.0)
    camera_fov_deg: float = Field(gt=1.0, lt=179.0)
    marker_scale_px_m: float = Field(gt=0.0)
    desired_altitude_m: float = Field(gt=0.0)
    min_altitude_m: float = Field(gt=0.0)
    max_altitude_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_altitude_bounds(self) -> "LocalWorldSection":
        if self.min_altitude_m >= self.max_altitude_m:
            raise ValueError("local_world.min_altitude_m must be less than local_world.max_altitude_m")
        if not (self.min_altitude_m <= self.desired_altitude_m <= self.max_altitude_m):
            raise ValueError(
                "local_world.desired_altitude_m must be within [local_world.min_altitude_m, local_world.max_altitude_m]"
            )
        return self


class RecordingSection(StrictBaseModel):
    output_dir: str = Field(min_length=1)
    save_debug_frames: bool
    debug_frame_interval_s: float = Field(gt=0.0)


class AirSimSection(StrictBaseModel):
    host: str = Field(min_length=1)
    port: int = Field(gt=0)
    vehicle_name: str
    timeout_seconds: float = Field(gt=0.0)
    takeoff_timeout_seconds: float = Field(gt=0.0)
    land_timeout_seconds: float = Field(gt=0.0)
    auto_takeoff_on_start: bool
    smoke_test_hover_duration_seconds: float = Field(gt=0.0)


class CameraSection(StrictBaseModel):
    rgb_camera_name: str = Field(min_length=1)
    depth_camera_name: str = Field(min_length=1)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    debug_output_dir: str = Field(min_length=1)


class ArucoSection(StrictBaseModel):
    dictionary: str = Field(min_length=1)
    marker_id: int = Field(ge=0)
    marker_length_m: float = Field(gt=0.0)


class ControlSection(StrictBaseModel):
    loop_hz: float = Field(gt=0.0)
    max_velocity_xy: float = Field(gt=0.0)
    max_velocity_z: float = Field(gt=0.0)
    yaw_rate_deg_s: float = Field(gt=0.0)
    servo_command_duration_s: float = Field(gt=0.0)
    servo_max_lateral_velocity_m_s: float = Field(gt=0.0)
    servo_max_vertical_velocity_m_s: float = Field(gt=0.0)
    servo_max_yaw_rate_deg_s: float = Field(gt=0.0)
    servo_yaw_error_deadband_px: float = Field(ge=0.0)
    servo_vertical_error_deadband_px: float = Field(ge=0.0)
    servo_lateral_kp: float = Field(ge=0.0)
    servo_lateral_ki: float = Field(ge=0.0)
    servo_lateral_kd: float = Field(ge=0.0)
    servo_vertical_kp: float = Field(ge=0.0)
    servo_vertical_ki: float = Field(ge=0.0)
    servo_vertical_kd: float = Field(ge=0.0)
    servo_yaw_kp: float = Field(ge=0.0)
    servo_yaw_ki: float = Field(ge=0.0)
    servo_yaw_kd: float = Field(ge=0.0)
    approach_center_tolerance_px: float = Field(ge=0.0)
    approach_target_marker_area: float = Field(gt=0.0)
    approach_forward_speed_m_s: float = Field(gt=0.0)
    approach_command_duration_s: float = Field(gt=0.0)


class DepthSection(StrictBaseModel):
    obstacle_distance_m: float = Field(gt=0.0)
    min_valid_depth_m: float = Field(gt=0.0)
    max_valid_depth_m: float = Field(gt=0.0)
    avoidance_speed_m_s: float = Field(gt=0.0)
    avoidance_yaw_rate_deg_s: float = Field(gt=0.0)
    avoidance_command_duration_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_depth_range(self) -> "DepthSection":
        if self.min_valid_depth_m >= self.max_valid_depth_m:
            raise ValueError("depth.min_valid_depth_m must be less than depth.max_valid_depth_m")
        return self


class LandingSection(StrictBaseModel):
    final_descent_rate_m_s: float = Field(gt=0.0)
    touchdown_altitude_m: float = Field(gt=0.0)
    alignment_tolerance_px: float = Field(ge=0.0)
    final_marker_area: float = Field(gt=0.0)
    descend_step_duration_s: float = Field(gt=0.0)
    lost_marker_limit: int = Field(ge=0)
    loop_pause_s: float = Field(gt=0.0)
    max_steps: int = Field(gt=0)


class TelemetrySection(StrictBaseModel):
    poll_interval_s: float = Field(gt=0.0)
    sample_count: int = Field(gt=0)


class OverlaysSection(StrictBaseModel):
    mode: Literal["image", "window"]
    output_dir: str = Field(min_length=1)
    output_filename: str = Field(min_length=1)
    window_name: str = Field(min_length=1)
    mission_state: str = Field(min_length=1)


class MissionSection(StrictBaseModel):
    search_yaw_rate_deg_s: float = Field(gt=0.0)
    search_step_duration_s: float = Field(gt=0.0)
    search_timeout_s: float = Field(gt=0.0)
    track_timeout_s: float = Field(gt=0.0)
    total_timeout_s: float = Field(gt=0.0)
    marker_lost_limit: int = Field(ge=0)
    descend_marker_area_threshold: float = Field(gt=0.0)
    descend_rate_m_s: float = Field(gt=0.0)
    descend_step_duration_s: float = Field(gt=0.0)
    auto_descend_on_target: bool
    landing_altitude_m: float = Field(gt=0.0)
    auto_land_on_target: bool
    complete_wait_s: float = Field(gt=0.0)
    loop_pause_s: float = Field(gt=0.0)
    max_steps: int = Field(gt=0)


class AppSettings(StrictBaseModel):
    app: AppSection
    runtime: RuntimeSection
    watchdog: WatchdogSection
    freshness: FreshnessSection
    local_ui: UiSection
    dev_ui: UiSection
    local_world: LocalWorldSection
    recording: RecordingSection
    airsim: AirSimSection
    camera: CameraSection
    aruco: ArucoSection
    control: ControlSection
    depth: DepthSection
    landing: LandingSection
    telemetry: TelemetrySection
    overlays: OverlaysSection
    mission: MissionSection


def format_settings_validation_error(exc: ValidationError) -> list[str]:
    formatted: list[str] = []
    for error in exc.errors():
        path = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        formatted.append(f"{path}: {error.get('msg', 'invalid value')}")
    return formatted
