from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mission.states import MissionState


@dataclass
class RuntimeControlCommand:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    duration_s: float = 0.0
    source: str = "idle"
    reason: str = "no command"


@dataclass
class RuntimeFrameState:
    rgb_frame: Any | None = None
    depth_frame: Any | None = None
    rgb_timestamp: int = 0
    depth_timestamp: int = 0
    local_world_snapshot: Any | None = None


@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass
class TelemetrySnapshot:
    timestamp: int
    position_m: Vector3
    velocity_m_s: Vector3
    altitude_m: float
    orientation: Quaternion
    speed_m_s: float


@dataclass
class LocalObstacleState:
    x_m: float
    y_m: float
    radius_m: float


@dataclass
class LocalWorldState:
    width_m: float = 10.0
    height_m: float = 8.0
    min_altitude_m: float = 0.15
    max_altitude_m: float = 3.0
    drone_x_m: float = 0.0
    drone_y_m: float = 0.0
    altitude_m: float = 1.5
    yaw_deg: float = 0.0
    velocity_x_m_s: float = 0.0
    velocity_y_m_s: float = 0.0
    velocity_z_m_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    marker_x_m: float = 6.0
    marker_y_m: float = 0.0
    marker_visible: bool = False
    marker_distance_m: float = 0.0
    obstacle_distance_m: float | None = None
    obstacle_side: str = "none"
    obstacle: LocalObstacleState = field(
        default_factory=lambda: LocalObstacleState(x_m=3.2, y_m=0.6, radius_m=0.55)
    )


@dataclass
class RuntimeSharedState:
    telemetry: TelemetrySnapshot | None = None
    frames: RuntimeFrameState | None = None
    detection: Any | None = None
    depth_analysis: Any | None = None
    local_world: LocalWorldState | None = None
    local_manual_yaw_rate_deg_s: float = 0.0
    local_manual_vz_m_s: float = 0.0
    local_manual_vy_m_s: float = 0.0
    local_manual_override_until_s: float = 0.0
    local_manual_status: str = "auto"
    local_spin_paused: bool = False
    mission_state: MissionState = MissionState.IDLE
    mission_detail: str = "initializing"
    desired_command: RuntimeControlCommand = field(default_factory=RuntimeControlCommand)
    control_applied: bool = False
