from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path
import threading
from typing import Any

from telemetry.models import Quaternion, TelemetrySnapshot, Vector3

try:
    import airsim
except ModuleNotFoundError:  # pragma: no cover - depends on local AirSim install
    airsim = None


@dataclass
class AirSimConnectionConfig:
    host: str
    port: int
    timeout_seconds: float = 10.0
    vehicle_name: str = ""


@dataclass
class DroneState:
    ready: bool
    landed_state: int
    position_xyz_m: tuple[float, float, float]
    linear_velocity_xyz_m_s: tuple[float, float, float]
    timestamp: int


@dataclass
class AirSimImageData:
    width: int
    height: int
    timestamp: int
    data_uint8: bytes | None = None
    data_float32: list[float] | None = None


@dataclass
class AirSimImagePair:
    rgb: AirSimImageData
    depth: AirSimImageData


class AirSimClientAdapter:
    """Wrapper around AirSim MultirotorClient for basic vehicle control."""

    def __init__(
        self,
        config: AirSimConnectionConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("drone_cv.airsim")
        self._client: Any | None = None
        self._resolved_host = config.host
        self._last_async_task: Any | None = None
        self._client_lock = threading.RLock()

    def connect(self) -> None:
        client_cls = self._require_airsim()
        resolved_host = self._resolve_host(self.config.host)
        self._resolved_host = resolved_host
        with self._client_lock:
            self._client = client_cls(
                ip=resolved_host,
                port=self.config.port,
                timeout_value=self.config.timeout_seconds,
            )
        self.logger.info(
            "Connecting to AirSim at %s:%s",
            resolved_host,
            self.config.port,
        )
        if resolved_host != self.config.host:
            self.logger.info(
                "AirSim host was auto-resolved from '%s' to '%s'",
                self.config.host,
                resolved_host,
            )

    def confirm_connection(self) -> None:
        with self._client_lock:
            client = self._require_client()
            client.confirmConnection()
        self.logger.info("AirSim connection confirmed")

    def enable_api_control(self, enabled: bool = True) -> None:
        with self._client_lock:
            client = self._require_client()
            client.enableApiControl(enabled, vehicle_name=self.config.vehicle_name)
        self.logger.info("API control set to %s", enabled)

    def arm(self, armed: bool = True) -> None:
        with self._client_lock:
            client = self._require_client()
            client.armDisarm(armed, vehicle_name=self.config.vehicle_name)
        self.logger.info("Arm state set to %s", armed)

    def takeoff(self, timeout_seconds: float = 20.0) -> None:
        self.logger.info("Takeoff started")
        with self._client_lock:
            client = self._require_client()
            self._await_last_async_task_locked()
            self._last_async_task = client.takeoffAsync(
                timeout_sec=timeout_seconds,
                vehicle_name=self.config.vehicle_name,
            )
            self._last_async_task.join()
            self._last_async_task = None
        self.logger.info("Takeoff completed")

    def hover(self, wait: bool = True) -> None:
        self.logger.info("Hover command sent")
        with self._client_lock:
            client = self._require_client()
            self._last_async_task = client.hoverAsync(vehicle_name=self.config.vehicle_name)
            if wait:
                self._last_async_task.join()
                self._last_async_task = None

    def land(self, timeout_seconds: float = 30.0) -> None:
        self.logger.info("Landing started")
        with self._client_lock:
            client = self._require_client()
            self._await_last_async_task_locked()
            self._last_async_task = client.landAsync(
                timeout_sec=timeout_seconds,
                vehicle_name=self.config.vehicle_name,
            )
            self._last_async_task.join()
            self._last_async_task = None
        self.logger.info("Landing completed")

    def move_by_velocity_body(
        self,
        vx_m_s: float,
        vy_m_s: float,
        vz_m_s: float,
        duration_s: float,
        yaw_rate_deg_s: float,
        wait: bool = False,
    ) -> None:
        airsim_module = self._require_airsim_module()
        self.logger.info(
            "Sending velocity command vx=%.3f vy=%.3f vz=%.3f duration=%.2f yaw_rate=%.3f",
            vx_m_s,
            vy_m_s,
            vz_m_s,
            duration_s,
            yaw_rate_deg_s,
        )
        with self._client_lock:
            client = self._require_client()
            self._last_async_task = client.moveByVelocityBodyFrameAsync(
                vx=vx_m_s,
                vy=vy_m_s,
                vz=vz_m_s,
                duration=duration_s,
                drivetrain=airsim_module.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim_module.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg_s),
                vehicle_name=self.config.vehicle_name,
            )
            if wait:
                self._last_async_task.join()
                self._last_async_task = None

    def get_state(self) -> DroneState:
        with self._client_lock:
            client = self._require_client()
            state = client.getMultirotorState(vehicle_name=self.config.vehicle_name)
        kinematics = state.kinematics_estimated
        return DroneState(
            ready=bool(state.ready),
            landed_state=int(state.landed_state),
            position_xyz_m=(
                kinematics.position.x_val,
                kinematics.position.y_val,
                kinematics.position.z_val,
            ),
            linear_velocity_xyz_m_s=(
                kinematics.linear_velocity.x_val,
                kinematics.linear_velocity.y_val,
                kinematics.linear_velocity.z_val,
            ),
            timestamp=state.timestamp,
        )

    def get_telemetry(self) -> TelemetrySnapshot:
        with self._client_lock:
            client = self._require_client()
            state = client.getMultirotorState(vehicle_name=self.config.vehicle_name)
        kinematics = state.kinematics_estimated

        position = Vector3(
            x=float(kinematics.position.x_val),
            y=float(kinematics.position.y_val),
            z=float(kinematics.position.z_val),
        )
        velocity = Vector3(
            x=float(kinematics.linear_velocity.x_val),
            y=float(kinematics.linear_velocity.y_val),
            z=float(kinematics.linear_velocity.z_val),
        )
        orientation = Quaternion(
            x=float(kinematics.orientation.x_val),
            y=float(kinematics.orientation.y_val),
            z=float(kinematics.orientation.z_val),
            w=float(kinematics.orientation.w_val),
        )
        speed_m_s = (
            (velocity.x * velocity.x)
            + (velocity.y * velocity.y)
            + (velocity.z * velocity.z)
        ) ** 0.5

        return TelemetrySnapshot(
            timestamp=int(state.timestamp),
            position_m=position,
            velocity_m_s=velocity,
            altitude_m=float(-position.z),
            orientation=orientation,
            speed_m_s=float(speed_m_s),
        )

    def fetch_rgb_and_depth(
        self,
        rgb_camera_name: str,
        depth_camera_name: str,
    ) -> AirSimImagePair:
        airsim_module = self._require_airsim_module()
        with self._client_lock:
            client = self._require_client()
            responses = client.simGetImages(
                [
                    airsim_module.ImageRequest(
                        rgb_camera_name,
                        airsim_module.ImageType.Scene,
                        pixels_as_float=False,
                        compress=False,
                    ),
                    airsim_module.ImageRequest(
                        depth_camera_name,
                        airsim_module.ImageType.DepthPerspective,
                        pixels_as_float=True,
                        compress=False,
                    ),
                ],
                vehicle_name=self.config.vehicle_name,
            )
        if len(responses) != 2:
            raise RuntimeError(
                f"Expected 2 AirSim image responses, received {len(responses)}."
            )

        rgb_response, depth_response = responses
        return AirSimImagePair(
            rgb=AirSimImageData(
                width=int(rgb_response.width),
                height=int(rgb_response.height),
                timestamp=int(rgb_response.time_stamp),
                data_uint8=bytes(rgb_response.image_data_uint8),
            ),
            depth=AirSimImageData(
                width=int(depth_response.width),
                height=int(depth_response.height),
                timestamp=int(depth_response.time_stamp),
                data_float32=list(depth_response.image_data_float),
            ),
        )

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("AirSim client is not connected. Call connect() first.")
        return self._client

    def _await_last_async_task_locked(self) -> None:
        if self._last_async_task is None:
            return
        try:
            self._last_async_task.join()
        finally:
            self._last_async_task = None

    @classmethod
    def _resolve_host(cls, configured_host: str) -> str:
        normalized = configured_host.strip().lower()
        if normalized not in {"auto", "wsl", "wsl-host", "auto-wsl"}:
            return configured_host

        nameserver_ip = cls._read_wsl_nameserver()
        if nameserver_ip:
            return nameserver_ip

        return "127.0.0.1"

    @staticmethod
    def _read_wsl_nameserver() -> str | None:
        resolv_conf = Path("/etc/resolv.conf")
        if not resolv_conf.exists():
            return None

        try:
            content = resolv_conf.read_text(encoding="utf-8")
        except OSError:
            return None

        for line in content.splitlines():
            match = re.match(r"^\s*nameserver\s+([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\s*$", line)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _require_airsim() -> Any:
        if airsim is None:
            raise RuntimeError(
                "The 'airsim' package is not installed. Install dependencies with "
                "'python3 -m pip install -r requirements.txt'."
            )
        return airsim.MultirotorClient

    @staticmethod
    def _require_airsim_module() -> Any:
        if airsim is None:
            raise RuntimeError(
                "The 'airsim' package is not installed. Install dependencies with "
                "'python3 -m pip install -r requirements.txt'."
            )
        return airsim
