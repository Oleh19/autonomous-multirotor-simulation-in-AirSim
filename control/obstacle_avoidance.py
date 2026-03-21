from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.airsim_client import AirSimClientAdapter, AirSimConnectionConfig
from app.bootstrap import bootstrap_app
from vision.depth_analyzer import DepthAnalysis, DepthAnalyzer
from vision.frame_fetcher import FrameFetcher


@dataclass
class ObstacleAvoidanceCommand:
    stop: bool = False
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    duration_s: float = 0.0
    chosen_side: str = "none"
    reason: str = "path clear"


class ObstacleAvoidanceController:
    def __init__(
        self,
        avoidance_speed_m_s: float,
        yaw_rate_deg_s: float,
        command_duration_s: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self.avoidance_speed_m_s = avoidance_speed_m_s
        self.yaw_rate_deg_s = yaw_rate_deg_s
        self.command_duration_s = command_duration_s
        self.logger = logger or logging.getLogger("drone_cv.obstacle_avoidance")

    def compute_command(self, analysis: DepthAnalysis) -> ObstacleAvoidanceCommand:
        if not analysis.obstacle_detected:
            return ObstacleAvoidanceCommand(
                stop=False,
                duration_s=self.command_duration_s,
                reason="center path clear",
            )

        if analysis.safer_side == "left":
            return ObstacleAvoidanceCommand(
                stop=False,
                vy=-self.avoidance_speed_m_s,
                yaw_rate=-self.yaw_rate_deg_s,
                duration_s=self.command_duration_s,
                chosen_side="left",
                reason="center blocked; left side has more clearance",
            )

        return ObstacleAvoidanceCommand(
            stop=False,
            vy=self.avoidance_speed_m_s,
            yaw_rate=self.yaw_rate_deg_s,
            duration_s=self.command_duration_s,
            chosen_side="right",
            reason="center blocked; right side has more clearance",
        )


def run_obstacle_avoidance_step() -> int:
    context = bootstrap_app()
    logger = context["logger"]
    settings = context["settings"]
    airsim_settings = settings.get("airsim", {})
    camera_settings = settings.get("camera", {})
    depth_settings = settings.get("depth", {})

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
    depth_analyzer = DepthAnalyzer(
        obstacle_distance_m=float(depth_settings.get("obstacle_distance_m", 2.0)),
        min_valid_depth_m=float(depth_settings.get("min_valid_depth_m", 0.2)),
        max_valid_depth_m=float(depth_settings.get("max_valid_depth_m", 20.0)),
    )
    avoidance = ObstacleAvoidanceController(
        avoidance_speed_m_s=float(depth_settings.get("avoidance_speed_m_s", 0.5)),
        yaw_rate_deg_s=float(depth_settings.get("avoidance_yaw_rate_deg_s", 12.0)),
        command_duration_s=float(depth_settings.get("avoidance_command_duration_s", 0.25)),
        logger=logger,
    )

    frame_bundle = frame_fetcher.fetch()
    analysis = depth_analyzer.analyze(frame_bundle.depth_m)
    command = avoidance.compute_command(analysis)

    if analysis.obstacle_detected:
        adapter.move_by_velocity_body(
            vx_m_s=command.vx,
            vy_m_s=command.vy,
            vz_m_s=command.vz,
            duration_s=command.duration_s,
            yaw_rate_deg_s=command.yaw_rate,
        )

    print(
        "Obstacle avoidance decision: "
        f"blocked={analysis.obstacle_detected} "
        f"left={analysis.left_clearance_m:.2f}m "
        f"center={analysis.center_clearance_m:.2f}m "
        f"right={analysis.right_clearance_m:.2f}m "
        f"side={command.chosen_side} reason={command.reason}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_obstacle_avoidance_step())
