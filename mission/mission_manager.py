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

from adapters.airsim_client import AirSimClientAdapter
from app.bootstrap import (
    PROJECT_ROOT,
    bootstrap_app,
    build_airsim_adapter,
    build_aruco_detector,
    build_frame_fetcher,
)
from control.visual_servo import VisualServoController, build_visual_servo_controller
from mission.search_pattern import SearchPattern
from mission.states import MissionState
from vision.aruco_detector import ArucoDetection, ArucoDetector
from vision.frame_fetcher import FrameFetcher


@dataclass
class MissionConfig:
    search_timeout_s: float
    track_timeout_s: float
    total_timeout_s: float
    marker_lost_limit: int
    descend_marker_area_threshold: float
    descend_rate_m_s: float
    descend_step_duration_s: float
    landing_altitude_m: float
    complete_wait_s: float
    loop_pause_s: float


@dataclass
class MissionStepResult:
    state: MissionState
    detection: ArucoDetection
    debug_image_path: Path
    detail: str


class MissionManager:
    def __init__(
        self,
        config: MissionConfig,
        visual_servo: VisualServoController,
        search_pattern: SearchPattern,
        target_marker_id: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.visual_servo = visual_servo
        self.search_pattern = search_pattern
        self.target_marker_id = target_marker_id
        self.logger = logger or logging.getLogger("drone_cv.mission")
        self.state = MissionState.IDLE
        self.state_started_at = time.monotonic()
        self.mission_started_at = self.state_started_at
        self.marker_lost_count = 0
        self.land_started_at: float | None = None

    def transition_to(self, next_state: MissionState) -> MissionState:
        if next_state != self.state:
            self.logger.info(
                "Mission state transition: %s -> %s",
                self.state.value,
                next_state.value,
            )
            self.state = next_state
            self.state_started_at = time.monotonic()
            if next_state in {MissionState.SEARCH, MissionState.TRACK, MissionState.DESCEND}:
                self.marker_lost_count = 0
            if next_state == MissionState.LAND:
                self.land_started_at = self.state_started_at
        return self.state

    def run_step(
        self,
        adapter: AirSimClientAdapter,
        detector: ArucoDetector,
        frame_fetcher: FrameFetcher,
    ) -> MissionStepResult:
        now = time.monotonic()
        if now - self.mission_started_at > self.config.total_timeout_s:
            return self._failsafe(adapter, frame_fetcher, detector, "mission timeout exceeded")

        frame_bundle = frame_fetcher.fetch()
        detection = detector.detect(
            frame_bundle.rgb_bgr,
            target_marker_id=self.target_marker_id,
        )
        telemetry = adapter.get_telemetry()

        if self.state == MissionState.IDLE:
            self.transition_to(MissionState.TAKEOFF)
            adapter.enable_api_control(True)
            adapter.arm(True)
            adapter.takeoff()
            self.transition_to(MissionState.SEARCH)
            return self._result(
                frame_bundle.rgb_bgr,
                detector,
                detection,
                "takeoff complete; entering search",
            )

        if self.state == MissionState.SEARCH:
            if now - self.state_started_at > self.config.search_timeout_s:
                return self._failsafe(adapter, frame_bundle.rgb_bgr, detector, detection, "search timeout exceeded")
            if detection.detected:
                self.transition_to(MissionState.TRACK)
                return self._result(
                    frame_bundle.rgb_bgr,
                    detector,
                    detection,
                    "marker detected; entering track",
                )

            search_command = self.search_pattern.next_command()
            adapter.move_by_velocity_body(
                vx_m_s=search_command.vx,
                vy_m_s=search_command.vy,
                vz_m_s=search_command.vz,
                duration_s=search_command.duration_s,
                yaw_rate_deg_s=search_command.yaw_rate,
            )
            return self._result(
                frame_bundle.rgb_bgr,
                detector,
                detection,
                "marker not visible; executing search yaw scan",
            )

        if self.state == MissionState.TRACK:
            if now - self.state_started_at > self.config.track_timeout_s:
                return self._failsafe(adapter, frame_bundle.rgb_bgr, detector, detection, "track timeout exceeded")
            if not detection.detected:
                self.marker_lost_count += 1
                if self.marker_lost_count >= self.config.marker_lost_limit:
                    self.visual_servo.reset()
                    self.transition_to(MissionState.SEARCH)
                    return self._result(
                        frame_bundle.rgb_bgr,
                        detector,
                        detection,
                        "marker lost; returning to search",
                    )
                return self._result(
                    frame_bundle.rgb_bgr,
                    detector,
                    detection,
                    f"marker temporarily lost ({self.marker_lost_count}/{self.config.marker_lost_limit})",
                )

            self.marker_lost_count = 0
            if detection.area >= self.config.descend_marker_area_threshold:
                self.transition_to(MissionState.DESCEND)
                return self._result(
                    frame_bundle.rgb_bgr,
                    detector,
                    detection,
                    "marker large enough; entering descend",
                )

            command = self.visual_servo.compute_command(
                detection=detection,
                frame_width=frame_bundle.rgb_bgr.shape[1],
                frame_height=frame_bundle.rgb_bgr.shape[0],
            )
            adapter.move_by_velocity_body(
                vx_m_s=command.vx,
                vy_m_s=command.vy,
                vz_m_s=command.vz,
                duration_s=command.duration_s,
                yaw_rate_deg_s=command.yaw_rate,
            )
            return self._result(
                frame_bundle.rgb_bgr,
                detector,
                detection,
                "tracking marker with visual servo",
            )

        if self.state == MissionState.DESCEND:
            if not detection.detected:
                self.marker_lost_count += 1
                if self.marker_lost_count >= self.config.marker_lost_limit:
                    self.visual_servo.reset()
                    self.transition_to(MissionState.SEARCH)
                    return self._result(
                        frame_bundle.rgb_bgr,
                        detector,
                        detection,
                        "marker lost during descend; returning to search",
                    )
            else:
                self.marker_lost_count = 0

            if telemetry.altitude_m <= self.config.landing_altitude_m:
                self.transition_to(MissionState.LAND)
                adapter.land()
                return self._result(
                    frame_bundle.rgb_bgr,
                    detector,
                    detection,
                    "landing altitude reached; issuing land",
                )

            command = self.visual_servo.compute_command(
                detection=detection,
                frame_width=frame_bundle.rgb_bgr.shape[1],
                frame_height=frame_bundle.rgb_bgr.shape[0],
            )
            adapter.move_by_velocity_body(
                vx_m_s=command.vx,
                vy_m_s=command.vy,
                vz_m_s=self.config.descend_rate_m_s,
                duration_s=self.config.descend_step_duration_s,
                yaw_rate_deg_s=command.yaw_rate,
            )
            return self._result(
                frame_bundle.rgb_bgr,
                detector,
                detection,
                "descending while maintaining marker alignment",
            )

        if self.state == MissionState.LAND:
            if self.land_started_at is not None and now - self.land_started_at >= self.config.complete_wait_s:
                adapter.arm(False)
                adapter.enable_api_control(False)
                self.transition_to(MissionState.COMPLETE)
                return self._result(
                    frame_bundle.rgb_bgr,
                    detector,
                    detection,
                    "landing complete",
                )
            return self._result(
                frame_bundle.rgb_bgr,
                detector,
                detection,
                "waiting for landing completion",
            )

        if self.state in {MissionState.COMPLETE, MissionState.FAILSAFE}:
            return self._result(
                frame_bundle.rgb_bgr,
                detector,
                detection,
                f"mission ended in state {self.state.value}",
            )

        return self._failsafe(adapter, frame_bundle.rgb_bgr, detector, detection, "unexpected mission state")

    def _failsafe(
        self,
        adapter: AirSimClientAdapter,
        frame_source: np.ndarray | FrameFetcher,
        detector: ArucoDetector,
        detection: ArucoDetection | None = None,
        detail: str = "failsafe",
    ) -> MissionStepResult:
        self.transition_to(MissionState.FAILSAFE)
        self.visual_servo.reset()
        try:
            adapter.hover()
        except RuntimeError:
            self.logger.exception("Failsafe hover command failed")

        if isinstance(frame_source, FrameFetcher):
            frame_bundle = frame_source.fetch()
            frame_bgr = frame_bundle.rgb_bgr
            if detection is None:
                detection = detector.detect(frame_bgr, target_marker_id=self.target_marker_id)
        else:
            frame_bgr = frame_source
        assert detection is not None
        return self._result(frame_bgr, detector, detection, detail)

    def _result(
        self,
        frame_bgr: np.ndarray,
        detector: ArucoDetector,
        detection: ArucoDetection,
        detail: str,
    ) -> MissionStepResult:
        debug_frame = detector.draw_overlay(frame_bgr, detection)
        _draw_state_banner(debug_frame, self.state, detail)
        debug_image_path = _save_mission_debug_frame(debug_frame)
        return MissionStepResult(
            state=self.state,
            detection=detection,
            debug_image_path=debug_image_path,
            detail=detail,
        )


def run_mission_demo() -> int:
    context = bootstrap_app()
    logger = context["logger"]
    settings = context["settings"]
    aruco_settings = settings.get("aruco", {})
    control_settings = settings.get("control", {})
    mission_settings = settings.get("mission", {})

    adapter = build_airsim_adapter(settings, logger)
    adapter.connect()
    adapter.confirm_connection()

    frame_fetcher = build_frame_fetcher(settings, adapter, logger)
    detector = build_aruco_detector(settings)
    visual_servo = build_visual_servo_controller(control_settings, logger=logger)
    search_pattern = SearchPattern(
        yaw_rate_deg_s=float(mission_settings.get("search_yaw_rate_deg_s", 8.0)),
        step_duration_s=float(mission_settings.get("search_step_duration_s", 0.35)),
    )
    mission_manager = MissionManager(
        config=MissionConfig(
            search_timeout_s=float(mission_settings.get("search_timeout_s", 20.0)),
            track_timeout_s=float(mission_settings.get("track_timeout_s", 30.0)),
            total_timeout_s=float(mission_settings.get("total_timeout_s", 90.0)),
            marker_lost_limit=int(mission_settings.get("marker_lost_limit", 3)),
            descend_marker_area_threshold=float(
                mission_settings.get("descend_marker_area_threshold", 18000.0)
            ),
            descend_rate_m_s=float(mission_settings.get("descend_rate_m_s", 0.2)),
            descend_step_duration_s=float(
                mission_settings.get("descend_step_duration_s", 0.25)
            ),
            landing_altitude_m=float(mission_settings.get("landing_altitude_m", 0.4)),
            complete_wait_s=float(mission_settings.get("complete_wait_s", 2.0)),
            loop_pause_s=float(mission_settings.get("loop_pause_s", 0.1)),
        ),
        visual_servo=visual_servo,
        search_pattern=search_pattern,
        target_marker_id=int(aruco_settings.get("marker_id", 0)),
        logger=logger,
    )

    max_steps = int(mission_settings.get("max_steps", 100))
    for step_index in range(max_steps):
        result = mission_manager.run_step(
            adapter=adapter,
            detector=detector,
            frame_fetcher=frame_fetcher,
        )
        print(
            f"step={step_index + 1} state={result.state.value} "
            f"marker_detected={result.detection.detected} detail={result.detail} "
            f"debug={result.debug_image_path}"
        )
        if result.state in {MissionState.COMPLETE, MissionState.FAILSAFE}:
            return 0 if result.state == MissionState.COMPLETE else 1
        time.sleep(mission_manager.config.loop_pause_s)

    mission_manager.transition_to(MissionState.FAILSAFE)
    print("Mission step limit exceeded; entering failsafe")
    return 1


def _draw_state_banner(frame_bgr: np.ndarray, state: MissionState, detail: str) -> None:
    cv2.putText(
        frame_bgr,
        f"State: {state.value}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        detail,
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _save_mission_debug_frame(frame_bgr: np.ndarray) -> Path:
    output_dir = PROJECT_ROOT / "debug_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mission_debug.png"
    cv2.imwrite(str(output_path), frame_bgr)
    return output_path


if __name__ == "__main__":
    raise SystemExit(run_mission_demo())
