from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from control.visual_servo import VisualServoCommand
from mission.mission_manager import MissionConfig, MissionManager
from mission.search_pattern import SearchPattern
from mission.states import MissionState
from telemetry.models import Quaternion, TelemetrySnapshot, Vector3
from vision.aruco_detector import ArucoDetection


@dataclass
class _FrameBundle:
    rgb_bgr: np.ndarray


class _FakeAdapter:
    def __init__(self, altitude_m: float = 1.0) -> None:
        self.commands: list[tuple[float, float, float, float, float]] = []
        self.api_control_enabled = False
        self.armed = False
        self.took_off = False
        self.hovered = False
        self.landed = False
        self.telemetry = TelemetrySnapshot(
            timestamp=1,
            position_m=Vector3(0.0, 0.0, -altitude_m),
            velocity_m_s=Vector3(0.0, 0.0, 0.0),
            altitude_m=altitude_m,
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            speed_m_s=0.0,
        )

    def enable_api_control(self, enabled: bool) -> None:
        self.api_control_enabled = enabled

    def arm(self, armed: bool) -> None:
        self.armed = armed

    def takeoff(self) -> None:
        self.took_off = True

    def move_by_velocity_body(
        self,
        vx_m_s: float,
        vy_m_s: float,
        vz_m_s: float,
        duration_s: float,
        yaw_rate_deg_s: float,
    ) -> None:
        self.commands.append((vx_m_s, vy_m_s, vz_m_s, duration_s, yaw_rate_deg_s))

    def get_telemetry(self) -> TelemetrySnapshot:
        return self.telemetry

    def land(self) -> None:
        self.landed = True

    def hover(self) -> None:
        self.hovered = True


class _FakeDetector:
    def __init__(self, detections: list[ArucoDetection]) -> None:
        self._detections = detections

    def detect(self, frame_bgr: np.ndarray, target_marker_id: int | None = None) -> ArucoDetection:
        return self._detections.pop(0)

    def draw_overlay(self, frame_bgr: np.ndarray, detection: ArucoDetection) -> np.ndarray:
        return frame_bgr.copy()


class _FakeFrameFetcher:
    def __init__(self, frame_count: int) -> None:
        self._frames = [_FrameBundle(np.zeros((60, 80, 3), dtype=np.uint8)) for _ in range(frame_count)]

    def fetch(self) -> _FrameBundle:
        return self._frames.pop(0)


class _FakeVisualServo:
    def __init__(self, commands: list[VisualServoCommand]) -> None:
        self._commands = commands
        self.reset_calls = 0

    def compute_command(
        self,
        detection: ArucoDetection,
        frame_width: int,
        frame_height: int,
    ) -> VisualServoCommand:
        return self._commands.pop(0)

    def reset(self) -> None:
        self.reset_calls += 1


def _build_manager(visual_servo: _FakeVisualServo) -> MissionManager:
    return MissionManager(
        config=MissionConfig(
            search_timeout_s=5.0,
            track_timeout_s=5.0,
            total_timeout_s=20.0,
            marker_lost_limit=2,
            descend_marker_area_threshold=18000.0,
            descend_rate_m_s=0.2,
            descend_step_duration_s=0.25,
            landing_altitude_m=0.4,
            complete_wait_s=0.2,
            loop_pause_s=0.01,
        ),
        visual_servo=visual_servo,
        search_pattern=SearchPattern(yaw_rate_deg_s=8.0, step_duration_s=0.35),
        target_marker_id=7,
        logger=logging.getLogger("test.mission_manager"),
    )


def _detection(
    *,
    detected: bool,
    marker_id: int | None = None,
    area: float = 0.0,
    center_x: float | None = 40.0,
    center_y: float | None = 30.0,
) -> ArucoDetection:
    return ArucoDetection(
        detected=detected,
        marker_id=marker_id,
        center_x=center_x if detected else None,
        center_y=center_y if detected else None,
        corners=((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)) if detected else (),
        area=area,
    )


def test_mission_search_sends_yaw_scan_when_marker_not_visible(monkeypatch) -> None:
    monkeypatch.setattr(
        "mission.mission_manager._save_mission_debug_frame",
        lambda frame_bgr: Path("artifacts/test_mission_debug.png"),
    )
    manager = _build_manager(_FakeVisualServo(commands=[]))
    manager.state = MissionState.SEARCH
    adapter = _FakeAdapter()
    detector = _FakeDetector([_detection(detected=False)])
    frame_fetcher = _FakeFrameFetcher(frame_count=1)

    result = manager.run_step(adapter=adapter, detector=detector, frame_fetcher=frame_fetcher)

    assert result.state == MissionState.SEARCH
    assert "yaw scan" in result.detail
    assert adapter.commands == [(0.0, 0.0, 0.0, 0.35, 8.0)]


def test_mission_track_returns_to_search_after_repeated_marker_loss(monkeypatch) -> None:
    monkeypatch.setattr(
        "mission.mission_manager._save_mission_debug_frame",
        lambda frame_bgr: Path("artifacts/test_mission_debug.png"),
    )
    visual_servo = _FakeVisualServo(commands=[])
    manager = _build_manager(visual_servo)
    manager.state = MissionState.TRACK
    manager.marker_lost_count = 1
    adapter = _FakeAdapter()
    detector = _FakeDetector([_detection(detected=False)])
    frame_fetcher = _FakeFrameFetcher(frame_count=1)

    result = manager.run_step(adapter=adapter, detector=detector, frame_fetcher=frame_fetcher)

    assert result.state == MissionState.SEARCH
    assert "returning to search" in result.detail
    assert visual_servo.reset_calls == 1


def test_mission_descend_triggers_land_at_touchdown_altitude(monkeypatch) -> None:
    monkeypatch.setattr(
        "mission.mission_manager._save_mission_debug_frame",
        lambda frame_bgr: Path("artifacts/test_mission_debug.png"),
    )
    visual_servo = _FakeVisualServo(
        commands=[VisualServoCommand(vy=0.1, vz=0.0, yaw_rate=1.0, duration_s=0.25, marker_detected=True)]
    )
    manager = _build_manager(visual_servo)
    manager.state = MissionState.DESCEND
    adapter = _FakeAdapter(altitude_m=0.3)
    detector = _FakeDetector([_detection(detected=True, marker_id=7, area=20000.0)])
    frame_fetcher = _FakeFrameFetcher(frame_count=1)

    result = manager.run_step(adapter=adapter, detector=detector, frame_fetcher=frame_fetcher)

    assert result.state == MissionState.LAND
    assert "issuing land" in result.detail
    assert adapter.landed is True
