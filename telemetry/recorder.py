from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from telemetry.models import TelemetrySnapshot


@dataclass
class TelemetryRecorder:
    output_dir: Path
    save_debug_frames: bool = False
    debug_frame_interval_s: float = 1.0

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.output_dir / "telemetry.jsonl"
        self.events_path = self.output_dir / "mission_events.jsonl"
        self.frames_dir = self.output_dir / "debug_frames"
        if self.save_debug_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._last_frame_saved_at = 0.0

    def record_telemetry(self, snapshot: TelemetrySnapshot) -> None:
        payload = {
            "timestamp": snapshot.timestamp,
            "altitude_m": snapshot.altitude_m,
            "speed_m_s": snapshot.speed_m_s,
            "position_m": {
                "x": snapshot.position_m.x,
                "y": snapshot.position_m.y,
                "z": snapshot.position_m.z,
            },
            "velocity_m_s": {
                "x": snapshot.velocity_m_s.x,
                "y": snapshot.velocity_m_s.y,
                "z": snapshot.velocity_m_s.z,
            },
            "orientation": {
                "x": snapshot.orientation.x,
                "y": snapshot.orientation.y,
                "z": snapshot.orientation.z,
                "w": snapshot.orientation.w,
            },
        }
        self._append_jsonl(self.telemetry_path, payload)

    def record_event(self, event: str, detail: str, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "time": time.time(),
            "event": event,
            "detail": detail,
        }
        if extra:
            payload.update(extra)
        self._append_jsonl(self.events_path, payload)

    def maybe_save_debug_frame(
        self,
        frame_bgr: np.ndarray | None,
        prefix: str,
        timestamp: int,
    ) -> Path | None:
        if not self.save_debug_frames or frame_bgr is None:
            return None
        now = time.monotonic()
        if now - self._last_frame_saved_at < self.debug_frame_interval_s:
            return None
        self._last_frame_saved_at = now
        target = self.frames_dir / f"{prefix}_{timestamp}.png"
        cv2.imwrite(str(target), frame_bgr)
        return target

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
