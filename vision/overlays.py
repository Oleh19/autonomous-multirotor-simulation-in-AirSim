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
from mission.states import MissionState
from telemetry.models import TelemetrySnapshot
from vision.frame_fetcher import FrameFetcher


@dataclass
class OverlayLabel:
    text: str


class OverlayRenderer:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("drone_cv.overlays")

    def draw(
        self,
        frame_bgr: np.ndarray,
        mission_state: str,
        telemetry: TelemetrySnapshot,
    ) -> np.ndarray:
        output = frame_bgr.copy()
        self._draw_crosshair(output)
        self._draw_text_block(output, mission_state, telemetry)
        return output

    @staticmethod
    def _draw_crosshair(frame_bgr: np.ndarray) -> None:
        height, width = frame_bgr.shape[:2]
        center_x = width // 2
        center_y = height // 2
        color = (0, 255, 255)
        size = 18
        thickness = 2

        cv2.line(
            frame_bgr,
            (center_x - size, center_y),
            (center_x + size, center_y),
            color,
            thickness,
        )
        cv2.line(
            frame_bgr,
            (center_x, center_y - size),
            (center_x, center_y + size),
            color,
            thickness,
        )
        cv2.circle(frame_bgr, (center_x, center_y), 6, color, 1)

    @staticmethod
    def _draw_text_block(
        frame_bgr: np.ndarray,
        mission_state: str,
        telemetry: TelemetrySnapshot,
    ) -> None:
        lines = [
            f"State: {mission_state}",
            f"Altitude: {telemetry.altitude_m:.2f} m",
            f"Speed: {telemetry.speed_m_s:.2f} m/s",
            (
                "Position: "
                f"x={telemetry.position_m.x:.2f} "
                f"y={telemetry.position_m.y:.2f} "
                f"z={telemetry.position_m.z:.2f}"
            ),
            f"Timestamp: {telemetry.timestamp}",
        ]

        origin_x = 12
        origin_y = 28
        line_height = 24
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_color = (255, 255, 255)
        background_color = (32, 32, 32)
        thickness = 2

        max_width = 0
        for line in lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_size[0])

        block_height = line_height * len(lines) + 12
        cv2.rectangle(
            frame_bgr,
            (origin_x - 8, origin_y - 22),
            (origin_x + max_width + 12, origin_y - 22 + block_height),
            background_color,
            thickness=-1,
        )

        for index, line in enumerate(lines):
            y = origin_y + (index * line_height)
            cv2.putText(
                frame_bgr,
                line,
                (origin_x, y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )


def run_overlay_debug() -> int:
    context = bootstrap_app()
    logger = context["logger"]
    settings = context["settings"]
    airsim_settings = settings.get("airsim", {})
    camera_settings = settings.get("camera", {})
    overlay_settings = settings.get("overlays", {})

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

    frame_fetcher = FrameFetcher(
        adapter=adapter,
        rgb_camera_name=str(camera_settings.get("rgb_camera_name", "front_center")),
        depth_camera_name=str(camera_settings.get("depth_camera_name", "front_center")),
        logger=logger,
    )
    frame_bundle = frame_fetcher.fetch()
    telemetry = adapter.get_telemetry()
    mission_state = str(overlay_settings.get("mission_state", MissionState.IDLE.value))

    renderer = OverlayRenderer(logger=logger)
    overlay_frame = renderer.draw(
        frame_bgr=frame_bundle.rgb_bgr,
        mission_state=mission_state,
        telemetry=telemetry,
    )

    mode = str(overlay_settings.get("mode", "image")).lower()
    output_dir = _resolve_output_dir(str(overlay_settings.get("output_dir", "debug_overlays")))
    output_path = output_dir / str(overlay_settings.get("output_filename", "overlay_debug.png"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay_frame)
    logger.info("Saved overlay debug image to %s", output_path)

    if mode == "window":
        window_name = str(overlay_settings.get("window_name", "drone_cv overlay"))
        cv2.imshow(window_name, overlay_frame)
        logger.info("Displaying overlay window. Press any key in the window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Saved overlay debug image to {output_path}")
    return 0


def _resolve_output_dir(output_dir: str) -> Path:
    target_dir = Path(output_dir)
    if target_dir.is_absolute():
        return target_dir
    return PROJECT_ROOT / target_dir


if __name__ == "__main__":
    raise SystemExit(run_overlay_debug())
