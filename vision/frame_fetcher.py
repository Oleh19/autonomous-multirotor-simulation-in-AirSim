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


@dataclass
class FrameBundle:
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    rgb_timestamp: int
    depth_timestamp: int


class FrameFetcher:
    def __init__(
        self,
        adapter: AirSimClientAdapter,
        rgb_camera_name: str,
        depth_camera_name: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.adapter = adapter
        self.rgb_camera_name = rgb_camera_name
        self.depth_camera_name = depth_camera_name
        self.logger = logger or logging.getLogger("drone_cv.frame_fetcher")

    def fetch(self) -> FrameBundle:
        image_pair = self.adapter.fetch_rgb_and_depth(
            rgb_camera_name=self.rgb_camera_name,
            depth_camera_name=self.depth_camera_name,
        )

        rgb = self._decode_rgb(
            image_pair.rgb.data_uint8,
            image_pair.rgb.height,
            image_pair.rgb.width,
        )
        depth = self._decode_depth(
            image_pair.depth.data_float32,
            image_pair.depth.height,
            image_pair.depth.width,
        )
        self.logger.info(
            "Fetched frame bundle: rgb=%sx%s depth=%sx%s",
            image_pair.rgb.width,
            image_pair.rgb.height,
            image_pair.depth.width,
            image_pair.depth.height,
        )
        return FrameBundle(
            rgb_bgr=rgb,
            depth_m=depth,
            rgb_timestamp=image_pair.rgb.timestamp,
            depth_timestamp=image_pair.depth.timestamp,
        )

    def save_debug_frames(self, output_dir: Path | str) -> tuple[Path, Path]:
        frames = self.fetch()
        target_dir = Path(output_dir)
        if not target_dir.is_absolute():
            target_dir = PROJECT_ROOT / target_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = target_dir / "rgb_debug.png"
        depth_path = target_dir / "depth_debug.png"

        cv2.imwrite(str(rgb_path), frames.rgb_bgr)
        cv2.imwrite(str(depth_path), self._visualize_depth(frames.depth_m))
        self.logger.info("Saved debug RGB frame to %s", rgb_path)
        self.logger.info("Saved debug depth frame to %s", depth_path)
        return rgb_path, depth_path

    @staticmethod
    def _decode_rgb(data: bytes | None, height: int, width: int) -> np.ndarray:
        if not data:
            raise RuntimeError("AirSim returned an empty RGB frame.")
        rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _decode_depth(
        data: list[float] | None,
        height: int,
        width: int,
    ) -> np.ndarray:
        if data is None or not data:
            raise RuntimeError("AirSim returned an empty depth frame.")
        return np.asarray(data, dtype=np.float32).reshape(height, width)

    @staticmethod
    def _visualize_depth(depth_m: np.ndarray) -> np.ndarray:
        finite_mask = np.isfinite(depth_m)
        if not finite_mask.any():
            return np.zeros(depth_m.shape, dtype=np.uint8)

        valid_depth = depth_m[finite_mask]
        min_depth = float(valid_depth.min())
        max_depth = float(valid_depth.max())
        if max_depth <= min_depth:
            normalized = np.zeros(depth_m.shape, dtype=np.uint8)
            normalized[finite_mask] = 255
            return normalized

        clipped = np.clip(depth_m, min_depth, max_depth)
        normalized = ((clipped - min_depth) / (max_depth - min_depth) * 255.0).astype(
            np.uint8
        )
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def run_debug_capture() -> int:
    context = bootstrap_app()
    logger = context["logger"]
    settings = context["settings"]
    airsim_settings = settings.get("airsim", {})
    camera_settings = settings.get("camera", {})

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

    fetcher = FrameFetcher(
        adapter=adapter,
        rgb_camera_name=str(camera_settings.get("rgb_camera_name", "front_center")),
        depth_camera_name=str(camera_settings.get("depth_camera_name", "front_center")),
        logger=logger,
    )
    output_dir = str(camera_settings.get("debug_output_dir", "debug_frames"))
    rgb_path, depth_path = fetcher.save_debug_frames(output_dir)
    print(f"Saved RGB debug frame to {rgb_path}")
    print(f"Saved depth debug frame to {depth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_debug_capture())
