from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np


@dataclass
class ArucoDetection:
    detected: bool
    marker_id: int | None
    center_x: float | None
    center_y: float | None
    corners: tuple[tuple[float, float], ...]
    area: float


ARUCO_DICTIONARIES: Final[dict[str, int]] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}


class ArucoDetector:
    def __init__(self, dictionary_name: str = "DICT_4X4_50") -> None:
        if dictionary_name not in ARUCO_DICTIONARIES:
            raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")

        dictionary_id = ARUCO_DICTIONARIES[dictionary_name]
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect(self, frame_bgr: np.ndarray, target_marker_id: int | None = None) -> ArucoDetection:
        corners, ids, _ = self.detector.detectMarkers(frame_bgr)
        if ids is None or len(ids) == 0:
            return ArucoDetection(
                detected=False,
                marker_id=None,
                center_x=None,
                center_y=None,
                corners=(),
                area=0.0,
            )

        flattened_ids = ids.flatten().tolist()
        selected_index = 0
        if target_marker_id is not None:
            try:
                selected_index = flattened_ids.index(target_marker_id)
            except ValueError:
                return ArucoDetection(
                    detected=False,
                    marker_id=None,
                    center_x=None,
                    center_y=None,
                    corners=(),
                    area=0.0,
                )

        marker_corners = corners[selected_index].reshape(4, 2).astype(np.float32)
        center = marker_corners.mean(axis=0)
        area = float(cv2.contourArea(marker_corners))
        return ArucoDetection(
            detected=True,
            marker_id=int(flattened_ids[selected_index]),
            center_x=float(center[0]),
            center_y=float(center[1]),
            corners=tuple((float(x), float(y)) for x, y in marker_corners),
            area=area,
        )

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        detection: ArucoDetection,
    ) -> np.ndarray:
        output = frame_bgr.copy()
        if not detection.detected:
            return output

        points = np.array(detection.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        if detection.center_x is not None and detection.center_y is not None:
            center = (int(round(detection.center_x)), int(round(detection.center_y)))
            cv2.circle(output, center, 5, (0, 0, 255), thickness=-1)

        label = f"id={detection.marker_id} area={detection.area:.1f}"
        text_origin = (
            int(round(detection.corners[0][0])),
            max(20, int(round(detection.corners[0][1])) - 10),
        )
        cv2.putText(
            output,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return output
