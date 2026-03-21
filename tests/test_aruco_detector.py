import cv2
import numpy as np

from vision.aruco_detector import ArucoDetection, ArucoDetector


def test_detects_aruco_marker_from_generated_image() -> None:
    frame = _generate_marker_frame(marker_id=7, dictionary_name="DICT_4X4_50")
    detector = ArucoDetector(dictionary_name="DICT_4X4_50")

    detection = detector.detect(frame, target_marker_id=7)

    assert detection.detected is True
    assert detection.marker_id == 7
    assert detection.center_x is not None
    assert detection.center_y is not None
    assert len(detection.corners) == 4
    assert detection.area > 0.0


def test_draw_overlay_changes_frame_when_marker_detected() -> None:
    frame = _generate_marker_frame(marker_id=3, dictionary_name="DICT_4X4_50")
    detector = ArucoDetector(dictionary_name="DICT_4X4_50")
    detection = detector.detect(frame, target_marker_id=3)

    overlay_frame = detector.draw_overlay(frame, detection)

    assert detection.detected is True
    assert np.any(overlay_frame != frame)


def test_returns_empty_detection_when_marker_not_found() -> None:
    blank_frame = np.full((400, 400, 3), 255, dtype=np.uint8)
    detector = ArucoDetector(dictionary_name="DICT_4X4_50")

    detection = detector.detect(blank_frame, target_marker_id=1)

    assert detection == ArucoDetection(
        detected=False,
        marker_id=None,
        center_x=None,
        center_y=None,
        corners=(),
        area=0.0,
    )


def _generate_marker_frame(marker_id: int, dictionary_name: str) -> np.ndarray:
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    marker_size_px = 200
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)

    frame = np.full((400, 400), 255, dtype=np.uint8)
    frame[100:300, 100:300] = marker_image
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
