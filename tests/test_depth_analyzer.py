import numpy as np

from vision.depth_analyzer import DepthAnalysis, DepthAnalyzer


def test_detects_blocked_center_and_chooses_right_side() -> None:
    analyzer = DepthAnalyzer(
        obstacle_distance_m=2.0,
        min_valid_depth_m=0.2,
        max_valid_depth_m=20.0,
    )
    depth = np.full((6, 9), 5.0, dtype=np.float32)
    depth[:, 3:6] = 1.0
    depth[:, :3] = 2.5
    depth[:, 6:] = 4.5

    analysis = analyzer.analyze(depth)

    assert analysis.obstacle_detected is True
    assert analysis.center_clearance_m == 1.0
    assert analysis.left_clearance_m == 2.5
    assert analysis.right_clearance_m == 4.5
    assert analysis.safer_side == "right"
    assert analysis.nearest_distance_m == 1.0


def test_reports_clear_path_when_center_zone_is_safe() -> None:
    analyzer = DepthAnalyzer(
        obstacle_distance_m=2.0,
        min_valid_depth_m=0.2,
        max_valid_depth_m=20.0,
    )
    depth = np.full((6, 9), 6.0, dtype=np.float32)

    analysis = analyzer.analyze(depth)

    assert analysis == DepthAnalysis(
        obstacle_detected=False,
        nearest_distance_m=6.0,
        left_clearance_m=6.0,
        center_clearance_m=6.0,
        right_clearance_m=6.0,
        safer_side="left",
    )


def test_ignores_invalid_depth_values() -> None:
    analyzer = DepthAnalyzer(
        obstacle_distance_m=2.0,
        min_valid_depth_m=0.2,
        max_valid_depth_m=20.0,
    )
    depth = np.array(
        [
            [np.nan, 30.0, 5.0, 1.5, 1.5, 1.5, 4.0, 4.0, 4.0],
            [np.nan, 30.0, 5.0, 1.5, 1.5, 1.5, 4.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )

    analysis = analyzer.analyze(depth)

    assert analysis.obstacle_detected is True
    assert analysis.left_clearance_m == 5.0
    assert analysis.center_clearance_m == 1.5
    assert analysis.right_clearance_m == 4.0
