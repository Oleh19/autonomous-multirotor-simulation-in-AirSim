from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthAnalysis:
    obstacle_detected: bool
    nearest_distance_m: float
    left_clearance_m: float
    center_clearance_m: float
    right_clearance_m: float
    safer_side: str


class DepthAnalyzer:
    def __init__(
        self,
        obstacle_distance_m: float,
        min_valid_depth_m: float,
        max_valid_depth_m: float,
    ) -> None:
        self.obstacle_distance_m = obstacle_distance_m
        self.min_valid_depth_m = min_valid_depth_m
        self.max_valid_depth_m = max_valid_depth_m

    def analyze(self, depth_m: np.ndarray) -> DepthAnalysis:
        if depth_m.ndim != 2:
            raise ValueError("Depth image must be a 2D array.")

        filtered = self._filter_depth(depth_m)
        if filtered.size == 0:
            raise RuntimeError("Depth image has no valid values in the configured range.")

        left_zone, center_zone, right_zone = self._split_zones(filtered)
        left_clearance = self._zone_clearance(left_zone)
        center_clearance = self._zone_clearance(center_zone)
        right_clearance = self._zone_clearance(right_zone)
        nearest_distance = min(left_clearance, center_clearance, right_clearance)
        safer_side = self._choose_safer_side(left_clearance, right_clearance)

        return DepthAnalysis(
            obstacle_detected=center_clearance <= self.obstacle_distance_m,
            nearest_distance_m=nearest_distance,
            left_clearance_m=left_clearance,
            center_clearance_m=center_clearance,
            right_clearance_m=right_clearance,
            safer_side=safer_side,
        )

    def _filter_depth(self, depth_m: np.ndarray) -> np.ndarray:
        valid_mask = (
            np.isfinite(depth_m)
            & (depth_m >= self.min_valid_depth_m)
            & (depth_m <= self.max_valid_depth_m)
        )
        filtered = np.where(valid_mask, depth_m, np.nan)
        return filtered

    @staticmethod
    def _split_zones(depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        width = depth_m.shape[1]
        third = width // 3
        left = depth_m[:, :third]
        center = depth_m[:, third : third * 2]
        right = depth_m[:, third * 2 :]
        return left, center, right

    @staticmethod
    def _zone_clearance(zone: np.ndarray) -> float:
        valid = zone[np.isfinite(zone)]
        if valid.size == 0:
            return float("inf")
        return float(np.nanpercentile(valid, 30))

    @staticmethod
    def _choose_safer_side(left_clearance_m: float, right_clearance_m: float) -> str:
        if left_clearance_m >= right_clearance_m:
            return "left"
        return "right"
