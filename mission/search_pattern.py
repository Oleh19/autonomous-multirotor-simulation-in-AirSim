from __future__ import annotations

from dataclasses import dataclass

from control.visual_servo import VisualServoCommand


@dataclass
class SearchPattern:
    name: str = "yaw_scan"
    yaw_rate_deg_s: float = 8.0
    step_duration_s: float = 0.35

    def next_command(self) -> VisualServoCommand:
        return VisualServoCommand(
            vx=0.0,
            vy=0.0,
            vz=0.0,
            yaw_rate=self.yaw_rate_deg_s,
            duration_s=self.step_duration_s,
            marker_detected=False,
        )
