from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    output_min: float | None = None
    output_max: float | None = None
    integral: float = 0.0
    previous_error: float = 0.0
    has_previous_error: bool = False

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            output = self.kp * error
            return self._clamp(output)

        self.integral += error * dt
        derivative = 0.0
        if self.has_previous_error:
            derivative = (error - self.previous_error) / dt

        self.previous_error = error
        self.has_previous_error = True

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return self._clamp(output)

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0
        self.has_previous_error = False

    def _clamp(self, value: float) -> float:
        if self.output_min is not None:
            value = max(self.output_min, value)
        if self.output_max is not None:
            value = min(self.output_max, value)
        return value
