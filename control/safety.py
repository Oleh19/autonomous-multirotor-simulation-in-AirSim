from __future__ import annotations

from dataclasses import dataclass

from telemetry.models import RuntimeControlCommand


@dataclass(frozen=True)
class CommandSafetyLimits:
    max_velocity_xy_m_s: float
    max_velocity_z_m_s: float
    max_yaw_rate_deg_s: float
    min_command_duration_s: float
    max_command_duration_s: float


class CommandSafetyLimiter:
    def __init__(self, limits: CommandSafetyLimits) -> None:
        self.limits = limits

    def clamp(self, command: RuntimeControlCommand) -> RuntimeControlCommand:
        limited_vx = self._clamp_abs(command.vx, self.limits.max_velocity_xy_m_s)
        limited_vy = self._clamp_abs(command.vy, self.limits.max_velocity_xy_m_s)
        limited_vz = self._clamp_abs(command.vz, self.limits.max_velocity_z_m_s)
        limited_yaw_rate = self._clamp_abs(command.yaw_rate, self.limits.max_yaw_rate_deg_s)
        limited_duration = min(
            self.limits.max_command_duration_s,
            max(self.limits.min_command_duration_s, command.duration_s),
        )

        reason = command.reason
        if (
            limited_vx != command.vx
            or limited_vy != command.vy
            or limited_vz != command.vz
            or limited_yaw_rate != command.yaw_rate
            or limited_duration != command.duration_s
        ):
            reason = f"{command.reason} | safety clamped"

        return RuntimeControlCommand(
            vx=limited_vx,
            vy=limited_vy,
            vz=limited_vz,
            yaw_rate=limited_yaw_rate,
            duration_s=limited_duration,
            source=command.source,
            reason=reason,
        )

    @staticmethod
    def _clamp_abs(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))
