from control.safety import CommandSafetyLimits, CommandSafetyLimiter
from telemetry.models import RuntimeControlCommand


def test_clamps_command_components_to_limits() -> None:
    limiter = CommandSafetyLimiter(
        CommandSafetyLimits(
            max_velocity_xy_m_s=1.0,
            max_velocity_z_m_s=0.5,
            max_yaw_rate_deg_s=15.0,
            min_command_duration_s=0.2,
            max_command_duration_s=0.4,
        )
    )
    command = RuntimeControlCommand(
        vx=2.0,
        vy=-2.5,
        vz=0.8,
        yaw_rate=-20.0,
        duration_s=0.1,
        reason="unsafe command",
    )

    safe_command = limiter.clamp(command)

    assert safe_command.vx == 1.0
    assert safe_command.vy == -1.0
    assert safe_command.vz == 0.5
    assert safe_command.yaw_rate == -15.0
    assert safe_command.duration_s == 0.2
    assert "safety clamped" in safe_command.reason


def test_preserves_safe_command_without_reason_noise() -> None:
    limiter = CommandSafetyLimiter(
        CommandSafetyLimits(
            max_velocity_xy_m_s=1.0,
            max_velocity_z_m_s=0.5,
            max_yaw_rate_deg_s=15.0,
            min_command_duration_s=0.2,
            max_command_duration_s=0.4,
        )
    )
    command = RuntimeControlCommand(
        vx=0.4,
        vy=-0.3,
        vz=0.1,
        yaw_rate=8.0,
        duration_s=0.25,
        reason="already safe",
    )

    safe_command = limiter.clamp(command)

    assert safe_command == command
