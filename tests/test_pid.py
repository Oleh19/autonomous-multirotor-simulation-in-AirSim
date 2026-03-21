from control.pid import PIDController


def test_pid_update_returns_float() -> None:
    controller = PIDController(kp=1.0, ki=0.1, kd=0.01)
    output = controller.update(error=0.5, dt=0.1)
    assert isinstance(output, float)


def test_pid_integral_and_derivative_terms_are_applied() -> None:
    controller = PIDController(kp=2.0, ki=1.0, kd=0.5)

    first_output = controller.update(error=1.0, dt=0.5)
    second_output = controller.update(error=1.5, dt=0.5)

    assert first_output == 2.5
    assert second_output == 4.25


def test_pid_output_is_clamped() -> None:
    controller = PIDController(
        kp=10.0,
        ki=0.0,
        kd=0.0,
        output_min=-5.0,
        output_max=5.0,
    )

    high_output = controller.update(error=1.0, dt=0.1)
    low_output = controller.update(error=-1.0, dt=0.1)

    assert high_output == 5.0
    assert low_output == -5.0


def test_pid_reset_clears_internal_state() -> None:
    controller = PIDController(kp=1.0, ki=1.0, kd=1.0)

    controller.update(error=2.0, dt=0.5)
    controller.reset()

    assert controller.integral == 0.0
    assert controller.previous_error == 0.0
    assert controller.has_previous_error is False
