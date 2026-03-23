from __future__ import annotations

import logging
import threading
import time

import adapters.airsim_client as airsim_client_module
from adapters.airsim_client import AirSimClientAdapter, AirSimConnectionConfig


class _FakeFuture:
    def __init__(self) -> None:
        self.join_calls = 0

    def join(self) -> None:
        self.join_calls += 1


class _FakeYawMode:
    def __init__(self, is_rate: bool, yaw_or_rate: float) -> None:
        self.is_rate = is_rate
        self.yaw_or_rate = yaw_or_rate


class _FakeClient:
    def __init__(self) -> None:
        self.move_future = _FakeFuture()
        self.hover_future = _FakeFuture()
        self.land_future = _FakeFuture()
        self.takeoff_future = _FakeFuture()

    def moveByVelocityBodyFrameAsync(self, **kwargs):
        self.move_kwargs = kwargs
        return self.move_future

    def hoverAsync(self, **kwargs):
        self.hover_kwargs = kwargs
        return self.hover_future

    def landAsync(self, **kwargs):
        self.land_kwargs = kwargs
        return self.land_future

    def takeoffAsync(self, **kwargs):
        self.takeoff_kwargs = kwargs
        return self.takeoff_future


class _FakeKinematics:
    class _Value:
        def __init__(self, value: float) -> None:
            self.x_val = value
            self.y_val = value
            self.z_val = value
            self.w_val = value

    def __init__(self) -> None:
        self.position = self._Value(0.0)
        self.linear_velocity = self._Value(0.0)
        self.orientation = self._Value(0.0)


class _FakeState:
    def __init__(self) -> None:
        self.ready = True
        self.landed_state = 0
        self.timestamp = 123
        self.kinematics_estimated = _FakeKinematics()


class _ConcurrencyCheckingClient(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self._active_calls = 0
        self.max_active_calls = 0
        self._counter_lock = threading.Lock()

    def _enter_call(self) -> None:
        with self._counter_lock:
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)

    def _leave_call(self) -> None:
        with self._counter_lock:
            self._active_calls -= 1

    def getMultirotorState(self, **kwargs):
        self._enter_call()
        try:
            time.sleep(0.02)
            return _FakeState()
        finally:
            self._leave_call()

    def simGetImages(self, requests, **kwargs):
        self._enter_call()
        try:
            time.sleep(0.02)
            return []
        finally:
            self._leave_call()


class _FakeAirSimModule:
    class DrivetrainType:
        MaxDegreeOfFreedom = "max-dof"

    YawMode = _FakeYawMode


def test_move_by_velocity_body_is_non_blocking_by_default() -> None:
    adapter = AirSimClientAdapter(
        AirSimConnectionConfig(host="127.0.0.1", port=41451),
        logger=logging.getLogger("test.airsim"),
    )
    adapter._client = _FakeClient()
    original_airsim = airsim_client_module.airsim
    airsim_client_module.airsim = _FakeAirSimModule

    try:
        adapter.move_by_velocity_body(0.4, -0.2, 0.1, 0.25, 8.0)
    finally:
        airsim_client_module.airsim = original_airsim

    assert adapter._client.move_future.join_calls == 0
    assert adapter._client.move_kwargs["duration"] == 0.25
    assert adapter._client.move_kwargs["yaw_mode"].yaw_or_rate == 8.0


def test_hover_wait_false_does_not_block_and_land_waits_for_previous_command() -> None:
    adapter = AirSimClientAdapter(
        AirSimConnectionConfig(host="127.0.0.1", port=41451),
        logger=logging.getLogger("test.airsim"),
    )
    adapter._client = _FakeClient()

    adapter.hover(wait=False)
    adapter.land(timeout_seconds=3.0)

    assert adapter._client.hover_future.join_calls == 1
    assert adapter._client.land_future.join_calls == 1


def test_adapter_serializes_sensor_rpc_calls_across_threads() -> None:
    adapter = AirSimClientAdapter(
        AirSimConnectionConfig(host="127.0.0.1", port=41451),
        logger=logging.getLogger("test.airsim"),
    )
    adapter._client = _ConcurrencyCheckingClient()

    def call_telemetry() -> None:
        adapter.get_telemetry()

    threads = [threading.Thread(target=call_telemetry) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert adapter._client.max_active_calls == 1
