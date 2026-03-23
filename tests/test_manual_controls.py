from __future__ import annotations

import asyncio

from app.local_runtime import apply_manual_key_input, normalize_manual_key
from telemetry.models import RuntimeFrameState, RuntimeSharedState
from vision.aruco_detector import ArucoDetection


def test_forward_key_sets_manual_forward_override() -> None:
    shared_state = RuntimeSharedState()
    state_lock = asyncio.Lock()

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("i")))

    assert shared_state.local_manual_vx_m_s == 1.0
    assert shared_state.local_manual_status == "forward"
    assert shared_state.control_applied is False
    assert shared_state.watchdog_triggered is False


def test_backward_key_sets_manual_backward_override() -> None:
    shared_state = RuntimeSharedState()
    state_lock = asyncio.Lock()

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("k")))

    assert shared_state.local_manual_vx_m_s == -1.0
    assert shared_state.local_manual_status == "backward"


def test_manual_input_clears_sticky_watchdog_and_rearms_control() -> None:
    shared_state = RuntimeSharedState()
    shared_state.watchdog_triggered = True
    shared_state.watchdog_reason = "stale loop"
    shared_state.control_applied = True
    state_lock = asyncio.Lock()

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("i")))

    assert shared_state.watchdog_triggered is False
    assert shared_state.watchdog_reason == ""
    assert shared_state.control_applied is False


def test_manual_mode_toggle_switches_between_manual_and_auto() -> None:
    shared_state = RuntimeSharedState()
    state_lock = asyncio.Lock()

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("m")))
    assert shared_state.local_manual_mode_enabled is True
    assert shared_state.local_autopilot_enabled is False
    assert shared_state.local_manual_status == "manual_mode"

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("m")))
    assert shared_state.local_manual_mode_enabled is False
    assert shared_state.local_autopilot_enabled is True
    assert shared_state.local_manual_status == "auto"


def test_cyrillic_forward_key_sets_manual_forward_override() -> None:
    shared_state = RuntimeSharedState()
    state_lock = asyncio.Lock()

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("ш")))

    assert shared_state.local_manual_vx_m_s == 1.0
    assert shared_state.local_manual_status == "forward"


def test_cyrillic_manual_mode_toggle_switches_between_manual_and_auto() -> None:
    shared_state = RuntimeSharedState()
    state_lock = asyncio.Lock()

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("ь")))
    assert shared_state.local_manual_mode_enabled is True
    assert shared_state.local_autopilot_enabled is False
    assert shared_state.local_manual_status == "manual_mode"

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("ь")))
    assert shared_state.local_manual_mode_enabled is False
    assert shared_state.local_autopilot_enabled is True
    assert shared_state.local_manual_status == "auto"


def test_autopilot_toggle_switches_between_on_and_off() -> None:
    shared_state = RuntimeSharedState()
    state_lock = asyncio.Lock()

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("p")))
    assert shared_state.local_autopilot_enabled is True
    assert shared_state.local_autopilot_target_locked is False
    assert shared_state.local_manual_status == "autopilot_on"

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("p")))
    assert shared_state.local_autopilot_enabled is False
    assert shared_state.local_manual_status == "autopilot_off"


def test_autopilot_toggle_captures_visible_target() -> None:
    shared_state = RuntimeSharedState()
    shared_state.frames = RuntimeFrameState(rgb_frame=object(), depth_frame=object())
    shared_state.detection = ArucoDetection(
        detected=True,
        marker_id=0,
        center_x=320.0,
        center_y=240.0,
        corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        area=10000.0,
    )
    state_lock = asyncio.Lock()

    asyncio.run(apply_manual_key_input(shared_state, state_lock, ord("p")))

    assert shared_state.local_autopilot_enabled is True
    assert shared_state.local_autopilot_target_locked is True
    assert shared_state.local_autopilot_target_marker_id == 0


def test_normalize_manual_key_accepts_raw_ascii_low_byte() -> None:
    assert normalize_manual_key(0x01000061) == ord("a")


def test_normalize_manual_key_preserves_unicode_manual_keys() -> None:
    assert normalize_manual_key(ord("ш")) == ord("ш")
