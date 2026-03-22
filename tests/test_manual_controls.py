from __future__ import annotations

import asyncio

from app.main import _apply_manual_key_input
from telemetry.models import RuntimeSharedState


def test_forward_key_sets_manual_forward_override() -> None:
    shared_state = RuntimeSharedState()
    state_lock = asyncio.Lock()

    asyncio.run(_apply_manual_key_input(shared_state, state_lock, ord("i")))

    assert shared_state.local_manual_vx_m_s == 1.0
    assert shared_state.local_manual_status == "forward"


def test_backward_key_sets_manual_backward_override() -> None:
    shared_state = RuntimeSharedState()
    state_lock = asyncio.Lock()

    asyncio.run(_apply_manual_key_input(shared_state, state_lock, ord("k")))

    assert shared_state.local_manual_vx_m_s == -1.0
    assert shared_state.local_manual_status == "backward"


def test_manual_mode_toggle_switches_between_manual_and_auto() -> None:
    shared_state = RuntimeSharedState()
    state_lock = asyncio.Lock()

    asyncio.run(_apply_manual_key_input(shared_state, state_lock, ord("m")))
    assert shared_state.local_manual_mode_enabled is True
    assert shared_state.local_manual_status == "manual_mode"

    asyncio.run(_apply_manual_key_input(shared_state, state_lock, ord("m")))
    assert shared_state.local_manual_mode_enabled is False
    assert shared_state.local_manual_status == "auto"
