from __future__ import annotations

import asyncio
import logging

from app.main import initialize_runtime_shared_state, wait_for_runtime_shutdown
from telemetry.models import RuntimeSharedState


def test_initialize_runtime_shared_state_enables_autopilot_and_heartbeats() -> None:
    shared_state = RuntimeSharedState()

    initialize_runtime_shared_state(shared_state)

    assert shared_state.local_autopilot_enabled is True
    assert set(shared_state.loop_heartbeats) == {
        "telemetry",
        "frame",
        "vision",
        "mission",
        "control",
    }


def test_wait_for_runtime_shutdown_tolerates_finished_interaction_task() -> None:
    async def scenario() -> None:
        shared_state = RuntimeSharedState()
        initialize_runtime_shared_state(shared_state)

        async def core_task() -> None:
            await asyncio.sleep(0.03)

        async def interaction_task() -> None:
            return

        core = asyncio.create_task(core_task(), name="core-test")
        interaction = asyncio.create_task(interaction_task(), name="ui-test")
        await wait_for_runtime_shutdown(
            shared_state=shared_state,
            core_tasks=[core],
            interaction_tasks=[interaction],
            logger=logging.getLogger("test.runtime"),
            run_duration_s=0.01,
            poll_interval_s=0.002,
        )
        core.cancel()
        await asyncio.gather(core, return_exceptions=True)

    asyncio.run(scenario())
