from app.runtime_loops import WATCHDOG_LOOP_NAMES, find_stale_loop_names


def test_watchdog_detects_missing_and_stale_loops() -> None:
    heartbeats = {
        "telemetry": 10.0,
        "frame": 10.2,
        "vision": 8.5,
        "mission": 10.1,
    }

    stale_loop_names = find_stale_loop_names(
        heartbeats=heartbeats,
        tracked_loop_names=WATCHDOG_LOOP_NAMES,
        now_s=10.6,
        stale_after_s=1.0,
    )

    assert stale_loop_names == ["vision", "control"]


def test_watchdog_accepts_fresh_heartbeats() -> None:
    heartbeats = {loop_name: 25.0 for loop_name in WATCHDOG_LOOP_NAMES}

    stale_loop_names = find_stale_loop_names(
        heartbeats=heartbeats,
        tracked_loop_names=WATCHDOG_LOOP_NAMES,
        now_s=25.5,
        stale_after_s=1.0,
    )

    assert stale_loop_names == []
