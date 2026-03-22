from __future__ import annotations

from pathlib import Path
import sys
import time


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bootstrap import bootstrap_app, build_airsim_adapter


def main() -> int:
    context = bootstrap_app()
    logger = context["logger"]
    settings = context["settings"]
    airsim_settings = settings.get("airsim", {})

    adapter = build_airsim_adapter(settings, logger)

    hover_duration_seconds = float(
        airsim_settings.get("smoke_test_hover_duration_seconds", 2.0)
    )
    takeoff_timeout_seconds = float(
        airsim_settings.get("takeoff_timeout_seconds", 20.0)
    )
    land_timeout_seconds = float(airsim_settings.get("land_timeout_seconds", 30.0))

    adapter.connect()
    adapter.confirm_connection()
    adapter.enable_api_control(True)
    adapter.arm(True)
    adapter.takeoff(timeout_seconds=takeoff_timeout_seconds)
    adapter.hover()
    logger.info("Holding hover for %.1f seconds", hover_duration_seconds)
    time.sleep(hover_duration_seconds)
    logger.info("Current drone state: %s", adapter.get_state())
    adapter.land(timeout_seconds=land_timeout_seconds)
    adapter.arm(False)
    adapter.enable_api_control(False)
    print("AirSim smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
