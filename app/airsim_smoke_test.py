from __future__ import annotations

from pathlib import Path
import sys
import time


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.airsim_client import AirSimClientAdapter, AirSimConnectionConfig
from app.bootstrap import bootstrap_app


def main() -> int:
    context = bootstrap_app()
    logger = context["logger"]
    settings = context["settings"]
    airsim_settings = settings.get("airsim", {})

    adapter = AirSimClientAdapter(
        config=AirSimConnectionConfig(
            host=str(airsim_settings.get("host", "127.0.0.1")),
            port=int(airsim_settings.get("port", 41451)),
            timeout_seconds=float(airsim_settings.get("timeout_seconds", 10.0)),
            vehicle_name=str(airsim_settings.get("vehicle_name", "")),
        ),
        logger=logger,
    )

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
