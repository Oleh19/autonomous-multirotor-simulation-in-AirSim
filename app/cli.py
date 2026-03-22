from __future__ import annotations

import argparse
import asyncio
import copy
from typing import Callable

from app.bootstrap import bootstrap_app, validate_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local runtime for drone_cv")
    parser.add_argument(
        "--mode",
        choices=("dev", "smoke", "local"),
        default="dev",
        help="Run dev, smoke, or local mode",
    )
    parser.add_argument(
        "--profile",
        choices=("default", "safe", "ci"),
        default="default",
        help="Apply runtime overrides for default, safe, or ci execution",
    )
    return parser.parse_args()


def apply_runtime_profile(settings: dict[str, object], profile: str, mode: str) -> dict[str, object]:
    if profile == "default":
        return settings

    profiled_settings = copy.deepcopy(settings)
    runtime_settings = profiled_settings.setdefault("runtime", {})
    recording_settings = profiled_settings.setdefault("recording", {})
    local_ui_settings = profiled_settings.setdefault("local_ui", {})
    dev_ui_settings = profiled_settings.setdefault("dev_ui", {})
    watchdog_settings = profiled_settings.setdefault("watchdog", {})
    freshness_settings = profiled_settings.setdefault("freshness", {})

    if profile == "safe":
        runtime_settings["control_interval_s"] = min(
            float(runtime_settings.get("control_interval_s", 0.2)),
            0.15,
        )
        watchdog_settings["stale_after_s"] = min(
            float(watchdog_settings.get("stale_after_s", 1.0)),
            0.75,
        )
        freshness_settings["telemetry_max_age_s"] = min(
            float(freshness_settings.get("telemetry_max_age_s", 1.5)),
            1.0,
        )
        freshness_settings["frames_max_age_s"] = min(
            float(freshness_settings.get("frames_max_age_s", 1.0)),
            0.75,
        )
        freshness_settings["detection_max_age_s"] = min(
            float(freshness_settings.get("detection_max_age_s", 1.0)),
            0.75,
        )
        freshness_settings["depth_analysis_max_age_s"] = min(
            float(freshness_settings.get("depth_analysis_max_age_s", 1.0)),
            0.75,
        )
    elif profile == "ci":
        runtime_settings["run_duration_s"] = min(
            float(runtime_settings.get("run_duration_s", 120.0)),
            5.0,
        )
        recording_settings["save_debug_frames"] = False
        local_ui_settings["enabled"] = False
        dev_ui_settings["enabled"] = False
        watchdog_settings["enabled"] = True
        if mode == "dev":
            raise ValueError("The 'ci' profile is intended for --mode local or --mode smoke only.")

    return profiled_settings


def print_startup_info(settings: dict[str, object], mode: str) -> None:
    app_name = settings.get("app", {}).get("name", "drone_cv")
    airsim_settings = settings.get("airsim", {})
    print(f"{app_name} starting in {mode} mode")
    if mode == "local":
        print("AirSim target: disabled in local mode")
    else:
        print(
            "AirSim target: "
            f"{airsim_settings.get('host', '127.0.0.1')}:{airsim_settings.get('port', 41451)}"
        )


def fail_with_actionable_error(message: str) -> int:
    print(f"Startup error: {message}")
    print(
        "Check config/settings.yaml and install dependencies with: "
        "python3 -m pip install -r requirements.txt"
    )
    return 1


def run_smoke_mode(profile: str) -> int:
    context = bootstrap_app()
    settings = apply_runtime_profile(context["settings"], profile, mode="smoke")
    logger = context["logger"]
    errors = validate_settings(settings)
    if errors:
        return fail_with_actionable_error("; ".join(errors))

    print_startup_info(settings, mode="smoke")
    logger.info("Smoke mode starting")

    try:
        from app.bootstrap import build_airsim_adapter
    except ModuleNotFoundError as exc:
        return fail_with_actionable_error(f"Missing dependency: {exc.name}")

    airsim_settings = settings.get("airsim", {})
    adapter = build_airsim_adapter(settings, logger)

    try:
        adapter.connect()
        print("AirSim adapter initialized")
        adapter.confirm_connection()
        print("AirSim connection confirmed")
    except Exception as exc:  # pragma: no cover - depends on local AirSim
        return fail_with_actionable_error(
            f"Could not connect to AirSim at {airsim_settings.get('host')}:{airsim_settings.get('port')}: {exc}"
        )

    print("Smoke check passed")
    return 0


def run_dev_mode(
    profile: str,
    run_runtime: Callable[..., object],
) -> int:
    context = bootstrap_app()
    settings = apply_runtime_profile(context["settings"], profile, mode="dev")
    logger = context["logger"]
    errors = validate_settings(settings)
    if errors:
        return fail_with_actionable_error("; ".join(errors))

    print_startup_info(settings, mode="dev")
    logger.info("Dev mode validated config successfully")

    try:
        return asyncio.run(run_runtime(settings=settings, logger=logger))
    except ModuleNotFoundError as exc:
        return fail_with_actionable_error(f"Missing dependency: {exc.name}")
    except Exception as exc:  # pragma: no cover - depends on local AirSim/runtime
        return fail_with_actionable_error(str(exc))


def run_local_mode(
    profile: str,
    run_local_runtime: Callable[..., object],
) -> int:
    context = bootstrap_app()
    settings = apply_runtime_profile(context["settings"], profile, mode="local")
    logger = context["logger"]
    errors = validate_settings(settings)
    if errors:
        return fail_with_actionable_error("; ".join(errors))

    print_startup_info(settings, mode="local")
    logger.info("Local mode validated config successfully")

    try:
        return asyncio.run(run_local_runtime(settings=settings, logger=logger))
    except ModuleNotFoundError as exc:
        return fail_with_actionable_error(f"Missing dependency: {exc.name}")
    except Exception as exc:
        return fail_with_actionable_error(str(exc))
