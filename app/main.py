from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.bootstrap import bootstrap_app, build_runtime_components
    from app.cli import parse_args, run_dev_mode, run_local_mode, run_smoke_mode
    from app.interaction import (
        dev_ui_loop,
        terminal_control_loop,
        terminal_controls_available,
    )
    from app.local_world import (
        build_initial_local_world,
    )
    from app.local_runtime import (
        apply_manual_key_input,
        local_control_loop,
        local_frame_loop,
        local_telemetry_loop,
        local_ui_loop,
    )
    from app.runtime_loops import (
        WATCHDOG_LOOP_NAMES,
        control_loop,
        frame_loop,
        mission_loop,
        telemetry_loop,
        vision_loop,
        watchdog_loop,
    )
else:
    from app.bootstrap import bootstrap_app, build_runtime_components
    from app.cli import parse_args, run_dev_mode, run_local_mode, run_smoke_mode
    from app.interaction import (
        dev_ui_loop,
        terminal_control_loop,
        terminal_controls_available,
    )
    from app.local_world import (
        build_initial_local_world,
    )
    from app.local_runtime import (
        apply_manual_key_input,
        local_control_loop,
        local_frame_loop,
        local_telemetry_loop,
        local_ui_loop,
    )
    from app.runtime_loops import (
        WATCHDOG_LOOP_NAMES,
        control_loop,
        frame_loop,
        mission_loop,
        telemetry_loop,
        vision_loop,
        watchdog_loop,
    )


async def run_runtime(settings: dict[str, object] | None = None, logger=None) -> int:
    if settings is None or logger is None:
        context = bootstrap_app()
        settings = context["settings"]
        logger = context["logger"]
    app_name = settings["app"]["name"]
    runtime = settings.get("runtime", {})
    watchdog_settings = settings.get("watchdog", {})
    airsim_settings = settings.get("airsim", {})
    dev_ui_settings = settings.get("dev_ui", {})

    logger.info("Starting %s asyncio runtime", app_name)
    print(f"{app_name} startup complete")

    components = build_runtime_components(settings, logger)
    adapter = components["adapter"]
    frame_fetcher = components["frame_fetcher"]
    detector = components["detector"]
    visual_servo = components["visual_servo"]
    depth_analyzer = components["depth_analyzer"]
    obstacle_avoidance = components["obstacle_avoidance"]
    safety_limiter = components["safety_limiter"]
    recorder = components["recorder"]
    shared_state = components["shared_state"]
    state_lock = components["state_lock"]
    now_s = time.monotonic()
    shared_state.loop_heartbeats = {loop_name: now_s for loop_name in WATCHDOG_LOOP_NAMES}

    try:
        await asyncio.to_thread(adapter.connect)
        await asyncio.to_thread(adapter.confirm_connection)
        await asyncio.to_thread(adapter.enable_api_control, True)
        await asyncio.to_thread(adapter.arm, True)
        if bool(airsim_settings.get("auto_takeoff_on_start", True)):
            await asyncio.to_thread(
                adapter.takeoff,
                float(airsim_settings.get("takeoff_timeout_seconds", 20.0)),
            )
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to AirSim at "
            f"{airsim_settings.get('host')}:{airsim_settings.get('port')}: {exc}. "
            "Start AirSim locally and confirm the RPC host/port in config/settings.yaml."
        ) from exc

    marker_id = int(settings.get("aruco", {}).get("marker_id", 0))
    telemetry_interval_s = float(runtime.get("telemetry_interval_s", 0.5))
    frame_interval_s = float(runtime.get("frame_interval_s", 0.2))
    vision_interval_s = float(runtime.get("vision_interval_s", 0.2))
    mission_interval_s = float(runtime.get("mission_interval_s", 0.2))
    control_interval_s = float(runtime.get("control_interval_s", 0.2))
    run_duration_s = float(runtime.get("run_duration_s", 3.0))
    profiling_enabled = bool(runtime.get("profiling_enabled", False))
    profiling_warn_threshold_ms = float(runtime.get("profiling_warn_threshold_ms", 8.0))
    watchdog_enabled = bool(watchdog_settings.get("enabled", True))
    watchdog_interval_s = float(watchdog_settings.get("loop_interval_s", 0.1))
    watchdog_stale_after_s = float(watchdog_settings.get("stale_after_s", 1.0))
    dev_ui_enabled = bool(dev_ui_settings.get("enabled", True))
    dev_ui_interval_s = float(dev_ui_settings.get("interval_s", 0.03))
    dev_ui_window_name = str(dev_ui_settings.get("window_name", "drone_cv dev"))

    tasks = [
        asyncio.create_task(telemetry_loop(adapter, recorder, shared_state, state_lock, logger, telemetry_interval_s, profiling_enabled, profiling_warn_threshold_ms), name="telemetry"),
        asyncio.create_task(frame_loop(frame_fetcher, shared_state, state_lock, logger, frame_interval_s, profiling_enabled, profiling_warn_threshold_ms), name="frame"),
        asyncio.create_task(vision_loop(detector, depth_analyzer, recorder, marker_id, shared_state, state_lock, logger, vision_interval_s, profiling_enabled, profiling_warn_threshold_ms), name="vision"),
        asyncio.create_task(mission_loop(visual_servo, obstacle_avoidance, recorder, settings, shared_state, state_lock, logger, mission_interval_s), name="mission"),
        asyncio.create_task(control_loop(adapter, safety_limiter, shared_state, state_lock, logger, control_interval_s, profiling_enabled, profiling_warn_threshold_ms), name="control"),
    ]
    interaction_tasks = []
    if watchdog_enabled:
        tasks.append(
            asyncio.create_task(
                watchdog_loop(shared_state, state_lock, logger, watchdog_interval_s, watchdog_stale_after_s),
                name="watchdog",
            )
        )
    if dev_ui_enabled:
        dev_ui_task = asyncio.create_task(
            dev_ui_loop(
                shared_state,
                state_lock,
                logger,
                dev_ui_interval_s,
                dev_ui_window_name,
            ),
            name="dev_ui",
        )
        tasks.append(dev_ui_task)
        interaction_tasks.append(dev_ui_task)
    if terminal_controls_available():
        terminal_task = asyncio.create_task(
            terminal_control_loop(
                shared_state,
                state_lock,
                logger,
                0.05,
                "dev",
            ),
            name="terminal_control",
        )
        tasks.append(terminal_task)
        interaction_tasks.append(terminal_task)

    try:
        if interaction_tasks:
            done, pending = await asyncio.wait(interaction_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, return_exceptions=True)
        else:
            await asyncio.sleep(run_duration_s)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.to_thread(adapter.hover)
        await asyncio.to_thread(adapter.arm, False)
        await asyncio.to_thread(adapter.enable_api_control, False)
        if dev_ui_enabled:
            import cv2

            cv2.destroyAllWindows()
    return 0


async def run_local_runtime(settings: dict[str, object] | None = None, logger=None) -> int:
    if settings is None or logger is None:
        context = bootstrap_app()
        settings = context["settings"]
        logger = context["logger"]
    runtime = settings.get("runtime", {})
    watchdog_settings = settings.get("watchdog", {})
    local_ui_settings = settings.get("local_ui", {})
    logger.info("Starting local asyncio runtime without AirSim")
    print(f"{settings['app']['name']} startup complete")

    from app.bootstrap import (
        build_aruco_detector,
        build_depth_analyzer,
        build_obstacle_avoidance_controller,
    )
    from control.safety import CommandSafetyLimits, CommandSafetyLimiter
    from control.visual_servo import build_visual_servo_controller
    from telemetry.models import RuntimeSharedState
    from telemetry.recorder import TelemetryRecorder

    aruco_settings = settings.get("aruco", {})
    control_settings = settings.get("control", {})
    depth_settings = settings.get("depth", {})
    recording_settings = settings.get("recording", {})

    shared_state = RuntimeSharedState()
    shared_state.local_world = build_initial_local_world(settings)
    now_s = time.monotonic()
    shared_state.loop_heartbeats = {loop_name: now_s for loop_name in WATCHDOG_LOOP_NAMES}
    state_lock = asyncio.Lock()
    recorder = TelemetryRecorder(
        output_dir=Path("artifacts"),
        save_debug_frames=bool(recording_settings.get("save_debug_frames", True)),
        debug_frame_interval_s=float(recording_settings.get("debug_frame_interval_s", 1.0)),
    )
    detector = build_aruco_detector(settings)
    visual_servo = build_visual_servo_controller(control_settings, logger=logger)
    depth_analyzer = build_depth_analyzer(settings)
    obstacle_avoidance = build_obstacle_avoidance_controller(settings, logger)
    mission_settings = settings.get("mission", {})
    safety_limiter = CommandSafetyLimiter(
        CommandSafetyLimits(
            max_velocity_xy_m_s=float(control_settings.get("max_velocity_xy", 1.0)),
            max_velocity_z_m_s=float(control_settings.get("max_velocity_z", 0.5)),
            max_yaw_rate_deg_s=float(control_settings.get("yaw_rate_deg_s", 15.0)),
            min_command_duration_s=min(
                float(runtime.get("control_interval_s", 0.2)),
                float(control_settings.get("approach_command_duration_s", 0.25)),
            ),
            max_command_duration_s=max(
                float(mission_settings.get("search_step_duration_s", 0.35)),
                float(mission_settings.get("descend_step_duration_s", 0.25)),
                float(control_settings.get("servo_command_duration_s", 0.2)),
                float(depth_settings.get("avoidance_command_duration_s", 0.25)),
                float(control_settings.get("approach_command_duration_s", 0.25)),
            ),
        )
    )

    marker_id = int(aruco_settings.get("marker_id", 0))
    telemetry_interval_s = float(runtime.get("telemetry_interval_s", 0.5))
    frame_interval_s = float(runtime.get("frame_interval_s", 0.2))
    vision_interval_s = float(runtime.get("vision_interval_s", 0.2))
    mission_interval_s = float(runtime.get("mission_interval_s", 0.2))
    control_interval_s = float(runtime.get("control_interval_s", 0.2))
    run_duration_s = float(runtime.get("run_duration_s", 3.0))
    profiling_enabled = bool(runtime.get("profiling_enabled", False))
    profiling_warn_threshold_ms = float(runtime.get("profiling_warn_threshold_ms", 8.0))
    watchdog_enabled = bool(watchdog_settings.get("enabled", True))
    watchdog_interval_s = float(watchdog_settings.get("loop_interval_s", 0.1))
    watchdog_stale_after_s = float(watchdog_settings.get("stale_after_s", 1.0))
    local_ui_enabled = bool(local_ui_settings.get("enabled", True))
    local_ui_interval_s = float(local_ui_settings.get("interval_s", 0.03))
    local_ui_window_name = str(local_ui_settings.get("window_name", "drone_cv local"))

    tasks = [
        asyncio.create_task(local_telemetry_loop(recorder, shared_state, state_lock, logger, telemetry_interval_s), name="telemetry"),
        asyncio.create_task(local_frame_loop(recorder, settings, shared_state, state_lock, logger, frame_interval_s, profiling_enabled, profiling_warn_threshold_ms), name="frame"),
        asyncio.create_task(vision_loop(detector, depth_analyzer, recorder, marker_id, shared_state, state_lock, logger, vision_interval_s, profiling_enabled, profiling_warn_threshold_ms), name="vision"),
        asyncio.create_task(mission_loop(visual_servo, obstacle_avoidance, recorder, settings, shared_state, state_lock, logger, mission_interval_s), name="mission"),
        asyncio.create_task(local_control_loop(safety_limiter, shared_state, state_lock, logger, control_interval_s, profiling_enabled, profiling_warn_threshold_ms), name="control"),
    ]
    interaction_tasks = []
    if watchdog_enabled:
        tasks.append(
            asyncio.create_task(
                watchdog_loop(shared_state, state_lock, logger, watchdog_interval_s, watchdog_stale_after_s),
                name="watchdog",
            )
        )
    if local_ui_enabled:
        local_ui_task = asyncio.create_task(
            local_ui_loop(
                shared_state,
                state_lock,
                logger,
                local_ui_interval_s,
                local_ui_window_name,
            ),
            name="local_ui",
        )
        tasks.append(local_ui_task)
        interaction_tasks.append(local_ui_task)
    if terminal_controls_available():
        terminal_task = asyncio.create_task(
            terminal_control_loop(
                shared_state,
                state_lock,
                logger,
                0.05,
                "local",
            ),
            name="terminal_control",
        )
        tasks.append(terminal_task)
        interaction_tasks.append(terminal_task)

    try:
        if interaction_tasks:
            done, pending = await asyncio.wait(interaction_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, return_exceptions=True)
        else:
            await asyncio.sleep(run_duration_s)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if local_ui_enabled:
            import cv2

            cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        return run_smoke_mode(args.profile)
    if args.mode == "local":
        return run_local_mode(args.profile, run_local_runtime)
    return run_dev_mode(args.profile, run_runtime)


if __name__ == "__main__":
    raise SystemExit(main())
