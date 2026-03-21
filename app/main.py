from __future__ import annotations

import asyncio
import argparse
import copy
import math
from pathlib import Path
import sys
import time


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.bootstrap import bootstrap_app, build_runtime_components, validate_settings
else:
    from app.bootstrap import bootstrap_app, build_runtime_components, validate_settings


_LEFT_KEYS = {ord("a"), ord("A")}
_RIGHT_KEYS = {ord("d"), ord("D")}
_UP_KEYS = {ord("w"), ord("W")}
_DOWN_KEYS = {ord("s"), ord("S")}
_STRAFE_LEFT_KEYS = {ord("j"), ord("J")}
_STRAFE_RIGHT_KEYS = {ord("l"), ord("L")}
_STOP_KEYS = {32}


async def telemetry_loop(
    adapter,
    recorder,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    from telemetry.logger import format_snapshot

    while True:
        snapshot = await asyncio.to_thread(adapter.get_telemetry)
        await asyncio.to_thread(recorder.record_telemetry, snapshot)
        async with state_lock:
            shared_state.telemetry = snapshot
        logger.info("Telemetry loop | %s", format_snapshot(snapshot))
        await asyncio.sleep(interval_s)


async def frame_loop(
    frame_fetcher,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    from telemetry.models import RuntimeFrameState

    while True:
        frame_bundle = await asyncio.to_thread(frame_fetcher.fetch)
        async with state_lock:
            shared_state.frames = RuntimeFrameState(
                rgb_frame=frame_bundle.rgb_bgr.copy(),
                depth_frame=frame_bundle.depth_m.copy(),
                rgb_timestamp=frame_bundle.rgb_timestamp,
                depth_timestamp=frame_bundle.depth_timestamp,
            )
        logger.info(
            "Frame loop | rgb_ts=%s depth_ts=%s",
            frame_bundle.rgb_timestamp,
            frame_bundle.depth_timestamp,
        )
        await asyncio.sleep(interval_s)


async def vision_loop(
    detector,
    depth_analyzer,
    recorder,
    marker_id: int,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    while True:
        async with state_lock:
            frames = shared_state.frames
        if frames is not None and frames.rgb_frame is not None and frames.depth_frame is not None:
            detection = await asyncio.to_thread(
                detector.detect,
                frames.rgb_frame,
                marker_id,
            )
            depth_analysis = await asyncio.to_thread(depth_analyzer.analyze, frames.depth_frame)
            async with state_lock:
                shared_state.detection = detection
                shared_state.depth_analysis = depth_analysis
            await asyncio.to_thread(
                recorder.maybe_save_debug_frame,
                frames.rgb_frame,
                "vision",
                frames.rgb_timestamp,
            )
            logger.info(
                "Vision loop | detected=%s area=%.1f obstacle=%s",
                detection.detected,
                detection.area,
                depth_analysis.obstacle_detected,
            )
        await asyncio.sleep(interval_s)


async def mission_loop(
    visual_servo,
    obstacle_avoidance,
    recorder,
    settings,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    from mission.states import MissionState
    from telemetry.models import RuntimeControlCommand

    control_settings = settings.get("control", {})
    depth_settings = settings.get("depth", {})
    aruco_settings = settings.get("aruco", {})
    target_marker_area = float(control_settings.get("approach_target_marker_area", 12000.0))
    center_tolerance_px = float(control_settings.get("approach_center_tolerance_px", 30.0))
    command_duration_s = float(control_settings.get("approach_command_duration_s", 0.25))
    forward_speed_m_s = float(control_settings.get("approach_forward_speed_m_s", 0.4))
    target_marker_id = int(aruco_settings.get("marker_id", 0))
    last_event_key: tuple[str, str] | None = None

    while True:
        async with state_lock:
            detection = shared_state.detection
            depth_analysis = shared_state.depth_analysis
            frames = shared_state.frames
            telemetry = shared_state.telemetry
            current_state = shared_state.mission_state

        mission_state = current_state
        mission_detail = "waiting for sensor data"
        desired_command = RuntimeControlCommand(duration_s=command_duration_s)

        if telemetry is None or frames is None or detection is None or depth_analysis is None:
            mission_state = MissionState.IDLE
        elif not detection.detected or detection.marker_id != target_marker_id:
            mission_state = MissionState.SEARCH
            desired_command = RuntimeControlCommand(
                yaw_rate=float(settings.get("mission", {}).get("search_yaw_rate_deg_s", 8.0)),
                duration_s=float(settings.get("mission", {}).get("search_step_duration_s", 0.35)),
                source="mission_loop",
                reason="marker not visible; yaw scan",
            )
            mission_detail = desired_command.reason
        else:
            servo_command = visual_servo.compute_command(
                detection=detection,
                frame_width=frames.rgb_frame.shape[1],
                frame_height=frames.rgb_frame.shape[0],
            )
            aligned = (
                abs(servo_command.error_x_px) <= center_tolerance_px
                and abs(servo_command.error_y_px) <= center_tolerance_px
            )
            if detection.area >= float(settings.get("mission", {}).get("descend_marker_area_threshold", 18000.0)):
                mission_state = MissionState.DESCEND
                desired_command = RuntimeControlCommand(
                    vx=0.0,
                    vy=servo_command.vy,
                    vz=float(settings.get("mission", {}).get("descend_rate_m_s", 0.2)),
                    yaw_rate=servo_command.yaw_rate,
                    duration_s=float(settings.get("mission", {}).get("descend_step_duration_s", 0.25)),
                    source="mission_loop",
                    reason="marker large enough; controlled descend",
                )
                mission_detail = desired_command.reason
            elif not aligned:
                mission_state = MissionState.TRACK
                desired_command = RuntimeControlCommand(
                    vx=0.0,
                    vy=servo_command.vy,
                    vz=servo_command.vz,
                    yaw_rate=servo_command.yaw_rate,
                    duration_s=servo_command.duration_s,
                    source="mission_loop",
                    reason="marker off-center; align before approach",
                )
                mission_detail = desired_command.reason
            elif depth_analysis.obstacle_detected:
                avoidance_command = obstacle_avoidance.compute_command(depth_analysis)
                mission_state = MissionState.TRACK
                desired_command = RuntimeControlCommand(
                    vx=avoidance_command.vx,
                    vy=avoidance_command.vy,
                    vz=avoidance_command.vz,
                    yaw_rate=avoidance_command.yaw_rate,
                    duration_s=avoidance_command.duration_s,
                    source="mission_loop",
                    reason=f"front obstacle; avoid toward {avoidance_command.chosen_side}",
                )
                mission_detail = desired_command.reason
            elif detection.area < target_marker_area:
                mission_state = MissionState.TRACK
                desired_command = RuntimeControlCommand(
                    vx=forward_speed_m_s,
                    vy=servo_command.vy,
                    vz=servo_command.vz,
                    yaw_rate=servo_command.yaw_rate,
                    duration_s=command_duration_s,
                    source="mission_loop",
                    reason="marker centered and small; approach",
                )
                mission_detail = desired_command.reason
            elif telemetry.altitude_m <= float(settings.get("landing", {}).get("touchdown_altitude_m", 0.15)):
                mission_state = MissionState.LAND
                desired_command = RuntimeControlCommand(
                    source="mission_loop",
                    reason="touchdown altitude reached; await landing trigger",
                )
                mission_detail = desired_command.reason
            else:
                mission_state = MissionState.TRACK
                desired_command = RuntimeControlCommand(
                    vx=0.0,
                    vy=servo_command.vy,
                    vz=0.0,
                    yaw_rate=servo_command.yaw_rate,
                    duration_s=command_duration_s,
                    source="mission_loop",
                    reason="marker stable; hold alignment",
                )
                mission_detail = desired_command.reason

        async with state_lock:
            shared_state.mission_state = mission_state
            shared_state.mission_detail = mission_detail
            shared_state.desired_command = desired_command
            shared_state.control_applied = False
        event_key = (mission_state.value, mission_detail)
        if event_key != last_event_key:
            await asyncio.to_thread(
                recorder.record_event,
                "mission_state",
                mission_detail,
                {"state": mission_state.value},
            )
            last_event_key = event_key
        logger.info("Mission loop | state=%s detail=%s", mission_state.value, mission_detail)
        await asyncio.sleep(interval_s)


async def control_loop(
    adapter,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    from mission.states import MissionState
    from telemetry.models import RuntimeControlCommand

    while True:
        async with state_lock:
            command = shared_state.desired_command
            mission_state = shared_state.mission_state
            already_applied = shared_state.control_applied
            manual_yaw_rate_deg_s = shared_state.local_manual_yaw_rate_deg_s
            manual_vz_m_s = shared_state.local_manual_vz_m_s
            manual_vy_m_s = shared_state.local_manual_vy_m_s
            manual_override_until_s = shared_state.local_manual_override_until_s
            manual_status = shared_state.local_manual_status
            spin_paused = shared_state.local_spin_paused

        if not already_applied:
            manual_active = time.monotonic() < manual_override_until_s
            effective_command = command

            if manual_active:
                manual_vx = 0.0 if manual_status in {"strafe_left", "strafe_right"} else command.vx
                effective_command = RuntimeControlCommand(
                    vx=manual_vx,
                    vy=manual_vy_m_s if manual_status in {"strafe_left", "strafe_right"} else command.vy,
                    vz=manual_vz_m_s if manual_status in {"up", "down", "hold"} else command.vz,
                    yaw_rate=manual_yaw_rate_deg_s,
                    duration_s=max(command.duration_s, interval_s),
                    source=command.source,
                    reason=f"{command.reason} | manual yaw {manual_status}",
                )
            elif spin_paused:
                effective_command = RuntimeControlCommand(
                    vx=command.vx,
                    vy=command.vy,
                    vz=command.vz,
                    yaw_rate=0.0,
                    duration_s=max(command.duration_s, interval_s),
                    source=command.source,
                    reason=f"{command.reason} | auto spin paused",
                )

            if mission_state == MissionState.LAND and not manual_active:
                await asyncio.to_thread(adapter.land)
            elif mission_state == MissionState.IDLE and not manual_active:
                await asyncio.to_thread(adapter.hover)
            else:
                await asyncio.to_thread(
                    adapter.move_by_velocity_body,
                    effective_command.vx,
                    effective_command.vy,
                    effective_command.vz,
                    max(effective_command.duration_s, interval_s),
                    effective_command.yaw_rate,
                )
            async with state_lock:
                shared_state.control_applied = True
            logger.info(
                "Control loop | state=%s manual=%s vx=%.3f vy=%.3f vz=%.3f yaw_rate=%.3f command=%s",
                mission_state.value,
                manual_active,
                effective_command.vx,
                effective_command.vy,
                effective_command.vz,
                effective_command.yaw_rate,
                effective_command.reason,
            )
        await asyncio.sleep(interval_s)


async def dev_ui_loop(
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
    window_name: str,
) -> None:
    import cv2
    import numpy as np

    while True:
        async with state_lock:
            frames = shared_state.frames
            detection = shared_state.detection
            depth_analysis = shared_state.depth_analysis
            mission_state = shared_state.mission_state
            mission_detail = shared_state.mission_detail
            command = shared_state.desired_command
            telemetry = shared_state.telemetry
            manual_override_until_s = shared_state.local_manual_override_until_s
            spin_paused = shared_state.local_spin_paused

        frame = None
        if frames is not None and frames.rgb_frame is not None:
            frame = frames.rgb_frame.copy()
        if frame is None:
            frame = np.full((480, 640, 3), 24, dtype=np.uint8)

        _draw_dev_camera_overlay(
            frame_bgr=frame,
            mission_state=mission_state.value,
            mission_detail=mission_detail,
            command_reason=command.reason,
            marker_detected=detection.detected if detection is not None else False,
            obstacle_detected=depth_analysis.obstacle_detected if depth_analysis is not None else False,
            altitude_m=telemetry.altitude_m if telemetry is not None else 0.0,
            steering_mode=(
                "manual"
                if time.monotonic() < manual_override_until_s
                else ("paused" if spin_paused else "auto")
            ),
        )
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in {27, ord("q"), ord("Q")}:
            logger.info("Dev UI loop requested shutdown")
            return
        if key != 255:
            await _apply_manual_key_input(shared_state, state_lock, key)
        await asyncio.sleep(interval_s)


async def run_runtime() -> int:
    context = bootstrap_app()
    settings = context["settings"]
    logger = context["logger"]
    app_name = settings["app"]["name"]
    runtime = settings.get("runtime", {})
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
    recorder = components["recorder"]
    shared_state = components["shared_state"]
    state_lock = components["state_lock"]

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
    dev_ui_enabled = bool(dev_ui_settings.get("enabled", True))
    dev_ui_interval_s = float(dev_ui_settings.get("interval_s", 0.03))
    dev_ui_window_name = str(dev_ui_settings.get("window_name", "drone_cv dev"))

    tasks = [
        asyncio.create_task(telemetry_loop(adapter, recorder, shared_state, state_lock, logger, telemetry_interval_s), name="telemetry"),
        asyncio.create_task(frame_loop(frame_fetcher, shared_state, state_lock, logger, frame_interval_s), name="frame"),
        asyncio.create_task(vision_loop(detector, depth_analyzer, recorder, marker_id, shared_state, state_lock, logger, vision_interval_s), name="vision"),
        asyncio.create_task(mission_loop(visual_servo, obstacle_avoidance, recorder, settings, shared_state, state_lock, logger, mission_interval_s), name="mission"),
        asyncio.create_task(control_loop(adapter, shared_state, state_lock, logger, control_interval_s), name="control"),
    ]
    if dev_ui_enabled:
        tasks.append(
            asyncio.create_task(
                dev_ui_loop(
                    shared_state,
                    state_lock,
                    logger,
                    dev_ui_interval_s,
                    dev_ui_window_name,
                ),
                name="dev_ui",
            )
        )

    try:
        if dev_ui_enabled:
            await next(task for task in tasks if task.get_name() == "dev_ui")
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


async def local_telemetry_loop(
    recorder,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    from telemetry.logger import format_snapshot
    from telemetry.models import Quaternion, TelemetrySnapshot, Vector3

    while True:
        async with state_lock:
            world = shared_state.local_world
        if world is None:
            await asyncio.sleep(interval_s)
            continue

        yaw_rad = math.radians(world.yaw_deg)
        snapshot = TelemetrySnapshot(
            timestamp=int(time.time() * 1_000_000),
            position_m=Vector3(x=world.drone_x_m, y=world.drone_y_m, z=-world.altitude_m),
            velocity_m_s=Vector3(
                x=world.velocity_x_m_s,
                y=world.velocity_y_m_s,
                z=-world.velocity_z_m_s,
            ),
            altitude_m=world.altitude_m,
            orientation=Quaternion(
                x=0.0,
                y=0.0,
                z=math.sin(yaw_rad / 2.0),
                w=math.cos(yaw_rad / 2.0),
            ),
            speed_m_s=math.sqrt(
                world.velocity_x_m_s ** 2
                + world.velocity_y_m_s ** 2
                + world.velocity_z_m_s ** 2
            ),
        )
        await asyncio.to_thread(recorder.record_telemetry, snapshot)
        async with state_lock:
            shared_state.telemetry = snapshot
        logger.info("Local telemetry | %s", format_snapshot(snapshot))
        await asyncio.sleep(interval_s)


async def local_frame_loop(
    recorder,
    settings,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    from telemetry.models import RuntimeFrameState

    while True:
        async with state_lock:
            world = shared_state.local_world
        if world is None:
            await asyncio.sleep(interval_s)
            continue

        frame, depth, marker_visible, marker_distance_m, obstacle_distance_m, obstacle_side = (
            _render_local_world_frame(settings, world)
        )
        frame_world_snapshot = copy.deepcopy(world)
        frame_world_snapshot.marker_visible = marker_visible
        frame_world_snapshot.marker_distance_m = marker_distance_m
        frame_world_snapshot.obstacle_distance_m = obstacle_distance_m
        frame_world_snapshot.obstacle_side = obstacle_side

        timestamp = int(time.time() * 1_000_000)
        async with state_lock:
            shared_state.frames = RuntimeFrameState(
                rgb_frame=frame,
                depth_frame=depth,
                rgb_timestamp=timestamp,
                depth_timestamp=timestamp,
                local_world_snapshot=frame_world_snapshot,
            )
            shared_state.local_world.marker_visible = marker_visible
            shared_state.local_world.marker_distance_m = marker_distance_m
            shared_state.local_world.obstacle_distance_m = obstacle_distance_m
            shared_state.local_world.obstacle_side = obstacle_side
        await asyncio.to_thread(recorder.maybe_save_debug_frame, frame, "local", timestamp)
        logger.info(
            "Local frame | marker_visible=%s marker_distance=%.2f obstacle=%s",
            marker_visible,
            marker_distance_m,
            obstacle_side,
        )
        await asyncio.sleep(interval_s)


async def local_control_loop(
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    while True:
        async with state_lock:
            command = shared_state.desired_command
            mission_state = shared_state.mission_state
            already_applied = shared_state.control_applied
            world = shared_state.local_world
            manual_yaw_rate_deg_s = shared_state.local_manual_yaw_rate_deg_s
            manual_vz_m_s = shared_state.local_manual_vz_m_s
            manual_vy_m_s = shared_state.local_manual_vy_m_s
            manual_override_until_s = shared_state.local_manual_override_until_s
            manual_status = shared_state.local_manual_status
            spin_paused = shared_state.local_spin_paused
        if not already_applied:
            effective_command = command
            if time.monotonic() < manual_override_until_s:
                from telemetry.models import RuntimeControlCommand

                manual_vx = 0.0 if manual_status in {"strafe_left", "strafe_right"} else command.vx
                effective_command = RuntimeControlCommand(
                    vx=manual_vx,
                    vy=manual_vy_m_s if manual_status in {"strafe_left", "strafe_right"} else command.vy,
                    vz=manual_vz_m_s if manual_status in {"up", "down", "hold"} else command.vz,
                    yaw_rate=manual_yaw_rate_deg_s,
                    duration_s=command.duration_s,
                    source=command.source,
                    reason=f"{command.reason} | manual yaw {manual_status}",
                )
            elif spin_paused:
                from telemetry.models import RuntimeControlCommand

                effective_command = RuntimeControlCommand(
                    vx=command.vx,
                    vy=command.vy,
                    vz=command.vz,
                    yaw_rate=0.0,
                    duration_s=command.duration_s,
                    source=command.source,
                    reason=f"{command.reason} | auto spin paused",
                )
            if world is not None:
                _apply_local_command(world, effective_command, max(effective_command.duration_s, interval_s))
            logger.info(
                "Local control | state=%s x=%.2f y=%.2f alt=%.2f yaw=%.1f vx=%.3f vy=%.3f vz=%.3f yaw_rate=%.3f reason=%s",
                mission_state.value,
                world.drone_x_m if world is not None else 0.0,
                world.drone_y_m if world is not None else 0.0,
                world.altitude_m if world is not None else 0.0,
                world.yaw_deg if world is not None else 0.0,
                effective_command.vx,
                effective_command.vy,
                effective_command.vz,
                effective_command.yaw_rate,
                effective_command.reason,
            )
            async with state_lock:
                shared_state.control_applied = True
        await asyncio.sleep(interval_s)


async def local_ui_loop(
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
    window_name: str,
) -> None:
    import cv2

    while True:
        async with state_lock:
            frames = shared_state.frames
            detection = shared_state.detection
            depth_analysis = shared_state.depth_analysis
            mission_state = shared_state.mission_state
            mission_detail = shared_state.mission_detail
            command = shared_state.desired_command
            telemetry = shared_state.telemetry
            world = shared_state.local_world
            manual_override_until_s = shared_state.local_manual_override_until_s
            spin_paused = shared_state.local_spin_paused

        if frames is not None and frames.rgb_frame is not None and world is not None:
            view_world = frames.local_world_snapshot if frames.local_world_snapshot is not None else world
            frame = frames.rgb_frame.copy()
            _draw_local_camera_overlay(
                frame,
                mission_state.value,
                mission_detail,
                command.reason,
                detection.detected if detection is not None else False,
                depth_analysis.obstacle_detected if depth_analysis is not None else False,
            )
            canvas = _build_local_ui_canvas(
                frame,
                view_world,
                view_world.altitude_m,
                mission_state.value,
                command,
                "manual" if time.monotonic() < manual_override_until_s else ("paused" if spin_paused else "auto"),
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                logger.info("Local UI loop requested shutdown")
                return
            if key != 255:
                await _apply_manual_key_input(shared_state, state_lock, key)
        await asyncio.sleep(interval_s)


async def _apply_manual_key_input(shared_state, state_lock: asyncio.Lock, key: int) -> None:
    if key in _LEFT_KEYS:
        async with state_lock:
            shared_state.local_manual_yaw_rate_deg_s = -35.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + 0.35
            shared_state.local_manual_status = "left"
    elif key in _RIGHT_KEYS:
        async with state_lock:
            shared_state.local_manual_yaw_rate_deg_s = 35.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + 0.35
            shared_state.local_manual_status = "right"
    elif key in _UP_KEYS:
        async with state_lock:
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = -0.45
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + 0.35
            shared_state.local_manual_status = "up"
    elif key in _DOWN_KEYS:
        async with state_lock:
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.45
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + 0.35
            shared_state.local_manual_status = "down"
    elif key in _STRAFE_LEFT_KEYS:
        async with state_lock:
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = -0.45
            shared_state.local_manual_override_until_s = time.monotonic() + 0.35
            shared_state.local_manual_status = "strafe_left"
    elif key in _STRAFE_RIGHT_KEYS:
        async with state_lock:
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.45
            shared_state.local_manual_override_until_s = time.monotonic() + 0.35
            shared_state.local_manual_status = "strafe_right"
    elif key in _STOP_KEYS:
        async with state_lock:
            shared_state.local_spin_paused = not shared_state.local_spin_paused
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = 0.0
            shared_state.local_manual_status = "hold" if shared_state.local_spin_paused else "auto"


async def run_local_runtime() -> int:
    context = bootstrap_app()
    settings = context["settings"]
    logger = context["logger"]
    runtime = settings.get("runtime", {})
    local_ui_settings = settings.get("local_ui", {})
    logger.info("Starting local asyncio runtime without AirSim")
    print(f"{settings['app']['name']} startup complete")

    from control.obstacle_avoidance import ObstacleAvoidanceController
    from control.visual_servo import VisualServoConfig, VisualServoController
    from telemetry.models import RuntimeSharedState
    from telemetry.recorder import TelemetryRecorder
    from vision.aruco_detector import ArucoDetector
    from vision.depth_analyzer import DepthAnalyzer

    control_settings = settings.get("control", {})
    depth_settings = settings.get("depth", {})
    aruco_settings = settings.get("aruco", {})
    recording_settings = settings.get("recording", {})

    shared_state = RuntimeSharedState()
    shared_state.local_world = _build_initial_local_world(settings)
    state_lock = asyncio.Lock()
    recorder = TelemetryRecorder(
        output_dir=Path("artifacts"),
        save_debug_frames=bool(recording_settings.get("save_debug_frames", True)),
        debug_frame_interval_s=float(recording_settings.get("debug_frame_interval_s", 1.0)),
    )
    detector = ArucoDetector(
        dictionary_name=str(aruco_settings.get("dictionary", "DICT_4X4_50"))
    )
    visual_servo = VisualServoController(
        config=VisualServoConfig(
            command_duration_s=float(control_settings.get("servo_command_duration_s", 0.2)),
            max_lateral_velocity_m_s=float(control_settings.get("servo_max_lateral_velocity_m_s", 0.5)),
            max_vertical_velocity_m_s=float(control_settings.get("servo_max_vertical_velocity_m_s", 0.4)),
            max_yaw_rate_deg_s=float(control_settings.get("servo_max_yaw_rate_deg_s", 10.0)),
            yaw_error_deadband_px=float(control_settings.get("servo_yaw_error_deadband_px", 10.0)),
            lateral_kp=float(control_settings.get("servo_lateral_kp", 0.4)),
            lateral_ki=float(control_settings.get("servo_lateral_ki", 0.0)),
            lateral_kd=float(control_settings.get("servo_lateral_kd", 0.05)),
            vertical_kp=float(control_settings.get("servo_vertical_kp", 0.35)),
            vertical_ki=float(control_settings.get("servo_vertical_ki", 0.0)),
            vertical_kd=float(control_settings.get("servo_vertical_kd", 0.04)),
            yaw_kp=float(control_settings.get("servo_yaw_kp", 5.0)),
            yaw_ki=float(control_settings.get("servo_yaw_ki", 0.0)),
            yaw_kd=float(control_settings.get("servo_yaw_kd", 0.2)),
        ),
        logger=logger,
    )
    depth_analyzer = DepthAnalyzer(
        obstacle_distance_m=float(depth_settings.get("obstacle_distance_m", 2.0)),
        min_valid_depth_m=float(depth_settings.get("min_valid_depth_m", 0.2)),
        max_valid_depth_m=float(depth_settings.get("max_valid_depth_m", 20.0)),
    )
    obstacle_avoidance = ObstacleAvoidanceController(
        avoidance_speed_m_s=float(depth_settings.get("avoidance_speed_m_s", 0.5)),
        yaw_rate_deg_s=float(depth_settings.get("avoidance_yaw_rate_deg_s", 12.0)),
        command_duration_s=float(depth_settings.get("avoidance_command_duration_s", 0.25)),
        logger=logger,
    )

    marker_id = int(aruco_settings.get("marker_id", 0))
    telemetry_interval_s = float(runtime.get("telemetry_interval_s", 0.5))
    frame_interval_s = float(runtime.get("frame_interval_s", 0.2))
    vision_interval_s = float(runtime.get("vision_interval_s", 0.2))
    mission_interval_s = float(runtime.get("mission_interval_s", 0.2))
    control_interval_s = float(runtime.get("control_interval_s", 0.2))
    run_duration_s = float(runtime.get("run_duration_s", 3.0))
    local_ui_enabled = bool(local_ui_settings.get("enabled", True))
    local_ui_interval_s = float(local_ui_settings.get("interval_s", 0.03))
    local_ui_window_name = str(local_ui_settings.get("window_name", "drone_cv local"))

    tasks = [
        asyncio.create_task(local_telemetry_loop(recorder, shared_state, state_lock, logger, telemetry_interval_s), name="telemetry"),
        asyncio.create_task(local_frame_loop(recorder, settings, shared_state, state_lock, logger, frame_interval_s), name="frame"),
        asyncio.create_task(vision_loop(detector, depth_analyzer, recorder, marker_id, shared_state, state_lock, logger, vision_interval_s), name="vision"),
        asyncio.create_task(mission_loop(visual_servo, obstacle_avoidance, recorder, settings, shared_state, state_lock, logger, mission_interval_s), name="mission"),
        asyncio.create_task(local_control_loop(shared_state, state_lock, logger, control_interval_s), name="control"),
    ]
    if local_ui_enabled:
        tasks.append(
            asyncio.create_task(
                local_ui_loop(
                    shared_state,
                    state_lock,
                    logger,
                    local_ui_interval_s,
                    local_ui_window_name,
                ),
                name="local_ui",
            )
        )

    try:
        if local_ui_enabled:
            await next(task for task in tasks if task.get_name() == "local_ui")
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


def _apply_local_command(world, command, dt: float) -> None:
    if "manual yaw strafe_left" in command.reason:
        world.drone_y_m = max(-(world.height_m / 2.0), world.drone_y_m - (0.45 * dt))
        world.velocity_x_m_s = 0.0
        world.velocity_y_m_s = -0.45
        world.velocity_z_m_s = 0.0
        world.yaw_rate_deg_s = 0.0
        return

    if "manual yaw strafe_right" in command.reason:
        world.drone_y_m = min(world.height_m / 2.0, world.drone_y_m + (0.45 * dt))
        world.velocity_x_m_s = 0.0
        world.velocity_y_m_s = 0.45
        world.velocity_z_m_s = 0.0
        world.yaw_rate_deg_s = 0.0
        return

    yaw_rad = math.radians(world.yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    world_velocity_x = (command.vx * cos_yaw) - (command.vy * sin_yaw)
    world_velocity_y = (command.vx * sin_yaw) + (command.vy * cos_yaw)
    world.drone_x_m = min(world.width_m, max(0.0, world.drone_x_m + (world_velocity_x * dt)))
    world.drone_y_m = min(
        world.height_m / 2.0,
        max(-(world.height_m / 2.0), world.drone_y_m + (world_velocity_y * dt)),
    )
    world.altitude_m = min(
        world.max_altitude_m,
        max(world.min_altitude_m, world.altitude_m - (command.vz * dt)),
    )
    world.yaw_deg = ((world.yaw_deg + (command.yaw_rate * dt) + 180.0) % 360.0) - 180.0
    world.velocity_x_m_s = world_velocity_x
    world.velocity_y_m_s = world_velocity_y
    world.velocity_z_m_s = command.vz
    world.yaw_rate_deg_s = command.yaw_rate


def _build_initial_local_world(settings):
    from telemetry.models import LocalObstacleState, LocalWorldState

    local_world_settings = settings.get("local_world", {})
    return LocalWorldState(
        width_m=float(local_world_settings.get("width_m", 10.0)),
        height_m=float(local_world_settings.get("height_m", 8.0)),
        min_altitude_m=float(local_world_settings.get("min_altitude_m", 0.15)),
        max_altitude_m=float(local_world_settings.get("max_altitude_m", 3.0)),
        drone_x_m=0.0,
        drone_y_m=0.0,
        altitude_m=float(local_world_settings.get("desired_altitude_m", 1.2)),
        yaw_deg=0.0,
        marker_x_m=float(local_world_settings.get("marker_x_m", 6.0)),
        marker_y_m=float(local_world_settings.get("marker_y_m", 0.0)),
        obstacle=LocalObstacleState(
            x_m=float(local_world_settings.get("obstacle_x_m", 3.2)),
            y_m=float(local_world_settings.get("obstacle_y_m", 0.6)),
            radius_m=float(local_world_settings.get("obstacle_radius_m", 0.55)),
        ),
    )


def _render_local_world_frame(settings, world):
    import cv2
    import numpy as np

    aruco_settings = settings.get("aruco", {})
    camera_settings = settings.get("camera", {})
    local_world_settings = settings.get("local_world", {})
    dictionary_name = str(aruco_settings.get("dictionary", "DICT_4X4_50"))
    marker_id = int(aruco_settings.get("marker_id", 0))
    width = int(camera_settings.get("image_width", 640))
    height = int(camera_settings.get("image_height", 480))
    field_of_view_deg = float(local_world_settings.get("camera_fov_deg", 70.0))
    marker_scale_px_m = float(local_world_settings.get("marker_scale_px_m", 260.0))
    desired_altitude_m = float(local_world_settings.get("desired_altitude_m", 1.2))

    frame = np.full((height, width, 3), 238, dtype=np.uint8)
    depth = np.full((height, width), 8.0, dtype=np.float32)
    cv2.rectangle(frame, (0, 0), (width, height // 2), (225, 235, 245), thickness=-1)
    cv2.rectangle(frame, (0, height // 2), (width, height), (212, 228, 206), thickness=-1)

    half_fov_rad = math.radians(field_of_view_deg / 2.0)
    focal_px = (width / 2.0) / math.tan(half_fov_rad)
    yaw_rad = math.radians(world.yaw_deg)

    marker_body_x, marker_body_y = _world_to_body(
        world.marker_x_m - world.drone_x_m,
        world.marker_y_m - world.drone_y_m,
        yaw_rad,
    )
    marker_distance_m = math.sqrt(marker_body_x ** 2 + marker_body_y ** 2 + world.altitude_m ** 2)
    marker_visible = (
        marker_body_x > 0.35
        and abs(marker_body_y / max(marker_body_x, 0.35)) <= math.tan(half_fov_rad)
    )

    if marker_visible:
        marker_size_px = int(
            max(40.0, min(220.0, marker_scale_px_m / max(marker_distance_m, 0.8)))
        )
        center_x = int(round((width / 2.0) + (marker_body_y / marker_body_x) * focal_px))
        vertical_ratio = (world.altitude_m - desired_altitude_m) / max(world.max_altitude_m, 0.1)
        center_y = int(round((height / 2.0) + (vertical_ratio * height * 0.7)))
        top = center_y - (marker_size_px // 2)
        left = center_x - (marker_size_px // 2)
        if top >= 0 and left >= 0 and top + marker_size_px < height and left + marker_size_px < width:
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
            marker_bgr = np.full((marker_size_px, marker_size_px, 3), 255, dtype=np.uint8)
            marker_bgr[marker == 0] = (70, 170, 70)
            frame[top : top + marker_size_px, left : left + marker_size_px] = marker_bgr

    obstacle_distance_m = None
    obstacle_side = "none"
    obstacle_body_x, obstacle_body_y = _world_to_body(
        world.obstacle.x_m - world.drone_x_m,
        world.obstacle.y_m - world.drone_y_m,
        yaw_rad,
    )
    obstacle_distance_center_m = math.hypot(obstacle_body_x, obstacle_body_y)
    obstacle_visible = (
        obstacle_body_x > 0.2
        and abs(obstacle_body_y / max(obstacle_body_x, 0.2)) <= math.tan(half_fov_rad)
    )
    if obstacle_visible:
        obstacle_side = "right" if obstacle_body_y >= 0.0 else "left"
        obstacle_distance_m = obstacle_distance_center_m
        angular_half_width = math.atan2(world.obstacle.radius_m, max(obstacle_body_x, 0.1))
        obstacle_center_x = int(round((width / 2.0) + (obstacle_body_y / obstacle_body_x) * focal_px))
        obstacle_half_width_px = max(14, int(round(math.tan(angular_half_width) * focal_px)))
        obstacle_top = int(height * 0.22)
        obstacle_bottom = int(height * 0.92)
        left = max(0, obstacle_center_x - obstacle_half_width_px)
        right = min(width, obstacle_center_x + obstacle_half_width_px)
        frame[obstacle_top:obstacle_bottom, left:right] = (70, 95, 155)
        depth[obstacle_top:obstacle_bottom, left:right] = min(depth[obstacle_top:obstacle_bottom, left:right].min(), obstacle_body_x)

    return frame, depth, marker_visible, marker_distance_m, obstacle_distance_m, obstacle_side


def _world_to_body(delta_x_m: float, delta_y_m: float, yaw_rad: float) -> tuple[float, float]:
    body_x = (math.cos(yaw_rad) * delta_x_m) + (math.sin(yaw_rad) * delta_y_m)
    body_y = (-math.sin(yaw_rad) * delta_x_m) + (math.cos(yaw_rad) * delta_y_m)
    return body_x, body_y


def _draw_local_camera_overlay(
    frame_bgr,
    mission_state: str,
    mission_detail: str,
    command_reason: str,
    marker_detected: bool,
    obstacle_detected: bool,
) -> None:
    import cv2

    height, width = frame_bgr.shape[:2]
    cv2.line(frame_bgr, (width // 2, 0), (width // 2, height), (40, 185, 255), 1, cv2.LINE_AA)
    cv2.line(frame_bgr, (0, height // 2), (width, height // 2), (40, 185, 255), 1, cv2.LINE_AA)
    cv2.circle(frame_bgr, (width // 2, height // 2), 12, (40, 185, 255), 1, cv2.LINE_AA)

    lines = [
        f"State: {mission_state}",
        f"Marker: {'yes' if marker_detected else 'no'}",
        f"Obstacle: {'yes' if obstacle_detected else 'no'}",
        f"Mission: {mission_detail}",
        f"Command: {command_reason}",
        "A/D: yaw | W/S: altitude | J/L: strafe",
        "Space: toggle auto spin",
        "Press Q or Esc to close",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28


def _draw_dev_camera_overlay(
    frame_bgr,
    mission_state: str,
    mission_detail: str,
    command_reason: str,
    marker_detected: bool,
    obstacle_detected: bool,
    altitude_m: float,
    steering_mode: str,
) -> None:
    import cv2

    height, width = frame_bgr.shape[:2]
    cv2.line(frame_bgr, (width // 2, 0), (width // 2, height), (40, 185, 255), 1, cv2.LINE_AA)
    cv2.line(frame_bgr, (0, height // 2), (width, height // 2), (40, 185, 255), 1, cv2.LINE_AA)
    cv2.circle(frame_bgr, (width // 2, height // 2), 12, (40, 185, 255), 1, cv2.LINE_AA)

    lines = [
        "AirSim DEV control",
        f"mode={steering_mode} alt={altitude_m:.2f}m",
        f"state={mission_state} marker={'yes' if marker_detected else 'no'} obstacle={'yes' if obstacle_detected else 'no'}",
        f"mission={mission_detail}",
        f"command={command_reason}",
        "A/D: yaw | W/S: altitude | J/L: strafe",
        "Space: pause auto yaw | Q/Esc: exit",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 27


def _build_local_ui_canvas(frame_bgr, world, altitude_m: float, mission_state: str, command, steering_mode: str) -> object:
    import cv2
    import numpy as np

    panel_width = frame_bgr.shape[1]
    canvas = np.full((frame_bgr.shape[0], frame_bgr.shape[1] + panel_width, 3), 248, dtype=np.uint8)
    canvas[:, : frame_bgr.shape[1]] = frame_bgr
    map_panel = canvas[:, frame_bgr.shape[1] :]
    _draw_local_world_panel(map_panel, world, altitude_m, mission_state, command, steering_mode)
    return canvas


def _draw_local_world_panel(panel_bgr, world, altitude_m: float, mission_state: str, command, steering_mode: str) -> None:
    import cv2

    height, width = panel_bgr.shape[:2]
    panel_bgr[:] = (246, 246, 242)

    world_width_m = world.width_m
    world_height_m = world.height_m
    margin = 36
    map_left = margin
    map_top = 70
    map_width = width - (margin * 2)
    map_height = height - 130
    cv2.rectangle(panel_bgr, (map_left, map_top), (map_left + map_width, map_top + map_height), (45, 45, 45), 2)

    def to_panel_point(x_m: float, y_m: float) -> tuple[int, int]:
        px = map_left + int(round((x_m / world_width_m) * map_width))
        normalized_y = (y_m + (world_height_m / 2.0)) / world_height_m
        py = map_top + int(round(normalized_y * map_height))
        return px, py

    marker_point = to_panel_point(world.marker_x_m, world.marker_y_m)
    obstacle_point = to_panel_point(world.obstacle.x_m, world.obstacle.y_m)
    drone_point = to_panel_point(world.drone_x_m, world.drone_y_m)

    cv2.circle(panel_bgr, marker_point, 12, (20, 120, 40), thickness=2)
    cv2.putText(panel_bgr, "marker", (marker_point[0] + 14, marker_point[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 120, 40), 1, cv2.LINE_AA)

    obstacle_radius_px = max(8, int(round((world.obstacle.radius_m / world_width_m) * map_width)))
    cv2.circle(panel_bgr, obstacle_point, obstacle_radius_px, (60, 90, 170), thickness=2)
    cv2.putText(panel_bgr, "obstacle", (obstacle_point[0] + 12, obstacle_point[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 90, 170), 1, cv2.LINE_AA)

    cv2.circle(panel_bgr, drone_point, 10, (40, 40, 220), thickness=-1)
    heading_end = (
        int(round(drone_point[0] + (math.cos(math.radians(world.yaw_deg)) * 24))),
        int(round(drone_point[1] + (math.sin(math.radians(world.yaw_deg)) * 24))),
    )
    cv2.arrowedLine(panel_bgr, drone_point, heading_end, (40, 40, 220), 2, cv2.LINE_AA, tipLength=0.25)

    lines = [
        "Local 2D World",
        f"state={mission_state}",
        f"steering={steering_mode}",
        f"pos=({world.drone_x_m:.2f}, {world.drone_y_m:.2f})m",
        f"alt={altitude_m:.2f}m yaw={world.yaw_deg:.1f}deg",
        f"marker_visible={'yes' if world.marker_visible else 'no'} dist={world.marker_distance_m:.2f}m",
        f"obstacle={world.obstacle_side} dist={world.obstacle_distance_m or 0.0:.2f}m",
        f"cmd vx={command.vx:.2f} vy={command.vy:.2f} vz={command.vz:.2f} yaw={command.yaw_rate:.2f}",
    ]
    y = 28
    for index, line in enumerate(lines):
        scale = 0.72 if index == 0 else 0.52
        thickness = 2 if index == 0 else 1
        cv2.putText(panel_bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), thickness, cv2.LINE_AA)
        y += 24 if index == 0 else 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local runtime for drone_cv")
    parser.add_argument(
        "--mode",
        choices=("dev", "smoke", "local"),
        default="dev",
        help="Run dev, smoke, or local mode",
    )
    return parser.parse_args()


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
    print("Check config/settings.yaml and install dependencies with: python3 -m pip install -r requirements.txt")
    return 1


def run_smoke_mode() -> int:
    context = bootstrap_app()
    settings = context["settings"]
    logger = context["logger"]
    errors = validate_settings(settings)
    if errors:
        return fail_with_actionable_error("; ".join(errors))

    print_startup_info(settings, mode="smoke")
    logger.info("Smoke mode starting")

    try:
        from adapters.airsim_client import AirSimClientAdapter, AirSimConnectionConfig
    except ModuleNotFoundError as exc:
        return fail_with_actionable_error(f"Missing dependency: {exc.name}")

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


def run_dev_mode() -> int:
    context = bootstrap_app()
    settings = context["settings"]
    logger = context["logger"]
    errors = validate_settings(settings)
    if errors:
        return fail_with_actionable_error("; ".join(errors))

    print_startup_info(settings, mode="dev")
    logger.info("Dev mode validated config successfully")

    try:
        return asyncio.run(run_runtime())
    except ModuleNotFoundError as exc:
        return fail_with_actionable_error(f"Missing dependency: {exc.name}")
    except Exception as exc:  # pragma: no cover - depends on local AirSim/runtime
        return fail_with_actionable_error(str(exc))


def run_local_mode() -> int:
    context = bootstrap_app()
    settings = context["settings"]
    logger = context["logger"]
    errors = validate_settings(settings)
    if errors:
        return fail_with_actionable_error("; ".join(errors))

    print_startup_info(settings, mode="local")
    logger.info("Local mode validated config successfully")

    try:
        return asyncio.run(run_local_runtime())
    except ModuleNotFoundError as exc:
        return fail_with_actionable_error(f"Missing dependency: {exc.name}")
    except Exception as exc:
        return fail_with_actionable_error(str(exc))


def main() -> int:
    args = parse_args()
    if args.mode == "smoke":
        return run_smoke_mode()
    if args.mode == "local":
        return run_local_mode()
    return run_dev_mode()


if __name__ == "__main__":
    raise SystemExit(main())
