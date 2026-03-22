from __future__ import annotations

import asyncio
import time


WATCHDOG_LOOP_NAMES = ("telemetry", "frame", "vision", "mission", "control")


def build_manual_override_command(
    manual_status: str,
    manual_vx_m_s: float,
    manual_vy_m_s: float,
    manual_vz_m_s: float,
    manual_yaw_rate_deg_s: float,
    duration_s: float,
):
    from telemetry.models import RuntimeControlCommand

    vx = 0.0
    vy = 0.0
    vz = 0.0
    yaw_rate = 0.0

    if manual_status in {"forward", "backward"}:
        vx = manual_vx_m_s
    elif manual_status in {"strafe_left", "strafe_right"}:
        vy = manual_vy_m_s
    elif manual_status in {"up", "down"}:
        vz = manual_vz_m_s
    elif manual_status in {"left", "right"}:
        yaw_rate = manual_yaw_rate_deg_s

    return RuntimeControlCommand(
        vx=vx,
        vy=vy,
        vz=vz,
        yaw_rate=yaw_rate,
        duration_s=duration_s,
        source="manual",
        reason=f"manual {manual_status}",
    )


async def mark_loop_heartbeat(shared_state, state_lock: asyncio.Lock, loop_name: str) -> None:
    async with state_lock:
        shared_state.loop_heartbeats[loop_name] = time.monotonic()


def find_stale_loop_names(
    heartbeats: dict[str, float],
    tracked_loop_names: tuple[str, ...],
    now_s: float,
    stale_after_s: float,
) -> list[str]:
    stale_loop_names: list[str] = []
    for loop_name in tracked_loop_names:
        last_heartbeat_s = heartbeats.get(loop_name)
        if last_heartbeat_s is None or (now_s - last_heartbeat_s) > stale_after_s:
            stale_loop_names.append(loop_name)
    return stale_loop_names


def find_stale_sensor_names(
    shared_state,
    now_s: float,
    freshness_settings: dict[str, float],
) -> list[str]:
    sensor_checks = (
        (
            "telemetry",
            shared_state.telemetry,
            shared_state.telemetry_updated_at_s,
            float(freshness_settings.get("telemetry_max_age_s", 1.5)),
        ),
        (
            "frames",
            shared_state.frames,
            shared_state.frames_updated_at_s,
            float(freshness_settings.get("frames_max_age_s", 1.0)),
        ),
        (
            "detection",
            shared_state.detection,
            shared_state.detection_updated_at_s,
            float(freshness_settings.get("detection_max_age_s", 1.0)),
        ),
        (
            "depth_analysis",
            shared_state.depth_analysis,
            shared_state.depth_analysis_updated_at_s,
            float(freshness_settings.get("depth_analysis_max_age_s", 1.0)),
        ),
    )
    stale_sensor_names: list[str] = []
    for sensor_name, sensor_value, updated_at_s, max_age_s in sensor_checks:
        if sensor_value is None or updated_at_s <= 0.0 or (now_s - updated_at_s) > max_age_s:
            stale_sensor_names.append(sensor_name)
    return stale_sensor_names


async def watchdog_loop(
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
    stale_after_s: float,
) -> None:
    from mission.states import MissionState
    from telemetry.models import RuntimeControlCommand

    while True:
        now_s = time.monotonic()
        should_trip = False
        reason = ""
        async with state_lock:
            if not shared_state.watchdog_triggered:
                stale_loop_names = find_stale_loop_names(
                    shared_state.loop_heartbeats,
                    WATCHDOG_LOOP_NAMES,
                    now_s,
                    stale_after_s,
                )
                if stale_loop_names:
                    should_trip = True
                    reason = "watchdog detected stale loop heartbeat: " + ", ".join(
                        sorted(stale_loop_names)
                    )
                    shared_state.watchdog_triggered = True
                    shared_state.watchdog_reason = reason
                    shared_state.mission_state = MissionState.FAILSAFE
                    shared_state.mission_detail = reason
                    shared_state.desired_command = RuntimeControlCommand(
                        duration_s=interval_s,
                        source="watchdog",
                        reason=reason,
                    )
                    shared_state.control_applied = False
        if should_trip:
            logger.error("Watchdog trip | %s", reason)
        await asyncio.sleep(interval_s)


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
        await mark_loop_heartbeat(shared_state, state_lock, "telemetry")
        snapshot = await asyncio.to_thread(adapter.get_telemetry)
        await asyncio.to_thread(recorder.record_telemetry, snapshot)
        async with state_lock:
            shared_state.telemetry = snapshot
            shared_state.telemetry_updated_at_s = time.monotonic()
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
        await mark_loop_heartbeat(shared_state, state_lock, "frame")
        frame_bundle = await asyncio.to_thread(frame_fetcher.fetch)
        async with state_lock:
            shared_state.frames = RuntimeFrameState(
                rgb_frame=frame_bundle.rgb_bgr.copy(),
                depth_frame=frame_bundle.depth_m.copy(),
                rgb_timestamp=frame_bundle.rgb_timestamp,
                depth_timestamp=frame_bundle.depth_timestamp,
            )
            shared_state.frames_updated_at_s = time.monotonic()
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
        await mark_loop_heartbeat(shared_state, state_lock, "vision")
        async with state_lock:
            frames = shared_state.frames
        if frames is not None and frames.rgb_frame is not None and frames.depth_frame is not None:
            detection = await asyncio.to_thread(detector.detect, frames.rgb_frame, marker_id)
            depth_analysis = await asyncio.to_thread(depth_analyzer.analyze, frames.depth_frame)
            async with state_lock:
                shared_state.detection = detection
                shared_state.detection_updated_at_s = time.monotonic()
                shared_state.depth_analysis = depth_analysis
                shared_state.depth_analysis_updated_at_s = time.monotonic()
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
    aruco_settings = settings.get("aruco", {})
    freshness_settings = settings.get("freshness", {})
    mission_settings = settings.get("mission", {})
    target_marker_area = float(control_settings.get("approach_target_marker_area", 12000.0))
    center_tolerance_px = float(control_settings.get("approach_center_tolerance_px", 30.0))
    command_duration_s = float(control_settings.get("approach_command_duration_s", 0.25))
    forward_speed_m_s = float(control_settings.get("approach_forward_speed_m_s", 0.4))
    target_marker_id = int(aruco_settings.get("marker_id", 0))
    auto_descend_on_target = bool(mission_settings.get("auto_descend_on_target", False))
    auto_land_on_target = bool(mission_settings.get("auto_land_on_target", False))
    last_event_key: tuple[str, str] | None = None

    while True:
        await mark_loop_heartbeat(shared_state, state_lock, "mission")
        async with state_lock:
            detection = shared_state.detection
            depth_analysis = shared_state.depth_analysis
            frames = shared_state.frames
            telemetry = shared_state.telemetry
            current_state = shared_state.mission_state
            watchdog_triggered = shared_state.watchdog_triggered
            watchdog_reason = shared_state.watchdog_reason
            manual_mode_enabled = shared_state.local_manual_mode_enabled
            stale_sensor_names = find_stale_sensor_names(
                shared_state,
                time.monotonic(),
                freshness_settings,
            )

        mission_state = current_state
        mission_detail = "waiting for sensor data"
        desired_command = RuntimeControlCommand(duration_s=command_duration_s)

        if watchdog_triggered:
            mission_state = MissionState.FAILSAFE
            mission_detail = watchdog_reason or "watchdog tripped"
            desired_command = RuntimeControlCommand(
                duration_s=command_duration_s,
                source="watchdog",
                reason=mission_detail,
            )
        elif manual_mode_enabled:
            mission_state = MissionState.IDLE
            mission_detail = "manual mode enabled"
            desired_command = RuntimeControlCommand(
                duration_s=command_duration_s,
                source="manual_mode",
                reason=mission_detail,
            )
        elif stale_sensor_names:
            mission_state = MissionState.FAILSAFE
            mission_detail = "stale sensor data: " + ", ".join(stale_sensor_names)
            desired_command = RuntimeControlCommand(
                duration_s=command_duration_s,
                source="freshness_guard",
                reason=mission_detail,
            )
        elif telemetry is None or frames is None or detection is None or depth_analysis is None:
            mission_state = MissionState.IDLE
        elif not detection.detected or detection.marker_id != target_marker_id:
            mission_state = MissionState.SEARCH
            desired_command = RuntimeControlCommand(
                yaw_rate=float(mission_settings.get("search_yaw_rate_deg_s", 8.0)),
                duration_s=float(mission_settings.get("search_step_duration_s", 0.35)),
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
            if auto_descend_on_target and detection.area >= float(
                mission_settings.get("descend_marker_area_threshold", 18000.0)
            ):
                mission_state = MissionState.DESCEND
                desired_command = RuntimeControlCommand(
                    vx=0.0,
                    vy=servo_command.vy,
                    vz=float(mission_settings.get("descend_rate_m_s", 0.2)),
                    yaw_rate=servo_command.yaw_rate,
                    duration_s=float(mission_settings.get("descend_step_duration_s", 0.25)),
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
            elif auto_land_on_target and telemetry.altitude_m <= float(
                settings.get("landing", {}).get("touchdown_altitude_m", 0.15)
            ):
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
                    reason=(
                        "marker stable near target; hold without auto descend"
                        if not auto_descend_on_target
                        else (
                            "marker stable near target; hold without autoland"
                            if not auto_land_on_target
                            else "marker stable; hold alignment"
                        )
                    ),
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
    safety_limiter,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    from mission.states import MissionState
    from telemetry.models import RuntimeControlCommand

    while True:
        await mark_loop_heartbeat(shared_state, state_lock, "control")
        async with state_lock:
            command = shared_state.desired_command
            mission_state = shared_state.mission_state
            already_applied = shared_state.control_applied
            manual_vx_m_s = shared_state.local_manual_vx_m_s
            manual_yaw_rate_deg_s = shared_state.local_manual_yaw_rate_deg_s
            manual_vz_m_s = shared_state.local_manual_vz_m_s
            manual_vy_m_s = shared_state.local_manual_vy_m_s
            manual_override_until_s = shared_state.local_manual_override_until_s
            manual_status = shared_state.local_manual_status
            spin_paused = shared_state.local_spin_paused
            manual_mode_enabled = shared_state.local_manual_mode_enabled
            watchdog_triggered = shared_state.watchdog_triggered
            watchdog_reason = shared_state.watchdog_reason

        if not already_applied:
            manual_active = time.monotonic() < manual_override_until_s
            effective_command = command

            if watchdog_triggered:
                manual_active = False
                effective_command = RuntimeControlCommand(
                    duration_s=max(command.duration_s, interval_s),
                    source="watchdog",
                    reason=watchdog_reason or "watchdog tripped",
                )
            elif manual_active:
                effective_command = build_manual_override_command(
                    manual_status=manual_status,
                    manual_vx_m_s=manual_vx_m_s,
                    manual_vy_m_s=manual_vy_m_s,
                    manual_vz_m_s=manual_vz_m_s,
                    manual_yaw_rate_deg_s=manual_yaw_rate_deg_s,
                    duration_s=max(command.duration_s, interval_s),
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
            effective_command = safety_limiter.clamp(effective_command)

            if mission_state == MissionState.LAND and not manual_active:
                await asyncio.to_thread(adapter.land)
            elif mission_state in {MissionState.IDLE, MissionState.FAILSAFE} and not manual_active:
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
                "Control loop | state=%s manual=%s manual_mode=%s vx=%.3f vy=%.3f vz=%.3f yaw_rate=%.3f command=%s",
                mission_state.value,
                manual_active,
                manual_mode_enabled,
                effective_command.vx,
                effective_command.vy,
                effective_command.vz,
                effective_command.yaw_rate,
                effective_command.reason,
            )
        await asyncio.sleep(interval_s)
