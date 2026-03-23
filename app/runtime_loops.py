from __future__ import annotations

import asyncio
import time


WATCHDOG_LOOP_NAMES = ("telemetry", "frame", "vision", "mission", "control")


def _fetch_and_record_telemetry(adapter, recorder):
    snapshot = adapter.get_telemetry()
    recorder.record_telemetry(snapshot)
    return snapshot


def _process_vision_step(
    detector,
    depth_analyzer,
    recorder,
    marker_id: int,
    rgb_frame,
    depth_frame,
    rgb_timestamp: int,
):
    detection = detector.detect(rgb_frame, marker_id)
    depth_analysis = depth_analyzer.analyze(depth_frame)
    recorder.maybe_save_debug_frame(rgb_frame, "vision", rgb_timestamp)
    return detection, depth_analysis


def _build_local_partial_detection(settings, local_world_snapshot, target_marker_id: int):
    from app.local_world import project_local_marker
    from vision.aruco_detector import ArucoDetection

    projection = project_local_marker(settings, local_world_snapshot)
    if not projection.visible_in_frame or projection.marker_size_px <= 0:
        return None

    half_size = projection.marker_size_px / 2.0
    left = projection.center_x_px - half_size
    top = projection.center_y_px - half_size
    right = projection.center_x_px + half_size
    bottom = projection.center_y_px + half_size
    clipped_width = max(0.0, float(projection.clipped_right_px - projection.clipped_left_px))
    clipped_height = max(0.0, float(projection.clipped_bottom_px - projection.clipped_top_px))
    visible_area = clipped_width * clipped_height
    if visible_area <= 0.0:
        return None

    return ArucoDetection(
        detected=True,
        marker_id=target_marker_id,
        center_x=float(projection.center_x_px),
        center_y=float(projection.center_y_px),
        corners=(
            (float(left), float(top)),
            (float(right), float(top)),
            (float(right), float(bottom)),
            (float(left), float(bottom)),
        ),
        area=visible_area,
    )


def log_profile_if_slow(
    logger,
    profiling_enabled: bool,
    profiling_warn_threshold_ms: float,
    stage: str,
    elapsed_ms: float,
) -> None:
    if profiling_enabled and elapsed_ms >= profiling_warn_threshold_ms:
        logger.info("Profile | stage=%s elapsed_ms=%.2f", stage, elapsed_ms)


def build_target_tracking_command(
    servo_command,
    *,
    aligned: bool,
    approach_forward_speed_m_s: float,
    visible_tracking_forward_speed_m_s: float,
    frame_width: int,
    frame_height: int,
):
    from telemetry.models import RuntimeControlCommand

    horizontal_error_ratio = min(1.0, abs(servo_command.error_x_px) / max(frame_width / 2.0, 1.0))
    vertical_error_ratio = min(1.0, abs(servo_command.error_y_px) / max(frame_height / 2.0, 1.0))
    dominant_error_ratio = max(horizontal_error_ratio, vertical_error_ratio)
    forward_speed_m_s = visible_tracking_forward_speed_m_s + (
        (approach_forward_speed_m_s - visible_tracking_forward_speed_m_s)
        * max(0.0, 1.0 - dominant_error_ratio)
    )

    if aligned:
        return RuntimeControlCommand(
            vx=approach_forward_speed_m_s,
            vy=servo_command.vy,
            vz=servo_command.vz,
            yaw_rate=servo_command.yaw_rate,
            duration_s=servo_command.duration_s,
            source="mission_loop",
            reason="marker aligned; continuous approach",
        )

    return RuntimeControlCommand(
        vx=forward_speed_m_s,
        vy=servo_command.vy,
        vz=servo_command.vz,
        yaw_rate=servo_command.yaw_rate,
        duration_s=servo_command.duration_s,
        source="mission_loop",
        reason="marker visible; track and re-center while approaching",
    )


def should_keep_recent_target_lock(
    *,
    target_locked: bool,
    detection,
    last_target_detection,
    last_target_seen_at_s: float,
    now_s: float,
    target_memory_timeout_s: float,
):
    if not target_locked:
        return detection
    if detection is not None and detection.detected:
        return detection
    if (
        last_target_detection is not None
        and last_target_detection.detected
        and last_target_seen_at_s > 0.0
        and (now_s - last_target_seen_at_s) <= target_memory_timeout_s
    ):
        return last_target_detection
    return detection


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
    profiling_enabled: bool = False,
    profiling_warn_threshold_ms: float = 0.0,
) -> None:
    from telemetry.logger import format_snapshot

    while True:
        await mark_loop_heartbeat(shared_state, state_lock, "telemetry")
        started_at = time.perf_counter()
        snapshot = await asyncio.to_thread(_fetch_and_record_telemetry, adapter, recorder)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        async with state_lock:
            shared_state.telemetry = snapshot
            shared_state.telemetry_updated_at_s = time.monotonic()
        log_profile_if_slow(
            logger,
            profiling_enabled,
            profiling_warn_threshold_ms,
            "telemetry",
            elapsed_ms,
        )
        logger.info("Telemetry loop | %s", format_snapshot(snapshot))
        await asyncio.sleep(interval_s)


async def frame_loop(
    frame_fetcher,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
    profiling_enabled: bool = False,
    profiling_warn_threshold_ms: float = 0.0,
) -> None:
    from telemetry.models import RuntimeFrameState

    while True:
        await mark_loop_heartbeat(shared_state, state_lock, "frame")
        started_at = time.perf_counter()
        frame_bundle = await asyncio.to_thread(frame_fetcher.fetch)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        async with state_lock:
            # FrameFetcher already returns fresh numpy arrays for each fetch, so we can
            # publish them directly and let consumers copy only when they intend to draw.
            shared_state.frames = RuntimeFrameState(
                rgb_frame=frame_bundle.rgb_bgr,
                depth_frame=frame_bundle.depth_m,
                rgb_timestamp=frame_bundle.rgb_timestamp,
                depth_timestamp=frame_bundle.depth_timestamp,
            )
            shared_state.frames_updated_at_s = time.monotonic()
        log_profile_if_slow(
            logger,
            profiling_enabled,
            profiling_warn_threshold_ms,
            "frame_fetch",
            elapsed_ms,
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
    settings,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
    profiling_enabled: bool = False,
    profiling_warn_threshold_ms: float = 0.0,
) -> None:
    while True:
        await mark_loop_heartbeat(shared_state, state_lock, "vision")
        async with state_lock:
            frames = shared_state.frames
        if frames is not None and frames.rgb_frame is not None and frames.depth_frame is not None:
            started_at = time.perf_counter()
            detection, depth_analysis = await asyncio.to_thread(
                _process_vision_step,
                detector,
                depth_analyzer,
                recorder,
                marker_id,
                frames.rgb_frame,
                frames.depth_frame,
                frames.rgb_timestamp,
            )
            if (
                not detection.detected
                and frames.local_world_snapshot is not None
            ):
                partial_detection = _build_local_partial_detection(
                    settings=settings,
                    local_world_snapshot=frames.local_world_snapshot,
                    target_marker_id=marker_id,
                )
                if partial_detection is not None:
                    detection = partial_detection
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            async with state_lock:
                shared_state.detection = detection
                shared_state.detection_updated_at_s = time.monotonic()
                shared_state.depth_analysis = depth_analysis
                shared_state.depth_analysis_updated_at_s = time.monotonic()
            log_profile_if_slow(
                logger,
                profiling_enabled,
                profiling_warn_threshold_ms,
                "vision_process",
                elapsed_ms,
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
    center_tolerance_px = float(control_settings.get("approach_center_tolerance_px", 30.0))
    command_duration_s = float(control_settings.get("approach_command_duration_s", 0.25))
    forward_speed_m_s = float(control_settings.get("approach_forward_speed_m_s", 0.4))
    visible_tracking_forward_speed_m_s = float(
        control_settings.get("track_visible_forward_speed_m_s", 0.08)
    )
    target_memory_timeout_s = float(control_settings.get("target_memory_timeout_s", 0.75))
    target_marker_id = int(aruco_settings.get("marker_id", 0))
    auto_descend_on_target = bool(mission_settings.get("auto_descend_on_target", False))
    auto_land_on_target = bool(mission_settings.get("auto_land_on_target", False))
    avoidance_forward_speed_m_s = float(mission_settings.get("avoidance_forward_speed_m_s", 0.15))
    avoidance_clear_cycles = int(mission_settings.get("avoidance_clear_cycles", 3))
    last_event_key: tuple[str, str] | None = None
    avoidance_side_lock = "none"
    avoidance_clear_cycles_remaining = 0

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
            autopilot_enabled = shared_state.local_autopilot_enabled
            target_locked = shared_state.local_autopilot_target_locked
            locked_target_marker_id = shared_state.local_autopilot_target_marker_id
            last_target_detection = shared_state.last_target_detection
            last_target_seen_at_s = shared_state.last_target_seen_at_s
            stale_sensor_names = find_stale_sensor_names(
                shared_state,
                time.monotonic(),
                freshness_settings,
            )
        now_s = time.monotonic()
        detection_for_guidance = should_keep_recent_target_lock(
            target_locked=target_locked,
            detection=detection,
            last_target_detection=last_target_detection,
            last_target_seen_at_s=last_target_seen_at_s,
            now_s=now_s,
            target_memory_timeout_s=target_memory_timeout_s,
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
            avoidance_side_lock = "none"
            avoidance_clear_cycles_remaining = 0
            mission_state = MissionState.IDLE
            mission_detail = "manual mode enabled"
            desired_command = RuntimeControlCommand(
                duration_s=command_duration_s,
                source="manual_mode",
                reason=mission_detail,
            )
        elif not autopilot_enabled:
            avoidance_side_lock = "none"
            avoidance_clear_cycles_remaining = 0
            mission_state = MissionState.IDLE
            mission_detail = "autopilot disabled"
            desired_command = RuntimeControlCommand(
                duration_s=command_duration_s,
                source="autopilot_toggle",
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
        elif telemetry is None or frames is None or detection_for_guidance is None or depth_analysis is None:
            mission_state = MissionState.IDLE
        elif (
            not detection_for_guidance.detected
            or detection_for_guidance.marker_id not in {target_marker_id, locked_target_marker_id}
        ):
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
                detection=detection_for_guidance,
                frame_width=frames.rgb_frame.shape[1],
                frame_height=frames.rgb_frame.shape[0],
            )
            aligned = (
                abs(servo_command.error_x_px) <= center_tolerance_px
                and abs(servo_command.error_y_px) <= center_tolerance_px
            )
            if depth_analysis.obstacle_detected or avoidance_side_lock != "none":
                mission_state = MissionState.TRACK
                if depth_analysis.obstacle_detected:
                    avoidance_side_lock = depth_analysis.safer_side
                    avoidance_clear_cycles_remaining = avoidance_clear_cycles
                elif avoidance_clear_cycles_remaining > 0:
                    avoidance_clear_cycles_remaining -= 1
                else:
                    avoidance_side_lock = "none"

                if avoidance_side_lock == "none":
                    desired_command = RuntimeControlCommand(
                        vx=0.0,
                        vy=servo_command.vy,
                        vz=servo_command.vz,
                        yaw_rate=servo_command.yaw_rate,
                        duration_s=servo_command.duration_s,
                        source="mission_loop",
                        reason="avoidance complete; resume alignment",
                    )
                else:
                    lateral_direction = -1.0 if avoidance_side_lock == "left" else 1.0
                    avoidance_command = obstacle_avoidance.compute_command(depth_analysis)
                    desired_command = RuntimeControlCommand(
                        vx=avoidance_forward_speed_m_s if aligned else 0.0,
                        vy=lateral_direction * max(abs(avoidance_command.vy), 0.01),
                        vz=servo_command.vz,
                        yaw_rate=servo_command.yaw_rate,
                        duration_s=max(avoidance_command.duration_s, command_duration_s),
                        source="mission_loop",
                        reason=(
                            f"front obstacle; bypass toward {avoidance_side_lock}"
                            if depth_analysis.obstacle_detected
                            else f"path clearing; continue bypass toward {avoidance_side_lock}"
                        ),
                    )
                mission_detail = desired_command.reason
            elif auto_descend_on_target and detection_for_guidance.area >= float(
                mission_settings.get("descend_marker_area_threshold", 18000.0)
            ):
                avoidance_side_lock = "none"
                avoidance_clear_cycles_remaining = 0
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
                avoidance_side_lock = "none"
                avoidance_clear_cycles_remaining = 0
                mission_state = MissionState.TRACK
                desired_command = build_target_tracking_command(
                    servo_command,
                    aligned=False,
                    approach_forward_speed_m_s=forward_speed_m_s,
                    visible_tracking_forward_speed_m_s=visible_tracking_forward_speed_m_s,
                    frame_width=frames.rgb_frame.shape[1],
                    frame_height=frames.rgb_frame.shape[0],
                )
                mission_detail = desired_command.reason
            elif auto_land_on_target and telemetry.altitude_m <= float(
                settings.get("landing", {}).get("touchdown_altitude_m", 0.15)
            ):
                avoidance_side_lock = "none"
                avoidance_clear_cycles_remaining = 0
                mission_state = MissionState.LAND
                desired_command = RuntimeControlCommand(
                    source="mission_loop",
                    reason="touchdown altitude reached; await landing trigger",
                )
                mission_detail = desired_command.reason
            else:
                avoidance_side_lock = "none"
                avoidance_clear_cycles_remaining = 0
                mission_state = MissionState.TRACK
                desired_command = build_target_tracking_command(
                    servo_command,
                    aligned=True,
                    approach_forward_speed_m_s=forward_speed_m_s,
                    visible_tracking_forward_speed_m_s=visible_tracking_forward_speed_m_s,
                    frame_width=frames.rgb_frame.shape[1],
                    frame_height=frames.rgb_frame.shape[0],
                )
                mission_detail = desired_command.reason

        async with state_lock:
            if (
                autopilot_enabled
                and detection is not None
                and detection.detected
                and detection.marker_id == target_marker_id
            ):
                shared_state.local_autopilot_target_locked = True
                shared_state.local_autopilot_target_marker_id = detection.marker_id
                shared_state.last_target_detection = detection
                shared_state.last_target_seen_at_s = now_s
            elif not autopilot_enabled or manual_mode_enabled:
                shared_state.local_autopilot_target_locked = False
                shared_state.local_autopilot_target_marker_id = None
                shared_state.last_target_detection = None
                shared_state.last_target_seen_at_s = 0.0
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
    profiling_enabled: bool = False,
    profiling_warn_threshold_ms: float = 0.0,
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

            started_at = time.perf_counter()
            if mission_state == MissionState.LAND and not manual_active:
                await asyncio.to_thread(adapter.land)
            elif mission_state in {MissionState.IDLE, MissionState.FAILSAFE} and not manual_active:
                await asyncio.to_thread(adapter.hover, False)
            else:
                await asyncio.to_thread(
                    adapter.move_by_velocity_body,
                    effective_command.vx,
                    effective_command.vy,
                    effective_command.vz,
                    max(effective_command.duration_s, interval_s),
                    effective_command.yaw_rate,
                    False,
                )
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            async with state_lock:
                shared_state.control_applied = True
            log_profile_if_slow(
                logger,
                profiling_enabled,
                profiling_warn_threshold_ms,
                "control_apply",
                elapsed_ms,
            )
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
