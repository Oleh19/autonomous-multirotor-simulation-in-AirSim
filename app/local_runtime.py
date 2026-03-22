from __future__ import annotations

import asyncio
import copy
import math
import time

from app.local_world import (
    apply_local_command,
    build_local_ui_canvas,
    draw_local_camera_overlay,
    render_local_world_frame,
)
from app.runtime_loops import build_manual_override_command, mark_loop_heartbeat

LEFT_KEYS = {ord("a"), ord("A")}
RIGHT_KEYS = {ord("d"), ord("D")}
UP_KEYS = {ord("w"), ord("W")}
DOWN_KEYS = {ord("s"), ord("S")}
FORWARD_KEYS = {ord("i"), ord("I")}
BACKWARD_KEYS = {ord("k"), ord("K")}
STRAFE_LEFT_KEYS = {ord("j"), ord("J")}
STRAFE_RIGHT_KEYS = {ord("l"), ord("L")}
STOP_KEYS = {32}
MANUAL_MODE_TOGGLE_KEYS = {ord("m"), ord("M")}
MANUAL_OVERRIDE_DURATION_S = 0.8
MANUAL_XY_SPEED_M_S = 1.0
MANUAL_Z_SPEED_M_S = 0.5
MANUAL_YAW_RATE_DEG_S = 25.0


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
        await mark_loop_heartbeat(shared_state, state_lock, "telemetry")
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
            shared_state.telemetry_updated_at_s = time.monotonic()
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
        await mark_loop_heartbeat(shared_state, state_lock, "frame")
        async with state_lock:
            world = shared_state.local_world
        if world is None:
            await asyncio.sleep(interval_s)
            continue

        frame, depth, marker_visible, marker_distance_m, obstacle_distance_m, obstacle_side = (
            render_local_world_frame(settings, world)
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
            shared_state.frames_updated_at_s = time.monotonic()
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
    safety_limiter,
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
) -> None:
    while True:
        await mark_loop_heartbeat(shared_state, state_lock, "control")
        async with state_lock:
            command = shared_state.desired_command
            mission_state = shared_state.mission_state
            already_applied = shared_state.control_applied
            world = shared_state.local_world
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
            effective_command = command
            if watchdog_triggered:
                from telemetry.models import RuntimeControlCommand

                effective_command = RuntimeControlCommand(
                    duration_s=max(command.duration_s, interval_s),
                    source="watchdog",
                    reason=watchdog_reason or "watchdog tripped",
                )
            elif mission_state.value == "failsafe":
                from telemetry.models import RuntimeControlCommand

                effective_command = RuntimeControlCommand(
                    duration_s=max(command.duration_s, interval_s),
                    source="failsafe",
                    reason=command.reason,
                )
            elif time.monotonic() < manual_override_until_s:
                effective_command = build_manual_override_command(
                    manual_status=manual_status,
                    manual_vx_m_s=manual_vx_m_s,
                    manual_vy_m_s=manual_vy_m_s,
                    manual_vz_m_s=manual_vz_m_s,
                    manual_yaw_rate_deg_s=manual_yaw_rate_deg_s,
                    duration_s=max(command.duration_s, interval_s),
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
            effective_command = safety_limiter.clamp(effective_command)
            if world is not None:
                apply_local_command(world, effective_command, max(effective_command.duration_s, interval_s))
            logger.info(
                "Local control | state=%s manual_mode=%s x=%.2f y=%.2f alt=%.2f yaw=%.1f vx=%.3f vy=%.3f vz=%.3f yaw_rate=%.3f reason=%s",
                mission_state.value,
                manual_mode_enabled,
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
            world = shared_state.local_world
            manual_override_until_s = shared_state.local_manual_override_until_s
            spin_paused = shared_state.local_spin_paused
            manual_mode_enabled = shared_state.local_manual_mode_enabled

        if frames is not None and frames.rgb_frame is not None and world is not None:
            view_world = frames.local_world_snapshot if frames.local_world_snapshot is not None else world
            frame = frames.rgb_frame.copy()
            draw_local_camera_overlay(
                frame,
                mission_state.value,
                mission_detail,
                command.reason,
                detection.detected if detection is not None else False,
                depth_analysis.obstacle_detected if depth_analysis is not None else False,
            )
            canvas = build_local_ui_canvas(
                frame,
                view_world,
                view_world.altitude_m,
                mission_state.value,
                command,
                (
                    "manual-only"
                    if manual_mode_enabled
                    else ("manual" if time.monotonic() < manual_override_until_s else ("paused" if spin_paused else "auto"))
                ),
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                logger.info("Local UI loop requested shutdown")
                return
            if key != 255:
                await apply_manual_key_input(shared_state, state_lock, key)
        await asyncio.sleep(interval_s)


async def apply_manual_key_input(shared_state, state_lock: asyncio.Lock, key: int) -> None:
    if key in LEFT_KEYS:
        async with state_lock:
            shared_state.local_manual_vx_m_s = 0.0
            shared_state.local_manual_yaw_rate_deg_s = -MANUAL_YAW_RATE_DEG_S
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + MANUAL_OVERRIDE_DURATION_S
            shared_state.local_manual_status = "left"
    elif key in RIGHT_KEYS:
        async with state_lock:
            shared_state.local_manual_vx_m_s = 0.0
            shared_state.local_manual_yaw_rate_deg_s = MANUAL_YAW_RATE_DEG_S
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + MANUAL_OVERRIDE_DURATION_S
            shared_state.local_manual_status = "right"
    elif key in UP_KEYS:
        async with state_lock:
            shared_state.local_manual_vx_m_s = 0.0
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = -MANUAL_Z_SPEED_M_S
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + MANUAL_OVERRIDE_DURATION_S
            shared_state.local_manual_status = "up"
    elif key in DOWN_KEYS:
        async with state_lock:
            shared_state.local_manual_vx_m_s = 0.0
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = MANUAL_Z_SPEED_M_S
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + MANUAL_OVERRIDE_DURATION_S
            shared_state.local_manual_status = "down"
    elif key in FORWARD_KEYS:
        async with state_lock:
            shared_state.local_manual_vx_m_s = MANUAL_XY_SPEED_M_S
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + MANUAL_OVERRIDE_DURATION_S
            shared_state.local_manual_status = "forward"
    elif key in BACKWARD_KEYS:
        async with state_lock:
            shared_state.local_manual_vx_m_s = -MANUAL_XY_SPEED_M_S
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = time.monotonic() + MANUAL_OVERRIDE_DURATION_S
            shared_state.local_manual_status = "backward"
    elif key in STRAFE_LEFT_KEYS:
        async with state_lock:
            shared_state.local_manual_vx_m_s = 0.0
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = -MANUAL_XY_SPEED_M_S
            shared_state.local_manual_override_until_s = time.monotonic() + MANUAL_OVERRIDE_DURATION_S
            shared_state.local_manual_status = "strafe_left"
    elif key in STRAFE_RIGHT_KEYS:
        async with state_lock:
            shared_state.local_manual_vx_m_s = 0.0
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = MANUAL_XY_SPEED_M_S
            shared_state.local_manual_override_until_s = time.monotonic() + MANUAL_OVERRIDE_DURATION_S
            shared_state.local_manual_status = "strafe_right"
    elif key in MANUAL_MODE_TOGGLE_KEYS:
        async with state_lock:
            shared_state.local_manual_mode_enabled = not shared_state.local_manual_mode_enabled
            shared_state.local_manual_vx_m_s = 0.0
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = 0.0
            shared_state.local_manual_status = (
                "manual_mode" if shared_state.local_manual_mode_enabled else "auto"
            )
    elif key in STOP_KEYS:
        async with state_lock:
            shared_state.local_spin_paused = not shared_state.local_spin_paused
            shared_state.local_manual_vx_m_s = 0.0
            shared_state.local_manual_yaw_rate_deg_s = 0.0
            shared_state.local_manual_vz_m_s = 0.0
            shared_state.local_manual_vy_m_s = 0.0
            shared_state.local_manual_override_until_s = 0.0
            shared_state.local_manual_status = "hold" if shared_state.local_spin_paused else "auto"
