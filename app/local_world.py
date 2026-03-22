from __future__ import annotations

import math


def apply_local_command(world, command, dt: float) -> None:
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


def build_initial_local_world(settings):
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


def render_local_world_frame(settings, world):
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

    marker_body_x, marker_body_y = world_to_body(
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
    obstacle_body_x, obstacle_body_y = world_to_body(
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
        depth[obstacle_top:obstacle_bottom, left:right] = min(
            depth[obstacle_top:obstacle_bottom, left:right].min(),
            obstacle_body_x,
        )

    return frame, depth, marker_visible, marker_distance_m, obstacle_distance_m, obstacle_side


def world_to_body(delta_x_m: float, delta_y_m: float, yaw_rad: float) -> tuple[float, float]:
    body_x = (math.cos(yaw_rad) * delta_x_m) + (math.sin(yaw_rad) * delta_y_m)
    body_y = (-math.sin(yaw_rad) * delta_x_m) + (math.cos(yaw_rad) * delta_y_m)
    return body_x, body_y


def draw_local_camera_overlay(
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
        "A/D: yaw | W/S: altitude | I/K: forward/back | J/L: strafe",
        "M: toggle manual-only | Space: toggle auto spin",
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


def draw_dev_camera_overlay(
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
        "A/D: yaw | W/S: altitude | I/K: forward/back | J/L: strafe",
        "M: toggle manual-only | Space: pause auto yaw | Q/Esc: exit",
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


def build_local_ui_canvas(frame_bgr, world, altitude_m: float, mission_state: str, command, steering_mode: str) -> object:
    import numpy as np

    panel_width = frame_bgr.shape[1]
    canvas = np.full(
        (frame_bgr.shape[0], frame_bgr.shape[1] + panel_width, 3),
        248,
        dtype=np.uint8,
    )
    canvas[:, : frame_bgr.shape[1]] = frame_bgr
    map_panel = canvas[:, frame_bgr.shape[1] :]
    draw_local_world_panel(map_panel, world, altitude_m, mission_state, command, steering_mode)
    return canvas


def draw_local_world_panel(panel_bgr, world, altitude_m: float, mission_state: str, command, steering_mode: str) -> None:
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
    cv2.rectangle(
        panel_bgr,
        (map_left, map_top),
        (map_left + map_width, map_top + map_height),
        (45, 45, 45),
        2,
    )

    def to_panel_point(x_m: float, y_m: float) -> tuple[int, int]:
        px = map_left + int(round((x_m / world_width_m) * map_width))
        normalized_y = (y_m + (world_height_m / 2.0)) / world_height_m
        py = map_top + int(round(normalized_y * map_height))
        return px, py

    marker_point = to_panel_point(world.marker_x_m, world.marker_y_m)
    obstacle_point = to_panel_point(world.obstacle.x_m, world.obstacle.y_m)
    drone_point = to_panel_point(world.drone_x_m, world.drone_y_m)

    cv2.circle(panel_bgr, marker_point, 12, (20, 120, 40), thickness=2)
    cv2.putText(
        panel_bgr,
        "marker",
        (marker_point[0] + 14, marker_point[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 120, 40),
        1,
        cv2.LINE_AA,
    )

    obstacle_radius_px = max(8, int(round((world.obstacle.radius_m / world_width_m) * map_width)))
    cv2.circle(panel_bgr, obstacle_point, obstacle_radius_px, (60, 90, 170), thickness=2)
    cv2.putText(
        panel_bgr,
        "obstacle",
        (obstacle_point[0] + 12, obstacle_point[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (60, 90, 170),
        1,
        cv2.LINE_AA,
    )

    cv2.circle(panel_bgr, drone_point, 10, (40, 40, 220), thickness=-1)
    heading_end = (
        int(round(drone_point[0] + (math.cos(math.radians(world.yaw_deg)) * 24))),
        int(round(drone_point[1] + (math.sin(math.radians(world.yaw_deg)) * 24))),
    )
    cv2.arrowedLine(
        panel_bgr,
        drone_point,
        heading_end,
        (40, 40, 220),
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )

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
        cv2.putText(
            panel_bgr,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (30, 30, 30),
            thickness,
            cv2.LINE_AA,
        )
        y += 24 if index == 0 else 22
