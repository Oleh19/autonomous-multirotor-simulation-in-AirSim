from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.local_runtime import apply_manual_key_input, is_supported_manual_key
from app.local_world import draw_dev_camera_overlay


class TerminalKeyReader:
    def __init__(self) -> None:
        self._isatty = sys.stdin.isatty()
        self._fd: int | None = None
        self._termios_module = None
        self._saved_termios = None

    def __enter__(self) -> "TerminalKeyReader":
        if not self._isatty or os.name == "nt" or running_in_wsl():
            return self

        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._termios_module = termios
        self._saved_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._fd is not None and self._saved_termios is not None and self._termios_module is not None:
            self._termios_module.tcsetattr(
                self._fd,
                self._termios_module.TCSADRAIN,
                self._saved_termios,
            )

    def read_key(self, timeout_s: float) -> str | None:
        if not self._isatty:
            time.sleep(timeout_s)
            return None

        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    key = msvcrt.getwch()
                    if key in {"\x00", "\xe0"} and msvcrt.kbhit():
                        msvcrt.getwch()
                        return None
                    return key
                time.sleep(0.01)
            return None

        import select

        if running_in_wsl():
            readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
            if readable:
                line = sys.stdin.readline().strip()
                if line:
                    return line[0]
            return None

        readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if readable:
            return sys.stdin.read(1)
        return None


def terminal_controls_available() -> bool:
    return sys.stdin.isatty()


def running_in_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


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
            manual_mode_enabled = shared_state.local_manual_mode_enabled

        frame = None
        if frames is not None and frames.rgb_frame is not None:
            frame = frames.rgb_frame.copy()
        if frame is None:
            frame = np.full((480, 640, 3), 24, dtype=np.uint8)

        draw_dev_camera_overlay(
            frame_bgr=frame,
            mission_state=mission_state.value,
            mission_detail=mission_detail,
            command_reason=command.reason,
            marker_detected=detection.detected if detection is not None else False,
            obstacle_detected=depth_analysis.obstacle_detected if depth_analysis is not None else False,
            altitude_m=telemetry.altitude_m if telemetry is not None else 0.0,
            steering_mode=(
                "manual-only"
                if manual_mode_enabled
                else (
                    "manual"
                    if time.monotonic() < manual_override_until_s
                    else ("paused" if spin_paused else "auto")
                )
            ),
        )
        cv2.imshow(window_name, frame)
        key = cv2.waitKeyEx(1)

        if key in {27, ord("q"), ord("Q"), ord("й"), ord("Й")}:
            logger.info("Dev UI loop requested shutdown")
            return
        if key != -1:
            await apply_manual_key_input(shared_state, state_lock, key)
        await asyncio.sleep(interval_s)


async def terminal_control_loop(
    shared_state,
    state_lock: asyncio.Lock,
    logger,
    interval_s: float,
    mode_name: str,
) -> None:
    if not terminal_controls_available():
        logger.info("Terminal control unavailable because stdin is not a TTY")
        return

    controls_hint = (
        f"{mode_name} terminal controls: A/D yaw | W/S altitude | I/K forward/back | "
        "J/L strafe | M manual-only | Space pause auto yaw | Q quit"
    )
    controls_hint += " | ru-layout: Ф/В Ц/Ы Ш/Л О/Д Ь"
    if running_in_wsl():
        controls_hint += " | in WSL terminals, type the key and press Enter"
    print(controls_hint)
    with TerminalKeyReader() as key_reader:
        while True:
            key_text = await asyncio.to_thread(key_reader.read_key, interval_s)
            if key_text is None:
                await asyncio.sleep(interval_s)
                continue

            if key_text in {"q", "Q", "й", "Й", "\x1b"}:
                async with state_lock:
                    shared_state.shutdown_requested = True
                logger.info("Terminal control requested shutdown")
                return

            if len(key_text) == 1 and is_supported_manual_key(ord(key_text)):
                await apply_manual_key_input(shared_state, state_lock, ord(key_text))
