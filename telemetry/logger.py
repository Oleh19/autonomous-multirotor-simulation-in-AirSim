from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from telemetry.models import TelemetrySnapshot


def get_logger(name: str = "drone_cv.telemetry") -> logging.Logger:
    return logging.getLogger(name)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def format_snapshot(snapshot: TelemetrySnapshot) -> str:
    position = snapshot.position_m
    velocity = snapshot.velocity_m_s
    orientation = snapshot.orientation
    return (
        "ts=%s | pos=(%.2f, %.2f, %.2f)m | vel=(%.2f, %.2f, %.2f)m/s | "
        "alt=%.2fm | speed=%.2fm/s | orient=(x=%.3f, y=%.3f, z=%.3f, w=%.3f)"
    ) % (
        snapshot.timestamp,
        position.x,
        position.y,
        position.z,
        velocity.x,
        velocity.y,
        velocity.z,
        snapshot.altitude_m,
        snapshot.speed_m_s,
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )


def create_file_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger


def log_mission_event(
    logger: logging.Logger,
    event: str,
    detail: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"event": event, "detail": detail}
    if extra:
        payload.update(extra)
    logger.info(json.dumps(payload, ensure_ascii=True))
