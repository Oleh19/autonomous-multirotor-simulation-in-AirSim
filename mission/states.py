from __future__ import annotations

from enum import Enum


class MissionState(str, Enum):
    IDLE = "idle"
    TAKEOFF = "takeoff"
    SEARCH = "search"
    TRACK = "track"
    DESCEND = "descend"
    LAND = "land"
    COMPLETE = "complete"
    FAILSAFE = "failsafe"
