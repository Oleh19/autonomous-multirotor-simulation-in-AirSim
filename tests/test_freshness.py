from telemetry.models import RuntimeSharedState

from app.runtime_loops import find_stale_sensor_names


def test_freshness_detects_missing_or_old_sensor_updates() -> None:
    shared_state = RuntimeSharedState()
    shared_state.telemetry = object()
    shared_state.telemetry_updated_at_s = 10.0
    shared_state.frames = object()
    shared_state.frames_updated_at_s = 9.2
    shared_state.detection = object()
    shared_state.detection_updated_at_s = 10.1

    stale_sensor_names = find_stale_sensor_names(
        shared_state=shared_state,
        now_s=10.5,
        freshness_settings={
            "telemetry_max_age_s": 1.0,
            "frames_max_age_s": 1.0,
            "detection_max_age_s": 1.0,
            "depth_analysis_max_age_s": 1.0,
        },
    )

    assert stale_sensor_names == ["frames", "depth_analysis"]


def test_freshness_accepts_recent_sensor_updates() -> None:
    shared_state = RuntimeSharedState()
    shared_state.telemetry = object()
    shared_state.telemetry_updated_at_s = 20.0
    shared_state.frames = object()
    shared_state.frames_updated_at_s = 20.1
    shared_state.detection = object()
    shared_state.detection_updated_at_s = 20.2
    shared_state.depth_analysis = object()
    shared_state.depth_analysis_updated_at_s = 20.2

    stale_sensor_names = find_stale_sensor_names(
        shared_state=shared_state,
        now_s=20.7,
        freshness_settings={
            "telemetry_max_age_s": 1.0,
            "frames_max_age_s": 1.0,
            "detection_max_age_s": 1.0,
            "depth_analysis_max_age_s": 1.0,
        },
    )

    assert stale_sensor_names == []
