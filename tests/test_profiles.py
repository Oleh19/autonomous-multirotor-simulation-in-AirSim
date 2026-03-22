from app.bootstrap import load_settings
from app.main import apply_runtime_profile


def test_safe_profile_tightens_runtime_guards() -> None:
    settings = load_settings()

    profiled = apply_runtime_profile(settings, profile="safe", mode="local")

    assert profiled["watchdog"]["stale_after_s"] <= settings["watchdog"]["stale_after_s"]
    assert profiled["freshness"]["frames_max_age_s"] <= settings["freshness"]["frames_max_age_s"]


def test_ci_profile_disables_ui_and_debug_frames() -> None:
    settings = load_settings()

    profiled = apply_runtime_profile(settings, profile="ci", mode="local")

    assert profiled["local_ui"]["enabled"] is False
    assert profiled["dev_ui"]["enabled"] is False
    assert profiled["recording"]["save_debug_frames"] is False
    assert profiled["runtime"]["run_duration_s"] <= 5.0
