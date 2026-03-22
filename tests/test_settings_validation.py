from __future__ import annotations

from copy import deepcopy

from app.bootstrap import load_settings, validate_settings


def test_project_settings_are_valid() -> None:
    settings = load_settings()

    assert validate_settings(settings) == []


def test_validation_reports_nested_numeric_errors() -> None:
    settings = load_settings()
    broken_settings = deepcopy(settings)
    broken_settings["airsim"]["port"] = 0
    broken_settings["depth"]["min_valid_depth_m"] = 25.0
    broken_settings["depth"]["max_valid_depth_m"] = 20.0

    errors = validate_settings(broken_settings)

    assert any("airsim.port" in error for error in errors)
    assert any("depth.min_valid_depth_m" in error for error in errors)


def test_validation_reports_unknown_keys() -> None:
    settings = load_settings()
    broken_settings = deepcopy(settings)
    broken_settings["app"]["unexpected"] = "value"

    errors = validate_settings(broken_settings)

    assert any("app.unexpected" in error for error in errors)
