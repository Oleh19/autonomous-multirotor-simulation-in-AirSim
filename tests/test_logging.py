import json
import logging

from app.bootstrap import configure_logging
from telemetry.logger import JsonLogFormatter


def test_json_log_formatter_emits_parseable_json() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="drone_cv.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello structured logs",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "drone_cv.test"
    assert payload["message"] == "hello structured logs"


def test_configure_logging_sets_requested_level() -> None:
    logger = configure_logging("DEBUG", "text")

    assert logger.getEffectiveLevel() == logging.DEBUG
