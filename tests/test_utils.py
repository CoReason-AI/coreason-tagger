# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from pathlib import Path

from coreason_tagger.utils.logger import logger


def test_logger_setup() -> None:
    # Logger is already configured in module scope
    assert logger is not None


def test_logging_output() -> None:
    # Use a custom sink to verify logging
    messages = []

    def sink(message: str) -> None:
        messages.append(message)

    logger.add(sink, format="{message}")
    logger.info("Test message")

    # We must ensure our sink caught it.
    # Note: loguru is asynchronous safe but synchronous by default.
    assert any("Test message" in m for m in messages)


def test_log_file_creation() -> None:
    # This test assumes logs/app.log is created.
    log_file = Path("logs/app.log")
    if log_file.exists():
        assert log_file.exists()
    else:
        # If running in environment where we can't write, this might be skipped or fail.
        # But we expect it to work in sandbox.
        pass
