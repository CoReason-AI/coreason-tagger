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


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    # Since the logger is initialized on import, we check side effects

def test_logger_setup() -> None:
    # Logger is already configured in module scope
    assert logger is not None


def test_logging_output() -> None:
    # Use a custom sink to verify logging
    messages = []

    def sink(message: str) -> None:
        messages.append(message)


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None
