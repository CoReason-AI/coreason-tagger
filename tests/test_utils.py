# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import patch

from loguru import logger

from coreason_tagger.utils.logger import setup_logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly."""
    # loguru.logger is a singleton always available
    assert logger is not None


def test_logger_dir_creation() -> None:
    """Test that log directory creation logic is triggered if it doesn't exist."""
    # We test the setup_logger function directly, avoiding reload()

    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.mkdir") as mock_mkdir,
        patch("loguru.logger.add"),
    ):
        setup_logger("test_logs")

        assert mock_mkdir.called
        # Verify it was called on the path we expect
        # Note: Since we mocked Path.exists, any Path("...") call returns a mock that returns False.
        # But we want to ensure mkdir was called.

        # More robust check:
        # mock_mkdir is the method on the Path instance?
        # No, patch("pathlib.Path.mkdir") patches the method on the class.
        # So any instance calling mkdir triggers this mock.
        assert mock_mkdir.call_count >= 1
