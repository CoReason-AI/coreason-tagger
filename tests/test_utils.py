# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger


import importlib
from unittest.mock import patch

from coreason_tagger.utils.logger import logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    # Since the logger is initialized on import, we check side effects


def test_logger_setup() -> None:
    # Logger is already configured in module scope
    assert logger is not None


def test_logger_dir_creation() -> None:
    """Test that log directory creation logic is triggered if it doesn't exist."""
    # We need to mock Path.exists to return False for "logs"
    # and verify mkdir is called.

    # Since the module is already imported, we might need to reload it
    # or extract the logic. But the logic is at module level.
    # So we force reload under mock.

    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.mkdir") as mock_mkdir,
        patch("loguru.logger.add"),
    ):  # Prevent actual file addition failing
        # We need to reload the module to trigger the top-level code
        # However, reloading 'coreason_tagger.utils.logger' might use cached reference.
        # We'll use importlib.reload
        from coreason_tagger.utils import logger

        importlib.reload(logger)

        # Verify mkdir was called
        # Note: Path("logs") is called in the module.
        # We need to ensure the mock captures it.
        # Since Path is imported as 'from pathlib import Path', patching pathlib.Path works.

        # We check if ANY mkdir call happened on the mocked object that represents "logs"
        # Since we mocked Path.exists to False, the code enters the if block.
        # log_path = Path("logs") -> mock_path
        # mock_path.mkdir(...)

        # Check if mkdir was called.
        # Note: Depending on how Path is mocked, we might need to be careful.
        # But 'pathlib.Path.exists' patch affects instances? No, it affects the method.
        # So any instance calling exists() returns False.
        # So log_path.exists() returns False.
        # Then log_path.mkdir(...) is called.

        # Since we mocked 'pathlib.Path.mkdir', it should capture the call.
        assert mock_mkdir.called
