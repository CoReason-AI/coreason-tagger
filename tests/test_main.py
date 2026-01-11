# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import MagicMock

import pytest

from coreason_tagger.main import hello_world


def test_hello_world(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock logger to verify it's called
    mock_logger = MagicMock()
    monkeypatch.setattr("coreason_tagger.main.logger", mock_logger)

    result = hello_world()

    assert result == "Hello World!"
    mock_logger.info.assert_called_once_with("Hello World!")
