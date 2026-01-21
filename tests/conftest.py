# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import os
from typing import Any

# Set APP_ENV to testing before any other modules are loaded
os.environ["APP_ENV"] = "testing"


# Configure pytest-asyncio to treat all async tests as asyncio-driven
# This avoids decorating every single test with @pytest.mark.asyncio
def pytest_configure(config: Any) -> None:
    config.option.asyncio_mode = "auto"
