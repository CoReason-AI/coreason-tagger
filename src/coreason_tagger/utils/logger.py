# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import sys
from pathlib import Path

from loguru import logger

from coreason_tagger.config import settings

__all__ = ["logger", "setup_logger"]


def setup_logger(log_dir: str = "logs") -> None:
    """
    Configure the logger with console and file sinks.
    """
    # Remove default handler
    logger.remove()

    # Determine environment
    app_env = settings.APP_ENV.lower()
    should_enqueue = app_env != "testing"

    # Sink 1: Stdout (Human-readable)
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )

    # Ensure logs directory exists
    log_path = Path(log_dir)
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)

    # Sink 2: File (JSON, Rotation, Retention)
    # We use a path relative to the log_dir
    log_file = log_path / "app.log"
    logger.add(
        str(log_file),
        rotation="500 MB",
        retention="10 days",
        serialize=True,
        enqueue=should_enqueue,
        level=settings.LOG_LEVEL,
    )


# Auto-configure on import
setup_logger()
