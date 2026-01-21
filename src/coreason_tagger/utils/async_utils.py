# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import asyncio
import functools
from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def run_blocking(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Run a blocking function in a separate thread (executor) to avoid blocking the asyncio event loop.

    Args:
        func: The blocking function to run.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the blocking function.
    """
    loop = asyncio.get_running_loop()
    # functools.partial is used to pass keyword arguments, as run_in_executor only accepts *args
    call = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, call)
