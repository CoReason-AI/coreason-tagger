# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Deque, TypeVar

from loguru import logger

T = TypeVar("T")


class CircuitState(Enum):
    """Enumeration of circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(Exception):
    """Exception raised when the circuit breaker is open."""

    pass


class CircuitBreaker:
    """A Singleton Circuit Breaker implementation.

    Triggers after `max_failures` within `window_seconds`.
    Attempts reset after `reset_timeout_seconds`.
    """

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "CircuitBreaker":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super(CircuitBreaker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_failures: int = 5,
        window_seconds: int = 10,
        reset_timeout_seconds: int = 30,
    ) -> None:
        """Initialize the CircuitBreaker.

        Args:
            max_failures: Number of failures allowed before opening the circuit. Defaults to 5.
            window_seconds: Time window in seconds to track failures. Defaults to 10.
            reset_timeout_seconds: Time in seconds to wait before attempting recovery (HALF_OPEN). Defaults to 30.
        """
        if getattr(self, "_initialized", False):
            return

        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.reset_timeout_seconds = reset_timeout_seconds

        self._failures: Deque[float] = deque()
        self._state = CircuitState.CLOSED
        self._last_failure_time: float = 0.0
        self._next_attempt_time: float = 0.0

        # Lock to ensure thread/async safety for state updates if needed
        # In asyncio, we don't need threading locks, but we need to be careful with concurrency.
        # Since this is running in a single event loop, atomic operations are generally safe
        # between awaits.
        self._initialized = True

    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        return self._state

    def _clean_window(self, now: float) -> None:
        """Remove failures that are outside the sliding window.

        Args:
            now: Current monotonic time.
        """
        while self._failures and self._failures[0] < now - self.window_seconds:
            self._failures.popleft()

    def _record_failure(self) -> None:
        """Record a failure and update state."""
        now = time.monotonic()
        self._last_failure_time = now
        self._failures.append(now)
        self._clean_window(now)

        logger.warning(f"Circuit Breaker failure recorded. Count: {len(self._failures)}")

        if self._state == CircuitState.HALF_OPEN:
            # If failed during HALF_OPEN, go back to OPEN immediately
            self._state = CircuitState.OPEN
            self._next_attempt_time = now + self.reset_timeout_seconds
            logger.error(f"Circuit Breaker transition: HALF_OPEN -> OPEN. Retry in {self.reset_timeout_seconds}s")
        elif self._state == CircuitState.CLOSED:
            if len(self._failures) >= self.max_failures:
                self._state = CircuitState.OPEN
                self._next_attempt_time = now + self.reset_timeout_seconds
                logger.error("Circuit Breaker transition: CLOSED -> OPEN. Threshold reached.")

    def _record_success(self) -> None:
        """Record a success and update state."""
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failures.clear()
            logger.info("Circuit Breaker transition: HALF_OPEN -> CLOSED. Recovered.")

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute the async function wrapped in the circuit breaker.

        Args:
            func: The async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the function call.

        Raises:
            CircuitOpenError: If the circuit is open.
            Exception: Re-raises any exception from the wrapped function.
        """
        now = time.monotonic()

        # Check if we can execute
        if self._state == CircuitState.OPEN:
            if now >= self._next_attempt_time:
                logger.info("Circuit Breaker probing: OPEN -> HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            # If the exception is NOT a CircuitOpenError (which it shouldn't be here),
            # we consider it a failure of the backend system.
            # We record failure and re-raise.
            # Note: We should verify if we want to catch ALL exceptions.
            # Usually yes for a circuit breaker protecting against network/system failures.
            self._record_failure()
            raise e
