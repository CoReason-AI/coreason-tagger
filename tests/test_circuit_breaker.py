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

import pytest
from coreason_tagger.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    # Reset singleton instance for each test
    CircuitBreaker._instance = None
    cb = CircuitBreaker(max_failures=5, window_seconds=10, reset_timeout_seconds=30)
    return cb


async def successful_func() -> str:
    return "success"


async def failing_func() -> None:
    raise ValueError("Failed")


@pytest.mark.asyncio
async def test_circuit_breaker_happy_path(circuit_breaker: CircuitBreaker) -> None:
    """Test normal operation (CLOSED state)."""
    assert circuit_breaker.state == CircuitState.CLOSED
    result = await circuit_breaker.call(successful_func)
    assert result == "success"
    assert circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures(circuit_breaker: CircuitBreaker) -> None:
    """Test that circuit opens after threshold failures."""
    # 4 Failures (Threshold is 5)
    for _ in range(4):
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)
        assert circuit_breaker.state == CircuitState.CLOSED

    # 5th Failure
    with pytest.raises(ValueError):
        await circuit_breaker.call(failing_func)

    assert circuit_breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_when_open(circuit_breaker: CircuitBreaker) -> None:
    """Test that calls are rejected immediately when OPEN."""
    # Force OPEN state
    circuit_breaker._state = CircuitState.OPEN
    # Ensure next attempt is in the future
    with patch("time.monotonic", return_value=100.0):
        circuit_breaker._next_attempt_time = 200.0

        with pytest.raises(CircuitOpenError):
            await circuit_breaker.call(successful_func)


@pytest.mark.asyncio
async def test_circuit_breaker_recovery(circuit_breaker: CircuitBreaker) -> None:
    """Test transition from OPEN -> HALF_OPEN -> CLOSED."""
    # Force OPEN state
    circuit_breaker._state = CircuitState.OPEN

    with patch("time.monotonic", return_value=100.0):
        circuit_breaker._next_attempt_time = 130.0  # 30s timeout

    # Time passes (timeout expired)
    with patch("time.monotonic", return_value=140.0):
        # Should transition to HALF_OPEN and call function
        result = await circuit_breaker.call(successful_func)
        assert result == "success"
        # Should be CLOSED after success
        assert circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_relapse(circuit_breaker: CircuitBreaker) -> None:
    """Test transition from OPEN -> HALF_OPEN -> OPEN (on failure)."""
    # Force OPEN state
    circuit_breaker._state = CircuitState.OPEN

    with patch("time.monotonic", return_value=100.0):
        circuit_breaker._next_attempt_time = 130.0

    # Time passes (timeout expired)
    with patch("time.monotonic", return_value=140.0):
        # Should transition to HALF_OPEN, attempt call, fail, and go back to OPEN
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN
        # Check that timeout reset (we can't check exact value easily due to internal time call,
        # but we can verify it's pushed out)
        # In implementation: _next_attempt_time = now + reset_timeout
        assert circuit_breaker._next_attempt_time == 140.0 + 30.0


@pytest.mark.asyncio
async def test_sliding_window_expiration(circuit_breaker: CircuitBreaker) -> None:
    """Test that old failures expire and don't contribute to threshold."""
    # 4 Failures at t=100
    with patch("time.monotonic", return_value=100.0):
        for _ in range(4):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

    assert circuit_breaker.state == CircuitState.CLOSED
    assert len(circuit_breaker._failures) == 4

    # Time passes (window is 10s), so at t=111, old failures should expire
    with patch("time.monotonic", return_value=111.0):
        # 1 Failure (Threshold is 5)
        # Logic: Record failure, THEN clean window.
        # But `_clean_window` checks `_failures[0] < now - window`.
        # 100 < 111 - 10 (101) -> True. Old failures expire.

        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)

        # Should still be CLOSED because old 4 expired
        assert circuit_breaker.state == CircuitState.CLOSED
        # Should have 1 failure now
        assert len(circuit_breaker._failures) == 1


@pytest.mark.asyncio
async def test_singleton_behavior() -> None:
    """Test that multiple instantiations return the same object."""
    CircuitBreaker._instance = None
    cb1 = CircuitBreaker()
    cb2 = CircuitBreaker()
    assert cb1 is cb2

    cb1.max_failures = 99
    assert cb2.max_failures == 99
