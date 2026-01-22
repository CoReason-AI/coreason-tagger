# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import AsyncMock, patch

import pytest

from coreason_tagger.interfaces import CodexClient
from coreason_tagger.linker import VectorLinker
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy
from coreason_tagger.utils.circuit_breaker import CircuitBreaker, CircuitState


@pytest.fixture
def mock_codex() -> AsyncMock:
    return AsyncMock(spec=CodexClient)


@pytest.fixture
def linker(mock_codex: AsyncMock) -> VectorLinker:
    # Reset Circuit Breaker for each test
    CircuitBreaker._instance = None
    # We use small window for testing
    cb = CircuitBreaker(max_failures=2, window_seconds=10, reset_timeout_seconds=5)
    linker = VectorLinker(codex_client=mock_codex)
    linker.circuit_breaker = cb
    return linker


def create_candidate(text: str) -> EntityCandidate:
    return EntityCandidate(
        text=text,
        label="Symptom",
        start=0,
        end=len(text),
        confidence=1.0,
        source_model="test",
    )


@pytest.mark.asyncio
async def test_linker_circuit_breaker_activates(linker: VectorLinker, mock_codex: AsyncMock) -> None:
    """Test that linker activates circuit breaker after failures and enters Offline Mode."""
    candidate = create_candidate("fever")
    strategy = ExtractionStrategy.SPEED_GLINER

    # 1. Simulate failures
    mock_codex.search.side_effect = Exception("Network Error")

    # Failure 1
    # resolve catches Exception and returns unlinked entity, but CB records failure
    res1 = await linker.resolve(candidate, "context", strategy)
    assert res1.concept_id is None
    assert linker.circuit_breaker.state == CircuitState.CLOSED

    # Failure 2 (Threshold reached)
    res2 = await linker.resolve(candidate, "context", strategy)
    assert res2.concept_id is None
    # Now state should be OPEN (recorded 2 failures)
    assert linker.circuit_breaker.state == CircuitState.OPEN

    # 2. Verify Offline Mode (Circuit OPEN)
    # Next call should NOT call mock_codex.search, should fail fast
    mock_codex.search.reset_mock()

    res3 = await linker.resolve(candidate, "context", strategy)

    # Check result is unlinked
    assert res3.concept_id is None
    # Check codex was NOT called
    mock_codex.search.assert_not_called()


@pytest.mark.asyncio
async def test_linker_circuit_breaker_recovery(linker: VectorLinker, mock_codex: AsyncMock) -> None:
    """Test that linker recovers from Offline Mode."""
    candidate = create_candidate("fever")
    strategy = ExtractionStrategy.SPEED_GLINER

    # Force OPEN
    linker.circuit_breaker._state = CircuitState.OPEN
    linker.circuit_breaker._next_attempt_time = 0.0  # Ready to retry immediately

    # Setup mock to succeed now
    mock_codex.search.side_effect = None
    mock_codex.search.return_value = [{"concept_id": "C1", "concept_name": "Fever", "link_score": 1.0}]

    # Mocking _rerank to avoid model loading issues in unit test
    # (Since we are testing logic flow, not actual embedding)
    with patch.object(linker, "_rerank", new_callable=AsyncMock) as mock_rerank:
        mock_rerank.return_value = {"concept_id": "C1", "concept_name": "Fever", "link_score": 1.0}

        # Call resolve
        res = await linker.resolve(candidate, "context", strategy)

        # Verify call went through
        assert res.concept_id == "C1"
        assert linker.circuit_breaker.state == CircuitState.CLOSED
