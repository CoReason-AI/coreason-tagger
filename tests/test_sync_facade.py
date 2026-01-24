# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coreason_tagger.schema import ExtractionStrategy, LinkedEntity
from coreason_tagger.tagger import CoreasonTagger


@pytest.fixture
def mock_async_tagger() -> Generator[MagicMock, None, None]:
    with patch("coreason_tagger.tagger.CoreasonTaggerAsync") as MockClass:
        mock_instance = MockClass.return_value
        # Setup async context manager
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)

        # Setup async methods
        mock_instance.tag = AsyncMock()
        mock_instance.tag_batch = AsyncMock()

        yield mock_instance


def test_sync_facade_context_manager(mock_async_tagger: MagicMock) -> None:
    """Test that the sync context manager correctly drives the async one via anyio."""
    with CoreasonTagger() as tagger:
        assert isinstance(tagger, CoreasonTagger)

    # Verify async context manager methods were called
    mock_async_tagger.__aenter__.assert_called_once()
    mock_async_tagger.__aexit__.assert_called_once()


def test_sync_facade_tag(mock_async_tagger: MagicMock) -> None:
    """Test blocking tag method."""
    # Setup mock return
    mock_entity = LinkedEntity(
        text="test",
        label="L",
        start=0,
        end=4,
        confidence=1.0,
        source_model="m",
        strategy_used=ExtractionStrategy.SPEED_GLINER,
    )
    mock_async_tagger.tag.return_value = [mock_entity]

    with CoreasonTagger() as tagger:
        results = tagger.tag("test text", ["L"])

    assert len(results) == 1
    assert results[0] == mock_entity
    # Verify async call
    mock_async_tagger.tag.assert_called_once_with("test text", ["L"], ExtractionStrategy.SPEED_GLINER)


def test_sync_facade_tag_batch(mock_async_tagger: MagicMock) -> None:
    """Test blocking tag_batch method."""
    mock_async_tagger.tag_batch.return_value = []

    with CoreasonTagger() as tagger:
        results = tagger.tag_batch(["t1"], ["L"])

    assert results == []
    # Verify async call
    mock_async_tagger.tag_batch.assert_called_once_with(["t1"], ["L"], ExtractionStrategy.SPEED_GLINER)
