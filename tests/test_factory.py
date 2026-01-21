# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import AsyncMock, MagicMock

import pytest

from coreason_tagger.ner import ExtractorFactory, GLiNERExtractor, NuNERExtractor
from coreason_tagger.schema import ExtractionStrategy
from coreason_tagger.tagger import CoreasonTagger


def test_extractor_factory_strategies() -> None:
    """Test that factory returns correct extractor for each strategy."""
    factory = ExtractorFactory()

    # Test GLiNER
    extractor_speed = factory.get_extractor(ExtractionStrategy.SPEED_GLINER)
    assert isinstance(extractor_speed, GLiNERExtractor)

    # Test NuNER
    extractor_precision = factory.get_extractor(ExtractionStrategy.PRECISION_NUNER)
    assert isinstance(extractor_precision, NuNERExtractor)

    # Test Fallback (Reasoning) -> currently defaults to GLiNER
    extractor_reasoning = factory.get_extractor(ExtractionStrategy.REASONING_LLM)
    assert isinstance(extractor_reasoning, GLiNERExtractor)


def test_extractor_factory_caching() -> None:
    """Test that factory caches extractor instances."""
    factory = ExtractorFactory()

    e1 = factory.get_extractor(ExtractionStrategy.SPEED_GLINER)
    e2 = factory.get_extractor(ExtractionStrategy.SPEED_GLINER)

    assert e1 is e2


@pytest.mark.asyncio
async def test_tagger_with_factory() -> None:
    """Test CoreasonTagger using ExtractorFactory."""
    factory = ExtractorFactory()
    mock_assertion = AsyncMock()
    mock_linker = AsyncMock()

    tagger = CoreasonTagger(factory, mock_assertion, mock_linker)

    # Tag with SPEED_GLINER
    # We need to mock the extract method of the extractor returned by factory
    # But factory returns real instances. We can mock the factory.

    mock_factory = MagicMock(spec=ExtractorFactory)
    mock_extractor = AsyncMock()
    mock_extractor.extract.return_value = []
    # Mock extract_batch to return a list of lists corresponding to inputs
    # Input is ["test"], so return [[]] (one empty list of candidates)
    mock_extractor.extract_batch.return_value = [[]]

    mock_factory.get_extractor.return_value = mock_extractor

    tagger = CoreasonTagger(mock_factory, mock_assertion, mock_linker)

    await tagger.tag("test", ["Label"], strategy=ExtractionStrategy.SPEED_GLINER)

    mock_factory.get_extractor.assert_called_with(ExtractionStrategy.SPEED_GLINER)
    mock_extractor.extract.assert_called_once()

    # Tag Batch
    await tagger.tag_batch(["test"], ["Label"], strategy=ExtractionStrategy.PRECISION_NUNER)
    mock_factory.get_extractor.assert_called_with(ExtractionStrategy.PRECISION_NUNER)
    mock_extractor.extract_batch.assert_called_once()


@pytest.mark.asyncio
async def test_tagger_legacy_init() -> None:
    """Test CoreasonTagger initialized with single extractor (legacy)."""
    mock_extractor = AsyncMock()
    mock_extractor.extract.return_value = []
    mock_assertion = AsyncMock()
    mock_linker = AsyncMock()

    tagger = CoreasonTagger(mock_extractor, mock_assertion, mock_linker)

    # Strategy arg should be ignored for selection, but passed to processing
    await tagger.tag("test", ["Label"], strategy=ExtractionStrategy.PRECISION_NUNER)

    mock_extractor.extract.assert_called_once()
    # Ensure it didn't crash trying to call get_extractor on the mock
