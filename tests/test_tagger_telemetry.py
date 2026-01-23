# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import logging
from typing import Generator
from unittest.mock import AsyncMock

import pytest
from coreason_tagger.interfaces import BaseAssertionDetector, BaseExtractor, BaseLinker
from coreason_tagger.schema import (
    AssertionStatus,
    EntityCandidate,
    ExtractionStrategy,
    LinkedEntity,
)
from coreason_tagger.tagger import CoreasonTagger
from loguru import logger


@pytest.fixture
def mock_ner() -> AsyncMock:
    return AsyncMock(spec=BaseExtractor)


@pytest.fixture
def mock_assertion() -> AsyncMock:
    return AsyncMock(spec=BaseAssertionDetector)


@pytest.fixture
def mock_linker() -> AsyncMock:
    return AsyncMock(spec=BaseLinker)


@pytest.fixture
def tagger(mock_ner: AsyncMock, mock_assertion: AsyncMock, mock_linker: AsyncMock) -> CoreasonTagger:
    return CoreasonTagger(ner=mock_ner, assertion=mock_assertion, linker=mock_linker)


@pytest.fixture(autouse=True)
def propagate_loguru_to_caplog(caplog: pytest.LogCaptureFixture) -> Generator[None, None, None]:
    """
    Redirect loguru logs to the pytest caplog fixture.
    """
    handler_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    yield
    logger.remove(handler_id)


@pytest.mark.asyncio
async def test_telemetry_logs_emitted(
    tagger: CoreasonTagger,
    mock_ner: AsyncMock,
    mock_assertion: AsyncMock,
    mock_linker: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Verify that timing logs are emitted for Extraction, Assertion, and Linking,
    and that the strategy is logged.
    """
    # Configure caplog to capture DEBUG logs as we might use DEBUG for granular steps
    caplog.set_level(logging.DEBUG)

    text = "Patient has headache."
    labels = ["Symptom"]
    strategy = ExtractionStrategy.SPEED_GLINER

    # Mock returns
    candidate = EntityCandidate(
        text="headache",
        label="Symptom",
        start=12,
        end=20,
        confidence=0.9,
        source_model="mock",
    )
    mock_ner.extract.return_value = [candidate]
    mock_assertion.detect.return_value = AssertionStatus.PRESENT
    linked_entity = LinkedEntity(
        **candidate.model_dump(),
        strategy_used=strategy,
        concept_id="HP:123",
        concept_name="Headache",
    )
    mock_linker.resolve.return_value = linked_entity

    # Execute
    await tagger.tag(text, labels, strategy=strategy)

    # Verify Logs
    logs = caplog.text

    # 1. Strategy Log
    assert f"Strategy: {strategy.value}" in logs or f"strategy={strategy.value}" in logs

    # 2. Extraction Timing
    # Look for something like "Extraction took" or "extracted in"
    assert "Extraction" in logs and ("took" in logs or "ms" in logs or "seconds" in logs)

    # 3. Assertion Timing
    assert "Assertion" in logs and ("took" in logs or "ms" in logs or "seconds" in logs)

    # 4. Linking Timing
    assert "Linking" in logs and ("took" in logs or "ms" in logs or "seconds" in logs)


@pytest.mark.asyncio
async def test_telemetry_batch_logs_emitted(
    tagger: CoreasonTagger,
    mock_ner: AsyncMock,
    mock_assertion: AsyncMock,
    mock_linker: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Verify telemetry for batch processing.
    """
    caplog.set_level(logging.DEBUG)
    texts = ["Text 1"]
    labels = ["Symptom"]

    candidate = EntityCandidate(text="E", label="L", start=0, end=1, confidence=1.0, source_model="m")
    mock_ner.extract_batch.return_value = [[candidate]]
    mock_assertion.detect.return_value = AssertionStatus.PRESENT
    mock_linker.resolve.return_value = LinkedEntity(
        **candidate.model_dump(), strategy_used=ExtractionStrategy.SPEED_GLINER, concept_id="C"
    )

    await tagger.tag_batch(texts, labels)

    logs = caplog.text
    assert "Extraction" in logs
    assert "Assertion" in logs
    assert "Linking" in logs
