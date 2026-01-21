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
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from coreason_tagger.interfaces import BaseAssertionDetector, BaseLinker, BaseNERExtractor
from coreason_tagger.schema import (
    AssertionStatus,
    EntityCandidate,
    ExtractionStrategy,
    LinkedEntity,
)
from coreason_tagger.tagger import CoreasonTagger


@pytest.fixture
def mock_components() -> tuple[MagicMock, MagicMock, MagicMock]:
    ner = MagicMock(spec=BaseNERExtractor)
    ner.extract = AsyncMock()
    ner.extract_batch = AsyncMock()

    assertion = MagicMock(spec=BaseAssertionDetector)
    assertion.detect = AsyncMock(return_value=AssertionStatus.PRESENT)

    linker = MagicMock(spec=BaseLinker)
    linker.resolve = AsyncMock()

    return ner, assertion, linker


@pytest.mark.asyncio
async def test_async_concurrency_speed(mock_components: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """
    Test A: Concurrency Verification.
    Process 10 entities. Each linker call sleeps 0.1s.
    Sequential time: 1.0s.
    Expected Async time: ~0.1s + overhead.
    """
    ner, assertion, linker = mock_components
    tagger = CoreasonTagger(ner=ner, assertion=assertion, linker=linker)

    # 1. Setup: NER returns 10 candidates
    candidates = [
        EntityCandidate(text=f"ent{i}", label="Test", start=0, end=3, confidence=1.0, source_model="mock")
        for i in range(10)
    ]
    ner.extract.return_value = candidates

    # 2. Setup: Linker sleeps 0.1s then returns result
    async def slow_resolve(entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        await asyncio.sleep(0.1)
        return LinkedEntity(
            **entity.model_dump(),
            strategy_used=strategy,
            concept_id=f"ID-{entity.text}",
        )

    linker.resolve.side_effect = slow_resolve

    # 3. Execution
    start_time = time.perf_counter()
    results = await tagger.tag("dummy text", ["Test"])
    end_time = time.perf_counter()
    duration = end_time - start_time

    # 4. Assertion
    assert len(results) == 10
    # Should be much faster than sequential (1.0s)
    # Allow some buffer for overhead (0.3s is generous but safe for CI)
    assert duration < 0.4
    print(f"\nProcessed 10 items in {duration:.4f}s (Expected < 0.4s)")


@pytest.mark.asyncio
async def test_batch_order_preservation_with_variable_delays(
    mock_components: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """
    Test B: Order Preservation.
    Input: [Text A, Text B, Text C]
    Text A processing takes 0.2s
    Text B processing takes 0.0s
    Text C processing takes 0.1s

    Output order MUST be [Result A, Result B, Result C].
    """
    ner, assertion, linker = mock_components
    tagger = CoreasonTagger(ner=ner, assertion=assertion, linker=linker)

    texts = ["A", "B", "C"]

    # NER returns 1 candidate per text
    c1 = EntityCandidate(text="entA", label="L", start=0, end=1, confidence=1.0, source_model="mock")
    c2 = EntityCandidate(text="entB", label="L", start=0, end=1, confidence=1.0, source_model="mock")
    c3 = EntityCandidate(text="entC", label="L", start=0, end=1, confidence=1.0, source_model="mock")

    ner.extract_batch.return_value = [[c1], [c2], [c3]]

    # Assertion delays based on input text context
    async def variable_delay_detect(text: str, **kwargs: Any) -> AssertionStatus:
        if text == "A":
            await asyncio.sleep(0.2)
        elif text == "C":
            await asyncio.sleep(0.1)
        return AssertionStatus.PRESENT

    assertion.detect.side_effect = variable_delay_detect

    # Linker must return entity based on input to identify it
    async def dynamic_resolve(entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        return LinkedEntity(**entity.model_dump(), strategy_used=strategy, concept_id="ID")

    linker.resolve.side_effect = dynamic_resolve

    results = await tagger.tag_batch(texts, ["L"])

    assert len(results) == 3
    # Result 0 should correspond to input 0 (entA), despite being the slowest
    assert results[0][0].text == "entA"
    # Result 1 (entB) was fastest
    assert results[1][0].text == "entB"
    # Result 2 (entC)
    assert results[2][0].text == "entC"


@pytest.mark.asyncio
async def test_exception_propagation(mock_components: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """
    Test C: Exception Handling.
    If one item in the batch fails, the whole operation should raise that exception
    (fail-fast), rather than swallowing it or returning partial bad data.
    """
    ner, assertion, linker = mock_components
    tagger = CoreasonTagger(ner=ner, assertion=assertion, linker=linker)

    # 2 candidates
    c1 = EntityCandidate(text="good", label="L", start=0, end=1, confidence=1.0, source_model="mock")
    c2 = EntityCandidate(text="bad", label="L", start=0, end=1, confidence=1.0, source_model="mock")
    ner.extract.return_value = [c1, c2]

    # Linker fails for "bad"
    async def faulty_resolve(entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        if entity.text == "bad":
            raise ValueError("Simulated Linker Error")
        return LinkedEntity(**entity.model_dump(), strategy_used=strategy, concept_id="ID")

    linker.resolve.side_effect = faulty_resolve

    with pytest.raises(ValueError, match="Simulated Linker Error"):
        await tagger.tag("text", ["L"])
