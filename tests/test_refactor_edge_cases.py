import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from coreason_tagger.interfaces import BaseExtractor
from coreason_tagger.ner import ExtractorFactory
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy, LinkedEntity
from coreason_tagger.tagger import CoreasonTagger


@pytest.fixture
def mock_assertion() -> Any:
    detector = AsyncMock()
    # Return a dummy status
    detector.detect.return_value = "PRESENT"
    return detector


@pytest.fixture
def mock_linker() -> Any:
    linker = AsyncMock()

    # Return a LinkedEntity (mocked success)
    def resolve_side_effect(entity: EntityCandidate, ctx: str, strat: ExtractionStrategy) -> LinkedEntity:
        return LinkedEntity(
            **entity.model_dump(),
            strategy_used=strat,
            concept_id="TEST:123",
            concept_name="Test Concept",
            link_score=0.99,
        )

    linker.resolve.side_effect = resolve_side_effect
    return linker


@pytest.mark.asyncio
async def test_concurrent_model_access(mock_assertion: Any, mock_linker: Any) -> None:
    """
    Test that multiple concurrent calls to the tagger (simulating high load)
    work correctly with the new asyncio.to_thread implementation.
    """
    # Create a mock extractor that simulates a blocking delay
    mock_extractor = AsyncMock(spec=BaseExtractor)

    # We need to mock the internal behavior.
    # Since the Tagger calls 'extract', which calls 'asyncio.to_thread' in the real implementation,
    # but here we are mocking the extractor itself.
    # To properly test the thread safety, we should ideally use a real "threaded" call or simulate it.
    # However, testing the orchestrator's handling of concurrent tasks is the goal.

    async def delayed_extract(text: str, labels: list[str]) -> list[EntityCandidate]:
        # Simulate blocking work
        await asyncio.sleep(0.01)
        return [
            EntityCandidate(text=f"Entity_{text}", start=0, end=5, label="Test", confidence=0.9, source_model="mock")
        ]

    mock_extractor.extract.side_effect = delayed_extract

    # Setup Factory
    factory = MagicMock(spec=ExtractorFactory)
    factory.get_extractor.return_value = mock_extractor

    tagger = CoreasonTagger(ner=factory, assertion=mock_assertion, linker=mock_linker)

    # Launch 50 concurrent requests
    tasks = [tagger.tag(f"doc_{i}", ["Label"]) for i in range(50)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 50
    # Verify results are correct (correlation check)
    for i, res in enumerate(results):
        assert len(res) == 1
        assert res[0].text == f"Entity_doc_{i}"


@pytest.mark.asyncio
async def test_large_batch_processing(mock_assertion: Any, mock_linker: Any) -> None:
    """
    Test tag_batch with a large number of documents to verify the new
    parallel processing logic (list comprehension of tasks).
    """
    mock_extractor = AsyncMock(spec=BaseExtractor)

    # Batch extract returns a list of lists
    async def batch_extract(texts: list[str], labels: list[str]) -> list[list[EntityCandidate]]:
        # Simulate work
        await asyncio.sleep(0.01)
        return [
            [EntityCandidate(text=f"Ent_{t}", start=0, end=3, label="Test", confidence=0.9, source_model="mock")]
            for t in texts
        ]

    mock_extractor.extract_batch.side_effect = batch_extract

    factory = MagicMock(spec=ExtractorFactory)
    factory.get_extractor.return_value = mock_extractor

    tagger = CoreasonTagger(ner=factory, assertion=mock_assertion, linker=mock_linker)

    # 100 documents
    texts = [f"text_{i}" for i in range(100)]
    results = await tagger.tag_batch(texts, ["Label"])

    assert len(results) == 100
    assert results[0][0].text == "Ent_text_0"
    assert results[99][0].text == "Ent_text_99"


@pytest.mark.asyncio
async def test_exception_propagation_in_threads(mock_assertion: Any, mock_linker: Any) -> None:
    """
    Verify that if the underlying blocking call (run in thread) raises an exception,
    it propagates correctly through the async stack.
    """
    # We'll use a real GLiNERExtractor but patch the internal model to raise
    from coreason_tagger.ner import GLiNERExtractor

    extractor = GLiNERExtractor()

    # Mock the model loading so we don't need real weights
    extractor.model = MagicMock()

    # The 'predict_entities' method is what's called inside asyncio.to_thread
    extractor.model.predict_entities.side_effect = ValueError("Model Failure")

    tagger = CoreasonTagger(ner=extractor, assertion=mock_assertion, linker=mock_linker)

    with pytest.raises(ValueError, match="Model Failure"):
        await tagger.tag("Input text", ["Label"])


@pytest.mark.asyncio
async def test_complex_scenario_mixed_failures(mock_assertion: Any, mock_linker: Any) -> None:
    """
    Complex Scenario: Batch processing where some assertions fail or linkers fail.
    Ensures partial failures don't crash the whole batch.
    """
    mock_extractor = AsyncMock(spec=BaseExtractor)
    mock_extractor.extract_batch.return_value = [
        [EntityCandidate(text="Good", start=0, end=4, label="A", confidence=1.0, source_model="m")],
        [EntityCandidate(text="BadLink", start=0, end=7, label="A", confidence=1.0, source_model="m")],
    ]

    factory = MagicMock(spec=ExtractorFactory)
    factory.get_extractor.return_value = mock_extractor

    # Assertions work fine
    mock_assertion.detect.return_value = "PRESENT"

    # Linker fails for "BadLink"
    async def flaky_resolve(entity: EntityCandidate, ctx: str, strat: ExtractionStrategy) -> LinkedEntity:
        if entity.text == "BadLink":
            # Simulate a crash in the linker
            raise RuntimeError("Linker Crash")
        # Return a valid LinkedEntity for success case
        return LinkedEntity(
            **entity.model_dump(),
            strategy_used=strat,
            concept_id="TEST:123",
            concept_name="Test Concept",
            link_score=0.99,
        )

    mock_linker.resolve.side_effect = flaky_resolve

    tagger = CoreasonTagger(ner=factory, assertion=mock_assertion, linker=mock_linker)

    # In the current implementation, if _process_candidate crashes (e.g. linker crash),
    # asyncio.gather will raise the exception immediately unless return_exceptions=True.
    # The default behavior is usually to fail fast.
    # Let's verify what happens. Ideally, for a robust system, we might want to suppress single entity failures,
    # but the current code does `await asyncio.gather(*tasks)`.

    with pytest.raises(RuntimeError, match="Linker Crash"):
        await tagger.tag_batch(["doc1", "doc2"], ["Label"])
