# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Generator, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_tagger.ner import NuNERExtractor


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Mock the transformers pipeline."""
    pipe = MagicMock()
    return pipe


@pytest.fixture
def nuner_extractor(mock_pipeline: MagicMock) -> Generator[Tuple[NuNERExtractor, MagicMock], None, None]:
    """Fixture for NuNERExtractor with mocked pipeline loading."""
    with patch("coreason_tagger.ner.get_nuner_pipeline", new_callable=AsyncMock) as mock_get_pipe:
        mock_get_pipe.return_value = mock_pipeline
        extractor = NuNERExtractor(model_name="test-nuner")
        yield extractor, mock_pipeline


@pytest.mark.asyncio
async def test_whitespace_stripping_and_offsets(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    """
    Test that the extractor strips whitespace from the entity text
    but preserves the start/end indices provided by the pipeline.
    This ensures that 'text' is clean for linking, while start/end
    still map to the original document for highlighting.
    """
    extractor, mock_pipeline = nuner_extractor

    # Simulate pipeline returning a span with leading space (common in tokenizers)
    # Original text: "Take  aspirin" (start 6, end 13 implies "aspirin")
    # But sometimes pipeline might capture " aspirin" -> start 5, end 13
    mock_pipeline.side_effect = lambda x: [
        {"word": " aspirin ", "entity_group": "DRUG", "start": 5, "end": 14, "score": 0.99},
    ]

    text = "Take  aspirin "
    labels = ["DRUG"]

    candidates = await extractor.extract(text, labels)

    assert len(candidates) == 1
    c = candidates[0]

    # Text should be stripped
    assert c.text == "aspirin"
    # Indices should be preserved (pointing to " aspirin ")
    assert c.start == 5
    assert c.end == 14


@pytest.mark.asyncio
async def test_malformed_pipeline_output(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    """
    Test that the extractor handles missing keys in the pipeline output gracefully.
    """
    extractor, mock_pipeline = nuner_extractor

    mock_pipeline.side_effect = lambda x: [
        {"word": "valid", "entity_group": "TEST", "start": 0, "end": 5, "score": 0.9},
        {
            "word": "missing_score",
            "entity_group": "TEST",
            "start": 6,
            "end": 10,
        },  # Missing score -> defaults to 0.0 -> filtered out if threshold > 0
        {
            "no_word_key": "???",
            "entity_group": "TEST",
            "score": 0.9,
        },  # Missing word -> defaults to "" -> retained but empty text
    ]

    # Set threshold to 0.5, so missing score (0.0) gets dropped
    candidates = await extractor.extract("dummy", ["TEST"], threshold=0.5)

    # "valid" should be present
    assert len(candidates) >= 1
    assert candidates[0].text == "valid"

    # "missing_score" should be dropped (score 0.0 < 0.5)
    assert not any(c.text == "missing_score" for c in candidates)

    # "no_word_key" has score 0.9, so it passes threshold.
    # It defaults text to "".
    # Note: Logic in `ner.py` uses `entity.get("word", "").strip()`.
    # If text is empty, `tagger.py` usually drops it later, but extractor returns it.
    empty_candidates = [c for c in candidates if c.text == ""]
    assert len(empty_candidates) == 1
    assert empty_candidates[0].confidence == 0.9


@pytest.mark.asyncio
async def test_complex_medical_note(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    """
    Simulate a complex medical note with mixed entities, some relevant, some not.
    """
    extractor, mock_pipeline = nuner_extractor

    note = """
    Patient: John Doe.
    Reason: Complains of severe migraine and nausea.
    History: Mother had T2DM.
    Plan: Prescribed Ibuprofen 200mg and advised rest.
    """

    # Mock output:
    # - migraine (SYMPTOM)
    # - nausea (SYMPTOM)
    # - T2DM (DISEASE) -> Not in requested labels?
    # - Ibuprofen (DRUG)
    # - rest (TREATMENT) -> Low confidence

    mock_output = [
        {"word": "migraine", "entity_group": "SYMPTOM", "start": 45, "end": 53, "score": 0.95},
        {"word": "nausea", "entity_group": "SYMPTOM", "start": 58, "end": 64, "score": 0.92},
        {"word": "T2DM", "entity_group": "DISEASE", "start": 88, "end": 92, "score": 0.98},
        {"word": "Ibuprofen", "entity_group": "DRUG", "start": 110, "end": 119, "score": 0.99},
        {"word": "rest", "entity_group": "TREATMENT", "start": 140, "end": 144, "score": 0.4},  # Low score
    ]

    mock_pipeline.side_effect = lambda x: mock_output

    # Requesting SYMPTOM and DRUG. Threshold 0.5.
    labels = ["SYMPTOM", "DRUG"]
    candidates = await extractor.extract(note, labels, threshold=0.5)

    # Expected: migraine, nausea, Ibuprofen.
    # Excluded: T2DM (wrong label), rest (low score).

    texts = sorted([c.text for c in candidates])
    assert texts == ["Ibuprofen", "migraine", "nausea"]

    # Check labels
    for c in candidates:
        if c.text == "Ibuprofen":
            assert c.label == "DRUG"
        else:
            assert c.label == "SYMPTOM"


@pytest.mark.asyncio
async def test_batch_mixed_results(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    """
    Test batch extraction where some documents have no entities, some have errors.
    """
    extractor, mock_pipeline = nuner_extractor

    batch_output = [
        [],  # Doc 1: Empty
        [{"word": "aspirin", "entity_group": "DRUG", "start": 0, "end": 7, "score": 0.9}],  # Doc 2: One entity
        [  # Doc 3: Mixed valid/invalid
            {"word": "pain", "entity_group": "SYMPTOM", "start": 5, "end": 9, "score": 0.8},
            {"word": "noise", "entity_group": "NOISE", "start": 10, "end": 15, "score": 0.9},  # Wrong label
        ],
    ]

    mock_pipeline.side_effect = lambda x: batch_output

    texts = ["", "Take aspirin", "Has pain and noise"]
    labels = ["DRUG", "SYMPTOM"]

    results = await extractor.extract_batch(texts, labels, threshold=0.5)

    assert len(results) == 3
    assert len(results[0]) == 0

    assert len(results[1]) == 1
    assert results[1][0].text == "aspirin"

    assert len(results[2]) == 1
    assert results[2][0].text == "pain"  # noise filtered by label
