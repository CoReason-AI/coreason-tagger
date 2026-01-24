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
from coreason_tagger.registry import get_nuner_pipeline


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
async def test_load_model(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    extractor, mock_pipeline = nuner_extractor

    # First load
    await extractor.load_model()
    assert extractor.model == mock_pipeline

    # Second load (should be no-op)
    with patch("coreason_tagger.ner.get_nuner_pipeline", new_callable=AsyncMock) as mock_get_pipe:
        await extractor.load_model()
        mock_get_pipe.assert_not_called()


@pytest.mark.asyncio
async def test_extract(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    extractor, mock_pipeline = nuner_extractor

    # Mock pipeline output
    mock_pipeline.side_effect = lambda x: [
        {"word": "aspirin", "entity_group": "DRUG", "start": 0, "end": 7, "score": 0.99},
        {"word": "headache", "entity_group": "SYMPTOM", "start": 12, "end": 20, "score": 0.8},
    ]

    text = "Take aspirin for headache."
    labels = ["DRUG", "SYMPTOM"]

    candidates = await extractor.extract(text, labels)

    assert len(candidates) == 2
    assert candidates[0].text == "aspirin"
    assert candidates[0].label == "DRUG"
    assert candidates[0].confidence == 0.99
    assert candidates[1].text == "headache"
    assert candidates[1].label == "SYMPTOM"


@pytest.mark.asyncio
async def test_extract_filtering(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    extractor, mock_pipeline = nuner_extractor

    # Mock pipeline output
    mock_pipeline.side_effect = lambda x: [
        {"word": "aspirin", "entity_group": "DRUG", "start": 0, "end": 7, "score": 0.99},
        {"word": "low conf", "entity_group": "DRUG", "start": 10, "end": 18, "score": 0.1},
    ]

    text = "aspirin low conf"
    labels = ["DRUG"]

    # Test threshold
    candidates = await extractor.extract(text, labels, threshold=0.5)
    assert len(candidates) == 1
    assert candidates[0].text == "aspirin"

    # Test label filtering
    mock_pipeline.side_effect = lambda x: [
        {"word": "aspirin", "entity_group": "DRUG", "start": 0, "end": 7, "score": 0.99},
        {"word": "headache", "entity_group": "SYMPTOM", "start": 12, "end": 20, "score": 0.99},
    ]
    candidates = await extractor.extract(text, ["SYMPTOM"], threshold=0.5)
    assert len(candidates) == 1
    assert candidates[0].label == "SYMPTOM"


@pytest.mark.asyncio
async def test_extract_batch(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    extractor, mock_pipeline = nuner_extractor

    # Mock pipeline output for batch
    # pipeline returns list of lists
    mock_pipeline.side_effect = lambda x: [
        [
            {"word": "aspirin", "entity_group": "DRUG", "start": 0, "end": 7, "score": 0.99},
            {"word": "ignored", "entity_group": "IGNORE", "start": 10, "end": 17, "score": 0.99},
            {"word": "low", "entity_group": "DRUG", "start": 20, "end": 23, "score": 0.1},
        ],
        [],
    ]

    texts = ["Take aspirin.", "Nothing here."]
    labels = ["DRUG"]

    batch_candidates = await extractor.extract_batch(texts, labels, threshold=0.5)

    assert len(batch_candidates) == 2
    assert len(batch_candidates[0]) == 1
    assert batch_candidates[0][0].text == "aspirin"
    assert len(batch_candidates[1]) == 0


@pytest.mark.asyncio
async def test_extract_empty_input(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    extractor, mock_pipeline = nuner_extractor

    candidates = await extractor.extract("", ["DRUG"])
    assert candidates == []

    batch_candidates = await extractor.extract_batch([], ["DRUG"])
    assert batch_candidates == []


@pytest.mark.asyncio
async def test_validate_threshold(nuner_extractor: Tuple[NuNERExtractor, MagicMock]) -> None:
    extractor, _ = nuner_extractor

    with pytest.raises(ValueError):
        await extractor.extract("text", ["label"], threshold=1.5)


@pytest.mark.asyncio
async def test_registry_get_nuner_pipeline() -> None:
    """Test actual registry function (mocking pipeline creation)."""
    with patch("coreason_tagger.registry.pipeline") as mock_pipeline_func:
        mock_pipeline_func.return_value = "mock_pipeline_obj"

        # Clear cache to ensure we run the function
        get_nuner_pipeline.cache_clear()

        pipe = await get_nuner_pipeline("test-model")
        assert pipe == "mock_pipeline_obj"
        mock_pipeline_func.assert_called_once()
