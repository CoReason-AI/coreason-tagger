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

from coreason_tagger.assertion_detector import DistilBERTAssertionDetector
from coreason_tagger.schema import AssertionStatus


@pytest.fixture
def mock_pipeline() -> MagicMock:
    return MagicMock()


@pytest.fixture
def detector(mock_pipeline: MagicMock) -> Generator[Tuple[DistilBERTAssertionDetector, MagicMock], None, None]:
    with patch("coreason_tagger.assertion_detector.get_assertion_pipeline", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_pipeline
        det = DistilBERTAssertionDetector(model_name="test-model")
        yield det, mock_pipeline


@pytest.mark.asyncio
async def test_load_model(detector: Tuple[DistilBERTAssertionDetector, MagicMock]) -> None:
    det, mock_pipe = detector
    assert det.model is None
    await det.load_model()
    assert det.model == mock_pipe

    # Test caching (idempotency)
    # We can't easily check the mock call count of get_assertion_pipeline here due to fixture structure,
    # but we can ensure it doesn't crash or change the model.
    await det.load_model()
    assert det.model == mock_pipe


@pytest.mark.asyncio
async def test_detect_mapping_absent(detector: Tuple[DistilBERTAssertionDetector, MagicMock]) -> None:
    det, mock_pipe = detector
    # Mock pipeline output
    mock_pipe.side_effect = lambda x, **kwargs: [{"label": "absent", "score": 0.99}]

    status = await det.detect("Patient has no fever", "fever", 15, 20)

    assert status == AssertionStatus.ABSENT
    mock_pipe.assert_called_once()
    args, _ = mock_pipe.call_args
    # Verify input formatting: "Patient has no [entity] fever [/entity]"
    assert "[entity] fever [/entity]" in args[0]
    assert "Patient has no" in args[0]


@pytest.mark.asyncio
async def test_detect_mapping_family(detector: Tuple[DistilBERTAssertionDetector, MagicMock]) -> None:
    det, mock_pipe = detector
    mock_pipe.side_effect = lambda x, **kwargs: [{"label": "associated_with_someone_else", "score": 0.9}]

    status = await det.detect("Mother had cancer", "cancer", 11, 17)
    assert status == AssertionStatus.FAMILY


@pytest.mark.asyncio
async def test_detect_mapping_history(detector: Tuple[DistilBERTAssertionDetector, MagicMock]) -> None:
    det, mock_pipe = detector
    mock_pipe.side_effect = lambda x, **kwargs: [{"label": "history", "score": 0.9}]

    status = await det.detect("History of cancer", "cancer", 11, 17)
    assert status == AssertionStatus.HISTORY


@pytest.mark.asyncio
async def test_detect_mapping_fallback_label(detector: Tuple[DistilBERTAssertionDetector, MagicMock]) -> None:
    det, mock_pipe = detector
    # LABEL_0 -> ABSENT in our map
    mock_pipe.side_effect = lambda x, **kwargs: [{"label": "LABEL_0", "score": 0.9}]

    status = await det.detect("text", "span", 0, 4)
    assert status == AssertionStatus.ABSENT


@pytest.mark.asyncio
async def test_detect_unknown_label(detector: Tuple[DistilBERTAssertionDetector, MagicMock]) -> None:
    det, mock_pipe = detector
    mock_pipe.side_effect = lambda x, **kwargs: [{"label": "unknown_junk", "score": 0.9}]

    status = await det.detect("text", "span", 0, 4)
    assert status == AssertionStatus.PRESENT  # Default


@pytest.mark.asyncio
async def test_detect_empty_result(detector: Tuple[DistilBERTAssertionDetector, MagicMock]) -> None:
    det, mock_pipe = detector
    mock_pipe.side_effect = lambda x, **kwargs: []

    status = await det.detect("text", "span", 0, 4)
    assert status == AssertionStatus.PRESENT
