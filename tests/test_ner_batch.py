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
from unittest.mock import MagicMock, patch

import pytest
from coreason_tagger.ner import GLiNERExtractor
from coreason_tagger.schema import ExtractedSpan


@pytest.fixture
def mock_gliner_model() -> Generator[MagicMock, None, None]:
    with patch("coreason_tagger.ner.GLiNER.from_pretrained") as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        yield mock_model


def test_gliner_extract_batch_success(mock_gliner_model: MagicMock) -> None:
    """Test standard batch extraction."""
    extractor = GLiNERExtractor()
    texts = ["Patient has fever.", "No symptoms reported."]
    labels = ["Symptom"]

    # Mock return value for batch_predict_entities
    mock_gliner_model.batch_predict_entities.return_value = [
        [{"text": "fever", "label": "Symptom", "start": 12, "end": 17, "score": 0.99}],
        [],
    ]

    results = extractor.extract_batch(texts, labels)

    assert len(results) == 2
    assert len(results[0]) == 1
    assert len(results[1]) == 0

    span = results[0][0]
    assert isinstance(span, ExtractedSpan)
    assert span.text == "fever"
    assert span.label == "Symptom"
    assert span.context == "Patient has fever."

    mock_gliner_model.batch_predict_entities.assert_called_once_with(texts, labels)


def test_gliner_extract_batch_empty_input(mock_gliner_model: MagicMock) -> None:
    """Test batch extraction with empty input list."""
    extractor = GLiNERExtractor()
    results = extractor.extract_batch([], ["Symptom"])
    assert results == []
    mock_gliner_model.batch_predict_entities.assert_not_called()


def test_gliner_extract_batch_no_labels(mock_gliner_model: MagicMock) -> None:
    """Test batch extraction with no labels."""
    extractor = GLiNERExtractor()
    texts = ["Some text"]
    results = extractor.extract_batch(texts, [])
    # Should return empty list for each text
    assert results == [[]]
    mock_gliner_model.batch_predict_entities.assert_not_called()


def test_gliner_extract_batch_order_preservation(mock_gliner_model: MagicMock) -> None:
    """Test that output order matches input order strictly."""
    extractor = GLiNERExtractor()
    texts = ["One", "Two", "Three"]
    labels = ["Label"]

    mock_gliner_model.batch_predict_entities.return_value = [
        [{"text": "One", "label": "Label", "start": 0, "end": 3, "score": 0.9}],
        [],
        [{"text": "Three", "label": "Label", "start": 0, "end": 5, "score": 0.8}],
    ]

    results = extractor.extract_batch(texts, labels)

    assert len(results) == 3
    assert results[0][0].text == "One"
    assert results[1] == []
    assert results[2][0].text == "Three"
