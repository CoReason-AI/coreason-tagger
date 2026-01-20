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

    mock_gliner_model.batch_predict_entities.assert_called_once_with(texts, labels, threshold=0.5)


def test_gliner_extract_batch_with_custom_threshold(mock_gliner_model: MagicMock) -> None:
    """Test batch extraction with a custom confidence threshold."""
    extractor = GLiNERExtractor()
    texts = ["Patient has fever."]
    labels = ["Symptom"]
    custom_threshold = 0.2

    # Mock return value: Empty list of entities for the single input text
    mock_gliner_model.batch_predict_entities.return_value = [[]]

    extractor.extract_batch(texts, labels, threshold=custom_threshold)

    mock_gliner_model.batch_predict_entities.assert_called_once_with(texts, labels, threshold=custom_threshold)


def test_gliner_extract_batch_invalid_threshold(mock_gliner_model: MagicMock) -> None:
    """Test batch extraction with invalid threshold."""
    extractor = GLiNERExtractor()
    texts = ["Patient has fever."]
    labels = ["Symptom"]

    # Test > 1.0
    with pytest.raises(ValueError, match="Threshold must be between"):
        extractor.extract_batch(texts, labels, threshold=1.5)

    # Test < 0.0
    with pytest.raises(ValueError, match="Threshold must be between"):
        extractor.extract_batch(texts, labels, threshold=-0.1)


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


def test_gliner_extract_batch_duplicate_inputs(mock_gliner_model: MagicMock) -> None:
    """Test batch extraction with duplicate input texts to ensure 1:1 mapping."""
    extractor = GLiNERExtractor()
    texts = ["Dup", "Dup"]
    labels = ["Label"]

    # Simulating model finding entities independently for each call
    mock_gliner_model.batch_predict_entities.return_value = [
        [{"text": "Dup", "label": "Label", "start": 0, "end": 3, "score": 0.95}],
        [{"text": "Dup", "label": "Label", "start": 0, "end": 3, "score": 0.95}],
    ]

    results = extractor.extract_batch(texts, labels)

    assert len(results) == 2
    # Verify both have results
    assert len(results[0]) == 1
    assert len(results[1]) == 1
    # Verify strict order/context mapping
    assert results[0][0].context == "Dup"
    assert results[1][0].context == "Dup"
    # Ensure they are distinct objects
    assert results[0][0] is not results[1][0]


def test_gliner_extract_batch_complex_mix(mock_gliner_model: MagicMock) -> None:
    """Test a mix of valid texts, empty strings, and special characters."""
    extractor = GLiNERExtractor()
    texts = ["Normal text", "", "Special: @#$%", "Multiple types"]
    labels = ["TypeA", "TypeB"]

    mock_gliner_model.batch_predict_entities.return_value = [
        [{"text": "Normal", "label": "TypeA", "start": 0, "end": 6, "score": 0.9}],
        [],  # Empty input -> likely empty output
        [{"text": "@#$%", "label": "TypeB", "start": 9, "end": 13, "score": 0.8}],
        [
            {"text": "Multiple", "label": "TypeA", "start": 0, "end": 8, "score": 0.9},
            {"text": "types", "label": "TypeB", "start": 9, "end": 14, "score": 0.85},
        ],
    ]

    results = extractor.extract_batch(texts, labels)

    assert len(results) == 4
    # 1. Normal
    assert len(results[0]) == 1
    assert results[0][0].text == "Normal"
    # 2. Empty string input
    assert len(results[1]) == 0
    # 3. Special chars
    assert len(results[2]) == 1
    assert results[2][0].text == "@#$%"
    # 4. Multiple entities
    assert len(results[3]) == 2
    assert results[3][0].label == "TypeA"
    assert results[3][1].label == "TypeB"


def test_gliner_extract_batch_overlapping_spans(mock_gliner_model: MagicMock) -> None:
    """Test that overlapping spans (nested entities) are preserved."""
    extractor = GLiNERExtractor()
    texts = ["Patient has lung cancer."]
    labels = ["Condition", "BodyPart"]

    # Model returns "lung cancer" (Condition) and "lung" (BodyPart)
    mock_gliner_model.batch_predict_entities.return_value = [
        [
            {"text": "lung cancer", "label": "Condition", "start": 12, "end": 23, "score": 0.95},
            {"text": "lung", "label": "BodyPart", "start": 12, "end": 16, "score": 0.85},
        ]
    ]

    results = extractor.extract_batch(texts, labels)

    assert len(results) == 1
    assert len(results[0]) == 2
    # Verify both are present
    labels_found = {span.label for span in results[0]}
    texts_found = {span.text for span in results[0]}
    assert "Condition" in labels_found
    assert "BodyPart" in labels_found
    assert "lung cancer" in texts_found
    assert "lung" in texts_found


def test_gliner_extract_batch_length_mismatch(mock_gliner_model: MagicMock) -> None:
    """
    Test that ValueError is raised if the number of results returned by the model
    does not match the number of input texts (validating strict=True in zip).
    """
    extractor = GLiNERExtractor()
    texts = ["One", "Two"]
    labels = ["Label"]

    # Model returns only 1 result list instead of 2
    mock_gliner_model.batch_predict_entities.return_value = [
        [{"text": "One", "label": "Label", "start": 0, "end": 3, "score": 0.9}],
    ]

    with pytest.raises(ValueError):
        extractor.extract_batch(texts, labels)
