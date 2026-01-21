# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import MagicMock, patch

import pytest
from coreason_tagger.ner import GLiNERExtractor
from coreason_tagger.schema import EntityCandidate


@pytest.mark.asyncio
class TestGLiNERExtractor:
    """Test suite for GLiNERExtractor."""

    @patch("coreason_tagger.ner.GLiNER")
    async def test_initialization(self, mock_gliner_class: MagicMock) -> None:
        """Test that the model is initialized with the correct name."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        extractor = GLiNERExtractor(model_name="test-model")
        # Load explicitly
        await extractor.load_model()

        mock_gliner_class.from_pretrained.assert_called_once_with("test-model")
        assert extractor.model == mock_model_instance

    @patch("coreason_tagger.ner.GLiNER")
    async def test_load_model_idempotency(self, mock_gliner_class: MagicMock) -> None:
        """Test that load_model does not reload if model is already loaded."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        extractor = GLiNERExtractor(model_name="test-model")

        # First load
        await extractor.load_model()
        mock_gliner_class.from_pretrained.assert_called_once()

        # Second load - should do nothing
        await extractor.load_model()
        mock_gliner_class.from_pretrained.assert_called_once()

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_valid_entities(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction with valid return values from the model."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        # Mock the predict_entities method
        mock_model_instance.predict_entities.return_value = [
            {"text": "headache", "label": "Symptom", "start": 0, "end": 8, "score": 0.95},
            {"text": "ibuprofen", "label": "Drug", "start": 20, "end": 29, "score": 0.99},
        ]

        extractor = GLiNERExtractor()
        text = "I have a headache and took ibuprofen."
        labels = ["Symptom", "Drug"]

        results = await extractor.extract(text, labels)

        # Verify that the model was called with the default threshold
        mock_model_instance.predict_entities.assert_called_once_with(text, labels, threshold=0.5)

        assert len(results) == 2
        assert isinstance(results[0], EntityCandidate)
        assert results[0].text == "headache"
        assert results[0].label == "Symptom"
        assert results[0].start == 0
        assert results[0].end == 8
        assert results[0].confidence == 0.95

        assert results[1].text == "ibuprofen"
        assert results[1].label == "Drug"

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_with_custom_threshold(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction with a custom confidence threshold."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance
        mock_model_instance.predict_entities.return_value = []

        extractor = GLiNERExtractor()
        text = "some text"
        labels = ["Symptom"]
        custom_threshold = 0.15

        await extractor.extract(text, labels, threshold=custom_threshold)

        # Verify that the model was called with the custom threshold
        mock_model_instance.predict_entities.assert_called_once_with(text, labels, threshold=custom_threshold)

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_invalid_threshold(self, mock_gliner_class: MagicMock) -> None:
        """Test validation of threshold parameter."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        extractor = GLiNERExtractor()
        text = "some text"
        labels = ["Label"]

        # Test threshold > 1.0
        with pytest.raises(ValueError, match="Threshold must be between"):
            await extractor.extract(text, labels, threshold=1.1)

        # Test threshold < 0.0
        with pytest.raises(ValueError, match="Threshold must be between"):
            await extractor.extract(text, labels, threshold=-0.1)

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_boundary_thresholds(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction with boundary thresholds (0.0 and 1.0)."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance
        mock_model_instance.predict_entities.return_value = []

        extractor = GLiNERExtractor()
        text = "some text"
        labels = ["Label"]

        # Threshold 0.0
        await extractor.extract(text, labels, threshold=0.0)
        mock_model_instance.predict_entities.assert_called_with(text, labels, threshold=0.0)

        # Threshold 1.0
        await extractor.extract(text, labels, threshold=1.0)
        mock_model_instance.predict_entities.assert_called_with(text, labels, threshold=1.0)

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_empty_inputs(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction with empty text or labels."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        extractor = GLiNERExtractor()

        # Empty text
        assert await extractor.extract("", ["Symptom"]) == []

        # Empty labels
        assert await extractor.extract("some text", []) == []

        # Both empty
        assert await extractor.extract("", []) == []

        # Ensure model was not called for these cases
        mock_model_instance.predict_entities.assert_not_called()

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_no_entities_found(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction when model finds nothing."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance
        mock_model_instance.predict_entities.return_value = []

        extractor = GLiNERExtractor()
        results = await extractor.extract("Nothing here", ["Symptom"])

        assert results == []

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_overlapping_entities(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction of overlapping entities (e.g., 'lung cancer' and 'cancer')."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        mock_model_instance.predict_entities.return_value = [
            {"text": "lung cancer", "label": "Condition", "start": 10, "end": 21, "score": 0.98},
            {"text": "cancer", "label": "Condition", "start": 15, "end": 21, "score": 0.95},
        ]

        extractor = GLiNERExtractor()
        results = await extractor.extract("He has lung cancer", ["Condition"])

        assert len(results) == 2
        assert results[0].text == "lung cancer"
        assert results[1].text == "cancer"
        # Check proper indices
        assert results[0].start == 10
        assert results[1].start == 15

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_complex_scenario(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction with multiple entity types in one sentence."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        mock_model_instance.predict_entities.return_value = [
            {"text": "ibuprofen", "label": "Drug", "start": 13, "end": 22, "score": 0.99},
            {"text": "headache", "label": "Symptom", "start": 27, "end": 35, "score": 0.92},
            {"text": "Lisinopril", "label": "Drug", "start": 40, "end": 50, "score": 0.98},
            {"text": "hypertension", "label": "Condition", "start": 55, "end": 67, "score": 0.95},
        ]

        extractor = GLiNERExtractor()
        text = "Patient took ibuprofen for headache and Lisinopril for hypertension."
        labels = ["Drug", "Symptom", "Condition"]

        results = await extractor.extract(text, labels)

        assert len(results) == 4
        assert results[0].label == "Drug"
        assert results[1].label == "Symptom"
        assert results[2].label == "Drug"
        assert results[3].label == "Condition"

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_special_characters(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction from text containing special characters and emojis."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        mock_model_instance.predict_entities.return_value = [
            {"text": "COVID-19", "label": "Condition", "start": 9, "end": 17, "score": 0.99},
            {"text": "fever", "label": "Symptom", "start": 21, "end": 26, "score": 0.95},
        ]

        extractor = GLiNERExtractor()
        text = "Diagnosis: COVID-19 🦠, fever 🌡️."
        results = await extractor.extract(text, ["Condition", "Symptom"])

        assert len(results) == 2
        assert results[0].text == "COVID-19"
        assert results[1].text == "fever"

    @patch("coreason_tagger.ner.GLiNER")
    async def test_extract_low_confidence(self, mock_gliner_class: MagicMock) -> None:
        """Test that entities with low confidence scores are preserved (not filtered)."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        mock_model_instance.predict_entities.return_value = [
            {"text": "maybe a cold", "label": "Condition", "start": 0, "end": 12, "score": 0.15},
        ]

        extractor = GLiNERExtractor()
        results = await extractor.extract("maybe a cold", ["Condition"])

        assert len(results) == 1
        assert results[0].confidence == 0.15
