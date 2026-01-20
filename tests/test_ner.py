# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import unittest
from unittest.mock import MagicMock, patch

from coreason_tagger.ner import GLiNERExtractor
from coreason_tagger.schema import ExtractedSpan


class TestGLiNERExtractor(unittest.TestCase):
    """Test suite for GLiNERExtractor."""

    @patch("coreason_tagger.ner.GLiNER")
    def test_initialization(self, mock_gliner_class: MagicMock) -> None:
        """Test that the model is initialized with the correct name."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        extractor = GLiNERExtractor(model_name="test-model")

        mock_gliner_class.from_pretrained.assert_called_once_with("test-model")
        self.assertEqual(extractor.model, mock_model_instance)

    @patch("coreason_tagger.ner.GLiNER")
    def test_extract_valid_entities(self, mock_gliner_class: MagicMock) -> None:
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

        results = extractor.extract(text, labels)

        # Verify that the model was called with the default threshold
        mock_model_instance.predict_entities.assert_called_once_with(text, labels, threshold=0.5)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ExtractedSpan)
        self.assertEqual(results[0].text, "headache")
        self.assertEqual(results[0].label, "Symptom")
        self.assertEqual(results[0].start, 0)
        self.assertEqual(results[0].end, 8)
        self.assertEqual(results[0].score, 0.95)

        self.assertEqual(results[1].text, "ibuprofen")
        self.assertEqual(results[1].label, "Drug")

    @patch("coreason_tagger.ner.GLiNER")
    def test_extract_with_custom_threshold(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction with a custom confidence threshold."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance
        mock_model_instance.predict_entities.return_value = []

        extractor = GLiNERExtractor()
        text = "some text"
        labels = ["Symptom"]
        custom_threshold = 0.15

        extractor.extract(text, labels, threshold=custom_threshold)

        # Verify that the model was called with the custom threshold
        mock_model_instance.predict_entities.assert_called_once_with(text, labels, threshold=custom_threshold)

    @patch("coreason_tagger.ner.GLiNER")
    def test_extract_empty_inputs(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction with empty text or labels."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        extractor = GLiNERExtractor()

        # Empty text
        self.assertEqual(extractor.extract("", ["Symptom"]), [])

        # Empty labels
        self.assertEqual(extractor.extract("some text", []), [])

        # Both empty
        self.assertEqual(extractor.extract("", []), [])

        # Ensure model was not called for these cases
        mock_model_instance.predict_entities.assert_not_called()

    @patch("coreason_tagger.ner.GLiNER")
    def test_extract_no_entities_found(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction when model finds nothing."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance
        mock_model_instance.predict_entities.return_value = []

        extractor = GLiNERExtractor()
        results = extractor.extract("Nothing here", ["Symptom"])

        self.assertEqual(results, [])

    @patch("coreason_tagger.ner.GLiNER")
    def test_extract_overlapping_entities(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction of overlapping entities (e.g., 'lung cancer' and 'cancer')."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        mock_model_instance.predict_entities.return_value = [
            {"text": "lung cancer", "label": "Condition", "start": 10, "end": 21, "score": 0.98},
            {"text": "cancer", "label": "Condition", "start": 15, "end": 21, "score": 0.95},
        ]

        extractor = GLiNERExtractor()
        results = extractor.extract("He has lung cancer", ["Condition"])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "lung cancer")
        self.assertEqual(results[1].text, "cancer")
        # Check proper indices
        self.assertEqual(results[0].start, 10)
        self.assertEqual(results[1].start, 15)

    @patch("coreason_tagger.ner.GLiNER")
    def test_extract_complex_scenario(self, mock_gliner_class: MagicMock) -> None:
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

        results = extractor.extract(text, labels)

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].label, "Drug")
        self.assertEqual(results[1].label, "Symptom")
        self.assertEqual(results[2].label, "Drug")
        self.assertEqual(results[3].label, "Condition")

    @patch("coreason_tagger.ner.GLiNER")
    def test_extract_special_characters(self, mock_gliner_class: MagicMock) -> None:
        """Test extraction from text containing special characters and emojis."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        mock_model_instance.predict_entities.return_value = [
            {"text": "COVID-19", "label": "Condition", "start": 9, "end": 17, "score": 0.99},
            {"text": "fever", "label": "Symptom", "start": 21, "end": 26, "score": 0.95},
        ]

        extractor = GLiNERExtractor()
        text = "Diagnosis: COVID-19 🦠, fever 🌡️."
        results = extractor.extract(text, ["Condition", "Symptom"])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "COVID-19")
        self.assertEqual(results[1].text, "fever")

    @patch("coreason_tagger.ner.GLiNER")
    def test_extract_low_confidence(self, mock_gliner_class: MagicMock) -> None:
        """Test that entities with low confidence scores are preserved (not filtered)."""
        mock_model_instance = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model_instance

        mock_model_instance.predict_entities.return_value = [
            {"text": "maybe a cold", "label": "Condition", "start": 0, "end": 12, "score": 0.15},
        ]

        extractor = GLiNERExtractor()
        results = extractor.extract("maybe a cold", ["Condition"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score, 0.15)
