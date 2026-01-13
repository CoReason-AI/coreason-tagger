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
