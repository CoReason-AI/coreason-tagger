# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import List

from gliner import GLiNER

from coreason_tagger.interfaces import BaseNERExtractor
from coreason_tagger.schema import ExtractedSpan
from coreason_tagger.utils.logger import logger


class GLiNERExtractor(BaseNERExtractor):
    """
    Zero-Shot NER Extractor using the GLiNER library.
    Wraps the underlying model to provide a clean interface for extracting entities.
    """

    def __init__(self, model_name: str = "urchade/gliner_small-v2.1") -> None:
        """
        Initialize the GLiNER extractor.

        Args:
            model_name (str): The name of the GLiNER model to load.
                             Defaults to "urchade/gliner_small-v2.1" (lightweight).
        """
        self.model_name = model_name
        logger.info(f"Initializing GLiNERExtractor with model: {model_name}")
        # Load the model. Note: This might download weights on first run.
        # In a real production setup, we might want to lazy-load or use a singleton pattern,
        # but for this atomic unit, strict encapsulation is preferred.
        self.model = GLiNER.from_pretrained(model_name)

    def _build_span(self, entity: dict, context: str) -> ExtractedSpan:
        """
        Helper to convert a raw dictionary from GLiNER into an ExtractedSpan.

        Args:
            entity (dict): Raw entity dictionary containing 'text', 'label', 'start', 'end', 'score'.
            context (str): The source text.

        Returns:
            ExtractedSpan: The typed entity span.
        """
        return ExtractedSpan(
            text=entity["text"],
            label=entity["label"],
            start=entity["start"],
            end=entity["end"],
            score=entity["score"],
            context=context,
        )

    def extract(self, text: str, labels: list[str]) -> list[ExtractedSpan]:
        """
        Extract entities from text using the provided labels.

        Args:
            text (str): The input text to process.
            labels (list[str]): A list of entity types to detect.

        Returns:
            list[ExtractedSpan]: A list of detected entity spans.
        """
        if not text or not labels:
            return []

        # GLiNER returns a list of dicts:
        # [{'start': 0, 'end': 5, 'text': '...', 'label': '...', 'score': 0.95}, ...]
        raw_entities = self.model.predict_entities(text, labels)

        return [self._build_span(entity, text) for entity in raw_entities]

    def extract_batch(self, texts: list[str], labels: list[str]) -> list[list[ExtractedSpan]]:
        """
        Extract entities from a batch of texts using the provided labels.

        Args:
            texts (list[str]): The list of input texts to process.
            labels (list[str]): A list of entity types to detect.

        Returns:
            list[list[ExtractedSpan]]: A list of lists, where each inner list contains
                                       detected entity spans for the corresponding text.
        """
        if not texts or not labels:
            return [[] for _ in texts]

        # Use batch_predict_entities if available.
        # batch_predict_entities returns a list of lists of dicts.
        batch_raw_entities = self.model.batch_predict_entities(texts, labels)

        batch_extracted_spans: list[list[ExtractedSpan]] = []

        # Iterate over both the original texts (for context) and the results
        # Use strict=True to ensure the model returns results for every input text
        for text, raw_entities in zip(texts, batch_raw_entities, strict=True):
            extracted_spans = [self._build_span(entity, text) for entity in raw_entities]
            batch_extracted_spans.append(extracted_spans)

        return batch_extracted_spans
