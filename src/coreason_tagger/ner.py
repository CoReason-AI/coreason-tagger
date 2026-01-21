# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Any, Optional

from gliner import GLiNER
from loguru import logger

from coreason_tagger.config import settings
from coreason_tagger.interfaces import BaseNERExtractor
from coreason_tagger.schema import EntityCandidate


class GLiNERExtractor(BaseNERExtractor):
    """
    Zero-Shot NER Extractor using the GLiNER library.
    Wraps the underlying model to provide a clean interface for extracting entities.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the GLiNER extractor.

        Args:
            model_name (str, optional): The name of the GLiNER model to load.
                                        If None, uses the value from settings.NER_MODEL_NAME.
        """
        self.model_name = model_name or settings.NER_MODEL_NAME
        logger.info(f"Initializing GLiNERExtractor with model: {self.model_name}")
        # Load the model. Note: This might download weights on first run.
        self.model = GLiNER.from_pretrained(self.model_name)

    def _build_candidate(self, entity: dict[str, Any]) -> EntityCandidate:
        """
        Helper to convert a raw dictionary from GLiNER into an EntityCandidate.

        Args:
            entity (dict[str, Any]): Raw entity dictionary containing 'text', 'label', 'start', 'end', 'score'.

        Returns:
            EntityCandidate: The typed entity candidate.
        """
        return EntityCandidate(
            text=entity["text"],
            label=entity["label"],
            start=entity["start"],
            end=entity["end"],
            confidence=entity["score"],
            source_model=self.model_name,
        )

    def _validate_threshold(self, threshold: float) -> None:
        """
        Validate that the threshold is within the valid range [0.0, 1.0].

        Args:
            threshold (float): The threshold value to check.

        Raises:
            ValueError: If the threshold is out of bounds.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    def extract(self, text: str, labels: list[str], threshold: float = 0.5) -> list[EntityCandidate]:
        """
        Extract entities from text using the provided labels.

        Args:
            text (str): The input text to process.
            labels (list[str]): A list of entity types to detect.
            threshold (float): The confidence threshold. Defaults to 0.5.

        Returns:
            list[EntityCandidate]: A list of detected entity candidates.
        """
        self._validate_threshold(threshold)

        if not text or not labels:
            return []

        # GLiNER returns a list of dicts:
        # [{'start': 0, 'end': 5, 'text': '...', 'label': '...', 'score': 0.95}, ...]
        raw_entities = self.model.predict_entities(text, labels, threshold=threshold)

        return [self._build_candidate(entity) for entity in raw_entities]

    def extract_batch(self, texts: list[str], labels: list[str], threshold: float = 0.5) -> list[list[EntityCandidate]]:
        """
        Extract entities from a batch of texts using the provided labels.

        Args:
            texts (list[str]): The list of input texts to process.
            labels (list[str]): A list of entity types to detect.
            threshold (float): The confidence threshold. Defaults to 0.5.

        Returns:
            list[list[EntityCandidate]]: A list of lists, where each inner list contains
                                       detected entity candidates for the corresponding text.
        """
        self._validate_threshold(threshold)

        if not texts or not labels:
            return [[] for _ in texts]

        # Use batch_predict_entities if available.
        # batch_predict_entities returns a list of lists of dicts.
        batch_raw_entities = self.model.batch_predict_entities(texts, labels, threshold=threshold)

        batch_extracted_candidates: list[list[EntityCandidate]] = []

        # Iterate over results
        # Use strict=True to ensure the model returns results for every input text
        # (Though we don't strictly need 'texts' for iteration anymore, zip is safer validation)
        for _, raw_entities in zip(texts, batch_raw_entities, strict=True):
            extracted_candidates = [self._build_candidate(entity) for entity in raw_entities]
            batch_extracted_candidates.append(extracted_candidates)

        return batch_extracted_candidates
