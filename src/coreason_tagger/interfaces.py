# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol

from coreason_tagger.schema import (
    AssertionStatus,
    EntityCandidate,
    ExtractionStrategy,
    LinkedEntity,
)


class CodexClient(Protocol):
    """
    Protocol defining the interface for the Codex client.
    """

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for concepts in the codex.
        """
        ...

    def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific concept by ID.
        """
        ...


class BaseAssertionDetector(ABC):
    """Abstract base class for assertion detection strategies."""

    @abstractmethod
    def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
        """
        Determine the assertion status of an entity within a given context.

        Args:
            text (str): The full context text (e.g., the sentence or paragraph).
            span_text (str): The text of the entity itself (e.g., "headaches").
            span_start (int): The character start index of the entity in `text`.
            span_end (int): The character end index of the entity in `text`.

        Returns:
            AssertionStatus: The detected status (e.g., PRESENT, ABSENT).
        """
        pass  # pragma: no cover


class BaseNERExtractor(ABC):
    """Abstract base class for NER extraction strategies."""

    @abstractmethod
    def extract(self, text: str, labels: List[str], threshold: float = 0.5) -> List[EntityCandidate]:
        """
        Extract entities from text using the provided labels.

        Args:
            text (str): The input text to process.
            labels (List[str]): A list of entity types to detect (e.g., ["Symptom", "Drug"]).
            threshold (float): The confidence threshold for extraction. Defaults to 0.5.

        Returns:
            List[EntityCandidate]: A list of detected entity candidates.
        """
        pass  # pragma: no cover

    @abstractmethod
    def extract_batch(self, texts: List[str], labels: List[str], threshold: float = 0.5) -> List[List[EntityCandidate]]:
        """
        Extract entities from a batch of texts using the provided labels.

        Args:
            texts (List[str]): The list of input texts to process.
            labels (List[str]): A list of entity types to detect.
            threshold (float): The confidence threshold for extraction. Defaults to 0.5.

        Returns:
            List[List[EntityCandidate]]: A list of lists, where each inner list contains
                                       detected entity candidates for the corresponding text.
        """
        pass  # pragma: no cover


class BaseLinker(ABC):
    """Abstract base class for entity linking strategies."""

    @abstractmethod
    def resolve(self, entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        """
        Link an extracted entity to a concept in the codex.
        """
        pass  # pragma: no cover
