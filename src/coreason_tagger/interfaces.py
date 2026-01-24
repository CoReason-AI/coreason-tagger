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
from typing import Any, Dict, List, Protocol, TypeVar

from coreason_tagger.schema import (
    AssertionStatus,
    EntityCandidate,
    ExtractionStrategy,
    LinkedEntity,
)

T = TypeVar("T")


class CodexClient(Protocol):
    """Protocol defining the interface for the Codex client."""

    async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for concepts in the codex.

        Args:
            query: The search query string.
            top_k: The number of results to return. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the found concepts.
        """
        ...

    async def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """Retrieve a specific concept by ID.

        Args:
            concept_id: The unique identifier of the concept.

        Returns:
            Dict[str, Any]: A dictionary representing the concept.
        """
        ...


class BaseAssertionDetector(ABC):
    """Abstract base class for assertion detection strategies."""

    @abstractmethod
    async def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
        """Determine the assertion status of an entity within a given context.

        Args:
            text: The full context text (e.g., the sentence or paragraph).
            span_text: The text of the entity itself (e.g., "headaches").
            span_start: The character start index of the entity in `text`.
            span_end: The character end index of the entity in `text`.

        Returns:
            AssertionStatus: The detected status (e.g., PRESENT, ABSENT).
        """
        pass  # pragma: no cover


class BaseExtractor(ABC):
    """Contract for all NER backends."""

    @abstractmethod
    async def load_model(self) -> None:
        """Lazy loading of weights to VRAM."""
        pass  # pragma: no cover

    def validate_threshold(self, threshold: float) -> None:
        """Validate that the threshold is within the valid range [0.0, 1.0].

        Args:
            threshold: The threshold value to check.

        Raises:
            ValueError: If the threshold is out of bounds.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    @abstractmethod
    async def extract(self, text: str, labels: List[str], threshold: float = 0.5) -> List[EntityCandidate]:
        """Extract entities from text using the provided labels.

        Args:
            text: The input text to process.
            labels: A list of entity types to detect (e.g., ["Symptom", "Drug"]).
            threshold: The confidence threshold for extraction. Defaults to 0.5.

        Returns:
            List[EntityCandidate]: A list of detected entity candidates.
        """
        pass  # pragma: no cover

    @abstractmethod
    async def extract_batch(
        self, texts: List[str], labels: List[str], threshold: float = 0.5
    ) -> List[List[EntityCandidate]]:
        """Extract entities from a batch of texts using the provided labels.

        Args:
            texts: The list of input texts to process.
            labels: A list of entity types to detect.
            threshold: The confidence threshold for extraction. Defaults to 0.5.

        Returns:
            List[List[EntityCandidate]]: A list of lists, where each inner list contains
                detected entity candidates for the corresponding text.
        """
        pass  # pragma: no cover


class BaseLinker(ABC):
    """Abstract base class for entity linking strategies."""

    @abstractmethod
    async def resolve(self, entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        """Link an extracted entity to a concept in the codex.

        Args:
            entity: The extracted entity candidate.
            context: The context text surrounding the entity.
            strategy: The extraction strategy used to identify the entity.

        Returns:
            LinkedEntity: The entity with linking information populated.
        """
        pass  # pragma: no cover
