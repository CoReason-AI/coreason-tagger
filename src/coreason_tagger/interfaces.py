# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Protocol, TypeVar

from coreason_tagger.schema import (
    AssertionStatus,
    EntityCandidate,
    ExtractionStrategy,
    LinkedEntity,
)

T = TypeVar("T")


class CodexClient(Protocol):
    """
    Protocol defining the interface for the Codex client.
    """

    async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for concepts in the codex.
        """
        ...

    async def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific concept by ID.
        """
        ...


class BaseAssertionDetector(ABC):
    """Abstract base class for assertion detection strategies."""

    @abstractmethod
    async def detect(self, text: str, span_text: str, span_start: int, span_end: int) -> AssertionStatus:
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


class BaseExtractor(ABC):
    """
    Contract for all NER backends.
    """

    @abstractmethod
    async def load_model(self) -> None:
        """Lazy loading of weights to VRAM."""
        pass  # pragma: no cover

    def validate_threshold(self, threshold: float) -> None:
        """
        Validate that the threshold is within the valid range [0.0, 1.0].

        Args:
            threshold (float): The threshold value to check.

        Raises:
            ValueError: If the threshold is out of bounds.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    async def run_in_executor(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Run a blocking function in the default loop executor.

        Args:
            func: The blocking function to run.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    @abstractmethod
    async def extract(self, text: str, labels: List[str], threshold: float = 0.5) -> List[EntityCandidate]:
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
    async def extract_batch(
        self, texts: List[str], labels: List[str], threshold: float = 0.5
    ) -> List[List[EntityCandidate]]:
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
    async def resolve(self, entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        """
        Link an extracted entity to a concept in the codex.
        """
        pass  # pragma: no cover
