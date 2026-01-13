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
from typing import Any, Dict, List

from coreason_tagger.schema import AssertionStatus, ExtractedSpan


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
    def extract(self, text: str, labels: List[str]) -> List[ExtractedSpan]:
        """
        Extract entities from text using the provided labels.

        Args:
            text (str): The input text to process.
            labels (List[str]): A list of entity types to detect (e.g., ["Symptom", "Drug"]).

        Returns:
            List[ExtractedSpan]: A list of detected entity spans.
        """
        pass  # pragma: no cover


class BaseLinker(ABC):
    """Abstract base class for entity linking strategies."""

    @abstractmethod
    def link(self, text: str, label: str) -> Dict[str, Any]:
        """
        Link an extracted entity to a concept in the codex.
        """
        pass  # pragma: no cover
