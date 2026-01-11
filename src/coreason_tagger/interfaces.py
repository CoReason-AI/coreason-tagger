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

from coreason_tagger.schema import AssertionStatus


class BaseExtractor(ABC):
    """
    Abstract base class for Entity Extraction (NER).
    """

    @abstractmethod
    def extract(self, text: str, labels: List[str]) -> List[Dict[str, Any]]:
        """
        Extracts entities from the text based on provided labels.
        Returns a list of dictionaries with span information (start, end, text, label).
        """
        pass  # pragma: no cover


class BaseAssertionDetector(ABC):
    """
    Abstract base class for Assertion Detection (Contextualization).
    """

    @abstractmethod
    def detect(self, text: str, entity_span: Dict[str, Any]) -> AssertionStatus:
        """
        Determines the assertion status of an entity within the text.
        entity_span is a dict containing 'start', 'end', 'text'.
        """
        pass  # pragma: no cover


class BaseLinker(ABC):
    """
    Abstract base class for Entity Linking (Normalization).
    """

    @abstractmethod
    def link(self, text: str, label: str) -> Dict[str, Any]:
        """
        Links a text span (and its label) to a concept ID.
        Returns a dict containing 'concept_id', 'concept_name', 'confidence'.
        """
        pass  # pragma: no cover
