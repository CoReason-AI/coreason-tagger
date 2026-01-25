# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ExtractionStrategy(str, Enum):
    """Enumeration of supported extraction strategies."""

    SPEED_GLINER = "SPEED_GLINER"
    PRECISION_NUNER = "PRECISION_NUNER"
    REASONING_LLM = "REASONING_LLM"


class AssertionStatus(str, Enum):
    """Enumeration of assertion statuses for entities."""

    PRESENT = "PRESENT"  # Default
    ABSENT = "ABSENT"  # Negated ("No signs of...")
    POSSIBLE = "POSSIBLE"  # Speculative ("Rule out...")
    CONDITIONAL = "CONDITIONAL"  # ("If symptoms persist...")
    HISTORY = "HISTORY"  # ("History of...")
    FAMILY = "FAMILY"  # ("Mother had...")


class EntityCandidate(BaseModel):
    """Raw output from the NER layer."""

    text: str
    start: int
    end: int
    label: str
    confidence: float
    source_model: str  # e.g., "gliner_large_v2"


class LinkedEntity(EntityCandidate):
    """The final hydrated entity."""

    # Context
    assertion: AssertionStatus = Field(default=AssertionStatus.PRESENT)

    # Linking (NEN)
    concept_id: Optional[str] = None  # "SNOMED:12345"
    concept_name: Optional[str] = None  # "Viral Rhinitis"
    link_score: float = 0.0  # Cosine similarity

    # Traceability
    strategy_used: ExtractionStrategy


class BatchRequest(BaseModel):
    """Request model for batch processing."""

    texts: List[str]
    labels: List[str]
    config: dict[str, Any] = Field(default_factory=dict)  # Overrides


class TaggingRequest(BaseModel):
    """Request model for single text processing."""

    text: str
    labels: List[str]
    strategy: ExtractionStrategy = ExtractionStrategy.SPEED_GLINER
