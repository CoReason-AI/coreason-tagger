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

from pydantic import BaseModel, Field


class AssertionStatus(str, Enum):
    """Enumeration of possible assertion statuses for an entity."""

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    POSSIBLE = "POSSIBLE"
    FAMILY = "FAMILY_HISTORY"
    CONDITIONAL = "CONDITIONAL"
    ASSOCIATED_WITH_SOMEONE_ELSE = "ASSOCIATED_WITH_SOMEONE_ELSE"


class TaggedEntity(BaseModel):
    """
    Represents a normalized entity extracted from text with assertion and link.

    Attributes:
        span_text (str): The exact text extracted from the source.
        label (str): The entity type (e.g., "Symptom", "Condition").
        concept_id (str): The normalized ID from the codex (e.g., "SNOMED:123").
        concept_name (str): The normalized name of the concept.
        link_confidence (float): The confidence score of the vector link.
        assertion (AssertionStatus): The context of the entity (e.g., ABSENT, PRESENT).
    """

    span_text: str = Field(..., min_length=1, description="The extracted text span.")
    label: str = Field(..., min_length=1, description="The entity label/type.")
    concept_id: str = Field(..., min_length=1, description="The unique concept identifier.")
    concept_name: str = Field(..., min_length=1, description="The human-readable concept name.")
    link_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the linking (0.0 - 1.0).")
    assertion: AssertionStatus = Field(..., description="The assertion status of the entity.")
