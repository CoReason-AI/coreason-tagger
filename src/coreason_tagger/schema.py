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
    """
    Status of an assertion (e.g., Present, Absent, Possible).
    Combines requirements from Section 3.2 and Section 6 of the PRD.
    """

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    POSSIBLE = "POSSIBLE"
    CONDITIONAL = "CONDITIONAL"
    FAMILY = "FAMILY_HISTORY"
    ASSOCIATED_WITH_SOMEONE_ELSE = "ASSOCIATED_WITH_SOMEONE_ELSE"


class TaggedEntity(BaseModel):
    """
    Represents a normalized entity extracted from text, contextualized with assertion status,
    and linked to a concept ID.
    """

    span_text: str = Field(..., min_length=1, description="The extracted text span.")
    label: str = Field(..., min_length=1, description="The entity label (e.g., Symptom).")

    # The Link (What is it?)
    concept_id: str = Field(..., min_length=1, description="The unique concept ID (e.g., SNOMED:123).")
    concept_name: str = Field(..., min_length=1, description="The canonical name of the concept.")
    link_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")

    # The Context (Is it real?)
    assertion: AssertionStatus
