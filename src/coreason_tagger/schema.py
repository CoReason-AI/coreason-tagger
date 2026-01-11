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

from pydantic import BaseModel


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

    span_text: str  # e.g., "severe headaches"
    label: str  # e.g., "Symptom"

    # The Link (What is it?)
    concept_id: str  # e.g., "SNOMED:25064002"
    concept_name: str  # e.g., "Migraine"
    link_confidence: float  # e.g., 0.98

    # The Context (Is it real?)
    assertion: AssertionStatus
