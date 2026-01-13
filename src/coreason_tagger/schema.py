from enum import Enum

from pydantic import BaseModel, Field


class AssertionStatus(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    POSSIBLE = "POSSIBLE"
    CONDITIONAL = "CONDITIONAL"
    ASSOCIATED_WITH_SOMEONE_ELSE = "ASSOCIATED_WITH_SOMEONE_ELSE"
    # Alias for convenience or if used interchangeably in some contexts
    FAMILY = "FAMILY_HISTORY"


class ExtractedSpan(BaseModel):
    text: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)
    score: float = Field(..., ge=0.0, le=1.0)


class TaggedEntity(BaseModel):
    span_text: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)

    # The Link
    concept_id: str = Field(..., min_length=1)
    concept_name: str = Field(..., min_length=1)
    link_confidence: float = Field(..., ge=0.0, le=1.0)

    # The Context
    assertion: AssertionStatus
