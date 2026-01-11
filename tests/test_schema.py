# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import pytest
from pydantic import ValidationError

from coreason_tagger.schema import AssertionStatus, TaggedEntity


def test_assertion_status_enum() -> None:
    assert AssertionStatus.PRESENT.value == "PRESENT"
    assert AssertionStatus.ABSENT.value == "ABSENT"
    assert AssertionStatus.POSSIBLE.value == "POSSIBLE"
    assert AssertionStatus.CONDITIONAL.value == "CONDITIONAL"
    assert AssertionStatus.FAMILY.value == "FAMILY_HISTORY"
    assert AssertionStatus.ASSOCIATED_WITH_SOMEONE_ELSE.value == "ASSOCIATED_WITH_SOMEONE_ELSE"


def test_tagged_entity_valid() -> None:
    entity = TaggedEntity(
        span_text="severe headaches",
        label="Symptom",
        concept_id="SNOMED:25064002",
        concept_name="Migraine",
        link_confidence=0.98,
        assertion=AssertionStatus.PRESENT,
    )
    assert entity.span_text == "severe headaches"
    assert entity.label == "Symptom"
    assert entity.concept_id == "SNOMED:25064002"
    assert entity.concept_name == "Migraine"
    assert entity.link_confidence == 0.98
    assert entity.assertion == AssertionStatus.PRESENT


def test_tagged_entity_invalid_assertion() -> None:
    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="severe headaches",
            label="Symptom",
            concept_id="SNOMED:25064002",
            concept_name="Migraine",
            link_confidence=0.98,
            assertion="INVALID_STATUS",  # type: ignore
        )


def test_tagged_entity_missing_field() -> None:
    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="severe headaches",
            label="Symptom",
            concept_id="SNOMED:25064002",
            # concept_name is missing, so we omit it to trigger validation error
            link_confidence=0.98,
            assertion=AssertionStatus.PRESENT,
        )  # type: ignore[call-arg]


def test_tagged_entity_empty_string_validation() -> None:
    """Test that empty strings are rejected for critical fields."""
    # Empty span_text
    with pytest.raises(ValidationError) as excinfo:
        TaggedEntity(
            span_text="",
            label="Symptom",
            concept_id="ID:1",
            concept_name="Name",
            link_confidence=0.5,
            assertion=AssertionStatus.PRESENT,
        )
    assert "String should have at least 1 character" in str(excinfo.value)

    # Empty concept_id
    with pytest.raises(ValidationError) as excinfo:
        TaggedEntity(
            span_text="Text",
            label="Symptom",
            concept_id="",
            concept_name="Name",
            link_confidence=0.5,
            assertion=AssertionStatus.PRESENT,
        )
    assert "String should have at least 1 character" in str(excinfo.value)


def test_tagged_entity_confidence_bounds() -> None:
    """Test that link_confidence must be between 0.0 and 1.0."""
    # Too high
    with pytest.raises(ValidationError) as excinfo:
        TaggedEntity(
            span_text="Text",
            label="Symptom",
            concept_id="ID:1",
            concept_name="Name",
            link_confidence=1.1,
            assertion=AssertionStatus.PRESENT,
        )
    assert "Input should be less than or equal to 1" in str(excinfo.value)

    # Too low
    with pytest.raises(ValidationError) as excinfo:
        TaggedEntity(
            span_text="Text",
            label="Symptom",
            concept_id="ID:1",
            concept_name="Name",
            link_confidence=-0.1,
            assertion=AssertionStatus.PRESENT,
        )
    assert "Input should be greater than or equal to 0" in str(excinfo.value)
