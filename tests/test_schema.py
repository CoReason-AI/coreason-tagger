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
from coreason_tagger.schema import AssertionStatus, TaggedEntity
from pydantic import ValidationError


def test_assertion_status_enum() -> None:
    """Test that AssertionStatus has the expected values."""
    assert AssertionStatus.PRESENT.value == "PRESENT"
    assert AssertionStatus.ABSENT.value == "ABSENT"
    assert AssertionStatus.POSSIBLE.value == "POSSIBLE"
    assert AssertionStatus.FAMILY.value == "FAMILY_HISTORY"
    assert AssertionStatus.CONDITIONAL.value == "CONDITIONAL"
    assert AssertionStatus.ASSOCIATED_WITH_SOMEONE_ELSE.value == "ASSOCIATED_WITH_SOMEONE_ELSE"


def test_tagged_entity_valid() -> None:
    """Test creating a valid TaggedEntity."""
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
    assert entity.link_confidence == 0.98
    assert entity.assertion == AssertionStatus.PRESENT


def test_tagged_entity_invalid_confidence() -> None:
    """Test validation failure for invalid confidence scores."""
    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="test",
            label="test",
            concept_id="id",
            concept_name="name",
            link_confidence=1.5,  # Invalid: > 1.0
            assertion=AssertionStatus.PRESENT,
        )

    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="test",
            label="test",
            concept_id="id",
            concept_name="name",
            link_confidence=-0.1,  # Invalid: < 0.0
            assertion=AssertionStatus.PRESENT,
        )


def test_tagged_entity_empty_fields() -> None:
    """Test validation failure for empty strings."""
    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="",  # Invalid: empty
            label="test",
            concept_id="id",
            concept_name="name",
            link_confidence=0.5,
            assertion=AssertionStatus.PRESENT,
        )
