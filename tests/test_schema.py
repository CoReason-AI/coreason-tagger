# Prosperity Public License 3.0
# Copyright (c) 2024 CoReason AI

import pytest
from coreason_tagger.schema import AssertionStatus, TaggedEntity
from pydantic import ValidationError


def test_assertion_status_enum() -> None:
    """Test that all expected assertion statuses are present and correct."""
    assert AssertionStatus.PRESENT.value == "PRESENT"
    assert AssertionStatus.ABSENT.value == "ABSENT"
    assert AssertionStatus.POSSIBLE.value == "POSSIBLE"
    assert AssertionStatus.CONDITIONAL.value == "CONDITIONAL"
    assert AssertionStatus.ASSOCIATED_WITH_SOMEONE_ELSE.value == "ASSOCIATED_WITH_SOMEONE_ELSE"
    assert AssertionStatus.FAMILY.value == "FAMILY_HISTORY"


def test_tagged_entity_valid_creation() -> None:
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
    assert entity.concept_name == "Migraine"
    assert entity.link_confidence == 0.98
    assert entity.assertion == AssertionStatus.PRESENT


def test_tagged_entity_confidence_bounds() -> None:
    """Test that link_confidence enforces bounds."""
    # Test valid bounds
    TaggedEntity(
        span_text="test",
        label="test",
        concept_id="id",
        concept_name="name",
        link_confidence=0.0,
        assertion=AssertionStatus.PRESENT,
    )
    TaggedEntity(
        span_text="test",
        label="test",
        concept_id="id",
        concept_name="name",
        link_confidence=1.0,
        assertion=AssertionStatus.PRESENT,
    )

    # Test invalid upper bound
    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="test",
            label="test",
            concept_id="id",
            concept_name="name",
            link_confidence=1.01,
            assertion=AssertionStatus.PRESENT,
        )

    # Test invalid lower bound
    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="test",
            label="test",
            concept_id="id",
            concept_name="name",
            link_confidence=-0.01,
            assertion=AssertionStatus.PRESENT,
        )


def test_tagged_entity_empty_strings() -> None:
    """Test that empty strings are not allowed for text fields."""
    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="",
            label="test",
            concept_id="id",
            concept_name="name",
            link_confidence=0.5,
            assertion=AssertionStatus.PRESENT,
        )

    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="test",
            label="",
            concept_id="id",
            concept_name="name",
            link_confidence=0.5,
            assertion=AssertionStatus.PRESENT,
        )

    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="test",
            label="test",
            concept_id="",
            concept_name="name",
            link_confidence=0.5,
            assertion=AssertionStatus.PRESENT,
        )


def test_tagged_entity_invalid_assertion() -> None:
    """Test validation of assertion status."""
    with pytest.raises(ValidationError):
        TaggedEntity(
            span_text="test",
            label="test",
            concept_id="id",
            concept_name="name",
            link_confidence=0.5,
            assertion="INVALID_STATUS",  # type: ignore
        )
