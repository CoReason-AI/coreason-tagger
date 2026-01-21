# Prosperity Public License 3.0
# Copyright (c) 2024 CoReason AI

import pytest
from coreason_tagger.schema import (
    AssertionStatus,
    BatchRequest,
    EntityCandidate,
    ExtractionStrategy,
    LinkedEntity,
)
from pydantic import ValidationError


def test_assertion_status_enum() -> None:
    """Test that all expected assertion statuses are present and correct."""
    assert AssertionStatus.PRESENT.value == "PRESENT"
    assert AssertionStatus.ABSENT.value == "ABSENT"
    assert AssertionStatus.POSSIBLE.value == "POSSIBLE"
    assert AssertionStatus.CONDITIONAL.value == "CONDITIONAL"
    assert AssertionStatus.HISTORY.value == "HISTORY"
    assert AssertionStatus.FAMILY.value == "FAMILY"


def test_entity_candidate_creation() -> None:
    """Test creating a valid EntityCandidate."""
    candidate = EntityCandidate(
        text="headache",
        label="Symptom",
        start=0,
        end=8,
        confidence=0.95,
        source_model="gliner_small",
    )
    assert candidate.text == "headache"
    assert candidate.label == "Symptom"
    assert candidate.start == 0
    assert candidate.end == 8
    assert candidate.confidence == 0.95
    assert candidate.source_model == "gliner_small"


def test_linked_entity_valid_creation() -> None:
    """Test creating a valid LinkedEntity."""
    entity = LinkedEntity(
        text="severe headaches",
        label="Symptom",
        start=0,
        end=16,
        confidence=0.98,
        source_model="gliner_small",
        assertion=AssertionStatus.PRESENT,
        concept_id="SNOMED:25064002",
        concept_name="Migraine",
        link_score=0.98,
        strategy_used=ExtractionStrategy.SPEED_GLINER,
    )
    assert entity.text == "severe headaches"
    assert entity.label == "Symptom"
    assert entity.concept_id == "SNOMED:25064002"
    assert entity.concept_name == "Migraine"
    assert entity.link_score == 0.98
    assert entity.assertion == AssertionStatus.PRESENT
    assert entity.strategy_used == ExtractionStrategy.SPEED_GLINER


def test_linked_entity_optional_fields() -> None:
    """Test that optional fields can be None."""
    entity = LinkedEntity(
        text="unknown thing",
        label="Unknown",
        start=0,
        end=13,
        confidence=0.5,
        source_model="gliner_small",
        strategy_used=ExtractionStrategy.SPEED_GLINER,
        # concept_id, concept_name defaults to None
    )
    assert entity.concept_id is None
    assert entity.concept_name is None
    assert entity.link_score == 0.0  # Default


def test_linked_entity_invalid_assertion() -> None:
    """Test validation of assertion status."""
    with pytest.raises(ValidationError):
        LinkedEntity(
            text="test",
            label="test",
            start=0,
            end=4,
            confidence=0.9,
            source_model="test",
            strategy_used=ExtractionStrategy.SPEED_GLINER,
            assertion="INVALID_STATUS",
        )


def test_batch_request_model() -> None:
    """Test validation of the BatchRequest model."""
    # Valid
    req = BatchRequest(texts=["t1", "t2"], labels=["l1", "l2"], config={"threshold": 0.5})
    assert len(req.texts) == 2
    assert req.config["threshold"] == 0.5

    # Invalid types
    with pytest.raises(ValidationError):
        BatchRequest(texts="not a list", labels=[])

    # Missing fields
    with pytest.raises(ValidationError):
        BatchRequest(texts=["t1"])  # type: ignore
