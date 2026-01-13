# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import MagicMock

import pytest
from coreason_tagger.interfaces import BaseAssertionDetector, BaseLinker, BaseNERExtractor
from coreason_tagger.schema import AssertionStatus, ExtractedSpan, TaggedEntity
from coreason_tagger.tagger import CoreasonTagger


@pytest.fixture
def mock_ner() -> MagicMock:
    return MagicMock(spec=BaseNERExtractor)


@pytest.fixture
def mock_assertion() -> MagicMock:
    return MagicMock(spec=BaseAssertionDetector)


@pytest.fixture
def mock_linker() -> MagicMock:
    return MagicMock(spec=BaseLinker)


@pytest.fixture
def tagger(mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock) -> CoreasonTagger:
    return CoreasonTagger(ner=mock_ner, assertion=mock_assertion, linker=mock_linker)


def test_tag_happy_path(
    tagger: CoreasonTagger,
    mock_ner: MagicMock,
    mock_assertion: MagicMock,
    mock_linker: MagicMock,
) -> None:
    """Test the standard flow: extract -> detect -> link -> return."""
    text = "Patient has a headache."
    labels = ["Symptom"]

    # Mock NER return
    span = ExtractedSpan(text="headache", label="Symptom", start=14, end=22, score=0.99)
    mock_ner.extract.return_value = [span]

    # Mock Assertion return
    mock_assertion.detect.return_value = AssertionStatus.PRESENT

    # Mock Linker return
    mock_linker.link.return_value = {
        "concept_id": "HP:0002315",
        "concept_name": "Headache",
        "link_confidence": 0.95,
    }

    results = tagger.tag(text, labels)

    assert len(results) == 1
    entity = results[0]
    assert isinstance(entity, TaggedEntity)
    assert entity.span_text == "headache"
    assert entity.label == "Symptom"
    assert entity.concept_id == "HP:0002315"
    assert entity.assertion == AssertionStatus.PRESENT
    assert entity.link_confidence == 0.95

    # Verify calls
    mock_ner.extract.assert_called_once_with(text, labels)
    mock_assertion.detect.assert_called_once_with(text=text, span_text="headache", span_start=14, span_end=22)
    mock_linker.link.assert_called_once_with(span)


def test_tag_empty_text(tagger: CoreasonTagger, mock_ner: MagicMock) -> None:
    """Test that empty text returns empty list without calling extract."""
    assert tagger.tag("", ["Label"]) == []
    mock_ner.extract.assert_not_called()


def test_tag_no_entities_found(tagger: CoreasonTagger, mock_ner: MagicMock) -> None:
    """Test when NER finds nothing."""
    mock_ner.extract.return_value = []
    assert tagger.tag("Clean text", ["Label"]) == []


def test_tag_linking_failure(
    tagger: CoreasonTagger,
    mock_ner: MagicMock,
    mock_assertion: MagicMock,
    mock_linker: MagicMock,
) -> None:
    """Test that entities are dropped if linking fails (returns empty dict)."""
    text = "Unknown thing."
    span = ExtractedSpan(text="thing", label="Unknown", start=8, end=13, score=0.5)
    mock_ner.extract.return_value = [span]
    mock_assertion.detect.return_value = AssertionStatus.PRESENT
    # Linker returns empty dict
    mock_linker.link.return_value = {}

    results = tagger.tag(text, ["Unknown"])
    assert results == []


def test_multiple_entities(
    tagger: CoreasonTagger,
    mock_ner: MagicMock,
    mock_assertion: MagicMock,
    mock_linker: MagicMock,
) -> None:
    """Test processing of multiple entities in one text."""
    text = "Patient denies fever but has cough."
    span1 = ExtractedSpan(text="fever", label="Symptom", start=15, end=20, score=0.9)
    span2 = ExtractedSpan(text="cough", label="Symptom", start=29, end=34, score=0.9)
    mock_ner.extract.return_value = [span1, span2]

    # Side effects for assertion (fever -> ABSENT, cough -> PRESENT)
    mock_assertion.detect.side_effect = [AssertionStatus.ABSENT, AssertionStatus.PRESENT]

    # Side effects for linker
    mock_linker.link.side_effect = [
        {"concept_id": "C1", "concept_name": "Fever", "link_confidence": 0.8},
        {"concept_id": "C2", "concept_name": "Cough", "link_confidence": 0.85},
    ]

    results = tagger.tag(text, ["Symptom"])

    assert len(results) == 2
    assert results[0].span_text == "fever"
    assert results[0].assertion == AssertionStatus.ABSENT
    assert results[0].concept_id == "C1"

    assert results[1].span_text == "cough"
    assert results[1].assertion == AssertionStatus.PRESENT
    assert results[1].concept_id == "C2"
