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


def test_user_story_family_history(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Test User Story A: "Patient's mother died of breast cancer."
    Expectation: Assertion sets FAMILY_HISTORY, Linker maps to Breast Cancer.
    """
    text = "Patient's mother died of breast cancer."
    # 1. NER detects
    span = ExtractedSpan(text="breast cancer", label="Diagnosis", start=24, end=37, score=0.99)
    mock_ner.extract.return_value = [span]

    # 2. Assertion detects "mother" in path
    mock_assertion.detect.return_value = AssertionStatus.FAMILY

    # 3. Linker maps
    mock_linker.link.return_value = {
        "concept_id": "SNOMED:254837009",
        "concept_name": "Malignant neoplasm of breast",
        "link_confidence": 0.98,
    }

    results = tagger.tag(text, ["Diagnosis"])

    assert len(results) == 1
    entity = results[0]
    assert entity.span_text == "breast cancer"
    assert entity.assertion == AssertionStatus.FAMILY
    assert entity.concept_id == "SNOMED:254837009"


def test_user_story_ambiguous_drug(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Test User Story B: "Administered 50mg of Lasix."
    Expectation: Linker maps "Lasix" (Brand) to "Furosemide" (Generic).
    """
    text = "Administered 50mg of Lasix."
    # 1. NER detects
    span = ExtractedSpan(text="Lasix", label="Drug", start=21, end=26, score=0.99)
    mock_ner.extract.return_value = [span]

    # 2. Assertion detects PRESENT
    mock_assertion.detect.return_value = AssertionStatus.PRESENT

    # 3. Linker maps brand to generic ID
    mock_linker.link.return_value = {
        "concept_id": "RxNorm:4603",
        "concept_name": "Furosemide",
        "link_confidence": 0.95,
    }

    results = tagger.tag(text, ["Drug"])

    assert len(results) == 1
    entity = results[0]
    assert entity.span_text == "Lasix"
    assert entity.concept_id == "RxNorm:4603"
    assert entity.concept_name == "Furosemide"


def test_mixed_linking_success(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Test a batch where some entities link and others fail.
    Scenario: [Success, Fail, Success]
    """
    text = "A, B, C"
    spans = [
        ExtractedSpan(text="A", label="Test", start=0, end=1, score=1.0),
        ExtractedSpan(text="B", label="Test", start=3, end=4, score=1.0),
        ExtractedSpan(text="C", label="Test", start=6, end=7, score=1.0),
    ]
    mock_ner.extract.return_value = spans
    mock_assertion.detect.return_value = AssertionStatus.PRESENT

    # Linker side effects: Dict, Empty, Dict
    mock_linker.link.side_effect = [
        {"concept_id": "ID_A", "concept_name": "A", "link_confidence": 1.0},
        {},  # Failure
        {"concept_id": "ID_C", "concept_name": "C", "link_confidence": 1.0},
    ]

    results = tagger.tag(text, ["Test"])

    assert len(results) == 2
    assert results[0].span_text == "A"
    assert results[1].span_text == "C"


def test_identical_entity_different_contexts(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Test same entity string appearing twice with different contexts/assertions.
    "Mother has diabetes, but patient denies diabetes."
    """
    text = "Mother has diabetes, but patient denies diabetes."
    # NER finds two "diabetes"
    span1 = ExtractedSpan(text="diabetes", label="Condition", start=11, end=19, score=0.99)
    span2 = ExtractedSpan(text="diabetes", label="Condition", start=40, end=48, score=0.99)
    mock_ner.extract.return_value = [span1, span2]

    # Assertion logic: First is FAMILY, second is ABSENT
    mock_assertion.detect.side_effect = [AssertionStatus.FAMILY, AssertionStatus.ABSENT]

    # Linker returns same ID for both
    mock_linker.link.return_value = {
        "concept_id": "C_DIABETES",
        "concept_name": "Diabetes Mellitus",
        "link_confidence": 1.0,
    }

    results = tagger.tag(text, ["Condition"])

    assert len(results) == 2
    assert results[0].span_text == "diabetes"
    assert results[0].assertion == AssertionStatus.FAMILY
    assert results[1].span_text == "diabetes"
    assert results[1].assertion == AssertionStatus.ABSENT


def test_malformed_linker_result(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Test that if linker returns a dict without 'concept_id', it is treated as a failure.
    (Prevents validation error in TaggedEntity construction).
    """
    text = "Something weird."
    span = ExtractedSpan(text="weird", label="Thing", start=10, end=15, score=0.5)
    mock_ner.extract.return_value = [span]
    mock_assertion.detect.return_value = AssertionStatus.PRESENT

    # Linker returns malformed dict
    mock_linker.link.return_value = {"concept_name": "Only Name No ID", "link_confidence": 0.5}

    results = tagger.tag(text, ["Thing"])

    # Should be skipped
    assert results == []


# --- Batch Processing Tests ---


def test_tag_batch_happy_path(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """Test standard batch flow."""
    texts = ["Patient has fever.", "No cough detected."]
    labels = ["Symptom"]

    # Mock NER Batch Return
    # Text 1: Fever
    span1 = ExtractedSpan(text="fever", label="Symptom", start=12, end=17, score=0.99)
    # Text 2: Cough
    span2 = ExtractedSpan(text="cough", label="Symptom", start=3, end=8, score=0.99)
    mock_ner.extract_batch.return_value = [[span1], [span2]]

    # Mock Assertion
    # Call 1 (fever): PRESENT
    # Call 2 (cough): ABSENT
    mock_assertion.detect.side_effect = [AssertionStatus.PRESENT, AssertionStatus.ABSENT]

    # Mock Linker
    mock_linker.link.side_effect = [
        {"concept_id": "C_FEVER", "concept_name": "Fever", "link_confidence": 0.9},
        {"concept_id": "C_COUGH", "concept_name": "Cough", "link_confidence": 0.9},
    ]

    results = tagger.tag_batch(texts, labels)

    assert len(results) == 2

    # Check Result 1
    assert len(results[0]) == 1
    ent1 = results[0][0]
    assert ent1.span_text == "fever"
    assert ent1.assertion == AssertionStatus.PRESENT
    assert ent1.concept_id == "C_FEVER"

    # Check Result 2
    assert len(results[1]) == 1
    ent2 = results[1][0]
    assert ent2.span_text == "cough"
    assert ent2.assertion == AssertionStatus.ABSENT
    assert ent2.concept_id == "C_COUGH"

    # Verify calls
    mock_ner.extract_batch.assert_called_once_with(texts, labels)

    # Assertion should be called with correct contexts
    assert mock_assertion.detect.call_count == 2
    mock_assertion.detect.assert_any_call(text=texts[0], span_text="fever", span_start=12, span_end=17)
    mock_assertion.detect.assert_any_call(text=texts[1], span_text="cough", span_start=3, span_end=8)


def test_tag_batch_empty_input(tagger: CoreasonTagger, mock_ner: MagicMock) -> None:
    """Test empty input list."""
    assert tagger.tag_batch([], ["Label"]) == []
    mock_ner.extract_batch.assert_not_called()


def test_tag_batch_mixed_empty_results(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """Test batch where some texts have no entities."""
    texts = ["Has fever.", "Nothing here.", "Has cough."]
    labels = ["Symptom"]

    span1 = ExtractedSpan(text="fever", label="Symptom", start=4, end=9, score=0.9)
    span3 = ExtractedSpan(text="cough", label="Symptom", start=4, end=9, score=0.9)

    # Return: [ [span1], [], [span3] ]
    mock_ner.extract_batch.return_value = [[span1], [], [span3]]

    mock_assertion.detect.return_value = AssertionStatus.PRESENT
    mock_linker.link.return_value = {
        "concept_id": "CID", "concept_name": "Name", "link_confidence": 1.0
    }

    results = tagger.tag_batch(texts, labels)

    assert len(results) == 3
    assert len(results[0]) == 1
    assert len(results[1]) == 0
    assert len(results[2]) == 1
    assert results[0][0].span_text == "fever"
    assert results[2][0].span_text == "cough"


def test_tag_batch_context_alignment(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Verify that the correct context text is passed to assertion detector for each item in batch,
    even if texts are identical.
    """
    texts = ["Context A", "Context B"]
    spanA = ExtractedSpan(text="Entity", label="L", start=0, end=6, score=1.0)
    spanB = ExtractedSpan(text="Entity", label="L", start=0, end=6, score=1.0)

    mock_ner.extract_batch.return_value = [[spanA], [spanB]]
    mock_assertion.detect.return_value = AssertionStatus.PRESENT
    mock_linker.link.return_value = {"concept_id": "C", "concept_name": "N", "link_confidence": 1.0}

    tagger.tag_batch(texts, ["L"])

    # Check calls to assertion
    # The first call should use texts[0] ("Context A")
    # The second call should use texts[1] ("Context B")

    calls = mock_assertion.detect.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs['text'] == "Context A"
    assert calls[1].kwargs['text'] == "Context B"
