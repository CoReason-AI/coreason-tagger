# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from coreason_tagger.interfaces import BaseAssertionDetector, BaseLinker, BaseNERExtractor
from coreason_tagger.schema import (
    AssertionStatus,
    EntityCandidate,
    ExtractionStrategy,
    LinkedEntity,
)
from coreason_tagger.tagger import CoreasonTagger


@pytest.fixture
def mock_ner() -> MagicMock:
    ner = MagicMock(spec=BaseNERExtractor)
    ner.extract = AsyncMock()
    ner.extract_batch = AsyncMock()
    return ner


@pytest.fixture
def mock_assertion() -> MagicMock:
    assertion = MagicMock(spec=BaseAssertionDetector)
    assertion.detect = AsyncMock()
    return assertion


@pytest.fixture
def mock_linker() -> MagicMock:
    linker = MagicMock(spec=BaseLinker)
    linker.resolve = AsyncMock()
    return linker


@pytest.fixture
def tagger(mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock) -> CoreasonTagger:
    return CoreasonTagger(ner=mock_ner, assertion=mock_assertion, linker=mock_linker)


def create_linked_entity(
    text: str, label: str, concept_id: str, assertion: AssertionStatus = AssertionStatus.PRESENT
) -> LinkedEntity:
    return LinkedEntity(
        text=text,
        label=label,
        start=0,
        end=len(text),
        confidence=1.0,
        source_model="mock",
        assertion=assertion,
        concept_id=concept_id,
        concept_name="Name",
        link_score=0.9,
        strategy_used=ExtractionStrategy.SPEED_GLINER,
    )


@pytest.mark.asyncio
async def test_tag_happy_path(
    tagger: CoreasonTagger,
    mock_ner: MagicMock,
    mock_assertion: MagicMock,
    mock_linker: MagicMock,
) -> None:
    """Test the standard flow: extract -> detect -> link -> return."""
    text = "Patient has a headache."
    labels = ["Symptom"]

    # Mock NER return
    candidate = EntityCandidate(
        text="headache",
        label="Symptom",
        start=14,
        end=22,
        confidence=0.99,
        source_model="mock",
    )
    mock_ner.extract.return_value = [candidate]

    # Mock Assertion return
    mock_assertion.detect.return_value = AssertionStatus.PRESENT

    # Mock Linker return
    mock_linker.resolve.return_value = create_linked_entity("headache", "Symptom", "HP:0002315")

    results = await tagger.tag(text, labels)

    assert len(results) == 1
    entity = results[0]
    assert isinstance(entity, LinkedEntity)
    assert entity.text == "headache"
    assert entity.label == "Symptom"
    assert entity.concept_id == "HP:0002315"
    assert entity.assertion == AssertionStatus.PRESENT
    assert entity.link_score == 0.9

    # Verify calls
    mock_ner.extract.assert_awaited_once_with(text, labels)
    mock_assertion.detect.assert_awaited_once_with(text=text, span_text="headache", span_start=14, span_end=22)
    mock_linker.resolve.assert_awaited_once()


@pytest.mark.asyncio
async def test_tag_empty_text(tagger: CoreasonTagger, mock_ner: MagicMock) -> None:
    """Test that empty text returns empty list without calling extract."""
    assert await tagger.tag("", ["Label"]) == []
    mock_ner.extract.assert_not_called()


@pytest.mark.asyncio
async def test_tag_no_entities_found(tagger: CoreasonTagger, mock_ner: MagicMock) -> None:
    """Test when NER finds nothing."""
    mock_ner.extract.return_value = []
    assert await tagger.tag("Clean text", ["Label"]) == []


@pytest.mark.asyncio
async def test_tag_linking_failure(
    tagger: CoreasonTagger,
    mock_ner: MagicMock,
    mock_assertion: MagicMock,
    mock_linker: MagicMock,
) -> None:
    """Test that entities are dropped if linking fails (returns entity with no ID)."""
    text = "Unknown thing."
    candidate = EntityCandidate(text="thing", label="Unknown", start=8, end=13, confidence=0.5, source_model="mock")
    mock_ner.extract.return_value = [candidate]
    mock_assertion.detect.return_value = AssertionStatus.PRESENT
    # Linker returns entity with None concept_id
    linked_entity = LinkedEntity(
        **candidate.model_dump(),
        strategy_used=ExtractionStrategy.SPEED_GLINER,
        concept_id=None,
    )
    mock_linker.resolve.return_value = linked_entity

    results = await tagger.tag(text, ["Unknown"])
    assert results == []


@pytest.mark.asyncio
async def test_multiple_entities(
    tagger: CoreasonTagger,
    mock_ner: MagicMock,
    mock_assertion: MagicMock,
    mock_linker: MagicMock,
) -> None:
    """Test processing of multiple entities in one text."""
    text = "Patient denies fever but has cough."
    c1 = EntityCandidate(text="fever", label="Symptom", start=15, end=20, confidence=0.9, source_model="mock")
    c2 = EntityCandidate(text="cough", label="Symptom", start=29, end=34, confidence=0.9, source_model="mock")
    mock_ner.extract.return_value = [c1, c2]

    def assertion_side_effect(**kwargs: Any) -> AssertionStatus:
        if kwargs["span_text"] == "fever":
            return AssertionStatus.ABSENT
        return AssertionStatus.PRESENT

    mock_assertion.detect.side_effect = assertion_side_effect

    def linker_side_effect(entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        if entity.text == "fever":
            return create_linked_entity("fever", "Symptom", "C1")
        return create_linked_entity("cough", "Symptom", "C2")

    mock_linker.resolve.side_effect = linker_side_effect

    results = await tagger.tag(text, ["Symptom"])

    assert len(results) == 2
    # gather preserves order of results matching input candidates
    assert results[0].text == "fever"
    assert results[0].assertion == AssertionStatus.ABSENT
    assert results[0].concept_id == "C1"

    assert results[1].text == "cough"
    assert results[1].assertion == AssertionStatus.PRESENT
    assert results[1].concept_id == "C2"


@pytest.mark.asyncio
async def test_user_story_family_history(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Test User Story A: "Patient's mother died of breast cancer."
    Expectation: Assertion sets FAMILY, Linker maps to Breast Cancer.
    """
    text = "Patient's mother died of breast cancer."
    candidate = EntityCandidate(
        text="breast cancer",
        label="Diagnosis",
        start=24,
        end=37,
        confidence=0.99,
        source_model="mock",
    )
    mock_ner.extract.return_value = [candidate]

    # 2. Assertion detects "mother" in path
    mock_assertion.detect.return_value = AssertionStatus.FAMILY

    # 3. Linker maps
    mock_linker.resolve.return_value = create_linked_entity("breast cancer", "Diagnosis", "SNOMED:254837009")

    results = await tagger.tag(text, ["Diagnosis"])

    assert len(results) == 1
    entity = results[0]
    assert entity.text == "breast cancer"
    assert entity.assertion == AssertionStatus.FAMILY
    assert entity.concept_id == "SNOMED:254837009"


@pytest.mark.asyncio
async def test_user_story_ambiguous_drug(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Test User Story B: "Administered 50mg of Lasix."
    Expectation: Linker maps "Lasix" (Brand) to "Furosemide" (Generic).
    """
    text = "Administered 50mg of Lasix."
    candidate = EntityCandidate(text="Lasix", label="Drug", start=21, end=26, confidence=0.99, source_model="mock")
    mock_ner.extract.return_value = [candidate]

    mock_assertion.detect.return_value = AssertionStatus.PRESENT

    l_ent = create_linked_entity("Lasix", "Drug", "RxNorm:4603")
    l_ent.concept_name = "Furosemide"
    mock_linker.resolve.return_value = l_ent

    results = await tagger.tag(text, ["Drug"])

    assert len(results) == 1
    entity = results[0]
    assert entity.text == "Lasix"
    assert entity.concept_id == "RxNorm:4603"
    assert entity.concept_name == "Furosemide"


@pytest.mark.asyncio
async def test_robustness_empty_span_text(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Test robustness: If NER returns a span with empty text, it should be skipped.
    """
    text = "Some text."
    # Valid span + Empty span
    c1 = EntityCandidate(text="valid", label="L", start=0, end=5, confidence=1.0, source_model="mock")
    c2 = EntityCandidate(text="", label="L", start=6, end=6, confidence=0.0, source_model="mock")

    mock_ner.extract.return_value = [c1, c2]
    mock_assertion.detect.return_value = AssertionStatus.PRESENT
    mock_linker.resolve.return_value = create_linked_entity("valid", "L", "C")

    results = await tagger.tag(text, ["L"])

    # Should only contain the valid one
    assert len(results) == 1
    assert results[0].text == "valid"
    # Linker should NOT have been called for the empty one
    assert mock_linker.resolve.call_count == 1


# --- Batch Processing Tests ---


@pytest.mark.asyncio
async def test_tag_batch_happy_path(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """Test standard batch flow."""
    texts = ["Patient has fever.", "No cough detected."]
    labels = ["Symptom"]

    # Mock NER Batch Return
    # Text 1: Fever
    c1 = EntityCandidate(text="fever", label="Symptom", start=12, end=17, confidence=0.99, source_model="mock")
    # Text 2: Cough
    c2 = EntityCandidate(text="cough", label="Symptom", start=3, end=8, confidence=0.99, source_model="mock")
    mock_ner.extract_batch.return_value = [[c1], [c2]]

    async def assertion_side_effect(**kwargs: Any) -> AssertionStatus:
        if kwargs["span_text"] == "fever":
            return AssertionStatus.PRESENT
        return AssertionStatus.ABSENT

    mock_assertion.detect.side_effect = assertion_side_effect

    async def linker_side_effect(entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        if entity.text == "fever":
            return create_linked_entity("fever", "Symptom", "C_FEVER")
        return create_linked_entity("cough", "Symptom", "C_COUGH", assertion=AssertionStatus.ABSENT)

    mock_linker.resolve.side_effect = linker_side_effect

    results = await tagger.tag_batch(texts, labels)

    assert len(results) == 2

    # Check Result 1
    assert len(results[0]) == 1
    ent1 = results[0][0]
    assert ent1.text == "fever"
    assert ent1.assertion == AssertionStatus.PRESENT
    assert ent1.concept_id == "C_FEVER"

    # Check Result 2
    assert len(results[1]) == 1
    ent2 = results[1][0]
    assert ent2.text == "cough"
    assert ent2.assertion == AssertionStatus.ABSENT
    assert ent2.concept_id == "C_COUGH"

    # Verify calls
    mock_ner.extract_batch.assert_called_once_with(texts, labels)
    assert mock_assertion.detect.call_count == 2


@pytest.mark.asyncio
async def test_tag_batch_empty_input(tagger: CoreasonTagger, mock_ner: MagicMock) -> None:
    """Test empty input list."""
    assert await tagger.tag_batch([], ["Label"]) == []
    mock_ner.extract_batch.assert_not_called()


@pytest.mark.asyncio
async def test_tag_batch_mixed_empty_results(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """Test batch where some texts have no entities."""
    texts = ["Has fever.", "Nothing here.", "Has cough."]
    labels = ["Symptom"]

    c1 = EntityCandidate(text="fever", label="Symptom", start=4, end=9, confidence=0.9, source_model="mock")
    c3 = EntityCandidate(text="cough", label="Symptom", start=4, end=9, confidence=0.9, source_model="mock")

    # Return: [ [c1], [], [c3] ]
    mock_ner.extract_batch.return_value = [[c1], [], [c3]]

    mock_assertion.detect.return_value = AssertionStatus.PRESENT

    async def linker_side_effect(entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        if entity.text == "fever":
            return create_linked_entity("fever", "Symptom", "CID")
        return create_linked_entity("cough", "Symptom", "CID")

    mock_linker.resolve.side_effect = linker_side_effect

    results = await tagger.tag_batch(texts, labels)

    assert len(results) == 3
    assert len(results[0]) == 1
    assert len(results[1]) == 0
    assert len(results[2]) == 1
    assert results[0][0].text == "fever"
    assert results[2][0].text == "cough"


@pytest.mark.asyncio
async def test_tag_batch_context_alignment(
    tagger: CoreasonTagger, mock_ner: MagicMock, mock_assertion: MagicMock, mock_linker: MagicMock
) -> None:
    """
    Verify that the correct context text is passed to assertion detector for each item in batch,
    even if texts are identical.
    """
    texts = ["Context A", "Context B"]
    cA = EntityCandidate(text="Entity", label="L", start=0, end=6, confidence=1.0, source_model="mock")
    cB = EntityCandidate(text="Entity", label="L", start=0, end=6, confidence=1.0, source_model="mock")

    mock_ner.extract_batch.return_value = [[cA], [cB]]
    mock_assertion.detect.return_value = AssertionStatus.PRESENT
    ent = create_linked_entity("Entity", "L", "C")
    mock_linker.resolve.return_value = ent

    await tagger.tag_batch(texts, ["L"])

    # Check calls to assertion
    calls = mock_assertion.detect.call_args_list
    assert len(calls) == 2
    # Since order is not guaranteed with gather, we check that both contexts appeared
    contexts_called = sorted([call.kwargs["text"] for call in calls])
    assert contexts_called == ["Context A", "Context B"]
