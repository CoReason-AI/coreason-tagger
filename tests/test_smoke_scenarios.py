# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import AsyncMock, patch

import pytest

from coreason_tagger.assertion_detector import RegexBasedAssertionDetector
from coreason_tagger.codex_mock import MockCoreasonCodex
from coreason_tagger.linker import VectorLinker
from coreason_tagger.schema import AssertionStatus, EntityCandidate
from coreason_tagger.tagger import CoreasonTagger


@pytest.fixture
def tagger_components():  # type: ignore
    """
    Setup tagger components with MOCKED NER but REAL Assertion, Linker, and Codex.
    This ensures we test the Codex data and Linking logic without invoking heavy models.
    """
    mock_ner = AsyncMock()
    assertion = RegexBasedAssertionDetector()
    codex_client = MockCoreasonCodex()
    # Mock embedding model in VectorLinker to avoid loading sentence-transformers
    # We just need it to return *some* similarity, or we assume Linker behavior.
    # Wait, VectorLinker.resolve uses _rerank which uses self.model.encode.
    # To properly test VectorLinker without loading model, we should mock the model inside it.

    # However, VectorLinker uses `get_sentence_transformer` from registry.
    # We can mock that.

    with patch("coreason_tagger.linker.get_sentence_transformer", new_callable=AsyncMock):
        # We will mock `VectorLinker._rerank` to just return the first candidate.
        # This is sufficient for verifying that Codex *found* the right candidate.

        linker = VectorLinker(codex_client=codex_client)
        linker._rerank = AsyncMock(side_effect=lambda q, c: c[0] if c else {})  # type: ignore # Just pick top 1

        tagger = CoreasonTagger(ner=mock_ner, assertion=assertion, linker=linker)
        return tagger, mock_ner


@pytest.mark.asyncio
async def test_smoke_fever_absent(tagger_components) -> None:  # type: ignore
    """
    Scenario: "Patient denies fever."
    Expected: Entity 'fever', Assertion ABSENT, Concept SNOMED:386661006 (Fever).
    """
    tagger, mock_ner = tagger_components
    text = "Patient denies fever."

    # Mock NER output
    mock_ner.extract.return_value = [
        EntityCandidate(text="fever", start=15, end=20, label="Symptom", confidence=0.99, source_model="mock")
    ]

    results = await tagger.tag(text, labels=["Symptom"])

    assert len(results) == 1
    entity = results[0]

    assert entity.text == "fever"
    assert entity.assertion == AssertionStatus.ABSENT
    assert entity.concept_id == "SNOMED:386661006"
    assert entity.concept_name == "Fever"


@pytest.mark.asyncio
async def test_smoke_severe_headache_synonym(tagger_components) -> None:  # type: ignore
    """
    Scenario: "Patient has a severe headache."
    Expected: Entity 'severe headache', Assertion PRESENT, Concept SNOMED:25064002 (Headache).
    This tests the 'severe headache' -> 'Headache' synonym mapping.
    """
    tagger, mock_ner = tagger_components
    text = "Patient has a severe headache."

    # Mock NER output
    mock_ner.extract.return_value = [
        EntityCandidate(text="severe headache", start=14, end=29, label="Symptom", confidence=0.99, source_model="mock")
    ]

    results = await tagger.tag(text, labels=["Symptom"])

    assert len(results) == 1
    entity = results[0]

    assert entity.text == "severe headache"
    assert entity.assertion == AssertionStatus.PRESENT
    assert entity.concept_id == "SNOMED:25064002"
    assert entity.concept_name == "Headache"


@pytest.mark.asyncio
async def test_smoke_boston_location(tagger_components) -> None:  # type: ignore
    """
    Scenario: "Patient lives in Boston."
    Expected: Entity 'Boston', Assertion PRESENT, Concept GEO:BOSTON.
    """
    tagger, mock_ner = tagger_components
    text = "Patient lives in Boston."

    # Mock NER output
    mock_ner.extract.return_value = [
        EntityCandidate(text="Boston", start=17, end=23, label="City", confidence=0.99, source_model="mock")
    ]

    results = await tagger.tag(text, labels=["City"])

    assert len(results) == 1
    entity = results[0]

    assert entity.text == "Boston"
    assert entity.concept_id == "GEO:BOSTON"
    assert entity.concept_name == "Boston"
