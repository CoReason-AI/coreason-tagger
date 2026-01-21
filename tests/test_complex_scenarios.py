# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from coreason_tagger.assertion_detector import RegexBasedAssertionDetector
from coreason_tagger.codex_mock import MockCoreasonCodex
from coreason_tagger.linker import VectorLinker
from coreason_tagger.schema import AssertionStatus, EntityCandidate
from coreason_tagger.tagger import CoreasonTagger


@pytest.fixture
def mock_sentence_transformer_complex() -> Generator[MagicMock, None, None]:
    """
    Mock SentenceTransformer for complex scenarios.
    """
    # Patch coreason_tagger.registry.SentenceTransformer
    with patch("coreason_tagger.registry.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value

        def encode_side_effect(sentences: str | list[str], convert_to_tensor: bool = False) -> torch.Tensor:
            def get_vec(text: str) -> list[float]:
                t = str(text).lower()
                # Infection / Viral -> X axis
                if "viral" in t or "common cold" in t or "caught" in t:
                    return [1.0, 0.0, 0.0]
                # Sensation / Chills -> Y axis
                if "chills" in t or "feeling" in t or "shivering" in t:
                    return [0.0, 1.0, 0.0]
                # Ambiguous "cold" -> Between X and Y
                if t == "cold":
                    return [0.707, 0.707, 0.0]

                # Diabetes
                if "diabetes" in t:
                    return [0.5, 0.5, 0.5]
                # Hypertension
                if "hypertension" in t:
                    return [0.6, 0.6, 0.6]

                # Default
                return [0.0, 0.0, 0.0]

            if isinstance(sentences, str):
                return torch.tensor(get_vec(sentences)).float()
            elif isinstance(sentences, list):
                return torch.tensor([get_vec(s) for s in sentences]).float()
            return torch.tensor([])

        mock_instance.encode.side_effect = encode_side_effect
        yield MockClass


@pytest.mark.asyncio
async def test_mixed_assertion_in_sentence(mock_sentence_transformer_complex: MagicMock) -> None:
    """
    Test a complex sentence where two entities have different assertion statuses.
    Text: "Mother has diabetes, but patient denies hypertension."

    Entity 1: "diabetes" -> Context "Mother" -> FAMILY
    Entity 2: "hypertension" -> Context "patient denies" -> ABSENT
    """
    text = "Mother has diabetes, but patient denies hypertension."

    # Mock NER
    mock_ner = AsyncMock()
    mock_ner.extract.return_value = [
        EntityCandidate(
            text="diabetes",
            label="Condition",
            start=11,
            end=19,
            confidence=0.99,
            source_model="mock",
        ),
        EntityCandidate(
            text="hypertension",
            label="Condition",
            start=40,
            end=52,
            confidence=0.99,
            source_model="mock",
        ),
    ]

    assertion_detector = RegexBasedAssertionDetector()
    codex_client = MockCoreasonCodex()
    linker = VectorLinker(codex_client=codex_client)
    tagger = CoreasonTagger(ner=mock_ner, assertion=assertion_detector, linker=linker)

    results = await tagger.tag(text, labels=["Condition"])

    assert len(results) == 2

    # Processed in order of appearance usually, but let's find by name
    diabetes_entity = next(e for e in results if e.text == "diabetes")
    hypertension_entity = next(e for e in results if e.text == "hypertension")

    # Verify Assertions
    assert diabetes_entity.assertion == AssertionStatus.FAMILY

    # This is the critical check for "Edge Case":
    # If assertion uses the whole sentence, it sees "Mother" and marks Hypertension as FAMILY.
    # It should be ABSENT (denies).
    assert hypertension_entity.assertion == AssertionStatus.ABSENT


@pytest.mark.asyncio
async def test_duplicate_term_disambiguation(mock_sentence_transformer_complex: MagicMock) -> None:
    """
    Test disambiguating identical terms in the same text based on local context.
    Text: "Patient caught a cold and is feeling cold."

    Span 1: "cold" (start ~17) -> Context "caught a ..." -> Common Cold
    Span 2: "cold" (start ~37) -> Context "feeling ..." -> Chills
    """
    text = "Patient caught a cold and is feeling cold."

    mock_ner = AsyncMock()
    mock_ner.extract.return_value = [
        EntityCandidate(
            text="cold",
            label="Condition",
            start=17,
            end=21,
            confidence=0.9,
            source_model="mock",
        ),
        EntityCandidate(
            text="cold",
            label="Symptom",
            start=37,
            end=41,
            confidence=0.9,
            source_model="mock",
        ),
    ]

    assertion_detector = RegexBasedAssertionDetector()
    codex_client = MockCoreasonCodex()
    # Use a small window to ensure local context dominates for this short sentence
    linker = VectorLinker(codex_client=codex_client, window_size=15)
    tagger = CoreasonTagger(ner=mock_ner, assertion=assertion_detector, linker=linker)

    results = await tagger.tag(text, labels=["Condition", "Symptom"])

    assert len(results) == 2

    # Sort by appearance to match expectations
    # Note: tagger output order depends on NER output order which we mocked sequentially
    entity1 = results[0]  # first "cold"
    entity2 = results[1]  # second "cold"

    assert entity1.text == "cold"
    assert entity1.concept_name == "Common Cold"  # Infection

    assert entity2.text == "cold"
    assert entity2.concept_name == "Chills"  # Sensation
