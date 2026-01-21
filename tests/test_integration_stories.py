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
def mock_sentence_transformer_integration() -> Generator[MagicMock, None, None]:
    """
    Mock SentenceTransformer for integration stories.
    We return embeddings to ensure correct matching for 'Lasix' -> 'Furosemide'.
    """
    with patch("coreason_tagger.linker.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value

        def encode_side_effect(sentences: str | list[str], convert_to_tensor: bool = False) -> torch.Tensor:
            def get_vec(text: str) -> list[float]:
                t = str(text).lower()
                # Breast Cancer
                if "breast cancer" in t:
                    return [0.8, 0.1, 0.1]
                # Lasix / Furosemide -> Match
                if "lasix" in t or "furosemide" in t:
                    return [0.4, 0.5, 0.6]
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
async def test_story_a_family_history(mock_sentence_transformer_integration: MagicMock) -> None:
    """
    Test Story A: The "Family History" Trap
    Text: "Patient's mother died of breast cancer."
    Expected:
    - NER detects "breast cancer"
    - Assertion detects "mother" -> FAMILY
    """
    text = "Patient's mother died of breast cancer."

    # 1. Setup Dependencies
    # Mock NER to extract "breast cancer"
    mock_ner = AsyncMock()
    mock_ner.extract.return_value = [
        EntityCandidate(
            text="breast cancer",
            label="Condition",
            start=25,
            end=38,
            confidence=0.99,
            source_model="mock",
        )
    ]

    assertion_detector = RegexBasedAssertionDetector()
    codex_client = MockCoreasonCodex()
    linker = VectorLinker(codex_client=codex_client)
    tagger = CoreasonTagger(ner=mock_ner, assertion=assertion_detector, linker=linker)

    # 2. Run Tagger
    results = await tagger.tag(text, labels=["Condition"])

    # 3. Validation
    assert len(results) == 1
    entity = results[0]

    assert entity.text == "breast cancer"
    assert entity.label == "Condition"
    # Ensure it linked correctly
    assert entity.concept_name == "Breast Cancer"
    # CRITICAL: Verify Assertion Status
    assert entity.assertion == AssertionStatus.FAMILY


@pytest.mark.asyncio
async def test_story_b_ambiguous_drug(mock_sentence_transformer_integration: MagicMock) -> None:
    """
    Test Story B: The "Ambiguous Drug"
    Text: "Administered 50mg of Lasix."
    Expected:
    - NER detects "Lasix"
    - Linker maps "Lasix" -> "Furosemide" (via Codex mapping + Vector Re-ranking)
    """
    text = "Administered 50mg of Lasix."

    # 1. Setup Dependencies
    mock_ner = AsyncMock()
    mock_ner.extract.return_value = [
        EntityCandidate(
            text="Lasix",
            label="Drug",
            start=21,
            end=26,
            confidence=0.99,
            source_model="mock",
        )
    ]

    assertion_detector = RegexBasedAssertionDetector()
    codex_client = MockCoreasonCodex()
    linker = VectorLinker(codex_client=codex_client)
    tagger = CoreasonTagger(ner=mock_ner, assertion=assertion_detector, linker=linker)

    # 2. Run Tagger
    results = await tagger.tag(text, labels=["Drug"])

    # 3. Validation
    assert len(results) == 1
    entity = results[0]

    assert entity.text == "Lasix"
    assert entity.label == "Drug"
    # CRITICAL: Verify Linker Mapping
    assert entity.concept_name == "Furosemide"
    assert entity.concept_id == "RxNorm:4603"
    # Assertion should be PRESENT (default)
    assert entity.assertion == AssertionStatus.PRESENT
