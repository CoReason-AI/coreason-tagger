# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Any, Generator, List
from unittest.mock import MagicMock, patch

import pytest
import torch
from coreason_tagger.codex_mock import MockCoreasonCodex
from coreason_tagger.linker import VectorLinker
from coreason_tagger.schema import ExtractedSpan


@pytest.fixture
def mock_sentence_transformer_context() -> Generator[MagicMock, None, None]:
    """
    Mock SentenceTransformer that returns embeddings reflecting semantic meaning.
    We use orthogonal vectors to ensure cosine similarity differentiates them properly.
    """
    with patch("coreason_tagger.linker.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value

        def encode_side_effect(sentences: Any, convert_to_tensor: bool = False) -> Any:
            # Helper to generate vector
            def get_vec(text: Any) -> List[float]:
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
                # Default / Irrelevant -> Z axis
                return [0.0, 0.0, 1.0]

            if isinstance(sentences, str):
                return torch.tensor(get_vec(sentences)).float()
            elif isinstance(sentences, list):
                return torch.tensor([get_vec(s) for s in sentences]).float()
            return torch.tensor([])

        mock_instance.encode.side_effect = encode_side_effect
        yield MockClass


def test_contextual_linking_disambiguation(mock_sentence_transformer_context: MagicMock) -> None:
    """
    Test that the linker correctly disambiguates 'Cold' based on context.

    Case 1: "Patient caught a cold." -> Should link to Common Cold (Infection)
    Case 2: "Patient reports feeling cold." -> Should link to Chills (Sensation)
    """
    codex = MockCoreasonCodex()
    linker = VectorLinker(codex_client=codex)

    # Case 1: Infection Context
    # "caught" -> [1.0, 0.0, 0.0]
    # "Common Cold" -> [1.0, 0.0, 0.0] -> Sim 1.0
    # "Chills" -> [0.0, 1.0, 0.0] -> Sim 0.0
    text1 = "Patient caught a cold last week."
    span1 = ExtractedSpan(text="cold", label="Condition", start=17, end=21, score=0.9, context=text1)

    result1 = linker.link(span1)

    assert result1["concept_name"] == "Common Cold"
    assert result1["concept_id"] == "SNOMED:82272006"

    # Case 2: Sensation Context
    # "feeling" -> [0.0, 1.0, 0.0]
    # "Common Cold" -> [1.0, 0.0, 0.0] -> Sim 0.0
    # "Chills" -> [0.0, 1.0, 0.0] -> Sim 1.0
    text2 = "Patient reports feeling cold and shivering."
    span2 = ExtractedSpan(text="cold", label="Symptom", start=24, end=28, score=0.9, context=text2)

    result2 = linker.link(span2)
    assert result2["concept_name"] == "Chills"
    assert result2["concept_id"] == "SNOMED:44077006"


def test_contextual_linking_fallback(mock_sentence_transformer_context: MagicMock) -> None:
    """
    Test that if context is missing, it falls back to the text itself.
    """
    codex = MockCoreasonCodex()
    linker = VectorLinker(codex_client=codex)

    # Fallback uses text "cold" -> [0.7, 0.7, 0.0]
    # "Common Cold" [1.0, 0.0, 0.0] -> Sim ~0.7
    # "Chills" [0.0, 1.0, 0.0] -> Sim ~0.7
    # "Migraine" [0.0, 0.0, 1.0] -> Sim 0.0

    # Both targets are tied at ~0.7.
    # Should pick the first candidate from search results: Common Cold.

    span = ExtractedSpan(
        text="cold",
        label="Condition",
        start=0,
        end=4,
        score=0.9,
        context="",  # Empty context
    )

    result = linker.link(span)

    # Verify it picks one of the valid targets, not Migraine
    assert result["concept_name"] in ["Common Cold", "Chills"]
    # Specifically, due to search order:
    assert result["concept_name"] == "Common Cold"
