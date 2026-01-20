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
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy


@pytest.fixture
def mock_sentence_transformer_complex() -> Generator[MagicMock, None, None]:
    """
    Mock ST with ability to distinguish local context.
    Vectors:
    - Infection/Viral/Caught -> [1, 0, 0]
    - Sensation/Feeling/Shivering -> [0, 1, 0]
    - Cold (Ambiguous) -> [0.7, 0.7, 0]
    """
    with patch("coreason_tagger.linker.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value

        def encode_side_effect(sentences: Any, convert_to_tensor: bool = False) -> Any:
            def get_vec(text: Any) -> List[float]:
                t = str(text).lower()
                # If we see infection keywords
                score_x = 0.0
                if "caught" in t or "common cold" in t or "viral" in t:
                    score_x = 1.0

                # If we see sensation keywords
                score_y = 0.0
                if "feeling" in t or "shivering" in t or "chills" in t:
                    score_y = 1.0

                # If both are present (mixed context without windowing)
                if score_x > 0 and score_y > 0:
                    # Return a mix? Or maybe one dominates?
                    # Let's say mixed
                    return [0.5, 0.5, 0.0]

                if score_x > 0:
                    return [1.0, 0.0, 0.0]
                if score_y > 0:
                    return [0.0, 1.0, 0.0]

                # Default for "cold" alone or irrelevant
                if "cold" in t:
                    return [0.7, 0.7, 0.0]

                return [0.0, 0.0, 1.0]  # Z axis default

            if isinstance(sentences, str):
                return torch.tensor(get_vec(sentences)).float()
            elif isinstance(sentences, list):
                return torch.tensor([get_vec(s) for s in sentences]).float()
            return torch.tensor([])

        mock_instance.encode.side_effect = encode_side_effect
        yield MockClass


def test_mixed_meanings_in_same_sentence(mock_sentence_transformer_complex: MagicMock) -> None:
    """
    Test that the linker can disambiguate two identical mentions in the same document
    if it correctly uses windowing/local context.
    """
    codex = MockCoreasonCodex()
    # Explicitly set window size to 50 to match previous behavior/tests
    linker = VectorLinker(codex_client=codex, window_size=50)

    # Text length ~140 chars.
    # "caught a cold" is early. "feeling cold" is late.
    # Gap is ~100 chars.
    text = (
        "The patient stated that they caught a cold during their recent trip to the mountains. "
        "It was a long journey. Now, back home, they reported feeling cold in their fingers."
    )

    # Find indices
    idx1 = text.find("caught a cold") + len("caught a ")  # index of "cold"
    idx2 = text.find("feeling cold") + len("feeling ")  # index of "cold"

    c1 = EntityCandidate(
        text="cold",
        label="Condition",
        start=idx1,
        end=idx1 + 4,
        confidence=0.9,
        source_model="mock",
    )

    c2 = EntityCandidate(text="cold", label="Symptom", start=idx2, end=idx2 + 4, confidence=0.9, source_model="mock")

    # IF logic uses WINDOWED context (50 chars):
    # Span 1 window: "...caught a cold during..." -> Includes "caught". Excludes "feeling".
    # Span 2 window: "...reported feeling cold in..." -> Includes "feeling". Excludes "caught".

    result1 = linker.resolve(c1, text, ExtractionStrategy.SPEED_GLINER)
    result2 = linker.resolve(c2, text, ExtractionStrategy.SPEED_GLINER)

    assert result1.concept_name == "Common Cold"
    assert result2.concept_name == "Chills"


def test_configurable_window_size(mock_sentence_transformer_complex: MagicMock) -> None:
    """
    Test that the window size is configurable and affects the output.
    If we set a massive window size, it should fail to disambiguate (like the original failure case).
    """
    codex = MockCoreasonCodex()
    # Set window size to 500 (covers the whole text)
    linker = VectorLinker(codex_client=codex, window_size=500)

    text = "Patient caught a cold and later reported feeling cold."

    # Span 2: "feeling cold"
    c2 = EntityCandidate(text="cold", label="Symptom", start=49, end=53, confidence=0.9, source_model="mock")

    # With massive window, it sees "caught" (infection) and "feeling" (sensation).
    # Based on our mock logic for mixed signals, it returns [0.5, 0.5, 0.0] or similar ambiguity.
    # And due to tie-breaking, it likely picks "Common Cold" (first candidate) over "Chills".

    result = linker.resolve(c2, text, ExtractionStrategy.SPEED_GLINER)

    # It should effectively FAIL to identify "Chills" correctly due to context pollution
    # OR be ambiguous. In this mock setup, it defaults to Common Cold on ties/mixes.
    assert result.concept_name == "Common Cold"
