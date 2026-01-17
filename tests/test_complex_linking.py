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

    We use a longer text to ensure the 50-char window effectively separates the contexts.
    """
    codex = MockCoreasonCodex()
    linker = VectorLinker(codex_client=codex)

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

    span1 = ExtractedSpan(text="cold", label="Condition", start=idx1, end=idx1 + 4, score=0.9, context=text)

    span2 = ExtractedSpan(text="cold", label="Symptom", start=idx2, end=idx2 + 4, score=0.9, context=text)

    # IF logic uses WINDOWED context (50 chars):
    # Span 1 window: "...caught a cold during..." -> Includes "caught". Excludes "feeling".
    # Span 2 window: "...reported feeling cold in..." -> Includes "feeling". Excludes "caught".

    result1 = linker.link(span1)
    result2 = linker.link(span2)

    assert result1["concept_name"] == "Common Cold"
    assert result2["concept_name"] == "Chills"
