# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from coreason_tagger.linker import VectorLinker


# Mock the SentenceTransformer to avoid downloading/loading the model during tests
@pytest.fixture  # type: ignore
def mock_sentence_transformer() -> Generator[MagicMock, None, None]:
    with patch("coreason_tagger.linker.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value
        # Mock encode to return dummy tensors
        # We need something that supports util.cos_sim, which expects tensors or ndarrays.
        # Since we mock, we can just return lists and mock cos_sim too, or let it run if it handles lists?
        # sentence_transformers.util.cos_sim expects torch Tensors or numpy arrays.
        # Let's mock util.cos_sim as well to avoid torch dependency logic in the unit test if possible,
        # OR we just return numpy arrays from encode.
        import numpy as np

        def side_effect(sentences: Any, convert_to_tensor: bool = False) -> Any:
            if isinstance(sentences, str):
                return np.array([0.1, 0.2, 0.3])  # Dummy 3D vector
            elif isinstance(sentences, list):
                # Return list of vectors
                return np.array([[0.1, 0.2, 0.3] for _ in sentences])
            return np.array([])

        mock_instance.encode.side_effect = side_effect
        yield MockClass


@pytest.fixture  # type: ignore
def mock_codex() -> MagicMock:
    codex = MagicMock()
    # default behavior
    codex.search.return_value = []
    return codex


def test_initialization(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex, model_name="test-model")
    assert linker.model_name == "test-model"
    mock_sentence_transformer.assert_called_with("test-model")


def test_link_empty_text(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)
    result = linker.link("", "Symptom")
    assert result == {}
    mock_codex.search.assert_not_called()


def test_link_no_candidates(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)
    mock_codex.search.return_value = []

    result = linker.link("unknown term", "Symptom")
    assert result == {}
    mock_codex.search.assert_called_once()


def test_link_success(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)

    # Setup candidates
    candidates = [
        {"concept_id": "C1", "concept_name": "Headache", "score": 0.5},
        {"concept_id": "C2", "concept_name": "Migraine", "score": 0.4},
    ]
    mock_codex.search.return_value = candidates

    # We need to control the scoring to verify re-ranking.
    # Since we mocked encode to return identical vectors, cos_sim would return 1.0 for all.
    # To test re-ranking logic, we should probably mock util.cos_sim.

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        # Return a tensor where the second candidate (Migraine) has a higher score
        # Shape: (1, 2) -> [[score1, score2]]
        # Let's say C2 is the better match
        mock_cos_sim.return_value = torch.tensor([[0.8, 0.95]])

        result = linker.link("severe migraine", "Symptom")

        assert result["concept_id"] == "C2"
        assert result["concept_name"] == "Migraine"
        assert result["link_confidence"] == pytest.approx(0.95)


def test_link_integration_with_mock_codex() -> None:
    """
    Test using the actual MockCoreasonCodex (from the codebase) but mocked SentenceTransformer.
    This ensures the interaction between Linker and the specific MockCodex implementation works.
    """
    from coreason_tagger.codex_mock import MockCoreasonCodex

    codex = MockCoreasonCodex()

    with (
        patch("coreason_tagger.linker.SentenceTransformer") as MockST,
        patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim,
    ):
        mock_instance = MockST.return_value
        import torch

        # Mock encode: doesn't matter what it returns as long as cos_sim handles it
        mock_instance.encode.return_value = "dummy_embedding"

        linker = VectorLinker(codex_client=codex)

        # Scenario: "cold" -> Expect "Viral Rhinitis" (if it was in the mock) or similar.
        # But our MockCodex only has Migraine, Furosemide, Headache.
        # Let's search for "head ache"

        # Candidates from MockCodex for "head ache":
        # It has "Headache" (score 0.5 or 0.9 depending on exact match).
        # MockCodex logic: score 0.9 if query in name else 0.5.
        # "head ache" is not in "Headache", so score 0.5.

        # Let's mock cos_sim to pick "Headache"
        # The mock codex returns sorted list.
        # We just need to verify linker calls search and uses the result.

        # Suppose search returns 3 items
        mock_cos_sim.return_value = torch.tensor([[0.1, 0.9, 0.2]])  # 2nd item is best

        result = linker.link("head ache", "Symptom")

        # We expect a result
        assert result is not None
        assert "concept_id" in result
        assert "link_confidence" in result
        assert result["link_confidence"] == pytest.approx(0.9)
