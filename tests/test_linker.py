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
from coreason_tagger.schema import ExtractedSpan


# Mock the SentenceTransformer to avoid downloading/loading the model during tests
@pytest.fixture
def mock_sentence_transformer() -> Generator[MagicMock, None, None]:
    with patch("coreason_tagger.linker.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value
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


@pytest.fixture
def mock_codex() -> MagicMock:
    codex = MagicMock()
    # default behavior
    codex.search.return_value = []
    return codex


def create_span(text: str, label: str = "Symptom") -> ExtractedSpan:
    return ExtractedSpan(text=text, label=label, start=0, end=len(text), score=1.0)


def test_initialization(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex, model_name="test-model")
    assert linker.model_name == "test-model"
    mock_sentence_transformer.assert_called_with("test-model")


def test_link_empty_text(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)
    span = create_span("")
    result = linker.link(span)
    assert result == {}
    mock_codex.search.assert_not_called()


def test_link_no_candidates(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)
    mock_codex.search.return_value = []

    span = create_span("unknown term")
    result = linker.link(span)
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

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        # Return a tensor where the second candidate (Migraine) has a higher score
        mock_cos_sim.return_value = torch.tensor([[0.8, 0.95]])

        span = create_span("severe migraine")
        result = linker.link(span)

        assert result["concept_id"] == "C2"
        assert result["concept_name"] == "Migraine"
        assert result["link_confidence"] == pytest.approx(0.95)


def test_link_integration_with_mock_codex() -> None:
    from coreason_tagger.codex_mock import MockCoreasonCodex

    codex = MockCoreasonCodex()

    with (
        patch("coreason_tagger.linker.SentenceTransformer") as MockST,
        patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim,
    ):
        mock_instance = MockST.return_value
        import torch

        mock_instance.encode.return_value = "dummy_embedding"

        linker = VectorLinker(codex_client=codex)

        # Scenario: "cold" (not in mock) vs something close?
        # MockCodex has "Headache" (score 0.5 for mismatch query).
        # We assume search returns 3 items
        mock_cos_sim.return_value = torch.tensor([[0.1, 0.9, 0.2]])

        span = create_span("head ache")
        result = linker.link(span)

        # We expect a result
        assert result is not None
        assert "concept_id" in result
        assert "link_confidence" in result
        assert result["link_confidence"] == pytest.approx(0.9)


def test_link_malformed_candidates(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    """Test handling of candidates missing keys like 'concept_name'."""
    linker = VectorLinker(codex_client=mock_codex)

    # Candidate missing name
    candidates = [
        {"concept_id": "C1", "score": 0.5},  # No name
        {"concept_id": "C2", "concept_name": "Valid", "score": 0.4},
    ]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        # Make the first one (missing name) have higher vector score to test it can be selected
        mock_cos_sim.return_value = torch.tensor([[0.9, 0.1]])

        span = create_span("query")
        result = linker.link(span)

        assert result["concept_id"] == "C1"
        assert result["link_confidence"] == pytest.approx(0.9)


def test_link_tie_handling(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    """Test behavior when multiple candidates have the same score."""
    linker = VectorLinker(codex_client=mock_codex)

    candidates = [
        {"concept_id": "C1", "concept_name": "Same", "score": 0.5},
        {"concept_id": "C2", "concept_name": "Same", "score": 0.4},
    ]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        # Both have 0.8
        mock_cos_sim.return_value = torch.tensor([[0.8, 0.8]])

        span = create_span("query")
        result = linker.link(span)

        # Should pick the first one (index 0) due to argmax behavior
        assert result["concept_id"] == "C1"
        assert result["link_confidence"] == pytest.approx(0.8)


def test_link_reordering_logic(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    """Test that a lower-ranked candidate from search is promoted if it has better vector score."""
    linker = VectorLinker(codex_client=mock_codex)

    candidates = [
        {"concept_id": "Rank1", "concept_name": "PoorMatch", "score": 0.9},
        {"concept_id": "Rank2", "concept_name": "GoodMatch", "score": 0.3},
    ]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        # Rank2 has much better vector score
        mock_cos_sim.return_value = torch.tensor([[0.2, 0.95]])

        span = create_span("query")
        result = linker.link(span)

        assert result["concept_id"] == "Rank2"
        assert result["concept_name"] == "GoodMatch"
        # Verify score is the vector score, not original score
        assert result["link_confidence"] == pytest.approx(0.95)
