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
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy, LinkedEntity


# Mock the SentenceTransformer to avoid downloading/loading the model during tests
@pytest.fixture
def mock_sentence_transformer() -> Generator[MagicMock, None, None]:
    with patch("coreason_tagger.linker.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value
        import torch

        def side_effect(sentences: Any, convert_to_tensor: bool = False) -> Any:
            # Always return tensors to match real behavior for cos_sim
            if isinstance(sentences, str):
                return torch.tensor([0.1, 0.2, 0.3])
            elif isinstance(sentences, list):
                # Return tensor of shape (len, 3)
                return torch.tensor([[0.1, 0.2, 0.3] for _ in sentences])
            return torch.tensor([])

        mock_instance.encode.side_effect = side_effect
        yield MockClass


@pytest.fixture
def mock_codex() -> MagicMock:
    codex = MagicMock()
    # default behavior
    codex.search.return_value = []
    return codex


def create_candidate(text: str, label: str = "Symptom") -> EntityCandidate:
    return EntityCandidate(
        text=text,
        label=label,
        start=0,
        end=len(text),
        confidence=1.0,
        source_model="mock",
    )


def test_initialization(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex, model_name="test-model")
    assert linker.model_name == "test-model"
    mock_sentence_transformer.assert_called_with("test-model")


def test_resolve_empty_text(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)
    candidate = create_candidate("")
    result = linker.resolve(candidate, "", ExtractionStrategy.SPEED_GLINER)
    assert isinstance(result, LinkedEntity)
    assert result.concept_id is None
    mock_codex.search.assert_not_called()


def test_resolve_no_candidates(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)
    mock_codex.search.return_value = []

    candidate = create_candidate("unknown term")
    result = linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
    assert isinstance(result, LinkedEntity)
    assert result.concept_id is None
    mock_codex.search.assert_called_once()


def test_resolve_success(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
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

        candidate = create_candidate("severe migraine")
        result = linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        assert result.concept_id == "C2"
        assert result.concept_name == "Migraine"
        assert result.link_score == pytest.approx(0.95)


def test_resolve_integration_with_mock_codex() -> None:
    from coreason_tagger.codex_mock import MockCoreasonCodex

    codex = MockCoreasonCodex()

    with (
        patch("coreason_tagger.linker.SentenceTransformer") as MockST,
        patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim,
    ):
        mock_instance = MockST.return_value
        import torch

        mock_instance.encode.return_value = torch.tensor([0.1, 0.2, 0.3])

        linker = VectorLinker(codex_client=codex)

        # Scenario: "cold" (not in mock) vs something close?
        # MockCodex has "Headache" (score 0.5 for mismatch query).
        # We assume search returns 3 items
        mock_cos_sim.return_value = torch.tensor([[0.1, 0.9, 0.2]])

        candidate = create_candidate("head ache")
        result = linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        # We expect a result
        assert result is not None
        assert result.concept_id is not None
        assert result.link_score == pytest.approx(0.9)


def test_resolve_malformed_candidates(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
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

        candidate = create_candidate("query")
        result = linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        assert result.concept_id == "C1"
        assert result.link_score == pytest.approx(0.9)


def test_resolve_tie_handling(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
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

        candidate = create_candidate("query")
        result = linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        # Should pick the first one (index 0) due to argmax behavior
        assert result.concept_id == "C1"
        assert result.link_score == pytest.approx(0.8)


def test_resolve_reordering_logic(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
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

        candidate = create_candidate("query")
        result = linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        assert result.concept_id == "Rank2"
        assert result.concept_name == "GoodMatch"
        # Verify score is the vector score, not original score
        assert result.link_score == pytest.approx(0.95)


def test_linker_caching_basic(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    """Verify standard LRU caching behavior for repeated calls."""

    # Setup return values
    candidates = [{"concept_id": "C1", "concept_name": "Test", "score": 0.5}]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        mock_cos_sim.return_value = torch.tensor([[0.9]])

        linker = VectorLinker(codex_client=mock_codex)
        candidate = create_candidate("aspirin")

        # First call
        linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
        assert mock_codex.search.call_count == 1

        # Second call (same text)
        linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
        assert mock_codex.search.call_count == 1  # Should still be 1

        # Third call (different text)
        candidate2 = create_candidate("tylenol")
        linker.resolve(candidate2, "context", ExtractionStrategy.SPEED_GLINER)
        assert mock_codex.search.call_count == 2

        # Verify cache hits via introspection (implementation detail check)
        info = linker._cached_get_candidates.cache_info()
        assert info.hits >= 1


def test_linker_caching_instance_independence(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    """Verify that caches are independent per instance."""

    candidates = [{"concept_id": "C1", "concept_name": "Test", "score": 0.5}]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        mock_cos_sim.return_value = torch.tensor([[0.9]])

        linker1 = VectorLinker(codex_client=mock_codex)
        linker2 = VectorLinker(codex_client=mock_codex)

        candidate = create_candidate("aspirin")

        # Linker 1 called
        linker1.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
        info1 = linker1._cached_get_candidates.cache_info()
        assert info1.hits == 0
        assert info1.misses == 1

        # Linker 2 called (same text) -> Should be a miss for linker2
        linker2.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
        info2 = linker2._cached_get_candidates.cache_info()
        assert info2.hits == 0
        assert info2.misses == 1


def test_linker_caching_empty_bypass(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    """Verify that empty strings bypass the cache mechanism completely."""
    linker = VectorLinker(codex_client=mock_codex)
    candidate = create_candidate("")

    linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

    # Check cache stats - should be untouched
    info = linker._cached_get_candidates.cache_info()
    assert info.hits == 0
    assert info.misses == 0

def test_rerank_empty_candidates(mock_sentence_transformer: MagicMock, mock_codex: MagicMock) -> None:
    """Test _rerank directly with empty candidates to ensure defensive code coverage."""
    linker = VectorLinker(codex_client=mock_codex)
    result = linker._rerank("query", [])
    assert result == {}
