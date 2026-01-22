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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coreason_tagger.linker import VectorLinker
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy, LinkedEntity


# Mock the SentenceTransformer to avoid downloading/loading the model during tests
@pytest.fixture
def mock_sentence_transformer() -> Generator[MagicMock, None, None]:
    # We patch 'coreason_tagger.registry.SentenceTransformer' because linker calls registry
    with patch("coreason_tagger.registry.SentenceTransformer") as MockClass:
        mock_instance = MockClass.return_value
        import torch

        def side_effect(sentences: Any, convert_to_tensor: bool = False) -> Any:
            if isinstance(sentences, str):
                return torch.tensor([0.1, 0.2, 0.3])
            elif isinstance(sentences, list):
                return torch.tensor([[0.1, 0.2, 0.3] for _ in sentences])
            return torch.tensor([])

        mock_instance.encode.side_effect = side_effect
        yield MockClass


@pytest.fixture
def mock_codex() -> AsyncMock:
    codex = AsyncMock()
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


def test_initialization(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    # Initialization doesn't load the model immediately anymore (lazy loaded via registry)
    # But calling resolve or access would load it.
    linker = VectorLinker(codex_client=mock_codex, model_name="test-model")
    assert linker.model_name == "test-model"
    # Registry is called only when needed.


async def test_resolve_empty_text(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)
    candidate = create_candidate("")
    result = await linker.resolve(candidate, "", ExtractionStrategy.SPEED_GLINER)
    assert isinstance(result, LinkedEntity)
    assert result.concept_id is None
    mock_codex.search.assert_not_called()


async def test_resolve_no_candidates(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)
    mock_codex.search.return_value = []

    candidate = create_candidate("unknown term")
    result = await linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
    assert isinstance(result, LinkedEntity)
    assert result.concept_id is None
    mock_codex.search.assert_called_once()


async def test_resolve_success(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    linker = VectorLinker(codex_client=mock_codex)

    # Setup candidates
    candidates = [
        {"concept_id": "C1", "concept_name": "Headache", "score": 0.5},
        {"concept_id": "C2", "concept_name": "Migraine", "score": 0.4},
    ]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        mock_cos_sim.return_value = torch.tensor([[0.8, 0.95]])

        candidate = create_candidate("severe migraine")
        result = await linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        assert result.concept_id == "C2"
        assert result.concept_name == "Migraine"
        assert result.link_score == pytest.approx(0.95)


async def test_resolve_integration_with_mock_codex() -> None:
    from coreason_tagger.codex_mock import MockCoreasonCodex

    codex = MockCoreasonCodex()

    # Patch SentenceTransformer in registry for this test
    with (
        patch("coreason_tagger.registry.SentenceTransformer") as MockST,
        patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim,
    ):
        mock_instance = MockST.return_value
        import torch

        mock_instance.encode.return_value = torch.tensor([0.1, 0.2, 0.3])

        linker = VectorLinker(codex_client=codex)

        mock_cos_sim.return_value = torch.tensor([[0.1, 0.9, 0.2]])

        candidate = create_candidate("head ache")
        result = await linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        assert result is not None
        assert result.concept_id is not None
        assert result.link_score == pytest.approx(0.9)


async def test_resolve_malformed_candidates(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    """Test handling of candidates missing keys like 'concept_name'."""
    linker = VectorLinker(codex_client=mock_codex)

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
        result = await linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        assert result.concept_id == "C1"
        assert result.link_score == pytest.approx(0.9)


async def test_resolve_tie_handling(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    """Test behavior when multiple candidates have the same score."""
    linker = VectorLinker(codex_client=mock_codex)

    candidates = [
        {"concept_id": "C1", "concept_name": "Same", "score": 0.5},
        {"concept_id": "C2", "concept_name": "Same", "score": 0.4},
    ]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        mock_cos_sim.return_value = torch.tensor([[0.8, 0.8]])

        candidate = create_candidate("query")
        result = await linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        # Should pick the first one (index 0) due to argmax behavior
        assert result.concept_id == "C1"
        assert result.link_score == pytest.approx(0.8)


async def test_resolve_reordering_logic(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    """Test that a lower-ranked candidate from search is promoted if it has better vector score."""
    linker = VectorLinker(codex_client=mock_codex)

    candidates = [
        {"concept_id": "Rank1", "concept_name": "PoorMatch", "score": 0.9},
        {"concept_id": "Rank2", "concept_name": "GoodMatch", "score": 0.3},
    ]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        mock_cos_sim.return_value = torch.tensor([[0.2, 0.95]])

        candidate = create_candidate("query")
        result = await linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        assert result.concept_id == "Rank2"
        assert result.concept_name == "GoodMatch"
        assert result.link_score == pytest.approx(0.95)


async def test_linker_caching_basic(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    """Verify standard LRU caching behavior for repeated calls."""
    candidates = [{"concept_id": "C1", "concept_name": "Test", "score": 0.5}]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        mock_cos_sim.return_value = torch.tensor([[0.9]])

        linker = VectorLinker(codex_client=mock_codex)
        candidate = create_candidate("aspirin")

        # First call
        await linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
        assert mock_codex.search.call_count == 1

        # Second call (same text) - Should be cached by alru_cache on _get_candidates_impl
        await linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
        assert mock_codex.search.call_count == 1  # Should still be 1

        # Third call (different text)
        candidate2 = create_candidate("tylenol")
        await linker.resolve(candidate2, "context", ExtractionStrategy.SPEED_GLINER)
        assert mock_codex.search.call_count == 2

        # Check cache info
        info = linker._get_candidates_impl.cache_info()
        assert info.hits >= 1


async def test_linker_caching_instance_independence(
    mock_sentence_transformer: MagicMock, mock_codex: AsyncMock
) -> None:
    """
    Verify that caches are working.
    Note: @alru_cache on the method shares cache across instances if 'self' is part of the key.
    Since 'self' (VectorLinker instance) is distinct, the keys (self, text) are distinct.
    So caches are effectively independent per instance.
    """
    candidates = [{"concept_id": "C1", "concept_name": "Test", "score": 0.5}]
    mock_codex.search.return_value = candidates

    with patch("coreason_tagger.linker.util.cos_sim") as mock_cos_sim:
        import torch

        mock_cos_sim.return_value = torch.tensor([[0.9]])

        linker1 = VectorLinker(codex_client=mock_codex)
        linker2 = VectorLinker(codex_client=mock_codex)

        candidate = create_candidate("aspirin")

        # Linker 1 called
        await linker1.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)
        # Linker 2 called (same text)
        await linker2.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

        # Since instances are different, cache keys are different.
        # So search should be called twice.
        assert mock_codex.search.call_count == 2


async def test_linker_caching_empty_bypass(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    """Verify that empty strings bypass the cache mechanism completely."""
    linker = VectorLinker(codex_client=mock_codex)
    candidate = create_candidate("")

    await linker.resolve(candidate, "context", ExtractionStrategy.SPEED_GLINER)

    # Cache should not be touched for empty string because resolve returns early
    assert mock_codex.search.call_count == 0


async def test_rerank_empty_candidates(mock_sentence_transformer: MagicMock, mock_codex: AsyncMock) -> None:
    """Test _rerank directly with empty candidates to ensure defensive code coverage."""
    linker = VectorLinker(codex_client=mock_codex)
    result = await linker._rerank("query", [])
    assert result == {}
