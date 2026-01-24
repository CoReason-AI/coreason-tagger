# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

# mypy: ignore-errors

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_tagger.ner import ExtractorFactory, ReasoningExtractor
from coreason_tagger.schema import EntityCandidate


@pytest.fixture
def candidate_factory():
    def _create(text, start, end, label="test"):
        return EntityCandidate(text=text, start=start, end=end, label=label, confidence=0.9, source_model="test")

    return _create


@pytest.mark.asyncio
async def test_clustering_no_overlap(candidate_factory):
    extractor = ReasoningExtractor()

    # [0, 5], [10, 15]
    c1 = candidate_factory("hello", 0, 5)
    c2 = candidate_factory("world", 10, 15)

    clustered = extractor._cluster_candidates([c1, c2])

    assert len(clustered) == 2
    assert clustered[0] == c1
    assert clustered[1] == c2


@pytest.mark.asyncio
async def test_clustering_empty(candidate_factory):
    extractor = ReasoningExtractor()
    clustered = extractor._cluster_candidates([])
    assert clustered == []


@pytest.mark.asyncio
async def test_clustering_overlap_merge(candidate_factory):
    extractor = ReasoningExtractor()

    c1 = candidate_factory("history of breast cancer", 0, 24)
    c2 = candidate_factory("breast cancer", 11, 24)

    clustered = extractor._cluster_candidates([c1, c2])

    assert len(clustered) == 1
    assert clustered[0].text == "history of breast cancer"


@pytest.mark.asyncio
async def test_clustering_overlap_exact(candidate_factory):
    extractor = ReasoningExtractor()

    c1 = candidate_factory("test", 0, 4)
    c2 = candidate_factory("test", 0, 4)

    clustered = extractor._cluster_candidates([c1, c2])

    assert len(clustered) == 1
    assert clustered[0].text == "test"


@pytest.mark.asyncio
async def test_clustering_transitive(candidate_factory):
    extractor = ReasoningExtractor()

    c1 = candidate_factory("aaaaaaaaaa", 0, 10)
    c2 = candidate_factory("bbbbbbbbbb", 2, 12)
    c3 = candidate_factory("cccccccccc", 4, 14)

    clustered = extractor._cluster_candidates([c1, c2, c3])

    assert len(clustered) == 1
    assert len(clustered[0].text) == 10


@pytest.mark.asyncio
async def test_verification_empty():
    extractor = ReasoningExtractor()
    verified = await extractor._verify_with_llm("context", [])
    assert verified == []


@pytest.mark.asyncio
async def test_verification_success(candidate_factory):
    extractor = ReasoningExtractor()
    c1 = candidate_factory("valid", 0, 5)
    c2 = candidate_factory("invalid", 10, 17)

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value.choices = [MagicMock(message=MagicMock(content='{"valid_ids": [0]}'))]

        verified = await extractor._verify_with_llm("context", [c1, c2])

        assert len(verified) == 1
        assert verified[0] == c1


@pytest.mark.asyncio
async def test_verification_timeout(candidate_factory):
    extractor = ReasoningExtractor()
    c1 = candidate_factory("test", 0, 4)

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = asyncio.TimeoutError()

        verified = await extractor._verify_with_llm("context", [c1])

        # Fail open
        assert len(verified) == 1
        assert verified[0] == c1


@pytest.mark.asyncio
async def test_load_model():
    extractor = ReasoningExtractor()
    extractor.gliner.load_model = AsyncMock()
    await extractor.load_model()
    extractor.gliner.load_model.assert_called_once()


@pytest.mark.asyncio
async def test_extract_full_flow(candidate_factory):
    extractor = ReasoningExtractor()

    c1 = candidate_factory("history of breast cancer", 0, 24)
    c2 = candidate_factory("breast cancer", 11, 24)

    # Mock GLiNER
    extractor.gliner.extract = AsyncMock(return_value=[c1, c2])

    # Mock LLM
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value.choices = [MagicMock(message=MagicMock(content='{"valid_ids": [0]}'))]

        result = await extractor.extract("history of breast cancer", ["test"])

        assert len(result) == 1
        assert result[0].text == "history of breast cancer"


@pytest.mark.asyncio
async def test_extract_batch(candidate_factory):
    extractor = ReasoningExtractor()

    c1 = candidate_factory("c1", 0, 2)

    # Mock GLiNER batch extract
    # Returns list of lists of candidates
    extractor.gliner.extract_batch = AsyncMock(return_value=[[c1], []])

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value.choices = [MagicMock(message=MagicMock(content='{"valid_ids": [0]}'))]

        results = await extractor.extract_batch(["text1", "text2"], ["label"])

        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[0][0] == c1
        assert len(results[1]) == 0


def test_factory_fallback():
    factory = ExtractorFactory()
    # Cast string to ExtractionStrategy to simulate unknown/invalid enum (if types ignored) or future enum
    # Or just subclass Enum?
    # Since type hint is ExtractionStrategy, mypy complains if I pass string.
    # But runtime it might work if I force it.
    # However, Python Enums are strict.
    # I can't easily pass an invalid enum member unless I mock.
    # But wait, the `else` block in factory is unreachable if Enum is exhaustive and handled.
    # But `ExtractionStrategy` has 3 members, and `get_extractor` handles all 3 explicitely.
    # The `else` block is strictly for safety/mypy exhaustiveness if not all cases covered.
    # To hit it, I would need a strategy not in the `if/elif`.
    # I can add a fake strategy to the Enum via patching or assume it's unreachable (pragma: no cover).
    # But strict coverage requires it.
    # I'll try to pass a string "UNKNOWN" forcefully.

    extractor = factory.get_extractor("UNKNOWN")  # type: ignore
    from coreason_tagger.ner import GLiNERExtractor

    assert isinstance(extractor, GLiNERExtractor)
