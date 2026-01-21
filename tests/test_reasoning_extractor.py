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

from coreason_tagger.ner import ReasoningExtractor
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
async def test_clustering_overlap_merge(candidate_factory):
    extractor = ReasoningExtractor()

    # "history of breast cancer" (0-24)
    # "breast cancer" (11-24)
    # Intersection 13. Union 24. 13/24 = 0.54 > 0.5. MERGE.

    c1 = candidate_factory("history of breast cancer", 0, 24)
    c2 = candidate_factory("breast cancer", 11, 24)

    clustered = extractor._cluster_candidates([c1, c2])

    assert len(clustered) == 1
    # Should pick the longest text
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

    # A overlaps B, B overlaps C.
    # [0, 10], [2, 12], [4, 14]
    # A-B overlap: 8. Union 12. 8/12 = 0.66. Merge. -> [0, 12] (assuming we update tracking)
    # New cluster effectively [0, 12].
    # Next [4, 14]. Overlap with [0, 12] is [4, 12] len 8. Union [0, 14] len 14. 8/14 = 0.57. Merge.

    c1 = candidate_factory("aaaaaaaaaa", 0, 10)
    c2 = candidate_factory("bbbbbbbbbb", 2, 12)
    c3 = candidate_factory("cccccccccc", 4, 14)

    clustered = extractor._cluster_candidates([c1, c2, c3])

    assert len(clustered) == 1
    # Max by length.
    assert len(clustered[0].text) == 10


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

        # 1. Clustering should merge to c1
        # 2. LLM verifies c1 (index 0 of clustered)

        assert len(result) == 1
        assert result[0].text == "history of breast cancer"

        # Verify prompt content
        args, kwargs = mock_llm.call_args
        messages = kwargs["messages"]
        assert "history of breast cancer" in messages[0]["content"]
