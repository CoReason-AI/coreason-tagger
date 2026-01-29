# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import json
from unittest.mock import AsyncMock, patch

import pytest

from coreason_tagger.linker import VectorLinker


@pytest.fixture
def mock_codex() -> AsyncMock:
    codex = AsyncMock()
    codex.search.return_value = [{"concept_id": "C1", "concept_name": "Test", "score": 1.0}]
    return codex


@pytest.fixture
def mock_redis() -> AsyncMock:
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    return client


@pytest.mark.asyncio
async def test_redis_hit(mock_codex: AsyncMock, mock_redis: AsyncMock) -> None:
    # Simulate data in Redis
    cached_data = [{"concept_id": "C1", "concept_name": "Cached", "score": 1.0}]
    mock_redis.get.return_value = json.dumps(cached_data)

    with (
        patch("coreason_tagger.linker.get_redis_client", AsyncMock(return_value=mock_redis)),
        patch("coreason_tagger.config.settings.REDIS_URL", "redis://localhost:6379"),
    ):
        linker = VectorLinker(codex_client=mock_codex)
        # Verify L2
        # Note: alru_cache (L1) sits on top.
        # We need to call _get_candidates_impl directly or clear cache if testing logic.

        # Force clear L1 cache just in case
        linker._get_candidates_impl.cache_clear()

        result = await linker._get_candidates_impl("query")

        assert result == cached_data
        assert result[0]["concept_name"] == "Cached"
        mock_redis.get.assert_called_once()
        mock_codex.search.assert_not_called()  # Should not hit Codex


@pytest.mark.asyncio
async def test_redis_miss_and_write(mock_codex: AsyncMock, mock_redis: AsyncMock) -> None:
    # Simulate Redis miss
    mock_redis.get.return_value = None

    with (
        patch("coreason_tagger.linker.get_redis_client", AsyncMock(return_value=mock_redis)),
        patch("coreason_tagger.config.settings.REDIS_URL", "redis://localhost:6379"),
    ):
        linker = VectorLinker(codex_client=mock_codex)
        linker._get_candidates_impl.cache_clear()

        result = await linker._get_candidates_impl("query")

        assert len(result) == 1
        assert result[0]["concept_name"] == "Test"

        # Verify interaction
        mock_redis.get.assert_called_once()
        mock_codex.search.assert_called_once()
        mock_redis.set.assert_called_once()  # Should write back


@pytest.mark.asyncio
async def test_redis_not_configured(mock_codex: AsyncMock) -> None:
    with (
        patch("coreason_tagger.linker.get_redis_client", AsyncMock(return_value=None)),
        patch("coreason_tagger.config.settings.REDIS_URL", None),
    ):
        linker = VectorLinker(codex_client=mock_codex)
        linker._get_candidates_impl.cache_clear()

        result = await linker._get_candidates_impl("query")

        assert len(result) == 1
        mock_codex.search.assert_called_once()


@pytest.mark.asyncio
async def test_redis_failure_fail_open(mock_codex: AsyncMock, mock_redis: AsyncMock) -> None:
    # Simulate Redis throwing exception on get
    mock_redis.get.side_effect = Exception("Connection Refused")

    with (
        patch("coreason_tagger.linker.get_redis_client", AsyncMock(return_value=mock_redis)),
        patch("coreason_tagger.config.settings.REDIS_URL", "redis://localhost:6379"),
    ):
        linker = VectorLinker(codex_client=mock_codex)
        linker._get_candidates_impl.cache_clear()

        # Should not raise exception, but proceed to Codex
        result = await linker._get_candidates_impl("query")

        assert len(result) == 1
        mock_codex.search.assert_called_once()


@pytest.mark.asyncio
async def test_redis_write_failure_safe(mock_codex: AsyncMock, mock_redis: AsyncMock) -> None:
    # Simulate Redis miss, then failure on set
    mock_redis.get.return_value = None
    mock_redis.set.side_effect = Exception("Write Failed")

    with (
        patch("coreason_tagger.linker.get_redis_client", AsyncMock(return_value=mock_redis)),
        patch("coreason_tagger.config.settings.REDIS_URL", "redis://localhost:6379"),
    ):
        linker = VectorLinker(codex_client=mock_codex)
        linker._get_candidates_impl.cache_clear()

        # Should succeed returning data even if write fails
        result = await linker._get_candidates_impl("query")

        assert len(result) == 1
        mock_codex.search.assert_called_once()
        mock_redis.set.assert_called_once()
