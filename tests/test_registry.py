# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_tagger.registry import get_assertion_pipeline, get_redis_client


@pytest.mark.asyncio
async def test_get_redis_client_success() -> None:
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)

    with patch("redis.asyncio.from_url", return_value=mock_client) as mock_from_url:
        client = await get_redis_client("redis://localhost:6379")

        assert client is mock_client
        mock_from_url.assert_called_once_with("redis://localhost:6379", encoding="utf-8", decode_responses=True)
        mock_client.ping.assert_called_once()

    # Clear cache for other tests
    get_redis_client.cache_clear()


@pytest.mark.asyncio
async def test_get_redis_client_failure() -> None:
    with patch("redis.asyncio.from_url", side_effect=Exception("Connection failed")):
        client = await get_redis_client("redis://localhost:6379")
        assert client is None

    get_redis_client.cache_clear()


@pytest.mark.asyncio
async def test_get_redis_client_ping_failure() -> None:
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(side_effect=Exception("Ping failed"))

    with patch("redis.asyncio.from_url", return_value=mock_client):
        client = await get_redis_client("redis://localhost:6379")
        assert client is None

    get_redis_client.cache_clear()


@pytest.mark.asyncio
async def test_get_redis_client_empty_url() -> None:
    client = await get_redis_client("")
    assert client is None

    get_redis_client.cache_clear()


@pytest.mark.asyncio
async def test_get_redis_client_caching() -> None:
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)

    with patch("redis.asyncio.from_url", return_value=mock_client) as mock_from_url:
        # First call
        client1 = await get_redis_client("redis://cache:6379")

        # Second call
        client2 = await get_redis_client("redis://cache:6379")

        assert client1 is client2
        assert mock_from_url.call_count == 1

    get_redis_client.cache_clear()


@pytest.mark.asyncio
async def test_get_assertion_pipeline() -> None:
    mock_pipeline = MagicMock()

    # We mock coreason_tagger.registry.pipeline instead of transformers.pipeline
    # because that's where it's imported (if imported as `from transformers import pipeline`)
    # Wait, in registry.py it is `from transformers import pipeline`.
    # So we should patch "coreason_tagger.registry.pipeline"
    with patch("coreason_tagger.registry.pipeline", return_value=mock_pipeline) as mock_hf_pipeline:
        pipe = await get_assertion_pipeline("test-model")

        assert pipe is mock_pipeline
        mock_hf_pipeline.assert_called_once_with(
            "text-classification",
            model="test-model",
            device_map="auto",
        )

    get_assertion_pipeline.cache_clear()
