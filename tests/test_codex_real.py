# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import httpx
import pytest
import respx
from coreason_tagger.codex_real import RealCoreasonCodex


@pytest.fixture
def client() -> RealCoreasonCodex:
    return RealCoreasonCodex(api_url="http://test-api.com")


@pytest.mark.asyncio
async def test_search_success(client: RealCoreasonCodex) -> None:
    """Test successful search request using httpx mock (respx)."""
    with respx.mock(base_url="http://test-api.com") as respx_mock:
        route = respx_mock.get("/search").respond(status_code=200, json=[{"concept_id": "C1", "concept_name": "Test"}])

        results = await client.search("query", top_k=5)

        assert len(results) == 1
        assert results[0]["concept_id"] == "C1"
        assert route.called
        assert route.call_count == 1
        request = route.calls.last.request
        assert request.url.params["q"] == "query"
        assert request.url.params["top_k"] == "5"


@pytest.mark.asyncio
async def test_get_concept_success(client: RealCoreasonCodex) -> None:
    """Test successful get_concept request using httpx mock (respx)."""
    with respx.mock(base_url="http://test-api.com") as respx_mock:
        route = respx_mock.get("/concept/C1").respond(
            status_code=200, json={"concept_id": "C1", "concept_name": "Test"}
        )

        result = await client.get_concept("C1")

        assert result["concept_id"] == "C1"
        assert route.called


@pytest.mark.asyncio
async def test_search_failure(client: RealCoreasonCodex) -> None:
    """Test search failure (e.g. network error) using httpx mock."""
    with respx.mock(base_url="http://test-api.com") as respx_mock:
        respx_mock.get("/search").mock(side_effect=httpx.NetworkError("Network Error"))

        with pytest.raises(httpx.NetworkError):
            await client.search("query")


@pytest.mark.asyncio
async def test_get_concept_failure(client: RealCoreasonCodex) -> None:
    """Test get_concept failure using httpx mock."""
    with respx.mock(base_url="http://test-api.com") as respx_mock:
        respx_mock.get("/concept/C1").mock(side_effect=httpx.NetworkError("Network Error"))

        with pytest.raises(httpx.NetworkError):
            await client.get_concept("C1")


@pytest.mark.asyncio
async def test_search_http_error(client: RealCoreasonCodex) -> None:
    """Test HTTP error status (e.g., 500) raises exception."""
    with respx.mock(base_url="http://test-api.com") as respx_mock:
        respx_mock.get("/search").respond(status_code=500)

        with pytest.raises(httpx.HTTPStatusError):
            await client.search("query")
