# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import MagicMock, patch

import pytest
from coreason_tagger.codex_real import RealCoreasonCodex


@pytest.fixture
def client() -> RealCoreasonCodex:
    return RealCoreasonCodex(api_url="http://test-api.com")


@patch("requests.get")
async def test_search_success(mock_get: MagicMock, client: RealCoreasonCodex) -> None:
    """Test successful search request."""
    mock_response = MagicMock()
    mock_response.json.return_value = [{"concept_id": "C1", "concept_name": "Test"}]
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    # RealCoreasonCodex uses requests (blocking), so we mock it.
    # Note: the actual code calls requests.get directly.
    # However, since search is async, but implementation uses requests (blocking),
    # we need to be careful. The current implementation in codex_real.py
    # defines async def search(...) but calls requests.get inside.
    # This blocks the loop. Ideally we should use httpx or run_in_executor.
    # But for this test, we just verify the logic.

    results = await client.search("query", top_k=5)

    assert len(results) == 1
    assert results[0]["concept_id"] == "C1"

    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "http://test-api.com/search"
    assert kwargs["params"] == {"q": "query", "top_k": "5"}
    assert kwargs["timeout"] == 5


@patch("requests.get")
async def test_get_concept_success(mock_get: MagicMock, client: RealCoreasonCodex) -> None:
    """Test successful get_concept request."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"concept_id": "C1", "concept_name": "Test"}
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    result = await client.get_concept("C1")

    assert result["concept_id"] == "C1"

    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "http://test-api.com/concept/C1"
    assert kwargs["timeout"] == 5


@patch("requests.get")
async def test_search_failure(mock_get: MagicMock, client: RealCoreasonCodex) -> None:
    """Test search failure (e.g. network error)."""
    mock_get.side_effect = Exception("Network Error")

    with pytest.raises(Exception, match="Network Error"):
        await client.search("query")


@patch("requests.get")
async def test_get_concept_failure(mock_get: MagicMock, client: RealCoreasonCodex) -> None:
    """Test get_concept failure."""
    mock_get.side_effect = Exception("Network Error")

    with pytest.raises(Exception, match="Network Error"):
        await client.get_concept("C1")
