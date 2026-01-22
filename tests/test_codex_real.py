# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from unittest.mock import patch

import pytest
from coreason_tagger.codex_real import RealCoreasonCodex


@pytest.mark.asyncio
async def test_search_real() -> None:
    """Test RealCoreasonCodex search method with mocked requests."""
    client = RealCoreasonCodex("http://fake-api")

    mock_response = [{"concept_id": "ID:1", "concept_name": "Test"}]

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response

        results = await client.search("query", top_k=5)

        assert results == mock_response
        mock_get.assert_called_once_with("http://fake-api/search", params={"q": "query", "top_k": "5"}, timeout=5)


@pytest.mark.asyncio
async def test_get_concept_real() -> None:
    """Test RealCoreasonCodex get_concept method with mocked requests."""
    client = RealCoreasonCodex("http://fake-api")

    mock_response = {"concept_id": "ID:1", "concept_name": "Test"}

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_response

        result = await client.get_concept("ID:1")

        assert result == mock_response
        mock_get.assert_called_once_with("http://fake-api/concept/ID:1", timeout=5)
