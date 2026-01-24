# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Any, Dict, List, Optional

import httpx


class RealCoreasonCodex:
    """Real implementation of the Coreason Codex client.

    Connects to a real database service (e.g. Postgres/Vector).
    Uses httpx for asynchronous HTTP requests.
    """

    def __init__(self, api_url: str, client: Optional[httpx.AsyncClient] = None) -> None:
        """Initialize the RealCoreasonCodex client.

        Args:
            api_url: The base URL of the Codex API.
            client: Optional external httpx.AsyncClient. If None, one will be created.
        """
        self.api_url = api_url
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

    async def __aenter__(self) -> "RealCoreasonCodex":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._internal_client:
            await self._client.aclose()

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for concepts in the real database.

        Args:
            query: The search query.
            top_k: The number of results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of found concepts.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        response = await self._client.get(
            f"{self.api_url}/search",
            params={"q": query, "top_k": str(top_k)},
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()  # type: ignore

    async def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """Retrieve a specific concept by ID.

        Args:
            concept_id: The ID of the concept to retrieve.

        Returns:
            Dict[str, Any]: The concept data.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        response = await self._client.get(f"{self.api_url}/concept/{concept_id}", timeout=5.0)
        response.raise_for_status()
        return response.json()  # type: ignore
