# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

from typing import Any, Dict, List

import httpx


class RealCoreasonCodex:
    """
    Real implementation of the Coreason Codex client.
    Connects to a real database service (e.g. Postgres/Vector).
    Uses httpx for asynchronous HTTP requests.
    """

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for concepts in the real database.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_url}/search",
                params={"q": query, "top_k": str(top_k)},
                timeout=5.0,
            )
            response.raise_for_status()
            return response.json()  # type: ignore

    async def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific concept by ID.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_url}/concept/{concept_id}", timeout=5.0)
            response.raise_for_status()
            return response.json()  # type: ignore
