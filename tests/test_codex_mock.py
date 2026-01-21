# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import pytest
from coreason_tagger.codex_mock import MockCoreasonCodex


class TestMockCoreasonCodex:
    @pytest.fixture
    def codex(self) -> MockCoreasonCodex:
        return MockCoreasonCodex()

    @pytest.mark.asyncio
    async def test_search_hit(self, codex: MockCoreasonCodex) -> None:
        results = await codex.search("Migraine")
        assert len(results) > 0
        assert results[0]["concept_name"] == "Migraine"
        assert results[0]["score"] > 0.8

    @pytest.mark.asyncio
    async def test_search_miss_but_returns_candidates(self, codex: MockCoreasonCodex) -> None:
        # The mock returns everything but with lower score if not matched
        results = await codex.search("Zebra")
        assert len(results) > 0
        assert results[0]["score"] == 0.5

    @pytest.mark.asyncio
    async def test_get_concept_exists(self, codex: MockCoreasonCodex) -> None:
        concept = await codex.get_concept("SNOMED:37796009")
        assert concept["name"] == "Migraine"

    @pytest.mark.asyncio
    async def test_get_concept_missing(self, codex: MockCoreasonCodex) -> None:
        concept = await codex.get_concept("INVALID:ID")
        assert concept == {}
