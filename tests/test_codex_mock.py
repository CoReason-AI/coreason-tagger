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

    async def test_search_hit(self, codex: MockCoreasonCodex) -> None:
        # Note: The new MockCoreasonCodex implementation in src/coreason_tagger/codex_mock.py
        # might have different data than what was in this test ("Migraine" vs specific set).
        # We should use a concept that exists in the new mock.
        # "SNOMED:22298006": {"concept_id": "SNOMED:22298006", "concept_name": "Myocardial Infarction"}
        results = await codex.search("Myocardial Infarction")
        assert len(results) > 0
        assert results[0]["concept_name"] == "Myocardial Infarction"
        # The new mock doesn't guarantee 'score' key unless added in search method,
        # but the test expects it?
        # Let's check src/coreason_tagger/codex_mock.py again.
        # It just returns the dict from self.concepts. self.concepts entries have id and name.
        # It does NOT have score.
        # Wait, the previous test expected score.
        # I should fix the test to match the implementation.
        # Or fix the implementation to match the test?
        # The prompt replaced codex_mock.py entirely.
        # So I should adapt the test to the new mock.

    async def test_search_synonym(self, codex: MockCoreasonCodex) -> None:
        results = await codex.search("heart attack")
        assert len(results) > 0
        assert results[0]["concept_name"] == "Myocardial Infarction"

    async def test_search_by_id(self, codex: MockCoreasonCodex) -> None:
        """Test searching by exact Concept ID."""
        results = await codex.search("RxNorm:161")
        assert len(results) > 0
        assert results[0]["concept_id"] == "RxNorm:161"
        assert results[0]["concept_name"] == "Acetaminophen"

    async def test_search_miss_but_returns_candidates(self, codex: MockCoreasonCodex) -> None:
        # The mock returns random candidates if no match, to simulate dense retrieval noise
        # unless strict?
        # Implementation: if results, add noise.
        # If NO results (no match), it returns empty?
        # Let's check codex_mock.py logic:
        # 1. Exact ID
        # 2. Synonyms
        # 3. Name match
        # 4. Add noise IF results.
        # So if I search "Zebra" and it's not in concepts, synonyms, or names...
        # It returns empty list?
        # "if results: ... add noise".
        # So "Zebra" -> []
        results = await codex.search("Zebra")
        assert results == []

    async def test_get_concept_exists(self, codex: MockCoreasonCodex) -> None:
        concept = await codex.get_concept("SNOMED:22298006")
        assert concept["concept_name"] == "Myocardial Infarction"

    async def test_get_concept_missing(self, codex: MockCoreasonCodex) -> None:
        concept = await codex.get_concept("INVALID:ID")
        assert concept == {}

    async def test_search_with_limited_top_k(self, codex: MockCoreasonCodex) -> None:
        """Test that search respects top_k limit."""
        # Query that returns hits ("cold") to trigger noise addition
        results = await codex.search("cold", top_k=2)
        # Should contain "Common Cold" and "Chills" (based on mock logic)
        assert len(results) <= 2

    async def test_search_noise_addition(self, codex: MockCoreasonCodex) -> None:
        """Test that noise is added if top_k allows."""
        # Query that matches something but top_k is large
        results = await codex.search("cold", top_k=50)
        # Should have matches + extra noise
        assert len(results) > 2
        # Ensure 'cold' matches are there
        names = [r["concept_name"] for r in results]
        assert "Common Cold" in names
