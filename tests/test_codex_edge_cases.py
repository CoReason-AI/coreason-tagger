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


@pytest.mark.asyncio
async def test_substring_search() -> None:
    """Test that substring queries return the correct full concept."""
    codex = MockCoreasonCodex()
    # "Asp" is in "Aspirin"
    results = await codex.search("Asp")
    assert len(results) > 0
    # Should find Aspirin (RxNorm:1191)
    assert any(r["concept_name"] == "Aspirin" for r in results)


@pytest.mark.asyncio
async def test_case_insensitivity() -> None:
    """Test that search is case-insensitive."""
    codex = MockCoreasonCodex()
    # Upper case input
    results = await codex.search("IBUPROFEN")
    assert len(results) > 0
    assert any(r["concept_name"] == "Ibuprofen" for r in results)

    # Mixed case input
    results_mixed = await codex.search("iBuPrOfEn")
    assert len(results_mixed) > 0
    assert any(r["concept_name"] == "Ibuprofen" for r in results_mixed)


@pytest.mark.asyncio
async def test_complex_scenario_multi_drug() -> None:
    """
    Test finding multiple distinct entities that exist in the codex.
    Simulates a complex sentence context: "Patient took Aspirin for Headache."
    """
    codex = MockCoreasonCodex()

    # 1. Search for Aspirin
    r1 = await codex.search("Aspirin")
    assert any(r["concept_name"] == "Aspirin" for r in r1)

    # 2. Search for Headache
    r2 = await codex.search("Headache")
    # Should find both SNOMED and HP entries (due to synonym and name match)
    names = [r["concept_name"] for r in r2]
    ids = [r["concept_id"] for r in r2]
    assert "Headache" in names
    # Verify we get valid IDs
    assert "SNOMED:25064002" in ids or "HP:0002315" in ids


@pytest.mark.asyncio
async def test_unknown_entity() -> None:
    """Test searching for an entity that definitely does not exist."""
    codex = MockCoreasonCodex()
    results = await codex.search("Unobtainium")
    # Should return empty list (no matches, so no noise loop trigger logic for 'results' check)
    assert results == []


@pytest.mark.asyncio
async def test_synonym_missing_behavior() -> None:
    """
    Verify behavior for a common synonym NOT currently in the mock.
    'Advil' is a brand name for Ibuprofen, but it is NOT in self.synonyms.
    This test documents the current limitation/behavior (Edge Case).
    """
    codex = MockCoreasonCodex()
    results = await codex.search("Advil")
    # Expect failure because 'Advil' is not in synonyms map nor in concept names
    assert results == []
