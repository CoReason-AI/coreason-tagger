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


class MockCoreasonCodex:
    """Mock implementation of the Coreason Codex client for testing and local development.

    Simulates semantic search and concept retrieval.
    """

    def __init__(self) -> None:
        """Initialize the Mock Codex with a pre-defined set of concepts and synonyms."""
        self.concepts = {
            "RxNorm:161": {"concept_id": "RxNorm:161", "concept_name": "Acetaminophen"},
            "RxNorm:4603": {"concept_id": "RxNorm:4603", "concept_name": "Furosemide"},
            "SNOMED:22298006": {"concept_id": "SNOMED:22298006", "concept_name": "Myocardial Infarction"},
            "SNOMED:195967001": {"concept_id": "SNOMED:195967001", "concept_name": "Asthma"},
            "SNOMED:254837009": {"concept_id": "SNOMED:254837009", "concept_name": "Breast Cancer"},
            "SNOMED:73211009": {"concept_id": "SNOMED:73211009", "concept_name": "Diabetes"},
            "SNOMED:38341003": {"concept_id": "SNOMED:38341003", "concept_name": "Hypertension"},
            "SNOMED:49727002": {"concept_id": "SNOMED:49727002", "concept_name": "Cough"},
            "SNOMED:36971009": {"concept_id": "SNOMED:36971009", "concept_name": "Sinusitis"},
            "SNOMED:68566005": {"concept_id": "SNOMED:68566005", "concept_name": "Urinary tract infection"},
            # Added for complex linking tests
            "SNOMED:82272006": {"concept_id": "SNOMED:82272006", "concept_name": "Common Cold"},
            "SNOMED:44077006": {"concept_id": "SNOMED:44077006", "concept_name": "Chills"},
            # Added for missing drugs support (Aspirin, Ibuprofen)
            # Adapting user request to existing schema (concept_id, concept_name)
            "RxNorm:1191": {"concept_id": "RxNorm:1191", "concept_name": "Aspirin"},
            "RxNorm:5640": {"concept_id": "RxNorm:5640", "concept_name": "Ibuprofen"},
            # Ensure Headache and Migraine are present (User listed them)
            "SNOMED:37796009": {"concept_id": "SNOMED:37796009", "concept_name": "Migraine"},
            "SNOMED:25064002": {"concept_id": "SNOMED:25064002", "concept_name": "Headache"},
            "SNOMED:386661006": {"concept_id": "SNOMED:386661006", "concept_name": "Fever"},
            "GEO:BOSTON": {"concept_id": "GEO:BOSTON", "concept_name": "Boston"},
        }
        # Simple synonym map for search simulation
        self.synonyms = {
            "tylenol": "RxNorm:161",
            "paracetamol": "RxNorm:161",
            "lasix": "RxNorm:4603",
            "heart attack": "SNOMED:22298006",
            "mi": "SNOMED:22298006",
            "asthma attack": "SNOMED:195967001",
            "cold": "SNOMED:82272006",  # Default to common cold
            "shivering": "SNOMED:44077006",
            "head ache": "HP:0002315",  # Used in existing tests
            "headache": "HP:0002315",
            "severe headache": "SNOMED:25064002",
            "fever": "SNOMED:386661006",
            "boston": "GEO:BOSTON",
        }
        # Add HP headache as well to support existing tests while adding SNOMED headache
        self.concepts["HP:0002315"] = {"concept_id": "HP:0002315", "concept_name": "Headache"}

    async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Simulate search. Returns exact matches or synonym matches.

        Also returns some random candidates to simulate "dense retrieval noise" for testing re-ranking.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List[Dict[str, Any]]: List of concept dictionaries.
        """
        results = []
        query_lower = query.lower().strip()

        # 1. Check exact ID match
        if query in self.concepts:
            results.append(self.concepts[query])

        # 2. Check Synonyms
        if query_lower in self.synonyms:
            concept_id = self.synonyms[query_lower]
            results.append(self.concepts[concept_id])

        # 3. Check Name match (partial)
        for concept in self.concepts.values():
            if query_lower in str(concept["concept_name"]).lower():
                results.append(concept)

        # 4. Add some "noise" candidates if results are few, to test re-ranking
        # (Only if we have results, to simulate retrieving "related" things)
        # OR if we are searching for "cold", ensure we return both Common Cold and Chills to allow re-ranking
        if "cold" in query_lower:
            results.append(self.concepts["SNOMED:82272006"])  # Common Cold
            results.append(self.concepts["SNOMED:44077006"])  # Chills

        if results:
            for concept in self.concepts.values():
                if concept not in results:
                    results.append(concept)
                    if len(results) >= top_k:
                        break

        # Deduplicate
        seen = set()
        unique_results = []
        for r in results:
            if r["concept_id"] not in seen:
                unique_results.append(r)
                seen.add(r["concept_id"])

        return unique_results[:top_k]

    async def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """Retrieve concept by ID.

        Args:
            concept_id: The ID of the concept to retrieve.

        Returns:
            Dict[str, Any]: The concept data or empty dict if not found.
        """
        if concept_id in self.concepts:
            return self.concepts[concept_id]
        return {}
