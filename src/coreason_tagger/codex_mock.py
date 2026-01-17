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
    """
    Mock implementation of the coreason-codex interface.
    This simulates the ontology source and vector retrieval capabilities.
    """

    def __init__(self) -> None:
        self._concepts: Dict[str, Dict[str, Any]] = {
            "SNOMED:37796009": {"name": "Migraine", "embedding": [0.1, 0.2, 0.3]},
            "RxNorm:4603": {"name": "Furosemide", "embedding": [0.4, 0.5, 0.6]},
            "SNOMED:25064002": {"name": "Headache", "embedding": [0.1, 0.2, 0.4]},
            # Ambiguous terms for testing contextual linking
            "SNOMED:82272006": {"name": "Common Cold", "embedding": [0.2, 0.2, 0.2]},  # Infection
            "SNOMED:44077006": {"name": "Chills", "embedding": [0.9, 0.9, 0.9]},  # Sensation
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Simulates a semantic search against the codex.
        Returns a list of candidate concepts.
        """
        results: List[Dict[str, Any]] = []
        query_lower = query.lower()
        for cid, data in self._concepts.items():
            name = str(data["name"])
            name_lower = name.lower()

            # Simple keyword matching logic for the mock
            score = 0.5
            if query_lower in name_lower or name_lower in query_lower:
                score = 0.9

            # Special handling for "Cold" to return both meanings
            if query_lower == "cold":
                if cid in ["SNOMED:82272006", "SNOMED:44077006"]:
                    score = 0.9

            results.append({"concept_id": cid, "concept_name": name, "score": score})
        return sorted(results, key=lambda x: float(x["score"]), reverse=True)[:top_k]

    def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Retrieves a concept by ID.
        """
        return self._concepts.get(concept_id, {})
