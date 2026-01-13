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

from sentence_transformers import SentenceTransformer, util

from coreason_tagger.interfaces import BaseLinker


class VectorLinker(BaseLinker):
    """
    Vector-Based Entity Linker using Bi-Encoders.
    Implements the Candidate Generation -> Semantic Re-ranking pipeline.
    """

    def __init__(self, codex_client: Any, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the Vector Linker.

        Args:
            codex_client: An instance of the codex client (e.g., MockCoreasonCodex).
                          Must have a `search(query: str, top_k: int)` method.
            model_name (str): The name of the sentence-transformers model to use.
                              Defaults to "all-MiniLM-L6-v2" (fast and effective).
        """
        self.codex_client = codex_client
        self.model_name = model_name
        # Load the model.
        # Note: In production, this should be lazy-loaded or managed by a model registry.
        self.model = SentenceTransformer(model_name)

    def link(self, text: str, label: str) -> Dict[str, Any]:
        """
        Link an extracted entity to a concept in the codex.

        Pipeline:
        1. Candidate Generation: Query the codex (BM25/Sparse) to get top candidates.
        2. Semantic Re-ranking: Encode the mention and candidates using the Bi-Encoder
           and select the best match based on cosine similarity.

        Args:
            text (str): The entity text to link (e.g., "severe headaches").
            label (str): The entity label (e.g., "Symptom"). Not used in this basic implementation
                         but useful for filtering context.

        Returns:
            Dict[str, Any]: The linked concept data, including 'concept_id', 'concept_name',
                            and 'link_confidence'. Returns empty dict if no link found.
        """
        if not text:
            return {}

        # Step 1: Candidate Generation (using Codex's search)
        # In a real scenario, we might pass the label to filter candidates (e.g., only Drugs).
        candidates: List[Dict[str, Any]] = self.codex_client.search(text, top_k=10)

        if not candidates:
            return {}

        # Step 2: Semantic Re-ranking
        # Encode the query (mention)
        query_embedding = self.model.encode(text, convert_to_tensor=True)

        # Encode the candidates (definitions/names)
        # We use the 'concept_name' for encoding.
        candidate_names = [str(c.get("concept_name", "")) for c in candidates]
        candidate_embeddings = self.model.encode(candidate_names, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

        # Find the best match
        best_idx = int(cosine_scores.argmax())
        best_score = float(cosine_scores[best_idx])
        best_candidate = candidates[best_idx]

        # Merge the re-ranked score into the result
        # Note: The codex might return a 'score' (BM25), but we overwrite/augment with semantic score.
        result = best_candidate.copy()
        result["link_confidence"] = best_score

        return result
