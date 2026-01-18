# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import functools
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer, util

from coreason_tagger.interfaces import BaseLinker, CodexClient
from coreason_tagger.schema import ExtractedSpan


class VectorLinker(BaseLinker):
    """
    Vector-Based Entity Linker using Bi-Encoders.
    Implements the Candidate Generation -> Semantic Re-ranking pipeline.
    """

    def __init__(
        self,
        codex_client: CodexClient,
        model_name: str = "all-MiniLM-L6-v2",
        window_size: int = 50,
        candidate_top_k: int = 10,
    ) -> None:
        """
        Initialize the Vector Linker.

        Args:
            codex_client: An instance of the codex client.
                          Must strictly implement the CodexClient Protocol.
            model_name (str): The name of the sentence-transformers model to use.
                              Defaults to "all-MiniLM-L6-v2".
            window_size (int): The number of characters to include before and after the entity
                               when constructing the context window for re-ranking. Defaults to 50.
            candidate_top_k (int): The number of candidates to retrieve from Codex. Defaults to 10.
        """
        self.codex_client = codex_client
        self.model_name = model_name
        self.window_size = window_size
        self.candidate_top_k = candidate_top_k

        # Load the model.
        # Note: In production, this should be lazy-loaded or managed by a model registry.
        self.model = SentenceTransformer(model_name)

        # Create an instance-level cache for candidate generation.
        # We cache only the retrieval step, not the re-ranking step,
        # because re-ranking is now context-dependent.
        self._cached_get_candidates = functools.lru_cache(maxsize=1024)(self._get_candidates_impl)

    def _get_candidates_impl(self, text: str) -> List[Dict[str, Any]]:
        """
        Implementation of candidate generation using Codex.
        """
        # Step 1: Candidate Generation (using Codex's search)
        candidates: List[Dict[str, Any]] = self.codex_client.search(text, top_k=self.candidate_top_k)
        return candidates

    def _rerank(self, query_text: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform semantic re-ranking using the Bi-Encoder.
        """
        if not candidates:
            return {}

        # Step 2: Semantic Re-ranking
        # Encode the query (mention OR context)
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)

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

    def link(self, entity: ExtractedSpan) -> Dict[str, Any]:
        """
        Link an extracted entity to a concept in the codex.

        Pipeline:
        1. Candidate Generation: Query the codex (BM25/Sparse) to get top candidates.
        2. Semantic Re-ranking: Encode the mention and candidates using the Bi-Encoder
           and select the best match based on cosine similarity.

        Args:
            entity (ExtractedSpan): The entity to link.

        Returns:
            Dict[str, Any]: The linked concept data, including 'concept_id', 'concept_name',
                            and 'link_confidence'. Returns empty dict if no link found.
        """
        text = entity.text
        if not text:
            return {}

        # Step 1: Get Candidates (Cached based on mention text)
        candidates = self._cached_get_candidates(text)

        if not candidates:
            return {}

        # Step 2: Semantic Re-ranking (Context-Aware)
        # If context is available, we use it for the query embedding to disambiguate.
        # We apply windowing to focus on the immediate context around the mention.
        query_text = text
        if entity.context:
            # Windowing strategy: configurable chars before and after (approx sentence size)
            start_idx = max(0, entity.start - self.window_size)
            end_idx = min(len(entity.context), entity.end + self.window_size)
            query_text = entity.context[start_idx:end_idx]

        return self._rerank(query_text, candidates)
