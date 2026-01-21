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

from async_lru import alru_cache
from sentence_transformers import SentenceTransformer, util

from coreason_tagger.config import settings
from coreason_tagger.interfaces import BaseLinker, CodexClient
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy, LinkedEntity
from coreason_tagger.utils.async_utils import run_blocking


class VectorLinker(BaseLinker):
    """
    Vector-Based Entity Linker using Bi-Encoders.
    Implements the Candidate Generation -> Semantic Re-ranking pipeline.
    """

    def __init__(
        self,
        codex_client: CodexClient,
        model_name: Optional[str] = None,
        window_size: Optional[int] = None,
        candidate_top_k: Optional[int] = None,
    ) -> None:
        """
        Initialize the Vector Linker.

        Args:
            codex_client: An instance of the codex client.
                          Must strictly implement the CodexClient Protocol.
            model_name (str, optional): The name of the sentence-transformers model to use.
                                        If None, uses settings.LINKER_MODEL_NAME.
            window_size (int, optional): The number of characters to include before and after the entity
                                         when constructing the context window for re-ranking.
                                         If None, uses settings.LINKER_WINDOW_SIZE.
            candidate_top_k (int, optional): The number of candidates to retrieve from Codex.
                                             If None, uses settings.LINKER_CANDIDATE_TOP_K.
        """
        self.codex_client = codex_client
        self.model_name = model_name or settings.LINKER_MODEL_NAME
        self.window_size = window_size or settings.LINKER_WINDOW_SIZE
        self.candidate_top_k = candidate_top_k or settings.LINKER_CANDIDATE_TOP_K

        # Load the model.
        # Note: In production, this should be lazy-loaded or managed by a model registry.
        self.model = SentenceTransformer(self.model_name)

    @alru_cache(maxsize=1024)  # type: ignore
    async def _get_candidates_impl(self, text: str) -> List[Dict[str, Any]]:
        """
        Implementation of candidate generation using Codex.
        Results are cached using async_lru.
        """
        # Step 1: Candidate Generation (using Codex's search)
        candidates: List[Dict[str, Any]] = await self.codex_client.search(text, top_k=self.candidate_top_k)
        return candidates

    async def _rerank(self, query_text: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform semantic re-ranking using the Bi-Encoder.
        """
        if not candidates:
            return {}

        # Step 2: Semantic Re-ranking
        # Encode the query (mention OR context)
        query_embedding = await run_blocking(self.model.encode, query_text, convert_to_tensor=True)

        # Encode the candidates (definitions/names)
        # We use the 'concept_name' for encoding.
        candidate_names = [str(c.get("concept_name", "")) for c in candidates]
        candidate_embeddings = await run_blocking(self.model.encode, candidate_names, convert_to_tensor=True)

        # Compute cosine similarity
        # Operations on tensors are fast enough to run in main thread usually,
        # but if we wanted to be super safe we could offload. For now, keep it sync.
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

        # Find the best match
        best_idx = int(cosine_scores.argmax())
        best_score = float(cosine_scores[best_idx])
        best_candidate = candidates[best_idx]

        # Merge the re-ranked score into the result
        # Note: The codex might return a 'score' (BM25), but we overwrite/augment with semantic score.
        result = best_candidate.copy()
        result["link_score"] = best_score

        return result

    def _build_linked_entity(
        self,
        entity: EntityCandidate,
        strategy: ExtractionStrategy,
        match: Optional[Dict[str, Any]] = None,
    ) -> LinkedEntity:
        """
        Helper to construct a LinkedEntity from a candidate and an optional match.
        Reduces code duplication for returning results.
        """
        base_data = entity.model_dump()

        if match:
            return LinkedEntity(
                **base_data,
                strategy_used=strategy,
                concept_id=match.get("concept_id"),
                concept_name=match.get("concept_name"),
                link_score=match.get("link_score", 0.0),
            )

        return LinkedEntity(**base_data, strategy_used=strategy)

    async def resolve(self, entity: EntityCandidate, context: str, strategy: ExtractionStrategy) -> LinkedEntity:
        """
        Link an extracted entity to a concept in the codex.

        Pipeline:
        1. Candidate Generation: Query the codex (BM25/Sparse) to get top candidates.
        2. Semantic Re-ranking: Encode the mention and candidates using the Bi-Encoder
           and select the best match based on cosine similarity.

        Args:
            entity (EntityCandidate): The entity to link.
            context (str): The full context text.
            strategy (ExtractionStrategy): The strategy used for extraction.

        Returns:
            LinkedEntity: The linked entity.
        """
        text = entity.text
        if not text:
            # Return entity without link
            return self._build_linked_entity(entity, strategy)

        # Step 1: Get Candidates
        candidates = await self._get_candidates_impl(text)

        if not candidates:
            return self._build_linked_entity(entity, strategy)

        # Step 2: Semantic Re-ranking (Context-Aware)
        # If context is available, we use it for the query embedding to disambiguate.
        # We apply windowing to focus on the immediate context around the mention.
        query_text = text
        if context:
            # Windowing strategy: configurable chars before and after (approx sentence size)
            start_idx = max(0, entity.start - self.window_size)
            end_idx = min(len(context), entity.end + self.window_size)
            query_text = context[start_idx:end_idx]

        best_match = await self._rerank(query_text, candidates)

        if not best_match:  # pragma: no cover
            return self._build_linked_entity(entity, strategy)

        return self._build_linked_entity(entity, strategy, best_match)
