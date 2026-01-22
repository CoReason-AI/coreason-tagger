# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_tagger

import asyncio
import json
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from async_lru import alru_cache
from loguru import logger
from sentence_transformers import SentenceTransformer, util

from coreason_tagger.config import settings
from coreason_tagger.interfaces import BaseLinker, CodexClient
from coreason_tagger.registry import get_redis_client, get_sentence_transformer
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy, LinkedEntity
from coreason_tagger.utils.circuit_breaker import CircuitBreaker, CircuitOpenError


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
        self.circuit_breaker = CircuitBreaker()

        # Model is loaded lazily via registry
        self.model: Optional[SentenceTransformer] = None

        # Redis is loaded lazily via registry if URL is set
        self.redis_client: Optional[redis.Redis[Any]] = None

    async def _get_model(self) -> SentenceTransformer:
        """Lazy load model via registry."""
        if self.model is None:
            self.model = await get_sentence_transformer(self.model_name)
        return self.model

    async def _get_redis(self) -> Optional[redis.Redis[Any]]:
        """Lazy load redis client via registry."""
        if self.redis_client is None and settings.REDIS_URL:
            self.redis_client = await get_redis_client(settings.REDIS_URL)
        return self.redis_client

    async def _check_redis_cache(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Check L2 Redis cache."""
        try:
            client = await self._get_redis()
            if not client:
                return None

            key = f"linker:candidates:{text}"
            data = await client.get(key)
            if data:
                return json.loads(data)  # type: ignore
        except Exception as e:
            logger.warning(f"Redis L2 read failed for '{text}': {e}")
        return None

    async def _write_redis_cache(self, text: str, candidates: List[Dict[str, Any]]) -> None:
        """Write to L2 Redis cache."""
        try:
            client = await self._get_redis()
            if not client:
                return

            key = f"linker:candidates:{text}"
            data = json.dumps(candidates)
            await client.set(key, data, ex=settings.REDIS_TTL)
        except Exception as e:
            logger.warning(f"Redis L2 write failed for '{text}': {e}")

    @alru_cache(maxsize=1024)
    async def _get_candidates_impl(self, text: str) -> List[Dict[str, Any]]:
        """
        Implementation of candidate generation using Codex with caching.
        L1 Cache: alru_cache (Memory)
        L2 Cache: Redis
        Fallback: Codex Search (Circuit Breaker protected)
        """

        # L2 Cache Check (Redis)
        cached_candidates = await self._check_redis_cache(text)
        if cached_candidates is not None:
            return cached_candidates

        # Fetch from Source (Codex)
        try:
            candidates: List[Dict[str, Any]] = await self.circuit_breaker.call(
                self.codex_client.search, text, top_k=self.candidate_top_k
            )

            # Write back to L2 Cache
            # (We do this asynchronously but await it to ensure order/consistency, or fire-and-forget?
            # Awaiting is safer for now to avoid complexity, though slightly slower on miss.)
            await self._write_redis_cache(text, candidates)

            return candidates
        except CircuitOpenError:
            logger.warning("Circuit Breaker is OPEN. Skipping candidate generation (Offline Mode).")
            raise  # Re-raise to be handled by caller if needed, or handle here
        except Exception:
            # Other exceptions are already logged/handled by circuit breaker record failure logic
            # but we might want to re-raise them or return empty list
            raise

    async def _rerank(self, query_text: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform semantic re-ranking using the Bi-Encoder.
        """
        if not candidates:
            return {}

        model = await self._get_model()

        # Step 2: Semantic Re-ranking
        loop = asyncio.get_running_loop()

        # Encode the query (mention OR context)
        query_embedding = await loop.run_in_executor(None, lambda: model.encode(query_text, convert_to_tensor=True))

        # Encode the candidates (definitions/names)
        # We use the 'concept_name' for encoding.
        candidate_names = [str(c.get("concept_name", "")) for c in candidates]
        candidate_embeddings = await loop.run_in_executor(
            None, lambda: model.encode(candidate_names, convert_to_tensor=True)
        )

        # Compute cosine similarity
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]

        # Find the best match
        best_idx = int(cosine_scores.argmax())
        best_score = float(cosine_scores[best_idx])
        best_candidate = candidates[best_idx]

        # Merge the re-ranked score into the result
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
        """
        text = entity.text
        if not text:
            return self._build_linked_entity(entity, strategy)

        # Step 1: Get Candidates (Cached)
        try:
            candidates = await self._get_candidates_impl(text)
        except CircuitOpenError:
            # Offline Mode: Return raw entity without linking
            return self._build_linked_entity(entity, strategy)
        except Exception as e:
            logger.error(f"Failed to retrieve candidates for '{text}': {e}")
            return self._build_linked_entity(entity, strategy)

        if not candidates:
            return self._build_linked_entity(entity, strategy)

        # Step 2: Semantic Re-ranking (Context-Aware)
        query_text = text
        if context:
            start_idx = max(0, entity.start - self.window_size)
            end_idx = min(len(context), entity.end + self.window_size)
            query_text = context[start_idx:end_idx]

        best_match = await self._rerank(query_text, candidates)

        if not best_match:  # pragma: no cover
            return self._build_linked_entity(entity, strategy)

        return self._build_linked_entity(entity, strategy, best_match)
