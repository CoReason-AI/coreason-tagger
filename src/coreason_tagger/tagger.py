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
import time
from typing import Optional, Union

from loguru import logger

from coreason_tagger.interfaces import BaseAssertionDetector, BaseExtractor, BaseLinker
from coreason_tagger.ner import ExtractorFactory
from coreason_tagger.schema import EntityCandidate, ExtractionStrategy, LinkedEntity


class CoreasonTagger:
    """
    The orchestrator for the coreason-tagger pipeline.
    Implements the Extract-Contextualize-Link loop.
    """

    def __init__(
        self,
        ner: Union[BaseExtractor, ExtractorFactory],
        assertion: BaseAssertionDetector,
        linker: BaseLinker,
    ) -> None:
        """
        Initialize the Tagger with its dependencies.

        Args:
            ner: The NER extractor (e.g., GLiNER) OR an ExtractorFactory.
                 If BaseExtractor is passed, it acts as a single-strategy pipeline (legacy support).
                 If ExtractorFactory is passed, it supports dynamic strategy switching.
            assertion: The assertion detector.
            linker: The entity linker.
        """
        self.ner_or_factory = ner
        self.assertion = assertion
        self.linker = linker

    def _get_extractor(self, strategy: ExtractionStrategy) -> BaseExtractor:
        """Helper to resolve the correct extractor."""
        if isinstance(self.ner_or_factory, ExtractorFactory):
            return self.ner_or_factory.get_extractor(strategy)
        return self.ner_or_factory

    async def _process_candidate(
        self, text: str, candidate: EntityCandidate, strategy: ExtractionStrategy
    ) -> Optional[LinkedEntity]:
        """
        Process a single candidate: contextualize (assertion) and link.

        Args:
            text: The full context text.
            candidate: The extracted entity candidate.
            strategy: The extraction strategy used.

        Returns:
            Optional[LinkedEntity]: The processed entity, or None if linking failed.
        """
        # Guard: If span text is empty, it's useless and will fail validation.
        if not candidate.text or not candidate.text.strip():
            return None

        # 2. Contextualize (Assertion)
        start_time = time.monotonic()
        assertion_status = await self.assertion.detect(
            text=text,
            span_text=candidate.text,
            span_start=candidate.start,
            span_end=candidate.end,
        )
        logger.debug(f"Assertion detection for '{candidate.text}' took {(time.monotonic() - start_time) * 1000:.2f}ms")

        # 3. Link (Vector Linking)
        start_time = time.monotonic()
        linked_entity = await self.linker.resolve(candidate, text, strategy)
        logger.debug(f"Linking for '{candidate.text}' took {(time.monotonic() - start_time) * 1000:.2f}ms")

        # If linking fails (concept_id is None), we log a warning but still return the entity
        if not linked_entity.concept_id:
            logger.warning(f"Returning unlinked entity: {linked_entity.text}")

        # Update assertion status (Linker returns default PRESENT)
        linked_entity.assertion = assertion_status

        return linked_entity

    async def _process_candidates(
        self,
        text: str,
        candidates: list[EntityCandidate],
        strategy: ExtractionStrategy,
    ) -> list[LinkedEntity]:
        """
        Helper to process a list of candidates concurrently: contextualize and link.

        Args:
            text: The full context text.
            candidates: The list of extracted candidates.
            strategy: The extraction strategy used.

        Returns:
            list[LinkedEntity]: The list of fully processed entities.
        """
        if not candidates:
            return []

        # Process candidates concurrently
        tasks = [self._process_candidate(text, candidate, strategy) for candidate in candidates]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        return [entity for entity in results if entity is not None]

    async def tag(
        self,
        text: str,
        labels: list[str],
        strategy: ExtractionStrategy = ExtractionStrategy.SPEED_GLINER,
    ) -> list[LinkedEntity]:
        """
        Process text to extract, contextualize, and link entities.

        Args:
            text: The input text.
            labels: The list of labels to extract (passed to NER).
            strategy: The strategy to attribute to the entities.

        Returns:
            list[LinkedEntity]: The list of fully processed entities.
        """
        if not text:
            return []

        logger.info(f"Starting extraction with strategy={strategy.value}")

        # 1. Resolve Extractor
        extractor = self._get_extractor(strategy)

        # 2. Extract (NER)
        start_time = time.monotonic()
        candidates = await extractor.extract(text, labels)
        logger.info(f"Extraction took {(time.monotonic() - start_time) * 1000:.2f}ms")

        # 3. Process Candidates
        return await self._process_candidates(text, candidates, strategy)

    async def tag_batch(
        self,
        texts: list[str],
        labels: list[str],
        strategy: ExtractionStrategy = ExtractionStrategy.SPEED_GLINER,
    ) -> list[list[LinkedEntity]]:
        """
        Process a batch of texts to extract, contextualize, and link entities.
        Optimized for batch NER processing.

        Args:
            texts: The list of input texts.
            labels: The list of labels to extract.
            strategy: The strategy to attribute to the entities.

        Returns:
            list[list[LinkedEntity]]: A list of lists of processed entities, corresponding to the input texts.
        """
        if not texts:
            return []

        logger.info(f"Starting batch extraction with strategy={strategy.value}")

        # 1. Resolve Extractor
        extractor = self._get_extractor(strategy)

        # 2. Batch Extract (NER)
        start_time = time.monotonic()
        batch_candidates = await extractor.extract_batch(texts, labels)
        logger.info(f"Batch Extraction took {(time.monotonic() - start_time) * 1000:.2f}ms")

        # 3. Process Each Text's Candidates
        tasks = [
            self._process_candidates(text, candidates, strategy)
            for text, candidates in zip(texts, batch_candidates, strict=True)
        ]
        batch_results = await asyncio.gather(*tasks)

        return list(batch_results)
